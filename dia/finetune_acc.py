import warnings
warnings.filterwarnings("ignore", message="`torch.nn.utils.weight_norm` is deprecated")

import argparse
import logging
import random
from pathlib import Path
import math
import gc

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Subset
# from torch.utils.tensorboard import SummaryWriter # THAY ĐỔI 1: Không cần import này nữa
from transformers import get_scheduler
import bitsandbytes as bnb
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets
from accelerate import Accelerator
from safetensors.torch import save_file

import dac
from .config import DiaConfig
from .layers import DiaModel
from .model import Dia
from .audio import build_delay_indices, apply_audio_delay
from .dataset import HFDiaDataset, HFDiaIterDataset, LocalDiaDataset

# --- Các phần còn lại giữ nguyên ---
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)
torch.backends.cudnn.benchmark = True

LANG2BYTE = { "en": 3, "de": 4, "fr": 5, "es": 6, "it": 7, "nl": 14, "pl": 15, "pt": 16, "tr": 17, "hu": 18 }
test_sentences = {"en": "In order to fully assess performance and the accuracy of language tags, this test sentence contains multiple subordinate clauses, varied punctuation, and a sufficient word count."}

class TrainConfig:
    epochs: int = 500
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    unconditional_frac: float = 0.15
    eval_step: int = 250
    save_step: int = 500
    split_ratio: float = 0.997
    shuffle_buffer_size: int = 10000
    seed: int = 42
    runs_dir: Path = Path("runs")
    run_name: str = "dia_finetune"
    output_dir: Path = Path("checkpoints/dia_finetune")

def get_args() -> argparse.Namespace:
    # Giữ nguyên tuyệt đối hàm get_args của bạn
    parser = argparse.ArgumentParser(description="Fine-tune mô hình Dia với Accelerate")
    parser.add_argument("--config", type=Path, required=True, help="Đường dẫn đến tệp DiaConfig JSON.")
    parser.add_argument("--dataset", type=str, default="Paradoxia/opendata-iisys-hui", help="Tên dataset trên HuggingFace Hub.")
    parser.add_argument("--dataset2", type=str, default=None, help="(Tùy chọn) Dataset thứ hai để interleave (chế độ streaming).")
    parser.add_argument("--streaming", action="store_true", help="Bật chế độ streaming cho dataset.")
    parser.add_argument("--hub_model", type=str, default="nari-labs/Dia-1.6B-0626", help="Tên model trên HuggingFace Hub để tải.")
    parser.add_argument("--csv_path", type=Path, default=None, help="Đường dẫn đến tệp CSV/TSV cục bộ (để train offline).")
    parser.add_argument("--audio_root", type=Path, default=None, help="Thư mục gốc cho audio (bắt buộc nếu dùng --csv_path).")
    parser.add_argument("--run_name", type=str, default=None, help="Tên của lần chạy này (ghi đè mặc định).")
    parser.add_argument("--output_dir", type=Path, default=None, help="Thư mục lưu checkpoint (ghi đè mặc định).")
    parser.add_argument("--shuffle_buffer_size", type=int, default=None, help="Buffer size cho streaming shuffle.")
    parser.add_argument("--seed", type=int, default=42, help="Seed ngẫu nhiên để tái lập kết quả.")
    parser.add_argument("--half", action="store_true", help="Tải model ở chế độ FP16.")
    parser.add_argument("--compile", action="store_true", help="Biên dịch model với torch.compile.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Số lượng mẫu tối đa cần lấy từ dataset để chạy thử. Bỏ qua để dùng toàn bộ dataset."
    )
    return parser.parse_args()

def collate_fn(batch, config: DiaConfig):
    texts, encodings, waveforms = zip(*batch)
    max_text = config.encoder_config.max_position_embeddings
    pad_tok = 0
    text_ids = []
    for txt in texts:
        b_full = txt.encode('utf-8')
        for code, val in LANG2BYTE.items():
            if b_full.startswith(f"[{code}]".encode('utf-8')):
                b_full = bytes([val]) + b_full[len(f"[{code}]"):]
                break
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids)
    max_audio = 860
    seq_lens = [min(e.size(0), max_audio) for e in encodings]
    batch_max = max(seq_lens) if seq_lens else 0
    padded = [F.pad(e, (0, 0, 0, batch_max - e.size(0))) if e.size(0) < batch_max else e[:batch_max] for e in encodings]
    codes = torch.stack(padded)
    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.delay_pattern)
    delayed = apply_audio_delay(codes, config.pad_token_id, config.bos_token_id, (t_idx, idxs))
    delayed = delayed[:, :max_audio, :]
    max_tgt_len = max_audio + 2
    tgt = torch.full((B, max_tgt_len, C), config.pad_token_id, dtype=torch.long)
    tgt[:, 0, :] = config.bos_token_id
    tgt_lens = []
    for i, L in enumerate(seq_lens):
        tgt[i, 1:1 + L, :] = delayed[i, :L, :]
        tgt[i, 1 + L, 0] = config.eos_token_id
        tgt_lens.append(1 + L + 1)
    return {
        'src_tokens': src, 
        'tgt_tokens': tgt, 
        'waveforms': waveforms, 
        'raw_text': texts[0] if texts else "", 
        'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long)
    }


def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig):
    collate = lambda b: collate_fn(b, dia_cfg)
    num_workers = 4
    if isinstance(dataset, Subset) or isinstance(dataset, torch.utils.data.Dataset) and not isinstance(dataset, HFDiaIterDataset):
        ds_len = len(dataset)
        n_train = int(train_cfg.split_ratio * ds_len)
        n_val = ds_len - n_train
        if n_val == 0 and ds_len > 0:
            n_val = 1
            n_train = ds_len - 1
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
    else:
        total = getattr(dataset, "total_examples", 100000)
        n_train = int(train_cfg.split_ratio * total)
        n_val = total - n_train
        base = dataset.dataset.shuffle(buffer_size=train_cfg.shuffle_buffer_size, seed=train_cfg.seed)
        train_ds = HFDiaIterDataset(base.skip(n_val), dia_cfg, dataset.dac_model)
        val_ds = HFDiaIterDataset(base.take(n_val), dia_cfg, dataset.dac_model)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=not isinstance(train_ds, HFDiaIterDataset), collate_fn=collate, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate, num_workers=num_workers)
    return train_loader, val_loader

def train_step(accelerator, model, batch, dia_cfg, train_cfg):
    if random.random() < train_cfg.unconditional_frac:
        batch['src_tokens'].fill_(0)
    logits = model(src_BxS=batch['src_tokens'], tgt_BxTxC=batch['tgt_tokens'])
    lens = batch['tgt_lens']
    max_L = int(lens.max().item())
    logits, target = logits[:,:max_L-1], batch['tgt_tokens'][:,1:max_L,:]
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=dia_cfg.pad_token_id)
    accelerator.backward(loss / train_cfg.grad_accum_steps)
    return loss.item()
def eval_and_generate(accelerator, model, val_loader, dia_cfg, dac_model, global_step):
    model.eval()
    eval_losses = []
    for batch in tqdm(val_loader, desc="Evaluating", disable=not accelerator.is_main_process):
        with torch.no_grad():
            logits = model(src_BxS=batch['src_tokens'], tgt_BxTxC=batch['tgt_tokens'])
            lens = batch['tgt_lens']
            max_L = int(lens.max().item())
            logits, target = logits[:, :max_L - 1], batch['tgt_tokens'][:, 1:max_L, :]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=dia_cfg.pad_token_id)
            eval_losses.append(accelerator.gather(loss).mean().item())
            
    if accelerator.is_main_process:
        avg_eval_loss = sum(eval_losses) / len(eval_losses) if eval_losses else 0
        accelerator.log({"Loss/eval": avg_eval_loss}, step=global_step)
        logger.info(f"Step {global_step}: Eval Loss = {avg_eval_loss:.4f}")

        unwrapped_model = accelerator.unwrap_model(model)
        dia_gen = Dia(config=dia_cfg, device=accelerator.device, load_dac=False)
        dia_gen.model, dia_gen.dac_model = unwrapped_model, dac_model
        with torch.no_grad():
            for lang_code, sentence in test_sentences.items():
                try:
                    audio_np = dia_gen.generate(text=f"[{lang_code}]{sentence}", verbose=False)
                    if audio_np is not None:
                        mono_audio = audio_np.squeeze()
                        if mono_audio.ndim > 1: mono_audio = mono_audio.mean(axis=0)
                        
                        tensorboard_tracker = accelerator.get_tracker("tensorboard")
                        tensorboard_tracker.writer.add_audio(f"Eval/{lang_code}", mono_audio, global_step, 44100)
                except Exception as e: logger.exception(f"Lỗi khi sinh audio mẫu: {e}")
        del unwrapped_model, dia_gen; gc.collect(); torch.cuda.empty_cache()
    model.train()

def train(accelerator, model, dia_cfg, dac_model, dataset, train_cfg):
    if accelerator.is_main_process:
        accelerator.init_trackers(train_cfg.run_name)

    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg)
    
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=train_cfg.learning_rate)
    steps_per_epoch = len(train_loader)
    total_training_steps = steps_per_epoch * train_cfg.epochs
    
    sched = get_scheduler('cosine', opt, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=total_training_steps)
    
    model, opt, train_loader, val_loader, sched = accelerator.prepare(model, opt, train_loader, val_loader, sched)
    model.gradient_checkpointing_enable()

    global_step = 0
    for epoch in range(train_cfg.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", total=steps_per_epoch, disable=not accelerator.is_main_process)
        
        for step, batch in enumerate(pbar):
            with accelerator.accumulate(model):
                loss = train_step(accelerator, model, batch, dia_cfg, train_cfg)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                    sched.step()
                    opt.zero_grad()

                if accelerator.is_main_process:
                    pbar.set_postfix({'loss': f"{loss:.4f}", 'lr': sched.get_last_lr()[0]})
                    accelerator.log({
                        "Loss/train": loss,
                        "LearningRate": sched.get_last_lr()[0]
                    }, step=global_step)

            if (global_step > 0 and (global_step + 1) % train_cfg.eval_step == 0):
                eval_and_generate(accelerator, model, val_loader, dia_cfg, dac_model, global_step)

            if (global_step > 0 and (global_step + 1) % train_cfg.save_step == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    save_directory = train_cfg.output_dir / f"ckpt_step_{global_step}"
                    unwrapped_model.save_pretrained(save_directory, max_shard_size="2GB", safe_serialization=True)
                    logger.info(f"Đã lưu checkpoint (sharded) tại: {save_directory}")
            
            global_step += 1

    accelerator.end_training()
    logger.info("Hoàn tất training.")


def main():
    args = get_args()
    accelerator = Accelerator(gradient_checkpointing=True, log_with="tensorboard")
    
    train_cfg = TrainConfig()
    if args.run_name: train_cfg.run_name = args.run_name
    if args.output_dir: train_cfg.output_dir = args.output_dir
    if args.shuffle_buffer_size: train_cfg.shuffle_buffer_size = args.shuffle_buffer_size
    if args.seed: train_cfg.seed = args.seed
    
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    # Giữ nguyên logic tải model và data của bạn
    logger.info(f"Đang tải model và config từ Hugging Face Hub: {args.hub_model}")
    compute_dtype = torch.float16 if args.half else torch.float32
    
    model = DiaModel.from_pretrained(args.hub_model, compute_dtype=compute_dtype)
    dia_cfg = model.config

    dac_model = dac.DAC.load(dac.utils.download()).to(accelerator.device)

    dataset = None
    if args.csv_path:
        if not args.audio_root: 
            raise ValueError("Phải cung cấp --audio_root khi dùng --csv_path")
        dataset = LocalDiaDataset(args.csv_path, args.audio_root, dia_cfg, dac_model)
        
        if args.num_samples:
            logger.info(f"Đang giảm dataset xuống còn {args.num_samples} mẫu.")
            indices = range(min(args.num_samples, len(dataset)))
            dataset = Subset(dataset, indices)

    elif args.dataset:
        ds1 = load_dataset(args.dataset, split="train", streaming=args.streaming)
        if args.num_samples:
            logger.info(f"Đang giảm dataset xuống còn {args.num_samples} mẫu.")
            if args.streaming:
                ds1 = ds1.take(args.num_samples)
            else:
                ds1 = ds1.select(range(min(args.num_samples, len(ds1))))
        if args.streaming:
            dataset = HFDiaIterDataset(ds1, dia_cfg, dac_model)
            if args.dataset2:
                ds2 = load_dataset(args.dataset2, split="train", streaming=True)
                hf_ds = interleave_datasets([ds1, ds2])
                dataset = HFDiaIterDataset(hf_ds, dia_cfg, dac_model)
        else:
            dataset = HFDiaDataset(ds1, dia_cfg, dac_model)
    else:
        raise ValueError("Bạn phải cung cấp một dataset qua --csv_path hoặc --dataset.")

    if args.compile:
        logger.info("Đang biên dịch model với torch.compile...")
        model = torch.compile(model, backend="inductor")

    logger.info("Bắt đầu quá trình training...")
    train(accelerator, model, dia_cfg, dac_model, dataset, train_cfg)

if __name__ == "__main__":
    main()