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
from torch.utils.data import DataLoader, random_split
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from transformers import get_scheduler
import bitsandbytes as bnb
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets
from accelerate import Accelerator

import dac
from .config import DiaConfig
from .layers import DiaModel
from .model import Dia
from .audio import build_delay_indices, apply_audio_delay
from .dataset import HFDiaDataset, HFDiaIterDataset, LocalDiaDataset

# --- Cấu hình logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True

# --- Ánh xạ ngôn ngữ sang byte để thay thế tag ---
LANG2BYTE = { "en": 3, "de": 4, "fr": 5, "es": 6, "it": 7, "nl": 14, "pl": 15, "pt": 16, "tr": 17, "hu": 18 }

# --- Các câu mẫu để kiểm tra trong quá trình đánh giá ---
test_sentences = {
    "en": "In order to fully assess performance and the accuracy of language tags, this test sentence contains multiple subordinate clauses, varied punctuation, and a sufficient word count.",
}

# --- Cấu hình cho quá trình training ---
class TrainConfig:
    epochs: int = 50
    batch_size: int = 1
    grad_accum_steps: int = 8
    learning_rate: float = 1e-5
    warmup_steps: int = 500
    unconditional_frac: float = 0.15
    eval_step: int = 200
    save_step: int = 2000
    split_ratio: float = 0.997
    shuffle_buffer_size: int = 10000 # Buffer size mặc định cho streaming shuffle
    seed: int = 42
    runs_dir: Path = Path("runs")
    run_name: str = "dia_finetune"
    output_dir: Path = Path("checkpoints/dia_finetune")

def get_args() -> argparse.Namespace:
    """Xử lý các tham số dòng lệnh."""
    parser = argparse.ArgumentParser(description="Fine-tune mô hình Dia với Accelerate")
    parser.add_argument("--dataset", type=str, default=None, help="Tên dataset trên HuggingFace Hub. Bắt buộc nếu không dùng --csv_path.")
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
    return parser.parse_args()

def collate_fn(batch, config: DiaConfig):
    """
    Hàm xử lý batch, chuyển đổi text và audio thành tensor.
    **Lưu ý**: Đã loại bỏ `.to(device)` để Accelerate tự quản lý.
    """
    texts, encodings, waveforms = zip(*batch)

    # --- Xử lý Text inputs ---
    max_text = config.encoder_config.max_position_embeddings
    pad_tok = 0  # Padding cho text thường là 0
    text_ids = []
    for txt in texts:
        b_full = txt.encode('utf-8')
        for code, val in LANG2BYTE.items():
            prefix = f"[{code}]".encode('utf-8')
            if b_full.startswith(prefix):
                b_full = bytes([val]) + b_full[len(prefix):]
                break
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids)

    # --- Xử lý Audio codes ---
    # max_audio = config.decoder_config.max_position_embeddings
    max_audio = 860
    seq_lens = [min(e.size(0), max_audio) for e in encodings]
    batch_max = max(seq_lens) if seq_lens else 0

    padded = [F.pad(e, (0, 0, 0, batch_max - e.size(0))) if e.size(0) < batch_max else e[:batch_max] for e in encodings]
    codes = torch.stack(padded)

    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.delay_pattern)
    delayed = apply_audio_delay(codes, config.pad_token_id, config.bos_token_id, (t_idx, idxs))
    delayed = delayed[:, :max_audio, :]

    # --- Xử lý Targets (thêm BOS/EOS) ---
    max_tgt_len = max_audio + 2
    tgt = torch.full((B, max_tgt_len, C), config.pad_token_id, dtype=torch.long)
    tgt[:, 0, :] = config.bos_token_id
    tgt_lens = []
    for i, L in enumerate(seq_lens):
        tgt[i, 1:1 + L, :] = delayed[i, :L, :]
        tgt[i, 1 + L, 0] = config.eos_token_id # EOS chỉ ở kênh đầu tiên
        tgt_lens.append(1 + L + 1)

    return {
        'src_tokens': src,
        'tgt_tokens': tgt,
        'waveforms': waveforms,
        'raw_text': texts[0] if texts else "",
        'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long),
    }

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig):
    """Thiết lập DataLoader cho training và validation."""
    collate = lambda b: collate_fn(b, dia_cfg)

    # Xử lý cho dataset streaming
    if isinstance(dataset, HFDiaIterDataset):
        total = getattr(dataset, "total_examples", None)
        try:
            if total is None:
                total = dataset.dataset.info.splits["train"].num_examples
        except Exception:
            logger.warning("Không thể xác định tổng số mẫu cho streaming dataset.")
            total = 100000 # Giả định một số lớn

        n_train = int(train_cfg.split_ratio * total)
        n_val = total - n_train
        if n_val <= 0 : n_val = 100 # Đảm bảo có ít nhất 100 mẫu val

        base = dataset.dataset.shuffle(buffer_size=train_cfg.shuffle_buffer_size, seed=train_cfg.seed) if train_cfg.shuffle_buffer_size else dataset.dataset

        val_stream = base.take(n_val)
        train_stream = base.skip(n_val)

        train_ds = HFDiaIterDataset(train_stream, dia_cfg, dataset.dac_model)
        val_ds = HFDiaIterDataset(val_stream, dia_cfg, dataset.dac_model)

        train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, collate_fn=collate)
        train_loader.steps_per_epoch = math.ceil(n_train / train_cfg.batch_size)
        val_loader = DataLoader(val_ds, batch_size=1, collate_fn=collate)
        return train_loader, val_loader

    # Xử lý cho dataset map-style (tải vào bộ nhớ)
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    train_ds, val_ds = random_split(dataset, [n_train, ds_len - n_train])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    return train_loader, val_loader

def train_step(accelerator, model, batch, dia_cfg, train_cfg):
    """Thực hiện một bước training."""
    # Unconditional conditioning ngẫu nhiên
    if random.random() < train_cfg.unconditional_frac:
        batch['src_tokens'].fill_(0)
        # Tạo mask tương ứng cho unconditional
        src_pad = batch['src_tokens'].ne(0)
        enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)
        batch['enc_self_attn_mask'] = enc_self_attn_mask

    # Tạo các mask cần thiết
    src_pad = batch['src_tokens'].ne(0)
    tgt_pad = batch['tgt_tokens'].ne(dia_cfg.pad_token_id).any(-1)
    max_tgt_len = batch['tgt_tokens'].size(1)
    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len), dtype=torch.bool, device=accelerator.device))

    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)
    dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
    dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)
    
    # Forward pass
    logits = model(
        src_BxS=batch['src_tokens'],
        tgt_BxTxC=batch['tgt_tokens'],
        enc_self_attn_mask=enc_self_attn_mask,
        dec_self_attn_mask=dec_self_attn_mask,
        dec_cross_attn_mask=dec_cross_attn_mask
    )

    lens = batch['tgt_lens']
    max_L = int(lens.max().item())

    logits = logits[:, : max_L - 1]
    target = batch['tgt_tokens'][:, 1:max_L, :]

    B, Tm1, C, V = logits.shape

    # Tạo mask để chỉ tính loss trên các token hợp lệ
    time_idx = torch.arange(Tm1, device=accelerator.device).unsqueeze(0)
    valid_time = time_idx < (lens.unsqueeze(1) - 1)
    mask = valid_time.unsqueeze(-1).expand(-1, -1, C)

    # Tính loss
    loss = F.cross_entropy(logits.view(-1, V), target.view(-1), ignore_index=dia_cfg.pad_token_id)

    # Backward và tối ưu hóa
    loss_for_backward = loss / train_cfg.grad_accum_steps
    accelerator.backward(loss_for_backward)

    return loss.item()

def eval_and_generate(accelerator, model, val_loader, dia_cfg, dac_model, writer, global_step):
    """Đánh giá model trên tập validation và sinh audio mẫu."""
    model.eval()
    eval_losses = []
    
    # Tính loss trên tập validation
    for batch in tqdm(val_loader, desc="Evaluating", disable=not accelerator.is_main_process, total=len(val_loader)):
        with torch.no_grad():
            src_pad = batch['src_tokens'].ne(0)
            tgt_pad = batch['tgt_tokens'].ne(dia_cfg.pad_token_id).any(-1)
            max_tgt_len = batch['tgt_tokens'].size(1)
            causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len), dtype=torch.bool, device=accelerator.device))
            enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)
            dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
            dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

            logits = model(
                src_BxS=batch['src_tokens'], tgt_BxTxC=batch['tgt_tokens'],
                enc_self_attn_mask=enc_self_attn_mask, dec_self_attn_mask=dec_self_attn_mask,
                dec_cross_attn_mask=dec_cross_attn_mask
            )
            lens = batch['tgt_lens']
            max_L = int(lens.max().item())
            logits = logits[:, : max_L - 1]
            target = batch['tgt_tokens'][:, 1:max_L, :]
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1), ignore_index=dia_cfg.pad_token_id)
            eval_losses.append(accelerator.gather(loss).mean().item())

    if accelerator.is_main_process and eval_losses:
        avg_eval_loss = sum(eval_losses) / len(eval_losses)
        writer.add_scalar('Loss/eval', avg_eval_loss, global_step)
        logger.info(f"Step {global_step}: Eval Loss = {avg_eval_loss:.4f}")

    # Sinh audio mẫu
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        dia_gen = Dia(config=dia_cfg, device=accelerator.device, load_dac=False)
        dia_gen.model = unwrapped_model
        dia_gen.dac_model = dac_model
        
        with torch.no_grad():
            for lang_code, sentence in test_sentences.items():
                text = f"[{lang_code}]{sentence}"
                try:
                    audio_np = dia_gen.generate(text=text, verbose=False)
                    if audio_np is not None:
                         writer.add_audio(f"Eval/{lang_code}", audio_np, global_step, 44100)
                except Exception as e:
                     logger.exception(f"Lỗi khi sinh audio mẫu cho ngôn ngữ {lang_code}: {e}")

        del unwrapped_model, dia_gen
        gc.collect()
        torch.cuda.empty_cache()

    model.train()

def train(accelerator, model, dia_cfg, dac_model, dataset, train_cfg):
    """Vòng lặp training chính."""
    if accelerator.is_main_process:
        train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
        (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(train_cfg.runs_dir / train_cfg.run_name))
    else:
        writer = None
    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg)
    
    opt = bnb.optim.AdamW8bit(model.parameters(), lr=train_cfg.learning_rate)
    steps_per_epoch = getattr(train_loader, 'steps_per_epoch', len(train_loader))
    total_training_steps = steps_per_epoch * train_cfg.epochs
    
    sched = get_scheduler('cosine', opt, num_warmup_steps=train_cfg.warmup_steps, num_training_steps=total_training_steps)

    model, opt, train_loader, val_loader, sched = accelerator.prepare(
        model, opt, train_loader, val_loader, sched
    )
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
                    if writer:
                        writer.add_scalar('Loss/train', loss, global_step)
                        writer.add_scalar('LearningRate', sched.get_last_lr()[0], global_step)

            if (global_step > 0 and (global_step + 1) % train_cfg.eval_step == 0):
                eval_and_generate(accelerator, model, val_loader, dia_cfg, dac_model, writer, global_step)

            if (global_step > 0 and (global_step + 1) % train_cfg.save_step == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    unwrapped_model = accelerator.unwrap_model(model)
                    ckpt_path = train_cfg.output_dir / f"ckpt_step_{global_step}.pth"
                    torch.save(unwrapped_model.state_dict(), ckpt_path)
                    logger.info(f"Đã lưu checkpoint: {ckpt_path}")
            
            global_step += 1

    if writer:
        writer.close()
    logger.info("Hoàn tất training.")


def main():
    args = get_args()
    accelerator = Accelerator()
    
    train_cfg = TrainConfig()
    if args.run_name: train_cfg.run_name = args.run_name
    if args.output_dir: train_cfg.output_dir = args.output_dir
    if args.shuffle_buffer_size: train_cfg.shuffle_buffer_size = args.shuffle_buffer_size
    train_cfg.seed = args.seed
    random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    logger.info(f"Đang tải model và config từ Hugging Face Hub: {args.hub_model}")
    compute_dtype = torch.float16 if args.half else torch.float32
    
    model = DiaModel.from_pretrained(args.hub_model, compute_dtype=compute_dtype)
    dia_cfg = model.config

    dac_model = dac.DAC.load(dac.utils.download()).to(accelerator.device)

    # --- Tải dataset ---
    if args.csv_path:
        if not args.audio_root:
            raise ValueError("Phải cung cấp --audio_root khi sử dụng --csv_path")
        dataset = LocalDiaDataset(args.csv_path, args.audio_root, dia_cfg, dac_model)
    elif args.dataset:
        ds1 = load_dataset(args.dataset, split="train", streaming=args.streaming)
        if args.streaming:
            dataset = HFDiaIterDataset(ds1, dia_cfg, dac_model)
            if args.dataset2:
                ds2 = load_dataset(args.dataset2, split="train", streaming=True)
                total = ds1.info.splits['train'].num_examples + ds2.info.splits['train'].num_examples
                hf_ds = interleave_datasets([ds1, ds2])
                dataset = HFDiaIterDataset(hf_ds, dia_cfg, dac_model)
                dataset.total_examples = total
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