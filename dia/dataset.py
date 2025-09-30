from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

import dac
from .config import DiaConfig




class LocalDiaDataset(Dataset):
    """Tải từ một tệp CSV cục bộ (phân tách bằng '|') và một thư mục âm thanh."""
    def __init__(self, csv_path: Path, audio_root: Path, config: DiaConfig, dac_model: dac.DAC):
        self.audio_root = audio_root
        self.config = config
        self.dac_model = dac_model
        
        try:
            self.df = pd.read_csv(
                csv_path,
                sep='|',
                header=None,
                on_bad_lines='warn'
            )
            self.df.columns = ['audio', 'text', 'normalized_text']
        except Exception as e:
            print(f"LỖI NGHIÊM TRỌNG KHI ĐỌC TỆP CSV: {e}")
            raise

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        lang = "en"
        text = f"[{lang}]" + row["text"]
        
        audio_path = self.audio_root / f'{row["audio"]}.wav'
        waveform, sr = torchaudio.load(audio_path)
        
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        
        # --- THAY ĐỔI QUAN TRỌNG Ở ĐÂY ---
        # Đảm bảo waveform luôn có 3 chiều: (Batch, Channels, Time)
        if waveform.ndim == 1:
            # Nếu là mono 1D, thêm cả chiều batch và channel
            waveform = waveform.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, Time)
        elif waveform.ndim == 2:
            # Nếu là (Channels, Time), chỉ cần thêm chiều batch
            waveform = waveform.unsqueeze(0)  # Shape: (1, Channels, Time)
        
        with torch.no_grad():
            device = next(self.dac_model.parameters()).device
            waveform = waveform.to(device)
            
            audio_tensor = self.dac_model.preprocess(waveform, 44100)
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)
        return text, encoded, waveform


class HFDiaDataset(Dataset):
    def __init__(self, hf_dataset, config: DiaConfig, dac_model: dac.DAC):
        self.dataset = hf_dataset
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        lang = sample.get("language", None)
        text = f"[{lang}]" + sample["text"] if lang else sample["text"]
        audio_info = sample["audio"]
        waveform = torch.tensor(audio_info["array"], dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0).unsqueeze(0)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        sr = audio_info.get("sampling_rate", 44100)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        with torch.no_grad():
            audio_tensor = (
                self.dac_model.preprocess(waveform, 44100)
                .to(next(self.dac_model.parameters()).device)
            )
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)
        return text, encoded, waveform



class HFDiaIterDataset(torch.utils.data.IterableDataset):
    """Iterable wrapper for a HF streaming Dataset that has `audio.array` & `text`."""
    def __init__(self, hf_iterable, config: DiaConfig, dac_model: dac.DAC):
        super().__init__()
        self.dataset = hf_iterable
        self.config = config
        self.dac_model = dac_model

    def __iter__(self):
        for sample in self.dataset:
            lang = sample.get("language", None)
            text = f"[{lang}]" + sample["text"] if lang else sample["text"]
            audio_info = sample['audio']
            waveform = torch.tensor(audio_info['array'], dtype=torch.float32)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0).unsqueeze(0)
            elif waveform.ndim == 2:
                waveform = waveform.unsqueeze(0)
            sr = audio_info.get('sampling_rate', 44100)
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
            with torch.no_grad():
                audio_tensor = (
                    self.dac_model.preprocess(waveform, 44100)
                    .to(next(self.dac_model.parameters()).device)
                )
                _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
                encoded = encoded.squeeze(0).transpose(0, 1)
            yield text, encoded, waveform
