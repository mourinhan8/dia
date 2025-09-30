from pathlib import Path

import torch
import torchaudio
import pandas as pd
from torch.utils.data import Dataset

import dac
from .config import DiaConfig




class LocalDiaDataset(Dataset):
    """Load from a local CSV (sep='|') + an audio folder."""
    def __init__(self, csv_path: Path, audio_root: Path, config: DiaConfig, dac_model: dac.DAC):
        self.df = pd.read_csv(csv_path, sep=r"\|", engine="python",
                      names=["text","audio"], header=None)
        self.audio_root = audio_root
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        lang = row.get("language", None)
        text = f"[{lang}]" + row["text"] if lang else row["text"]
        audio_path = self.audio_root / f'{row["audio"]}.wav'
        waveform, sr = torchaudio.load(audio_path)
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        waveform = waveform.unsqueeze(0)
        with torch.no_grad():
            audio_tensor = self.dac_model.preprocess(
                waveform, 44100
            ).to(next(self.dac_model.parameters()).device)
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
