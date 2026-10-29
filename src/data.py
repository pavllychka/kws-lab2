import os

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.datasets import SPEECHCOMMANDS

SAMPLE_RATE = 16000
N_SAMPLES = 16000
N_FFT = 512
HOP_LENGTH = 160
N_MELS = 40

_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT,
    hop_length=HOP_LENGTH, n_mels=N_MELS)
_db = torchaudio.transforms.AmplitudeToDB()


def preprocess(wave, sr):
    """wave: tensor [канали, відліки] -> тензор [1, 40, 101]."""
    if sr != SAMPLE_RATE:
        wave = torchaudio.functional.resample(wave, sr, SAMPLE_RATE)
    if wave.shape[0] > 1:
        wave = wave.mean(0, keepdim=True)
    if wave.shape[1] < N_SAMPLES:
        wave = torch.nn.functional.pad(wave, (0, N_SAMPLES - wave.shape[1]))
    else:
        wave = wave[:, :N_SAMPLES]
    spec = _db(_mel(wave))
    return (spec - spec.mean()) / (spec.std() + 1e-6)


def get_root(data_dir):
    root = os.path.join(data_dir, "SpeechCommands", "speech_commands_v0.02")
    # качаємо й розпаковуємо лише якщо датасету ще немає
    if not os.path.exists(os.path.join(root, "testing_list.txt")):
        os.makedirs(data_dir, exist_ok=True)
        SPEECHCOMMANDS(data_dir, download=True)
    return root


def read_list(root, name):
    with open(os.path.join(root, name)) as f:
        return {line.strip() for line in f}


class KWSDataset(Dataset):
    def __init__(self, subset, classes, data_dir="data"):
        self.root = get_root(data_dir)
        val = read_list(self.root, "validation_list.txt")
        test = read_list(self.root, "testing_list.txt")
        self.items = []
        for ci, c in enumerate(classes):
            for fn in sorted(os.listdir(os.path.join(self.root, c))):
                if not fn.endswith(".wav"):
                    continue
                rel = f"{c}/{fn}"
                if subset == "validation":
                    ok = rel in val
                elif subset == "testing":
                    ok = rel in test
                else:
                    ok = rel not in val and rel not in test
                if ok:
                    self.items.append((rel, ci))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        rel, label = self.items[i]
        data, sr = sf.read(os.path.join(self.root, rel), dtype="float32")
        wave = torch.from_numpy(data).unsqueeze(0)
        return preprocess(wave, sr), label