import numpy as np
from librosa.feature import mfcc
from librosa.core import load
from scipy.interpolate import interp1d
from torch.utils.data import Dataset

import pyworld as pw

def np_mulaw_encode(x, quantization_channels):
    if x.max() > 1 or x.min() < -1:
        print("mulaw encode: input value out of range.")
        return 1
    mu = float(quantization_channels - 1)
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    x_mu = np.floor((x_mu+1)*mu/2)
    return x_mu


def np_mulaw_decode(y, quantization_channels):
    mu = float(quantization_channels - 1)
    if y.max() > mu or y.min() < 0:
        print("mulaw decode: input value out of range.")
        return 1
    y_mu = (y/mu*2)-1
    y_mu = np.sign(y) / mu * (np.power(mu + 1, np.abs(y)) - 1)
    return y_mu


def get_wav_and_feature(filename, n_fft=400, hop_length=160, feature_size=26):
    y, sr = load(filename, sr=None)
    n_mfcc = feature_size - 1
    h = mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    f0, _ = pw.dio(y.astype(float), sr, frame_period=hop_length * 1000 // sr)

    if len(f0) > h.shape[1]:
        f0 = f0[:h.shape[1]]
    elif h.shape[1] > len(f0):
        h = h[:, :len(f0)]
    h = np.vstack((f0, h))

    # interpolation
    x = np.arange(h.shape[1]) * 160
    f = interp1d(x, h, copy=False)
    y = y[:x[-1]]
    h = f(np.arange(len(y)))
    return sr, y, h


class CMU_Dataset(Dataset):
    def __init__(self, filelist, feature_scaler, seq_M=5000, channels=256):
        self.data_files = filelist
        self.seq_M = seq_M
        self.channels = channels
        self.scaler = feature_scaler

    def __getitem__(self, idx):
        _, y, h = get_wav_and_feature(self.data_files[idx])
        y = y[:len(y) // self.seq_M * self.seq_M]
        h = h[:, :len(y)]

        y = mu_law_transform(y, self.channels)

        h = self.scaler.transform(h.T).T
        h = np.roll(h, shift=-1, axis=1)

        t = np.floor((y + 1) / 2 * (self.channels - 1))
        #y = t / (self.channels - 1) * 2 - 1
        t = np.roll(t, shift=-1).astype(int)
        return (y, h, t)

    def __len__(self):
        return len(self.data_files)
