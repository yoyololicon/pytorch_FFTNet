import numpy as np
from librosa.feature import mfcc
from librosa.core import load
from scipy.interpolate import interp1d
from aubio import pitch
import torch.nn.init as init
import torch.nn as nn


# import matplotlib.pyplot as plt

def mu_law_transform(x, quantization_channels):
    mu = float(quantization_channels - 1)
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return x_mu


def inv_mu_law_transform(y, quantization_channels):
    mu = float(quantization_channels - 1)
    y_mu = np.sign(y) / mu * (np.power(mu + 1, np.abs(y)) - 1)
    return y_mu


def get_wav_and_feature(filename, n_fft=400, hop_length=160, feature_size=26):
    y, sr = load(filename, sr=None)
    h = mfcc(y, sr, n_mfcc=feature_size-1, n_fft=n_fft, hop_length=hop_length)

    po = pitch("yin", n_fft, hop_length, sr)
    f0 = []

    for pos in range(hop_length, len(y), hop_length):
        samples = y[pos-hop_length:pos]
        p = po(samples.astype(np.float32))[0]
        f0.append(p)

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


def init_weights(m):
    if type(m) == nn.Conv2d:
        N = m.in_channels * np.prod(m.kernel_size)
        m.weight.data.normal_(0., np.sqrt(1 / N))
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        N = m.in_features
        m.weight.data.normal_(0., np.sqrt(1 / N))
        m.bias.data.fill_(0)


if __name__ == '__main__':
    get_wav_and_feature("sample.wav")
