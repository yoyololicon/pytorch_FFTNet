import numpy as np
from librosa.feature import mfcc
from librosa.core import load
from scipy.interpolate import interp1d
import torch.nn as nn

import pyworld as pw


# import matplotlib.pyplot as plt

def mu_law_transform(x, quantization_channels):
    mu = float(quantization_channels - 1)
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return x_mu


def inv_mu_law_transform(y, quantization_channels):
    mu = float(quantization_channels - 1)
    y_mu = np.sign(y) / mu * (np.power(mu + 1, np.abs(y)) - 1)
    return y_mu


def get_wav_and_feature(filename, n_fft=400, hop_length=160, feature_size=26, include_f0=True):
    y, sr = load(filename, sr=None)

    n_mfcc = feature_size
    if include_f0:
        n_mfcc -= 1

    h = mfcc(y, sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    if include_f0:
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


def init_weights(m):
    if type(m) == nn.Conv2d:
        N = m.in_channels * np.prod(m.kernel_size)
        m.weight.data.normal_(0., np.sqrt(1 / N))
        m.bias.data.fill_(0)
    elif type(m) == nn.Linear:
        N = m.in_features
        m.weight.data.normal_(0., np.sqrt(1 / N))
        m.bias.data.fill_(0)


def read_MAPS_txt(F0, fs=100):
    f = open(F0)
    lines = f.read().split('\n')
    data = []
    # remove head an tail
    lines.pop()
    lines.pop(0)
    for note in lines:
        t = [float(x) for x in note.split('\t')]
        data.append((t[0], 1, t[2] - 21))
        data.append((t[1], 0, t[2] - 21))
    data.sort()
    result = []
    current = []
    timer = 0.
    step = 1 / fs
    for event in data:
        while timer < event[0]:
            result.append(current[:])
            timer += step
        # print current
        if event[1] == 1:
            current.append(int(event[2]))
        else:
            current.remove(int(event[2]))
    result.append(current[:])

    pianoroll = np.zeros((88, len(result)))
    for i in range(len(result)):
        for note in result[i]:
            pianoroll[note, i] = 1.
    return pianoroll


if __name__ == '__main__':
    get_wav_and_feature("sample.wav")
