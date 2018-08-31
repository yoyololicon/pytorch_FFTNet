import numpy as np
import torch
from torch.nn import functional as F
from scipy.special import expn
from torchaudio.transforms import MuLawEncoding, MuLawExpanding


def encoder(quantization_channels):
    return MuLawEncoding(quantization_channels)


def decoder(quantization_channels):
    return MuLawExpanding(quantization_channels)


def np_mulaw(x, quantization_channels):
    mu = quantization_channels - 1
    x_mu = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
    return x_mu


def np_inv_mulaw(x, quantization_channels):
    mu = quantization_channels - 1
    x = np.sign(x) * (np.exp(np.abs(x) * np.log1p(mu)) - 1.) / mu
    return x


def float2class(x, classes):
    mu = classes - 1
    return np.rint((x + 1) / 2 * mu).astype(int)


def class2float(x, classes):
    mu = classes - 1
    return x.astype(float) / mu * 2 - 1.


def zero_padding(x, maxlen, dim=0):
    diff = maxlen - x.shape[dim]
    if diff <= 0:
        return x
    else:
        pad_shape = ()
        for i in range(len(x.shape)):
            if i != dim:
                pad_shape += ((0, 0),)
            else:
                pad_shape += ((0, diff),)

        return np.pad(x, pad_shape, 'constant')


def repeat_last_padding(x, maxlen):
    diff = maxlen - x.shape[-1]
    if diff <= 0:
        return x
    else:
        pad_value = np.tile(x[..., [-1]], diff)
        return np.concatenate((x, pad_value), axis=-1)


# this function is copied from https://github.com/braindead/logmmse/blob/master/logmmse.py
# change numpy to tensor

def logmmse(x, sr, noise_std=1 / 256):
    window_size = int(0.02 * sr)

    if window_size % 2 == 1:
        window_size += 1

    # noverlap = len1; hop_size = len2; window_size = len
    noverlap = int(window_size * 0.75)
    hop_size = window_size - noverlap

    win = torch.hann_window(window_size)
    win *= hop_size / win.sum()
    nfft = 2 ** (window_size - 1).bit_length()
    pad_pos = (nfft - window_size) // 2

    noise = torch.randn(6, window_size) * noise_std
    noise_fft = torch.rfft(F.pad(win * noise, (pad_pos, pad_pos)), 1)
    noise_mean = noise_fft.pow(2).sum(-1).sqrt()
    noise_mu = noise_mean.mean(0)
    noise_mu2 = noise_mu.pow(2)

    spec = torch.stft(x, nfft, hop_length=hop_size, win_length=window_size, window=win, center=False)
    spec_copy = spec.clone()
    sig2 = spec.pow(2).sum(-1)

    vad_curve = vad(x, S=spec).float()

    aa = 0.98
    ksi_min = 10 ** (-25 / 10)

    gammak = torch.min(sig2 / noise_mu2.unsqueeze(-1), torch.Tensor([40]))
    for n in range(spec.size(1)):
        gammak_n = gammak[:, n]
        if n == 0:
            ksi = aa + (1 - aa) * F.relu(gammak_n - 1)
        else:
            ksi = aa * spec_copy[:, n - 1].pow(2).sum(-1) / noise_mu2 + (1 - aa) * F.relu(gammak_n - 1)
            ksi = torch.max(ksi, torch.Tensor([ksi_min]))

        A = ksi / (1 + ksi)
        vk = A * gammak_n
        ei_vk = 0.5 * expint(vk)
        hw = A * ei_vk.exp()

        spec_copy[:, n] *= hw.unsqueeze(-1)

    xi_w = torch.irfft(spec_copy.transpose(0, 1), 1, signal_sizes=torch.Size([nfft]))[:, pad_pos:-pad_pos]
    origin = torch.irfft(spec.transpose(0, 1), 1, signal_sizes=torch.Size([nfft]))[:, pad_pos:-pad_pos]

    xi_w_mask = vad_curve / 2 + 0.5
    orign_mask = (1 - vad_curve) / 2

    final_framed = xi_w * xi_w_mask.unsqueeze(-1) + origin * orign_mask.unsqueeze(-1)

    xfinal = torch.zeros(final_framed.size(0) * hop_size + noverlap)
    k = 0
    for n in range(final_framed.size(0)):
        xfinal[k:k + window_size] += final_framed[n]
        k += hop_size
    return xfinal


def expint(x):
    x = x.detach().cpu().numpy()
    x = expn(1, x)
    return torch.from_numpy(x).float()


def vad(x, hop_size=256, S=None, k=5, med_num=9):
    if S is None:
        S = torch.stft(x, hop_size * 4, hop_length=hop_size)
    energy = S.pow(2).sum(-1).mean(0).sqrt()
    energy /= energy.max()

    sorted_E, _ = energy.sort()
    sorted_E_d = sorted_E[2:] - sorted_E[:-2]
    smoothed = F.pad(sorted_E_d, (7, 7)).unfold(0, 15, 1).mean(-1)
    sorted_E_d_peak = F.relu(smoothed[1:-1] - smoothed[:-2]) * F.relu(smoothed[1:-1] - smoothed[2:])

    first, *dummy = torch.nonzero(sorted_E_d_peak) + 2
    E_th = sorted_E[:first].mean() * k
    decision = torch.gt(energy, E_th)

    pad = (med_num // 2, med_num // 2)
    decision = F.pad(decision, pad)
    decision = decision.unfold(0, med_num, 1)
    decision, _ = decision.median(dim=-1)
    return decision
