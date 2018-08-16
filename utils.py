import numpy as np
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
