import numpy as np
from torchaudio.transforms import MuLawEncoding, MuLawExpanding

def encoder(quantization_channels):
    return MuLawEncoding(quantization_channels)


def decoder(quantization_channels):
    return MuLawExpanding(quantization_channels)


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

