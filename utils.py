import numpy as np
from torchaudio.transforms import MuLawEncoding, MuLawExpanding
from hparams import hparams

enc = MuLawEncoding(hparams.quantization_channels)
dec = MuLawExpanding(hparams.quantization_channels)

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


if __name__ == '__main__':
    x = np.random.rand(3, 5)
    print(x)
    x = zero_padding(x, 10, 1)
    print(x.shape, x)