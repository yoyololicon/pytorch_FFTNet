import numpy as np
import pyworld as pw
import pysptk as sptk
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


def get_mcc_and_f0(x, sr, winlen, minf0, maxf0, frame_period, n_mcep, alpha):
    f0, t = pw.dio(x, sr, f0_floor=minf0, f0_ceil=maxf0, frame_period=frame_period)  # can't adjust window size
    f0 = pw.stonemask(x, f0, t, sr)
    spec = pw.cheaptrick(x, f0, t, sr, fft_size=int(sr * winlen))
    if spec.min() == 0:
        # prevent overflow in the following log(x)
        spec[np.where(spec == 0)] = 1e-150
        print("have 0")
    mcep = sptk.sp2mc(spec, n_mcep - 1, alpha)

    # in future may need to check the length between spec and f0
    h = np.vstack((mcep.T, f0))
    return h


if __name__ == '__main__':
    x = np.random.rand(3, 5)
    print(x)
    x = zero_padding(x, 10, 1)
    print(x.shape, x)
