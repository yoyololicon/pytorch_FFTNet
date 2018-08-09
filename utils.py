import numpy as np
import pyworld as pw
import pysptk as sptk
from librosa import feature
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
    f0, t = pw.harvest(x, sr, f0_floor=minf0, f0_ceil=maxf0, frame_period=frame_period)  # can't adjust window size
    f0 = pw.stonemask(x, f0, t, sr)
    spec = pw.cheaptrick(x, f0, t, sr, f0_floor=minf0, fft_size=int(sr * winlen))
    if spec.min() == 0:
        # prevent overflow in the following log(x)
        spec[np.where(spec == 0)] = 1e-150
    mcep = sptk.sp2mc(spec, n_mcep - 1, alpha)
    h = np.vstack((mcep.T, f0))
    return h


def get_mfcc_and_f0(x, sr, winlen, minf0, maxf0, frame_period, n_mfcc):
    f0, t = pw.harvest(x, sr, f0_floor=minf0, f0_ceil=maxf0, frame_period=frame_period)  # can't adjust window size
    f0 = pw.stonemask(x, f0, t, sr)
    hopsize = int(sr * frame_period / 1000)
    h = feature.mfcc(x, sr, n_mfcc=n_mfcc, n_fft=int(sr * winlen), hop_length=hopsize)
    h = np.vstack((h, f0))
    return h




if __name__ == '__main__':
    x = np.random.rand(3, 5)
    print(x)
    x = zero_padding(x, 10, 1)
    print(x.shape, x)
