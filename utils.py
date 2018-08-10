import numpy as np
from torchaudio.transforms import MuLawEncoding, MuLawExpanding
from webrtcvad import Vad
from librosa.util import frame


def voice_unvoice(x, sr, winstep=0.01, winlen=0.03, mode=3):
    vad = Vad(mode)
    windowsize = int(sr * winlen)
    hopsize = int(sr * winstep)
    x = np.pad(x, (windowsize // 2,) * 2, 'constant')
    frames = frame(x, frame_length=windowsize, hop_length=hopsize)


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


if __name__ == '__main__':
    x = np.random.rand(3, 5)
    print(x)
    x = zero_padding(x, 10, 1)
    print(x.shape, x)
