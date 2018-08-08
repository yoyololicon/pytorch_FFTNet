import os
import sys
from librosa.core import load
from librosa.feature import mfcc
import pyworld as pw
import pysptk as sptk
import numpy as np
from utils import zero_padding, encoder, get_mcc_and_f0
from sklearn.preprocessing import StandardScaler


def _process_wav(file_list, outfile, winlen, winstep, n_mcep, mcep_alpha, minf0, maxf0, q_channels):
    data_dict = {}
    enc = encoder(q_channels)
    for f in file_list:
        wav, sr = load(f, sr=None)

        # hopsize = int(sr * winstep)
        # spec = mfcc(wav, sr, n_mfcc=n_mfcc, n_fft=int(sr * winlen), hop_length=hopsize)

        x = wav.astype(float)
        h = get_mcc_and_f0(x, sr, winlen, minf0, maxf0, winstep * 1000, n_mcep, mcep_alpha)
        # mulaw encode
        wav = enc(wav).astype(np.uint8)

        id = os.path.basename(f).replace(".wav", "")
        print("reading", id, "...")
        data_dict[id] = wav
        data_dict[id + "_h"] = h
    np.savez(outfile, **data_dict)


def calc_stats(npzfile, out_dir):
    scaler = StandardScaler()
    data_dict = np.load(npzfile)
    for name, x in data_dict.items():
        if name[-2:] == '_h':
            scaler.partial_fit(x.T)

    mean = scaler.mean_
    scale = scaler.scale_

    np.savez(os.path.join(out_dir, 'scaler.npz'), mean=np.float32(mean), scale=np.float32(scale))


def preprocess(wav_dir, output, **kwargs):
    in_dir = os.path.join(wav_dir)
    out_dir = os.path.join(output)
    # print(in_dir, out_dir)
    train_data = os.path.join(out_dir, 'train.npz')
    test_data = os.path.join(out_dir, 'test.npz')
    os.makedirs(out_dir, exist_ok=True)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    files.sort()
    train_files = files[:1032]
    test_files = files[1032:]

    _process_wav(train_files, train_data, **kwargs)
    _process_wav(test_files, test_data, **kwargs)

    calc_stats(train_data, out_dir)
