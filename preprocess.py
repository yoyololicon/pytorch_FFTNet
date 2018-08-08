import os
import sys
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import repeat
from librosa.core import load
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

        if np.isinf(h).any():
            print(f, "is infinite")
        if np.isnan(h).any():
            print(f, "is nan")

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


def preprocess_multi(wav_dir, output, winlen, winstep, n_mcep, mcep_alpha, minf0, maxf0, q_channels):
    in_dir = os.path.join(wav_dir)
    out_dir = os.path.join(output)
    train_data = os.path.join(out_dir, 'train.npz')
    test_data = os.path.join(out_dir, 'test.npz')
    os.makedirs(out_dir, exist_ok=True)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    files.sort()
    train_files = files[:1032]
    test_files = files[1032:]

    enc = encoder(q_channels)

    pool = Pool(cpu_count())
    print("Running", cpu_count(), "processes.")

    print("Processing training data ...")
    data_dict = {}
    it = pool.starmap(get_features,
                      zip(train_files, repeat(winlen), repeat(winstep), repeat(n_mcep), repeat(mcep_alpha),
                          repeat(minf0), repeat(maxf0)))

    for i, (id, data, feature) in enumerate(tqdm(it, total=len(train_files))):
        data_dict[id] = enc(data)
        data_dict[id + '_h'] = feature
    np.savez(train_data, **data_dict)

    print("Processing test data ...")
    data_dict = {}
    it = pool.starmap(get_features,
                      zip(test_files, repeat(winlen), repeat(winstep), repeat(n_mcep), repeat(mcep_alpha),
                          repeat(minf0), repeat(maxf0)))

    for i, (id, data, feature) in enumerate(tqdm(it, total=len(test_files))):
        data_dict[id] = enc(data)
        data_dict[id + '_h'] = feature
    np.savez(test_data, **data_dict)

    calc_stats(train_data, out_dir)


def get_features(filename, winlen, winstep, n_mcep, mcep_alpha, minf0, maxf0):
    wav, sr = load(filename, sr=None)

    x = wav.astype(float)
    h = get_mcc_and_f0(x, sr, winlen, minf0, maxf0, winstep * 1000, n_mcep, mcep_alpha)
    id = os.path.basename(filename).replace(".wav", "")
    if np.isinf(h).any():
        print(filename, "is infinite")
    if np.isnan(h).any():
        print(filename, "is nan")
    return (id, wav, h)


if __name__ == '__main__':
    preprocess_multi("/media/ycy/Shared/Datasets/cmu_us_rms_arctic/wav", "training_data", winlen=0.025, winstep=0.01,
                     n_mcep=25, mcep_alpha=0.42, minf0=40, maxf0=500,
                     q_channels=256)
