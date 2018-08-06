import argparse
import os
from librosa.core import load
from librosa.feature import mfcc
# from python_speech_features import mfcc
from pyworld import dio
import numpy as np
from hparams import hparams
from utils import zero_padding, enc
from sklearn.preprocessing import StandardScaler


def _process_wav(file_list, outfile):
    data_dict = {}
    for f in file_list:
        wav, sr = load(f, sr=None)

        hopsize = int(sr * hparams.winstep)
        spec = mfcc(wav, sr, n_mfcc=hparams.n_mfcc, n_fft=int(sr * hparams.winlen), hop_length=hopsize)
        f0, _ = dio(wav.astype(float), sr, f0_floor=hparams.minf0, f0_ceil=hparams.maxf0,
                    frame_period=hparams.winstep * 1000)

        # in future may need to check the lenght between spec and f0
        h = np.vstack((spec, f0))
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

    np.save(os.path.join(out_dir, 'mean'), np.float32(mean))
    np.save(os.path.join(out_dir, 'scale'), np.float32(scale))


def preprocess(args):
    in_dir = os.path.join(args.wav_dir)
    out_dir = os.path.join(args.output)
    # print(in_dir, out_dir)
    train_data = os.path.join(out_dir, 'train.npz')
    test_data = os.path.join(out_dir, 'test.npz')
    os.makedirs(out_dir, exist_ok=True)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    files.sort()
    train_files = files[:1032]
    test_files = files[1032:]

    _process_wav(train_files, train_data)
    _process_wav(test_files, test_data)

    calc_stats(train_data, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default='/host/data_dsk1/dataset/CMU_ARCTIC_Databases/cmu_us_rms_arctic/wav')
    parser.add_argument('--output', default='training_data')
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
    # calc_stats("training_data/train.npz", "training_data")
