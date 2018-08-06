import argparse
import os
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from librosa.core import load
from librosa.feature import mfcc
# from python_speech_features import mfcc
from pyworld import dio
import numpy as np
from scipy.interpolate import interp1d

from hparams import hparams
from utils import zero_padding


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
        id = os.path.basename(f).replace(".wav", "")
        print("reading", id, "...")
        data_dict[id] = wav
        data_dict[id + "_h"] = h
    np.savez(outfile, **data_dict)

def build_from_path(in_dir, audio_out_dir, mel_out_dir, num_workers, tqdm=lambda x: x):
    executor = ProcessPoolExecutor(max_workers=num_workers)
    futures = []
    wav_list = os.listdir(in_dir)
    for wav_path in wav_list:
        fid = os.path.basename(wav_path).replace('.wav', '.npy')
        audio_path = os.path.join(audio_out_dir, fid)
        mel_path = os.path.join(mel_out_dir, fid)
        futures.append(executor.submit(partial(_process_wav, wav_path, audio_path, mel_path)))

    return [future.result() for future in tqdm(futures)]


def preprocess(args):
    in_dir = os.path.join(args.wav_dir)
    out_dir = os.path.join(args.output)
    # print(in_dir, out_dir)
    train_data = os.path.join(out_dir, 'train')
    test_data = os.path.join(out_dir, 'test')
    os.makedirs(out_dir, exist_ok=True)

    files = [os.path.join(in_dir, f) for f in os.listdir(in_dir)]
    files.sort()
    train_files = files[:1032]
    test_files = files[1032:]

    _process_wav(train_files, train_data)
    _process_wav(test_files, test_data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default='/media/ycy/Shared/Datasets/cmu_us_rms_arctic/wav')
    parser.add_argument('--output', default='training_data')
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
