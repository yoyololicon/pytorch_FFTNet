import argparse
import os
from multiprocessing import cpu_count
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm
from librosa.core import load
from python_speech_features import mfcc
import numpy as np
from scipy.interpolate import interp1d

from hparams import hparams


def _process_wav(wav_path, audio_path, spc_path):
    wav, sr = load(wav_path, sr=None)
    spc = mfcc(wav, sr, winlen=hparams.winlen, winstep=hparams.winstep, numcep=hparams.n_mfcc,
               winfunc=np.hamming)

    # Align audios and mcc
    hop_length = int(hparams.frame_shift_ms / 1000 * hparams.sample_rate)
    length_diff = len(spc) * hop_length - len(wav)
    wav = wav.reshape(-1, 1)
    if length_diff > 0:
        wav = np.pad(wav, [[0, length_diff], [0, 0]], 'constant')
    elif length_diff < 0:
        wav = wav[: hop_length * spc.shape[0]]

    np.save(audio_path, wav)
    np.save(spc_path, spc)
    return (audio_path, spc_path, spc.shape[0])


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
    audio_out_dir = os.path.join(out_dir, 'audios')
    mcc_out_dir = os.path.join(out_dir, 'mcc')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(mcc_out_dir, exist_ok=True)
    metadata = build_from_path(in_dir, audio_out_dir, mel_out_dir, args.num_workers, tqdm=tqdm)
    write_metadata(metadata, out_dir)

    spc_list = find_files(mel_out_dir, "*.npy")
    calc_stats(spc_list, out_dir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wav_dir', default='cmu_us_rms_arctic/wav')
    parser.add_argument('--output', default='training_data')
    parser.add_argument('--num_workers', type=int, default=cpu_count())
    args = parser.parse_args()
    preprocess(args)


if __name__ == "__main__":
    main()
