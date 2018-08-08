import torch
import argparse
import os
from librosa.core import load
from sklearn.preprocessing import StandardScaler
import numpy as np
from torchaudio import save
from datetime import datetime
from scipy.interpolate import interp1d

from utils import get_mcc_and_f0, decoder

parser = argparse.ArgumentParser(description='FFTNet decoder.')
parser.add_argument('--infile', type=str, default=None)
parser.add_argument('--outfile', type=str, default=None)
parser.add_argument('--data_dir', type=str, default='training_data')
parser.add_argument('--num_mcep', type=int, default=25, help='number of mcc coefficients')
parser.add_argument('--mcep_alpha', type=float, default=0.42, help='all-pass filter constant.'
                                                                   '16khz: 0.42,'
                                                                   '10khz: 0.35,'
                                                                   '8khz: 0.31.')
parser.add_argument('--window_length', type=float, default=0.025)
parser.add_argument('--window_step', type=float, default=0.01)
parser.add_argument('--minimum_f0', type=float, default=40)
parser.add_argument('--maximum_f0', type=float, default=500)
parser.add_argument('--q_channels', type=int, default=256, help='quantization channels')
parser.add_argument('--interp_method', type=str, default='linear')
parser.add_argument('-c', type=float, default=2., help='a constant multiply before softmax.')
parser.add_argument('--model_file', type=str, default='fftnet_model.pth')

if __name__ == '__main__':
    args = parser.parse_args()
    net = torch.load("fftnet_model.pth")
    scaler = StandardScaler()
    scaler_info = np.load(os.path.join(args.data_dir, 'scaler.npz'))
    scaler.mean_ = scaler_info['mean']
    scaler.scale_ = scaler_info['scale']

    net.eval()
    net = net.cpu()
    with torch.no_grad():
        if args.infile is None:
            pass
        elif args.outfile is not None:
            wav, sr = load(args.infile, sr=None)
            x = wav.astype(float)
            h = get_mcc_and_f0(x, sr, args.window_length, args.minimum_f0, args.maximum_f0, args.window_step * 1000,
                               args.num_mcep,
                               args.mcep_alpha)
            h = scaler.transform(h.T).T

            # interpolation
            hopsize = int(sr * args.window_step)
            if args.interp_method == 'linear':
                xx = np.arange(h.shape[1]) * hopsize
                f = interp1d(xx, h, copy=False, axis=1)
                h = f(np.arange(xx[-1]))
            elif args.interp_method == 'repeat':
                h = np.repeat(h, hopsize, axis=1)
            else:
                print("interpolation method", args.interp_method, "is not implemented.")
                exit(1)

            h = torch.from_numpy(h).unsqueeze(0).float()
            print("Decoding file", args.infile)
            a = datetime.now().replace(microsecond=0)
            generation = net.fast_generate(h=h, c=args.c)

            dec = decoder(args.q_channels)
            generation = dec(generation)
            save(args.outfile, generation, sr)
            cost = datetime.now().replace(microsecond=0) - a
            print("Generation time cost:", cost)
            print("Speed:", generation.size(0) / cost.total_seconds(), "samples/sec.")
