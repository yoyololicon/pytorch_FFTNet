import torch
from torch.nn import functional as F
from tqdm import tqdm
import os
import argparse
import numpy as np
from torchaudio import save
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from datetime import datetime
from utils import decoder, logmmse, vad
from preprocess import get_features

parser = argparse.ArgumentParser()
parser.add_argument('--infile', type=str, default=None)
parser.add_argument('--outfile', type=str, default=None)
parser.add_argument('--data_dir', type=str, default='slt_mcc_data')
parser.add_argument('--feature_type', type=str, default='mcc')
parser.add_argument('--feature_dim', type=int, default=25, help='number of mcc coefficients')
parser.add_argument('--mcep_alpha', type=float, default=0.42, help='''all-pass filter constant.
                                                                   16khz: 0.42,
                                                                   10khz: 0.35,
                                                                   8khz: 0.31.''')
parser.add_argument('--window_length', type=float, default=0.025)
parser.add_argument('--window_step', type=float, default=0.01)
parser.add_argument('--minimum_f0', type=float, default=71)
parser.add_argument('--maximum_f0', type=float, default=800)
parser.add_argument('--q_channels', type=int, default=256, help='quantization channels')
parser.add_argument('--interp_method', type=str, default='linear')
parser.add_argument('-c', type=float, default=2., help='a constant multiply before softmax.')
parser.add_argument('--model_file', type=str, default='slt_fftnet.pth')
parser.add_argument('--cuda', action='store_true')
parser.add_argument('--denoise', action='store_true')
parser.add_argument('--noise_std', type=float, default=0.005)

sampling_rate = 16000

if __name__ == '__main__':
    args = parser.parse_args()
    net = torch.load(args.model_file)
    scaler = StandardScaler()
    scaler_info = np.load(os.path.join(args.data_dir, 'scaler.npz'))
    scaler.mean_ = scaler_info['mean']
    scaler.scale_ = scaler_info['scale']

    net.eval()
    if not args.cuda:
        net = net.cpu()
    else:
        net = net.cuda()

    print(args.model_file, "has", sum(p.numel() for p in net.parameters() if p.requires_grad), "of parameters.")

    with torch.no_grad():
        if args.infile is None:
            # haven't implement
            pass
        elif args.outfile is not None:
            id, x, h = get_features(args.infile, winlen=args.window_length, winstep=args.window_step,
                                    n_mcep=args.feature_dim, mcep_alpha=args.mcep_alpha, minf0=args.minimum_f0,
                                    maxf0=args.maximum_f0, type=args.feature_type)

            h = scaler.transform(h.T).T
            # interpolation
            hopsize = int(sampling_rate * args.window_step)
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
            r_field = net.get_receptive_field()
            pred_dist = net.get_predict_distance()

            vad_curve = vad(torch.from_numpy(x), hopsize).numpy()
            vad_curve = np.repeat(vad_curve, hopsize)

            output_buf = torch.empty(h.size(2)).long()
            h = F.pad(h, (r_field, 0))
            samples = torch.zeros(pred_dist).long()
            if args.cuda:
                h = h.cuda()
                samples = samples.cuda()

            net.init_buf()
            print("Decoding file", args.infile)
            a = datetime.now().replace(microsecond=0)
            for pos in tqdm(range(r_field + pred_dist, h.size(2) + 1, pred_dist)):
                out_pos = pos - r_field - pred_dist
                decision = np.mean(vad_curve[out_pos:out_pos + pred_dist])
                if decision > 0.5:
                    samples = net.one_sample_generate(samples, h=h[:, :, :pos], c=args.c)
                else:
                    samples = net.one_sample_generate(samples, h=h[:, :, :pos])

                output_buf[out_pos:out_pos + pred_dist] = samples
            cost = datetime.now().replace(microsecond=0) - a
            dec = decoder(args.q_channels)
            generation = dec(output_buf)
            if args.denoise:
                generation = logmmse(generation, sampling_rate, noise_std=args.noise_std)
            save(args.outfile, generation.view(-1, 1), sampling_rate)
            print("Speed:", generation.size(0) / cost.total_seconds(), "samples/sec.")
        else:
            print("Please enter output file name.")
