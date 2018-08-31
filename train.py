import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import argparse
import os

from preprocess import preprocess_cmu
from models import general_FFTNet
from dataset import CMU_Dataset
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--preprocess', action='store_true')
parser.add_argument('--wav_dir', type=str, default='/host/data_dsk1/dataset/CMU_ARCTIC_Databases/cmu_us_slt_arctic/wav')
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
parser.add_argument('--fft_channels', type=int, default=128, help='fftnet layer channels')
parser.add_argument('--seq_M', type=int, default=5000, help='training sequence length')
parser.add_argument('--radixs', nargs='+', type=int, default=[2] * 11)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--steps', type=int, default=100000, help='iteration number')
parser.add_argument('--injected_noise', action='store_true')
parser.add_argument('--model_file', type=str, default='slt_fftnet')
parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/',
                    help='Directory to save checkpoints.')
parser.add_argument('--checkpoint_step', type=int, default=5000)
parser.add_argument('--transpose', action='store_true')
parser.add_argument('--predict_dist', type=int, default=1)


def main():
    args = parser.parse_args()
    if args.preprocess:
        print('==> Preprocessing data ...')
        preprocess_cmu(args.wav_dir, args.data_dir, q_channels=args.q_channels, winlen=args.window_length,
                       winstep=args.window_step, n_mcep=args.feature_dim, mcep_alpha=args.mcep_alpha,
                       minf0=args.minimum_f0, maxf0=args.maximum_f0, type=args.feature_type)

    print('==> Loading Dataset..')
    training_dataset = CMU_Dataset(args.data_dir, args.seq_M, args.q_channels, int(16000 * args.window_step),
                                   args.interp_method, injected_noise=args.injected_noise,
                                   predict_dist=args.predict_dist)
    training_loader = DataLoader(training_dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

    print('==> Building model..')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = general_FFTNet(radixs=args.radixs, fft_channels=args.fft_channels, classes=args.q_channels,
                         aux_channels=args.feature_dim + 1, transpose=args.transpose,
                         predict_dist=args.predict_dist).to(device)

    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    if device == 'cuda':
        cudnn.benchmark = True

    print(sum(p.numel() for p in net.parameters() if p.requires_grad), "of parameters.")

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    print("Start Training.")
    a = datetime.now().replace(microsecond=0)
    global_step = 0
    while global_step < args.steps:
        for batch_idx, (inputs, targets, features) in enumerate(training_loader):
            inputs, targets, features = inputs.cuda(), targets.cuda(), features.cuda()

            optimizer.zero_grad()

            logits = net(inputs, features)
            loss = criterion(logits.unsqueeze(-1), targets.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            print(global_step, "{:.4f}".format(loss.item()))
            global_step += 1
            if global_step > args.steps:
                break

            if global_step % args.checkpoint_step == 0:
                model = net.module if isinstance(net, torch.nn.DataParallel) else net
                torch.save(model, os.path.join(args.checkpoint_dir, args.model_file + "_{}.pth".format(global_step)))
                print("Checkpoint saved.")

    print("Training time cost:", datetime.now().replace(microsecond=0) - a)

    net = net.module if isinstance(net, torch.nn.DataParallel) else net

    torch.save(net, args.model_file + ".pth")
    print("Model saved to", args.model_file)


if __name__ == '__main__':
    main()
