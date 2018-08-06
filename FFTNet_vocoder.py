import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import torchaudio
from torchaudio import transforms

import numpy as np
from scipy.signal import resample
from scipy.interpolate import interp1d
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import argparse
# import matplotlib.pyplot as plt

from models import general_FFTNet
from python_speech_features import mfcc
from pyworld import dio, harvest

parser = argparse.ArgumentParser(description='FFTNet vocoder.')
parser.add_argument('outfile', type=str, help='output file name')
parser.add_argument('--seq_M', type=int, default=2500, help='training sequence length')
parser.add_argument('--depth', type=int, default=10, help='model depth. The receptive field will be 2^depth.')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--channels', type=int, default=256, help='quantization channels')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--steps', type=int, default=80000, help='iteration number')
parser.add_argument('-c', type=float, default=2., help='a constant multiply before softmax layer in generation')
parser.add_argument('--file_size', type=float, default=5., help='generated wav file size (in seconds)')
parser.add_argument('--feature_size', type=int, default=26, help='generated wav file size (in seconds)')

sr = 8000
winstep = 0.01
winlen = 0.025

if __name__ == '__main__':
    args = parser.parse_args()

    seq_M = args.seq_M
    batch_size = args.batch_size
    depth = args.depth
    radixs = [2] * depth
    N = np.prod(radixs)
    channels = args.channels
    lr = args.lr
    steps = args.steps
    c = args.c
    generation_time = args.file_size
    filename = args.outfile
    features_size = args.feature_size

    print('==> Downloading YesNo Dataset..')
    transform = transforms.Compose([transforms.Scale()])
    data = torchaudio.datasets.YESNO('./data', download=True, transform=transform)
    data_loader = DataLoader(data, batch_size=1, num_workers=2)

    print('==> Extracting features..')
    train_wav = []
    train_features = []
    train_targets = []
    for batch_idx, (inputs, _) in enumerate(data_loader):
        inputs = inputs.view(-1).numpy()
        targets = np.roll(inputs, shift=-1)

        h = mfcc(inputs, sr, winlen=winlen, winstep=winstep, numcep=features_size - 1, winfunc=np.hamming)

        f0, _ = dio(inputs.astype(float), sr, f0_floor=40, f0_ceil=500, frame_period=winstep * 1000)

        if len(f0) > h.shape[0]:
            f0 = f0[:h.shape[0]]
        elif h.shape[0] > len(f0):
            h = h[:len(f0)]
        h = np.hstack((h, f0[:, None]))

        # interpolation
        x = np.arange(h.shape[0]) * winstep * sr
        f = interp1d(x, h, copy=False, axis=0)

        inputs = inputs[:x[-1].astype(int)]
        targets = targets[:x[-1].astype(int)]
        inputs = inputs[:len(inputs) // seq_M * seq_M]
        targets = targets[:len(targets) // seq_M * seq_M]

        h = f(np.arange(1, len(inputs) + 1))

        train_wav.append(inputs)
        train_features.append(h)
        train_targets.append(targets)

    train_wav = np.concatenate(train_wav)
    train_features = np.vstack(train_features)
    train_targets = np.concatenate(train_targets)

    enc = transforms.MuLawEncoding(channels)
    dec = transforms.MuLawExpanding(channels)

    train_wav = enc(train_wav).astype(float) / (channels - 1) * 2 - 1
    train_targets = enc(train_targets)

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_features)

    train_wav = train_wav.reshape(-1, 1, seq_M)
    train_features = np.rollaxis(train_features.reshape(-1, seq_M, features_size), 2, 1)
    train_targets = train_targets.reshape(-1, seq_M)

    train_wav = torch.from_numpy(train_wav).float()
    train_features = torch.from_numpy(train_features).float()
    train_targets = torch.from_numpy(train_targets).long()
    print(train_features.shape, train_wav.shape, train_targets.shape)

    test_features = train_features[:int(sr * generation_time / seq_M)]
    test_features = test_features.transpose(0, 1).contiguous().view(1, features_size, -1).cuda()
    print(test_features.shape)

    print('==> Construct Tensor Dataloader...')
    dataset = TensorDataset(train_wav, train_features, train_targets)
    data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

    print('==> Building model..')
    net = general_FFTNet(radixs, 1, channels, features_size, classes=channels).cuda()

    print(sum(p.numel() for p in net.parameters()), "of parameters.")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print("Start Training.")
    a = datetime.now().replace(microsecond=0)

    step = 0
    while step < steps:
        for batch_idx, (inputs, features, targets) in enumerate(data_loader):
            # inject guassion noise
            inputs += torch.randn_like(inputs) / channels
            inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()

            optimizer.zero_grad()

            logits = net(inputs, features)[:, :, 1:]
            loss = criterion(logits.unsqueeze(-1), targets.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            print(step, "{:.4f}".format(loss.item()))
            step += 1
            if step > steps:
                break

    print("Training time cost:", datetime.now().replace(microsecond=0) - a)

    print("Start to generate some noise...")
    net = net.cpu()
    net.eval()
    with torch.no_grad():
        a = datetime.now().replace(microsecond=0)
        generation = net.fast_generate(h=test_features, c=c)
        generation = dec(generation)
        torchaudio.save(filename, generation, sr)
        cost = datetime.now().replace(microsecond=0) - a
        print("Generation time cost:", cost, ". Speed:", generation.size(0)/cost.total_seconds(), "samples/sec.")