import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

import torchaudio
from torchaudio import transforms

import numpy as np
from datetime import datetime
import argparse
# import matplotlib.pyplot as plt

from models import general_FFTNet

parser = argparse.ArgumentParser(description='FFTNet audio generation.')
parser.add_argument('outfile', type=str, help='output file name')
parser.add_argument('--seq_M', type=int, default=2500, help='training sequence length')
parser.add_argument('--depth', type=int, default=10, help='model depth. The receptive field will be 2^depth.')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--channels', type=int, default=256, help='quantization channels')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--steps', type=int, default=10000, help='iteration number')
parser.add_argument('-c', type=float, default=2., help='a constant multiply before softmax layer in generation')
parser.add_argument('--file_size', type=float, default=5., help='generated wav file size (in seconds)')

sr = 8000

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

    maxlen = 50000
    print('==> Downloading YesNo Dataset..')
    transform = transforms.Compose(
        [transforms.Scale(),
         transforms.PadTrim(maxlen),
         transforms.MuLawEncoding(quantization_channels=channels)])
    data = torchaudio.datasets.YESNO('./data', download=True, transform=transform)
    data_loader = DataLoader(data, batch_size=batch_size, num_workers=4, shuffle=True)

    print('==> Building model..')
    net = general_FFTNet(radixs, 1, channels, classes=channels).cuda()

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print("Start Training.")
    a = datetime.now().replace(microsecond=0)

    step = 0
    seq_idx = torch.arange(seq_M).view(1, -1)
    while step < steps:
        for batch_idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.transpose(1, 2)
            targets = torch.cat((inputs[:, 0, 1:], inputs[:, 0, 0:1]), 1)
            inputs = inputs.float() / (channels - 1) * 2 - 1

            # random sample segments from batch
            randn_idx = torch.LongTensor(inputs.size(0)).random_(maxlen - seq_M)
            randn_seq_idx = seq_idx.expand(inputs.size(0), -1) + randn_idx.unsqueeze(-1)
            inputs = torch.gather(inputs, 2, randn_seq_idx.view(-1, 1, seq_M)).float().cuda()
            targets = torch.gather(targets, 1, randn_seq_idx).long().cuda()

            # inject guassion noise
            inputs += torch.randn_like(inputs) / channels

            optimizer.zero_grad()

            logits = net(inputs)[:, :, 1:]
            loss = criterion(logits.unsqueeze(-1), targets.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            print(step, "{:.4f}".format(loss.item()))
            step += 1
            if step > steps:
                break

            """
            x_sequences = list(torch.split(inputs, seq_M, 2))
            y_sequences = list(torch.split(targets, seq_M, 1))
            for x, y in zip(x_sequences, y_sequences):
                x, y = x.cuda(), y.cuda()
                optimizer.zero_grad()

                logits = net(x)[:, :, 1:]
                loss = criterion(logits.unsqueeze(-1), y.unsqueeze(-1))
                loss.backward()
                optimizer.step()

                print(step, "{:.4f}".format(loss.item()))
                step += 1
                if step > steps:
                    break
            """

    print("Training time cost:", datetime.now().replace(microsecond=0) - a)

    print("Start to generate some noise...")
    net = net.cpu()
    net.eval()
    with torch.no_grad():
        a = datetime.now().replace(microsecond=0)
        generation = net.fast_generate(int(sr * generation_time), c=c)
        decoder = transforms.MuLawExpanding(channels)
        generation = decoder(generation)
        torchaudio.save(filename, generation, sr)
        print("Generation time cost:", datetime.now().replace(microsecond=0) - a)
