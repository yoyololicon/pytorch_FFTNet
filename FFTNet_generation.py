import torch
import torch.backends.cudnn as cudnn
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
from utils import inv_mu_law_transform

parser = argparse.ArgumentParser(description='FFTNet audio generation.')
parser.add_argument('outfile', type=str, help='output file name')
parser.add_argument('--seq_M', type=int, default=2500, help='training sequence length')
parser.add_argument('--depth', type=int, default=10, help='model depth. The receptive field will be 2^depth.')
parser.add_argument('--batch_size', type=int, default=5)
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

    print('==> Downloading YesNo Dataset..')
    transform = transforms.Compose([transforms.Scale(), transforms.MuLawEncoding(quantization_channels=channels)])
    data = torchaudio.datasets.YESNO('./data', download=True, transform=transform)
    data_loader = DataLoader(data, batch_size=1, num_workers=2, shuffle=True)

    print('==> Building model..')
    net = general_FFTNet(radixs, 1, 0, channels, classes=channels)
    net = torch.nn.DataParallel(net.cuda())
    cudnn.benchmark = True

    print(sum(p.numel() for p in net.parameters()), "of parameters.")

    optimizer = optim.Adam(net.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    print("Start Training.")
    a = datetime.now().replace(microsecond=0)

    step = 0
    while step < steps:
        for batch_idx, (inputs, _) in enumerate(data_loader):
            inputs = inputs.view(-1)
            targets = torch.cat((inputs[1:], inputs[0:1]))

            inputs, targets = inputs[:inputs.size(0) // seq_M * seq_M], targets[:targets.size(0) // seq_M * seq_M]

            inputs, targets = inputs.view(-1, 1, seq_M).float().cuda(), targets.view(-1, seq_M).long().cuda()
            inputs = inputs / (channels - 1) * 2 - 1

            # inject guassion noise
            inputs += torch.randn_like(inputs) / channels

            for i in range(0, inputs.size(0), batch_size):
                optimizer.zero_grad()

                logits = net(inputs[i:i + batch_size])[:, :, 1:]
                loss = criterion(logits.unsqueeze(-1), targets[i:i + batch_size].unsqueeze(-1))
                loss.backward()
                optimizer.step()

                print(step, "{:.4f}".format(loss.item()))
                step += 1
                if step > steps:
                    break

    print("Training time cost:", datetime.now().replace(microsecond=0) - a)

    print("Start to generate some noise...")
    a = datetime.now().replace(microsecond=0)
    buffer = torch.zeros(1, 1, N).cuda()
    buffer[:, :, -1].uniform_(-1, 1)
    samples = []
    for i in range(int(generation_time * sr)):
        logit = net(buffer, zeropad=False)
        probs = F.softmax(logit * c, dim=1).view(-1)
        dist = torch.distributions.Categorical(probs)
        sample = dist.sample().float() / (channels - 1) * 2 - 1

        buffer = torch.cat((buffer[:, :, 1:], sample.view(1, 1, 1)), 2)
        samples.append(sample.item())

    generation = inv_mu_law_transform(np.array(samples), channels)
    torchaudio.save(filename, torch.from_numpy(generation).view(-1, 1), sr)
    print("Generation time cost:", datetime.now().replace(microsecond=0) - a)
