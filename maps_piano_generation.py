import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from librosa import load
from librosa.output import write_wav
from datetime import datetime
import matplotlib.pyplot as plt
import pretty_midi
from sklearn.preprocessing import scale, StandardScaler

from models import *
from utils import *

data_dir = '/media/ycy/Shared/Datasets/MAPS/ENSTDkCl/MUS'

files = os.listdir(data_dir)
files = [f[:-4] for f in files if f[-4:] == '.wav']

seq_M = 8000
batch_size = 5
lr = 1e-4
steps = 50000
channels = 256
# depth = 11
# N = 2 ** depth
radixs = [2] * 12
N = np.prod(radixs)
feature_size = 88
c = 2

fs = 100
sr = 16000

train_wav = []
train_feature = []
print("Reading training data...")
for f in files[:1]:
    y, sr = load(os.path.join(data_dir, f + ".wav"), sr=sr)
    h = read_MAPS_txt(os.path.join(data_dir, f + ".txt"), fs=fs)

    # interpolation
    x = np.arange(h.shape[1]) * sr // fs
    f = interp1d(x, h, copy=False)
    y = y[:x[-1]]
    h = f(np.arange(len(y)))

    y = y[:len(y) // seq_M * seq_M]
    h = h[:, :len(y)]
    train_wav += [y]
    train_feature += [h]

train_wav = np.concatenate(train_wav)
train_feature = np.hstack(train_feature)

# mu law mappgin
train_wav = mu_law_transform(train_wav, channels)
# quantize
train_label = np.floor((train_wav + 1) / 2 * (channels - 1)).astype(int)
train_label = np.roll(train_label, shift=-1)
train_feature = np.roll(train_feature, shift=-1, axis=1)

# reshaping
train_wav, train_feature, train_label = train_wav.reshape(-1, 1, seq_M), np.swapaxes(
    train_feature.reshape(feature_size, -1, seq_M), 0, 1), train_label.reshape(-1, seq_M)

# convert to tensor
train_wav, train_feature, train_label = torch.from_numpy(train_wav).float(), torch.from_numpy(
    train_feature).float(), torch.from_numpy(train_label).long()
print(train_wav.size(), train_feature.size(), train_label.size())

# construct data loader
dataset = TensorDataset(train_wav, train_feature, train_label)
loader = DataLoader(dataset, batch_size=batch_size, num_workers=4, shuffle=True)

print('==> Building model..')
net = general_FFTNet_aux(radixs=radixs, feature_width=feature_size, channels=channels, classes=channels).cuda()
net.apply(init_weights)
net = torch.nn.DataParallel(net)
cudnn.bscalerhmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)


if __name__ == '__main__':
    print("Start Training.")

    step = 0
    while step < steps:
        net.train()
        for batch_idx, (inputs, features, targets) in enumerate(loader):
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

