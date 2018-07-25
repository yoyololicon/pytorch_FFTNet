import matplotlib

matplotlib.use('Agg')

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from librosa import load
from librosa.output import write_wav
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler

from models import *
from utils import *

wav_dir = '/media/ycy/Shared/Datasets/cmu_us_rms_arctic/wav'
#wav_dir = '/host/data_dsk1/dataset/CMU_ARCTIC_Databases/cmu_us_rms_arctic/wav'

files = sorted(os.listdir(wav_dir))
train_files = files[:1032]
test_files = files[1032:]
decode_file = files[1032]

seq_M = 5000
batch_size = 5
lr = 0.001
steps = 40000
channels = 256
depth = 11
N = 2 ** depth

train_wav = []
test_wav = []

print("Reading training data...")
for f in train_files:
    y, sr = load(os.path.join(wav_dir, f))
    y = y[:len(y) // seq_M * seq_M]
    train_wav += [y]

train_wav = np.concatenate(train_wav)

# mu law mappgin
train_wav = mu_law_transform(train_wav, channels)
# quantize
train_label = np.floor((train_wav + 1) / 2 * (channels - 1)).astype(int)
# reshaping
train_wav, train_label = train_wav.reshape(-1, 1, seq_M)[:, :, :-1],  train_label.reshape(-1, seq_M)

# convert to tensor
train_wav, train_label = torch.from_numpy(train_wav).float(), torch.from_numpy(train_label).long()

# construct data loader
train_dataset = TensorDataset(train_wav, train_label)
train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=2, shuffle=True)

print('==> Building model..')
models = {"FFTNet": FFTNet(), "FFTNet_inv": FFTNet(inverse=True)}
optimizer = {}
train_loss = {}
test_loss = {}
for name, net in models.items():
    #m.apply(init_weights)
    models[name] = torch.nn.DataParallel(net.cuda())
    optimizer[name] = optim.Adam(models[name].parameters(), lr=lr)
    train_loss[name] = []
    net.train()

cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()

print("Start Training.")
a = datetime.now().replace(microsecond=0)

step = 0
while step < steps:
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        wav_zeropad = torch.zeros(inputs.size(0), 1, N)
        inputs = torch.cat((wav_zeropad, inputs), 2).cuda()
        targets = targets.cuda()

        loss_print = ""
        model_names = ""
        for name, net in models.items():
            optimizer[name].zero_grad()
            logits = net(inputs)
            loss = criterion(logits.unsqueeze(-1), targets.unsqueeze(-1))
            loss.backward()
            optimizer[name].step()

            train_loss[name] += [loss.item()]
            loss_print += str(train_loss[name][-1]) + "/"
            model_names += name + "/"

        print(step, model_names[:-1] + ":", loss_print[:-1])
        step += 1



for name, hist in train_loss.items():
    plt.plot(hist, label=name)
plt.legend()
plt.title("train loss")
plt.savefig("train loss.png", dpi=100)


"""
if step % 5000 == 0:
    # generation
    print("generating...")
    net.eval()

    output_buf = []
    x_buf = torch.zeros(1, 1, N).cuda()

    with torch.no_grad():
        for i in range(sr):
            probs = F.softmax(net(x_buf), dim=1).view(-1)
            dist = torch.distributions.Categorical(probs)

            predict = dist.sample().float() / (channels - 1) * 2 - 1
            output_buf.append(predict.item())
            x_buf = torch.cat((x_buf[:, :, 1:], predict.view(1, 1, 1)), 2)

    logits = inv_mu_law_transform(output_buf, channels)
    write_wav(str(step) + "_sample.wav", logits, sr=sr)
    net.train()
"""