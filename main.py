import matplotlib

matplotlib.use('Agg')
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from librosa import load
from librosa.output import write_wav
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler
from os import listdir, path

from models import *
from utils import *

# wav_dir = '/media/ycy/Shared/Datasets/cmu_us_rms_arctic/wav'
wav_dir = '/host/data_dsk1/dataset/CMU_ARCTIC_Databases/cmu_us_rms_arctic/wav'

files = listdir(wav_dir)
files.sort()
train_files = [path.join(wav_dir, f) for f in files[:1032]]
test_files = [path.join(wav_dir, f) for f in files[1032:]]

seq_M = 5000
batch_size = 5
lr = 5e-5
steps = 100000
channels = 256
n_fft = 400
hop_length = 160
radixs = [2] * 11
N = np.prod(radixs)
feature_size = 26
c = 2

features = []
print("Reading files to fit standard scaler.")
for f in train_files:
    # print(f)
    y, sr = load(f, sr=None)
    h = mfcc(y, sr, n_mfcc=feature_size - 1, n_fft=n_fft, hop_length=hop_length)
    f0, _ = pw.dio(y.astype(float), sr, frame_period=hop_length * 1000 // sr)

    if len(f0) > h.shape[1]:
        f0 = f0[:h.shape[1]]
    elif h.shape[1] > len(f0):
        h = h[:, :len(f0)]
    h = np.vstack((f0, h))
    features += [h]

features = np.hstack(features).T
scaler = StandardScaler()
train_feature = scaler.fit(features)

dataset = CMU_Dataset(train_files, scaler, seq_M, channels)
loader = DataLoader(dataset, num_workers=0, shuffle=True)

print('==> Building model..')
net = general_FFTNet(radixs=radixs, aux_channels=feature_size, channels=channels, classes=channels).cuda()
net = torch.nn.DataParallel(net)
cudnn.bscalerhmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)

print("Start Training.")

step = 0
while step < steps:
    net.train()
    for batch_idx, (inputs, features, targets) in enumerate(loader):
        inputs += torch.randn_like(inputs) / channels
        inputs, features, targets = inputs.view(-1, 1, seq_M).float().cuda(), \
                                    features.view(feature_size, -1, seq_M).transpose(0, 1).float().cuda(), \
                                    targets.view(-1, seq_M).long().cuda()

        for i in range(0, inputs.size(0), batch_size):
            optimizer.zero_grad()

            logits = net(inputs[i:i + batch_size], features[i:i + batch_size])[:, :, 1:]
            loss = criterion(logits.unsqueeze(-1), targets[i:i + batch_size].unsqueeze(-1))
            loss.backward()
            optimizer.step()

            print(step, "{:.4f}".format(loss.item()))
            step += 1
            if step > steps:
                break

    net.eval()
    a = datetime.now().replace(microsecond=0)
    print("decoding file", test_files[0])

    sr, _, features = get_wav_and_feature(test_files[0])
    features = scaler.transform(features.T).T
    features = torch.from_numpy(features).view(1, feature_size, -1).float().cuda()

    x_buf = torch.zeros(1, 1, N).cuda()
    h_buf = torch.zeros(1, feature_size, N).cuda()
    samples = []
    for i in range(features.size(2)):
        h_buf = torch.cat((h_buf[:, :, 1:], features[:, :, i].view(1, feature_size, 1)), 2)

        logits = net(x_buf, h_buf, zeropad=False)
        prob = F.softmax(logits * c, dim=1).view(-1)
        dist = torch.distributions.Categorical(prob)
        sample = dist.sample().float() / (channels - 1) * 2 - 1

        samples.append(sample.item())
        x_buf = torch.cat((x_buf[:, :, 1:], sample.view(1, 1, 1)), 2)

    print("Time cost", datetime.now().replace(microsecond=0) - a)
    samples = inv_mu_law_transform(np.array(samples), channels)
    write_wav(str(step) + ".wav", samples, sr)

"""
    net.eval()

    print("docoding...")

    plt.subplot(4, 1, 1)
    plt.plot(test_y)
    plt.ylim(-1, 1)
    plt.xlim(0, seq_M)

    with torch.no_grad():
        inputs, features = torch.Tensor(test_y.reshape(1, 1, -1)).float().cuda(), torch.Tensor(
            test_h.reshape(1, feature_size, -1)).float().cuda()

        logits = net(inputs, features)
        probs1 = F.softmax(logits, dim=1).transpose(1, 2).view(-1, channels)
        probs2 = F.softmax(logits * c, dim=1).transpose(1, 2).view(-1, channels)

        dist = torch.distributions.Categorical(probs1)
        samples = dist.sample().float() / (channels - 1) * 2 - 1

        plt.subplot(4, 1, 2)
        plt.plot(samples.detach().cpu().numpy())
        plt.xlim(0, seq_M)
        plt.ylim(-1, 1)

        dist = torch.distributions.Categorical(probs2)
        samples = dist.sample().float() / (channels - 1) * 2 - 1

        plt.subplot(4, 1, 3)
        plt.plot(samples.detach().cpu().numpy())
        plt.xlim(0, seq_M)
        plt.ylim(-1, 1)

        x_buf = torch.zeros(1, 1, N).cuda()
        h_buf = torch.zeros(1, feature_size, N).cuda()
        img = []
        for i in range(features.size(2)):
            h_buf = torch.cat((h_buf[:, :, 1:], features[:, :, i].view(1, feature_size, 1)), 2)

            logits = net(x_buf, h_buf, zeropad=False)
            _, predict = logits.max(1)

            # sample = predict.float() / (channels - 1) * 2 - 1
            prob = F.softmax(logits * c, dim=1).view(-1)
            dist = torch.distributions.Categorical(prob)
            sample = dist.sample().float() / (channels - 1) * 2 - 1

            img.append(sample.item())
            # img.append(prob.cpu().detach().numpy())
            x_buf = torch.cat((x_buf[:, :, 1:], sample.view(1, 1, 1)), 2)

        plt.subplot(4, 1, 4)
        # img = np.array(img).T
        # plt.imshow(img, aspect='auto', origin='lower')
        plt.plot(img)
        plt.xlim(0, seq_M)
        plt.ylim(-1, 1)

    plt.savefig(str(step) + ".png", dpi=150)
    plt.gcf().clear()
"""
