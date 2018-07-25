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

# wav_dir = '/media/ycy/Shared/Datasets/cmu_us_rms_arctic/wav'
wav_dir = '/host/data_dsk1/dataset/CMU_ARCTIC_Databases/cmu_us_rms_arctic/wav'

files = sorted(os.listdir(wav_dir))
train_files = files[:1032]
test_files = files[1032:]
decode_file = files[1032]

seq_M = 5000
batch_size = 5
lr = 1e-4
steps = 50000
channels = 256
# depth = 11
# N = 2 ** depth
radixs = [2] * 11
N = np.prod(radixs)
feature_size = 26
c = 2

train_wav = []
train_feature = []
print("Reading training data...")
for f in train_files:
    _, y, h = get_wav_and_feature(os.path.join(wav_dir, f), feature_size=feature_size)

    y = y[:len(y) // seq_M * seq_M]
    h = h[:, :len(y)]
    train_wav += [y]
    train_feature += [h]

train_wav = np.concatenate(train_wav)
train_feature = np.hstack(train_feature).T

# mu law mappgin
train_wav = mu_law_transform(train_wav, channels)
# quantize
train_label = np.floor((train_wav + 1) / 2 * (channels - 1)).astype(int)
train_label = np.roll(train_label, shift=-1)
# normalize
scaler = StandardScaler()
train_feature = scaler.fit_transform(train_feature)
train_feature = np.roll(train_feature, shift=-1, axis=0)

# reshaping
train_wav, train_feature, train_label = train_wav.reshape(-1, 1, seq_M), np.swapaxes(
    train_feature.reshape(-1, seq_M, feature_size), 1, 2), train_label.reshape(-1, seq_M)

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

sr, y, h = get_wav_and_feature(os.path.join(wav_dir, decode_file), feature_size=feature_size)
h = scaler.transform(h.T).T
test_y = mu_law_transform(y[sr:sr + seq_M], channels)
test_h = h[:, sr + 1:sr + seq_M + 1]

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

        net.eval()

        print("docoding...")

        plt.subplot(3, 1, 1)
        plt.plot(test_y)
        plt.ylim(-1, 1)
        plt.xlim(0, seq_M)

        with torch.no_grad():
            inputs, features = torch.Tensor(test_y.reshape(1, 1, -1)).float().cuda(), torch.Tensor(
                test_h.reshape(1, feature_size, -1)).float().cuda()

            logits = net(inputs, features)
            probs = F.softmax(logits * c, dim=1).view(channels, -1).cpu().detach().numpy()

            plt.subplot(3, 1, 2)
            plt.imshow(probs, aspect='auto', origin='lower')

            x_buf = torch.zeros(1, 1, N).cuda()
            h_buf = torch.zeros(1, feature_size, N).cuda()
            img = []
            for i in range(features.size(2)):
                h_buf = torch.cat((h_buf[:, :, 1:], features[:, :, i].view(1, feature_size, 1)), 2)

                logits = net(x_buf, h_buf, zeropad=False)
                _, predict = logits.max(1)

                #sample = predict.float() / (channels - 1) * 2 - 1
                prob = F.softmax(logits * c, dim=1).view(-1)
                dist = torch.distributions.Categorical(prob)
                sample = dist.sample().float() / (channels - 1) * 2 - 1

                img.append(sample.item())
                # img.append(prob.cpu().detach().numpy())
                x_buf = torch.cat((x_buf[:, :, 1:], sample.view(1, 1, 1)), 2)

            plt.subplot(3, 1, 3)
            # img = np.array(img).T
            # plt.imshow(img, aspect='auto', origin='lower')
            plt.plot(img)
            plt.xlim(0, seq_M)
            plt.ylim(-1, 1)

        plt.savefig(str(step) + ".png", dpi=150)
        plt.gcf().clear()
