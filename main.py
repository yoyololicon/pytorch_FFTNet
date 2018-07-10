import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import TensorDataset, DataLoader
import os
import numpy as np
from librosa import load
from librosa.output import write_wav
from datetime import datetime
# import matplotlib.pyplot as plt
from sklearn.preprocessing import scale, StandardScaler

from models import *
from utils import *

#wav_dir = '/media/ycy/Shared/Datasets/cmu_us_rms_arctic/wav'
wav_dir = '/host/data_dsk1/dataset/CMU_ARCTIC_Databases/cmu_us_rms_arctic/wav'

files = os.listdir(wav_dir)
train_files = files[:100]
test_files = files[1032:]
decode_file = files[1032]

seq_M = 5000
batch_size = 5
lr = 0.001
steps = 50000
channels = 256
depth = 11
N = 2 ** depth

train_wav = []
train_feature = []
print("Reading training data...")
for f in train_files:
    _, y, h = get_wav_and_feature(os.path.join(wav_dir, f))

    y = y[:len(y) // seq_M * seq_M]
    h = h[:, :len(y)]
    train_wav += [y]
    train_feature += [h]

train_wav = np.concatenate(train_wav)
train_feature = np.hstack(train_feature).T

# mu law mappgin
train_wav = mu_law_transform(train_wav, channels)
# quantize
train_wav = np.floor((train_wav + 1) / 2 * (channels - 1))
# get label data
train_label = train_wav.astype(int)
# normalize
# train_wav /= channels - 1
# scaler = StandardScaler()
# train_feature = scaler.fit_transform(train_feature)

# reshaping
train_wav, train_feature, train_label = train_wav.reshape(-1, 1, seq_M), np.swapaxes(
    train_feature.reshape(-1, seq_M, 26), 1, 2), train_label.reshape(-1, seq_M)

# padding zeros
train_wav = np.pad(train_wav, ((0, 0), (0, 0), (N, 0)), mode='constant')[:, :, :-1]
train_feature = np.pad(train_feature, ((0, 0), (0, 0), (N - 1, 0)), mode='constant')

# convert to tensor
train_wav, train_feature, train_label = torch.from_numpy(train_wav).float(), torch.from_numpy(
    train_feature).float(), torch.from_numpy(train_label).long()
print(train_wav.size(), train_feature.size(), train_label.size())

# construct data loader
dataset = TensorDataset(train_wav, train_feature, train_label)
loader = DataLoader(dataset, batch_size=batch_size, num_workers=2, shuffle=True)

print('==> Building model..')
net = FFTNet(depth=depth, feature_width=26, channels=channels, classes=channels).cuda()
net = torch.nn.DataParallel(net)
cudnn.bscalerhmark = True

criterion = nn.NLLLoss()
optimizer = optim.Adam(net.parameters(), lr=lr)
# scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

if __name__ == '__main__':
    print("Start Training.")
    a = datetime.now().replace(microsecond=0)

    step = 0
    while step < steps:
        net.train()
        for batch_idx, (inputs, features, targets) in enumerate(loader):
            inputs, features, targets = inputs.cuda(), features.cuda(), targets.cuda()
            optimizer.zero_grad()

            outputs = net(inputs, features)
            loss = criterion(outputs.unsqueeze(-1), targets.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            print(step, "{:.4f}".format(loss.item()))
            step += 1
            if step > steps:
                break
            elif step % 1000 == 0:
                # generation
                print("decode file", decode_file, "from mfcc features...")
                net.eval()

                sr, y, h = get_wav_and_feature(os.path.join(wav_dir, decode_file))
                # h = scaler.transform(h.T).T

                output_buf = []
                x_buf = np.zeros(N)
                h_buf = np.zeros((26, N))
                with torch.no_grad():
                    for i in range(len(y)):
                        features = np.hstack((h_buf[:, 1:], h[:, i, np.newaxis]))[np.newaxis, :]
                        features = torch.Tensor(features).cuda()
                        inputs = torch.Tensor(x_buf[np.newaxis, np.newaxis, :]).cuda()

                        outputs = net(inputs, features)
                        prob = np.squeeze(outputs.exp().cpu().detach().numpy())
                        pred = np.random.choice(channels, p=prob)
                        # pred /= channels - 1
                        x_buf = np.concatenate((x_buf[1:], [pred]))
                        output_buf.append(pred)

                outputs = np.array(output_buf) / (channels - 1) * 2 - 1
                outputs = inv_mu_law_transform(outputs, channels)
                write_wav(str(step) + "_sample.wav", outputs, sr=sr)
                net.train()
