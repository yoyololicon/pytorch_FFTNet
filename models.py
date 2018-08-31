import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import mul
from functools import reduce


class One_Hot(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.ones = nn.Parameter(torch.eye(depth).float(), requires_grad=False)

    def forward(self, x):
        return self.ones.index_select(0, x.view(-1)).view(x.size() + torch.Size([self.depth]))


class general_FFTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, N, *, radix=2, aux_channels=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radix = radix
        self.aux_channels = aux_channels

        # this should mathematically equal to having 2 1x1 kernel
        self.W_lr = nn.Conv1d(in_channels, out_channels, kernel_size=radix, dilation=N // radix)
        if aux_channels is not None:
            self.V_lr = nn.Conv1d(aux_channels, out_channels, kernel_size=radix, dilation=N // radix)
        self.W_o = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.pad = nn.ConstantPad1d((N - N // radix, 0), 0.)

    def forward(self, x, h=None, zeropad=True, input_onehot=False):
        M = x.size(-1)
        x = self.pad(x) if zeropad else x
        if input_onehot:
            x[:, self.in_channels // 2, :x.size(2) - M] = 1

        if h is None:
            z = F.relu(self.W_lr(x))
        else:
            h = self.pad(h[:, :, -M:]) if zeropad else h[:, :, -M:]
            z = F.relu(self.W_lr(x) + self.V_lr(h))
        return F.relu(self.W_o(z))


class general_FFTNet(nn.Module):
    def __init__(self, radixs=[2] * 11, fft_channels=128, classes=256, *, aux_channels=None, transpose=False,
                 predict_dist=1):
        super().__init__()
        self.channels = fft_channels
        self.aux_channels = aux_channels
        self.classes = classes
        self.predict_dist = predict_dist
        if transpose:
            N_seq = [reduce(mul, radixs[:i + 1]) for i in range(len(radixs))]
        else:
            N_seq = [reduce(mul, radixs[i:]) for i in range(len(radixs))]
        self.r_field = reduce(mul, radixs)
        self.radixs = radixs
        self.N_seq = N_seq

        # transform input to one hot
        self.one_hot = One_Hot(classes)

        self.fft_layers = nn.ModuleList()
        in_channels = classes
        for N, r in zip(N_seq, radixs):
            self.fft_layers.append(general_FFTLayer(in_channels, fft_channels, N, radix=r, aux_channels=aux_channels))
            in_channels = fft_channels
        self.fc_out = nn.Linear(in_channels, classes)

    def forward(self, x, h=None, zeropad=True):
        x = self.one_hot(x).transpose(1, 2)
        first_layer = True

        for fft_layer in self.fft_layers:
            x = fft_layer(x, h, zeropad, first_layer)
            first_layer = False

        x = self.fc_out(x.transpose(1, 2))
        return x.transpose(1, 2)

    def get_receptive_field(self):
        return self.r_field

    def get_predict_distance(self):
        return self.predict_dist

    def conditional_sampling(self, logits):
        probs = F.softmax(logits, dim=1)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

    def argmax(self, logits):
        _, sample = logits.max(1)
        return sample

    def init_buf(self):
        if next(self.parameters()).is_cuda:
            device = 'cuda'
        else:
            device = 'cpu'

        if hasattr(self, "buffers"):
            for buf in self.buffers:
                buf.fill_(0.).to(device)
        else:
            self.buffers = [
                torch.zeros(1, self.classes,
                            self.N_seq[0] - self.N_seq[0] // self.radixs[0] + self.predict_dist).float().to(device)]
            self.buffers += [torch.zeros(1, self.channels, N - N // r + self.predict_dist).float().to(device) for N, r
                             in zip(self.N_seq[1:], self.radixs[1:])]
        self.buffers[0][:, self.classes // 2] = 1

    def one_sample_generate(self, samples, h=None, c=1., method='sampling'):
        samples = self.one_hot(samples).t()
        for i in range(len(self.buffers)):
            torch.cat((self.buffers[i][:, :, self.predict_dist:], samples.view(1, -1, self.predict_dist)), 2,
                      out=self.buffers[i])
            samples = self.fft_layers[i](self.buffers[i], h, False)

        logits = self.fc_out(samples.transpose(1, 2)).view(self.predict_dist, self.classes) * c
        if method == 'argmax':
            samples = self.argmax(logits)
        else:
            samples = self.conditional_sampling(logits)
        return samples