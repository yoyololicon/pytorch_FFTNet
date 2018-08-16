import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import mul
from functools import reduce

from random import randint


class One_Hot(nn.Module):
    def __init__(self, depth):
        super().__init__()
        self.depth = depth
        self.ones = nn.Parameter(torch.eye(depth).float(), requires_grad=False)

    def forward(self, x):
        # assume input shape = (batch, seq), output shape = (batch, depth, seq)
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

    def forward(self, x, h=None, zeropad=True):
        M = x.size(-1)
        x = self.pad(x) if zeropad else x

        if h is None:
            z = F.relu(self.W_lr(x))
        else:
            h = self.pad(h[:, :, -M:]) if zeropad else h[:, :, -M:]
            z = F.relu(self.W_lr(x) + self.V_lr(h))
        return F.relu(self.W_o(z))


class general_FFTNet(nn.Module):
    def __init__(self, radixs=[2] * 11, fft_channels=128, classes=256, *, aux_channels=None):
        super().__init__()
        self.channels = fft_channels
        self.aux_channels = aux_channels
        self.classes = classes
        N_seq = [reduce(mul, radixs[i:]) for i in range(len(radixs))]
        self.r_field = N_seq[0]
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

        # only used when generating
        self.padding_layer = nn.ConstantPad1d((self.r_field, 0), 0.)
        self.init_buffer = nn.Parameter(torch.empty(1, self.classes, self.r_field).float(), requires_grad=False)

    def forward(self, x, h=None, zeropad=True):
        x = self.one_hot(x).transpose(1, 2)
        for fft_layer in self.fft_layers:
            x = fft_layer(x, h, zeropad)

        x = self.fc_out(x.transpose(1, 2))
        return x.transpose(1, 2)

    def conditional_sampling(self, logits):
        probs = F.softmax(logits, dim=0)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

    def argmax(self, logits):
        _, sample = logits.max(0)
        return sample

    def class2float(self, category):
        return category.float() / (self.classes - 1) * 2 - 1

    def fast_generate(self, num_samples=None, h=None, c=1, method='sampling'):
        # this method seems only 2~3 times faster
        buf = self.init_buffer.fill_(0.)

        if method == 'argmax':
            predict_fn = self.argmax
        else:
            predict_fn = self.conditional_sampling

        output_list = []
        buf_list = []
        if h is None:
            buf[:, randint(0, self.classes - 1), -1] = 1.
            for fft_layer, r, N in zip(self.fft_layers, self.radixs, self.N_seq):
                buf_list.append(buf[:, :, N // r - 1:])
                buf = fft_layer(buf, zeropad=False)

            # first sample
            logits = self.fc_out(buf.transpose(1, 2)).view(-1) * c
            sample = predict_fn(logits)
            output_list.append(sample.item())
            sample = self.one_hot(sample)

            for i in range(1, num_samples):
                for j in range(len(buf_list)):
                    torch.cat((buf_list[j][:, :, 1:], sample.view(1, -1, 1)), 2, out=buf_list[j])
                    sample = self.fft_layers[j](buf_list[j], zeropad=False)

                logits = self.fc_out(sample.transpose(1, 2)).view(-1) * c
                sample = predict_fn(logits)
                output_list.append(sample.item())
                sample = self.one_hot(sample)
        else:
            h = self.padding_layer(h)
            pos = self.r_field + 1
            for fft_layer, r, N in zip(self.fft_layers, self.radixs, self.N_seq):
                buf_list.append(buf[:, :, N // r - 1:])
                buf = fft_layer(buf, h[:, :, :pos], False)

            # first sample
            logits = self.fc_out(buf.transpose(1, 2)).view(-1) * c
            sample = predict_fn(logits)
            output_list.append(sample.item())
            sample = self.one_hot(sample)

            for pos in range(self.r_field + 2, h.size(2) + 1):
                for j in range(len(buf_list)):
                    torch.cat((buf_list[j][:, :, 1:], sample.view(1, -1, 1)), 2, out=buf_list[j])
                    sample = self.fft_layers[j](buf_list[j], h[:, :, :pos], False)

                logits = self.fc_out(sample.transpose(1, 2)).view(-1) * c
                sample = predict_fn(logits)
                output_list.append(sample.item())
                sample = self.one_hot(sample)

        outputs = torch.Tensor(output_list).view(-1, 1)
        return outputs
