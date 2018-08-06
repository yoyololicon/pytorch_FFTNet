import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import mul
from functools import reduce


class general_FFTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, N, radix=2, aux_channels=None):
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

    def forward(self, x, h=None):
        if h is None:
            z = F.relu(self.W_lr(x))
        else:
            z = F.relu(self.W_lr(x) + self.V_lr(h[:, :, -x.size(-1):]))
        return F.relu(self.W_o(z))


class general_FFTNet(nn.Module):
    def __init__(self, radixs=[2] * 11, in_channels=1, channels=256, aux_channels=None, classes=256):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.aux_channels = aux_channels
        self.classes = classes
        N_seq = [reduce(mul, radixs[i:]) for i in range(len(radixs))]
        self.r_field = N_seq[0]
        self.radixs = radixs
        self.N_seq = N_seq

        self.fft_layers = nn.ModuleList()
        for N, r in zip(N_seq, radixs):
            self.fft_layers.append(general_FFTLayer(in_channels, channels, N, r, aux_channels))
            in_channels = channels
        self.fc_out = nn.Linear(channels, classes)
        self.padding_layer = nn.ConstantPad1d((self.r_field, 0), 0.)

    def forward(self, x, h=None, zeropad=True):
        if zeropad:
            x = self.padding_layer(x)
            if h is not None:
                h = self.padding_layer(h)

        for fft_layer in self.fft_layers:
            x = fft_layer(x, h)

        x = self.fc_out(x.transpose(1, 2))
        return x.transpose(1, 2)

    def conditional_sampling(self, logits, c):
        out = self.fc_out(logits.transpose(1, 2)).view(-1)
        probs = F.softmax(out * c, dim=0)
        dist = torch.distributions.Categorical(probs)
        return dist.sample()

    def class2float(self, category):
        return category.float() / (self.classes - 1) * 2 - 1

    def fast_generate(self, num_samples=None, h=None, c=1):
        # this method seems only 2~3 times faster
        buf = torch.zeros(1, self.in_channels, self.r_field)
        if next(self.parameters()).is_cuda:
            buf = buf.cuda()

        output_list = []
        buf_list = []
        if h is None:
            buf[:, :, -1].uniform_(-1, 1)
            for fft_layer, r, N in zip(self.fft_layers, self.radixs, self.N_seq):
                buf_list.append(buf[:, :, N // r - 1:])
                buf = fft_layer(buf)

            # first sample
            sample = self.conditional_sampling(buf, c)
            output_list.append(sample.item())
            sample = self.class2float(sample)

            for i in range(1, num_samples):
                for j in range(len(buf_list)):
                    torch.cat((buf_list[j][:, :, 1:], sample.view(1, -1, 1)), 2, out=buf_list[j])
                    sample = self.fft_layers[j](buf_list[j])

                sample = self.conditional_sampling(sample, c)
                output_list.append(sample.item())
                sample = self.class2float(sample)
        else:
            h = self.padding_layer(h)
            pos = self.r_field + 1
            for fft_layer, r, N in zip(self.fft_layers, self.radixs, self.N_seq):
                buf_list.append(buf[:, :, N // r - 1:])
                buf = fft_layer(buf, h[:, :, :pos])

            # first sample
            sample = self.conditional_sampling(buf, c)
            output_list.append(sample.item())
            sample = self.class2float(sample)

            for pos in range(self.r_field + 2, h.size(2) + 1):
                for j in range(len(buf_list)):
                    torch.cat((buf_list[j][:, :, 1:], sample.view(1, -1, 1)), 2, out=buf_list[j])
                    sample = self.fft_layers[j](buf_list[j], h[:, :, :pos])

                sample = self.conditional_sampling(sample, c)
                output_list.append(sample.item())
                sample = self.class2float(sample)

        outputs = torch.Tensor(output_list).view(-1, 1)
        return outputs
