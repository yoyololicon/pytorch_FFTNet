import torch
import torch.nn as nn
import torch.nn.functional as F

from operator import mul
from functools import reduce


class FFTLayer_base(nn.Module):
    # I think its like division in frequency?
    def __init__(self, in_channels, out_channels, N):
        super().__init__()
        self.N = N
        self.W_l = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.W_r = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.W_o = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x_l = self.W_l(x[:, :, :-self.N // 2])
        x_r = self.W_r(x[:, :, self.N // 2:])
        x = F.relu(x_l + x_r)
        return F.relu(self.W_o(x))


class general_FFTLayer(nn.Module):
    def __init__(self, in_channels, out_channels, N, radix=2):
        super().__init__()
        self.radix = radix
        self.W_lr = nn.Conv1d(in_channels, out_channels, kernel_size=radix, dilation=N // radix)
        self.W_o = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.relu(self.W_o(F.relu(self.W_lr(x))))


class general_FFTLayer_aux(nn.Module):
    def __init__(self, in_channels, aux_channels, out_channels, N, radix=2, stationary=False):
        super().__init__()
        self.radix = radix
        self.W_lr = nn.Conv1d(in_channels, out_channels, kernel_size=radix, dilation=N // radix)
        self.stationary = stationary
        if stationary:
            self.V_n = nn.Conv1d(aux_channels, out_channels, kernel_size=1)
        else:
            self.V_lr = nn.Conv1d(aux_channels, out_channels, kernel_size=radix, dilation=N // radix)
        self.W_o = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, h):
        if self.stationary:
            h = h.view(1, -1, 1)
            h = self.V_n(h)
        else:
            h = self.V_lr(h)
        return F.relu(self.W_o(F.relu(self.W_lr(x) + h)))


class FFTNet(nn.Module):
    def __init__(self, in_channels=1, channels=256, depth=11, classes=256, inverse=False):
        super().__init__()
        self.classes = classes
        self.r_field = 2 ** depth
        N_seq = [2 ** i for i in range(1, depth + 1)]
        if not inverse:
            N_seq = reversed(N_seq)

        x = in_channels
        fft_layers = []
        for N in N_seq:
            fft_layers += [general_FFTLayer(x, channels, N)]
            x = channels
        self.fft_layers = nn.Sequential(*fft_layers)
        self.fc_out = nn.Linear(channels, classes)

        self.padding_layer = nn.ConstantPad1d((self.r_field, 0), 0.)
        # print('Receptive Field: %i samples' % self.r_field)

    def forward(self, x, zeropad=True):
        if zeropad:
            x = self.padding_layer(x)
        x = self.fft_layers(x)
        x = self.fc_out(x.transpose(1, 2))
        return x.transpose(1, 2)


class FFTNet_aux(nn.Module):
    def __init__(self, in_channels=1, feature_width=26, channels=256, depth=11, classes=256, inverse=False):
        super().__init__()
        self.classes = classes
        self.r_field = 2 ** depth
        if not inverse:
            N = 2 ** depth
        else:
            N = 2
        self.first_layer = general_FFTLayer_aux(in_channels, feature_width, channels, N)
        self.layers = FFTNet(channels, channels, depth - 1, classes, inverse)

        self.padding_layer = nn.ConstantPad1d((self.r_field, 0), 0.)
        # print('Receptive Field: %i samples' % self.r_field)

    def forward(self, x, h, zeropad=True):
        if zeropad:
            x = self.padding_layer(x)
            h = self.padding_layer(h)
        x = self.first_layer(x, h)
        return self.layers(x, zeropad=False)


class general_FFTNet(nn.Module):
    def __init__(self, radixs=[2] * 11, in_channels=1, feature_width=26, channels=256, classes=256, inverse=False,
                 stationary=False):
        super().__init__()
        self.classes = classes
        self.stationary = stationary
        N_seq = [reduce(mul, radixs[i:]) for i in range(len(radixs))]
        self.r_field = N_seq[0]
        if inverse:
            N_seq = reversed(N_seq)

        if feature_width > 0:
            self.first_layer = general_FFTLayer_aux(in_channels, feature_width, channels, N_seq[0], radixs[0],
                                                    stationary)
        else:
            self.first_layer = general_FFTLayer(in_channels, channels, N_seq[0], radixs[0])

        fft_layers = []
        for i in range(1, len(radixs)):
            fft_layers += [general_FFTLayer(channels, channels, N_seq[i], radix=radixs[i])]
        self.fft_layers = nn.Sequential(*fft_layers)
        self.fc_out = nn.Linear(channels, classes)
        self.padding_layer = nn.ConstantPad1d((self.r_field, 0), 0.)

    def forward(self, x, h=None, zeropad=True):
        if zeropad:
            x = self.padding_layer(x)

        if h is None:
            x = self.first_layer(x)
        else:
            if not self.stationary and zeropad:
                h = self.padding_layer(h)
            x = self.first_layer(x, h)
        x = self.fft_layers(x)
        x = self.fc_out(x.transpose(1, 2))
        return x.transpose(1, 2)

