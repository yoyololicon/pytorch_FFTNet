import torch
from torch import optim
import torch.nn as nn
import torch.nn.functional as F


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


class FFTLayer2(nn.Module):
    def __init__(self, in_channels, out_channels, N):
        super().__init__()
        self.N = N
        self.W_lr = nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=N // 2)
        self.W_o = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return F.relu(self.W_o(F.relu(self.W_lr(x))))


class FFTLayer_aux(nn.Module):
    def __init__(self, in_channels, out_channels, aux_channels, N):
        super().__init__()
        self.N = N
        self.W_l = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.W_r = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.V_l = nn.Conv1d(aux_channels, out_channels, kernel_size=1)
        self.V_r = nn.Conv1d(aux_channels, out_channels, kernel_size=1)
        self.W_o = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, h):
        x_l = self.W_l(x[:, :, :-self.N // 2])
        x_r = self.W_r(x[:, :, self.N // 2:])
        h_l = self.V_l(h[:, :, :-self.N // 2])
        h_r = self.V_r(h[:, :, self.N // 2:])
        x = F.relu(x_l + x_r + h_l + h_r)
        return F.relu(self.W_o(x))


class FFTLayer2_inv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation, aux_channels=None, ):
        super().__init__()
        self.dilation = dilation
        if aux_channels:
            self.aux = True
        else:
            self.aux = False
        self.W_lr = nn.Conv1d(in_channels, out_channels, kernel_size=2, dilation=dilation)
        if aux_channels:
            self.V_lr = nn.Conv1d(aux_channels, out_channels, kernel_size=2, dilation=dilation)
        self.W_o = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x, h=None):
        if self.aux:
            return F.relu(self.W_o(F.relu(self.W_lr(x) + self.V_lr(h))))
        else:
            return F.relu(self.W_o(F.relu(self.W_lr(x))))


class FFTNet(nn.Module):
    def __init__(self, channels=256, depth=11, classes=256, feature_width=26):
        super().__init__()
        self.classes = classes
        self.r_field = 2 ** depth
        N_seq = reversed([2 ** i for i in range(1, depth + 1)])

        in_channels = 1
        fft_layers = []
        for N in N_seq:
            if in_channels == 1:
                self.first_layer = FFTLayer_aux(in_channels, channels, feature_width, N)
            else:
                fft_layers += [FFTLayer_base(in_channels, channels, N)]
            in_channels = channels
        self.fft_layers = nn.Sequential(*fft_layers)
        self.fc_out = nn.Linear(channels, classes)
        print('Receptive Field: %i samples' % self.r_field)

    def forward(self, x, h):
        x = self.first_layer(x, h)
        x = self.fft_layers(x)
        x = self.fc_out(x.transpose(1, 2))
        return F.log_softmax(x, dim=-1).transpose(1, 2)


if __name__ == '__main__':
    net = FFTNet(classes=64)
    y = net(torch.randn(1, 1, 7048), torch.randn(1, 26, 7048))
    y_hat = y.transpose(1, 2).unsqueeze(-1)
    print(y.size(), y_hat.size())
