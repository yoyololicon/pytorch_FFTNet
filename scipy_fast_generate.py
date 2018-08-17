from scipy import linalg
import numpy as np
import torch
from torch.nn.modules import Conv1d, Linear
from models import general_FFTLayer
from datetime import datetime

net = torch.load("./rms_model.pth").cpu()
net.eval()

fft_layers = []
fc_out = None

with torch.no_grad():
    for name, m in net.named_children():
        if name == 'fft_layers':
            for mm in m:
                W1 = mm.W_lr.weight.detach().numpy()
                W1_bias = mm.W_lr.bias.detach().numpy()
                d = mm.W_lr.dilation[0]

                V = mm.V_lr.weight.detach().numpy()
                V_bias = mm.V_lr.bias.detach().numpy()

                W2 = np.squeeze(mm.W_o.weight.detach().numpy())
                W2_bias = mm.W_o.bias.detach().numpy()
                fft_layers += [(W1, W1_bias, V, V_bias, W2, W2_bias, d)]
        elif name == 'fc_out':
            fc_out = m.weight.detach().numpy()

receptive_field = 1

buf_list = []
for W1, W1_bias, V, V_bias, W2, W2_bias, d in fft_layers:
    buf_list.append(np.zeros((W1.shape[1], (W1.shape[2] - 1) * d + 1)))
    receptive_field += d

h = np.random.randn(26, receptive_field * 10)
sample = np.random.randn(256)
a = datetime.now().replace(microsecond=0)
for k in range(receptive_field + 1, h.shape[1]):
    for i, weights in enumerate(fft_layers):
        W1, W1_bias, V, V_bias, W2, W2_bias, d = weights
        buf_list[i][:, :-1] = buf_list[i][:, 1:]
        buf_list[i][:, -1] = sample

        sample = np.tensordot(W1, buf_list[i][:, ::d], ([1, 2], [0, 1])) + W1_bias
        #sample += np.tensordot(V, h[:, k - buf_list[i].shape[1]:k:d], ([1, 2], [0, 1])) + V_bias
        sample += W2 @ sample + W2_bias

    sample = fc_out @ sample
cost = datetime.now().replace(microsecond=0) - a
print("Speed:", (h.shape[1] - receptive_field) / cost.total_seconds(), "samples/sec.")
