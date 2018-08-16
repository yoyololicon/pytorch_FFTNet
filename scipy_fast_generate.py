from scipy import linalg
import numpy as np
import torch
from torch.nn.modules import Conv1d, Linear
from models import general_FFTLayer

net = torch.load("./rms_model_20000.pth").cpu()
net.eval()

fft_layers = []
fc_out = None

with torch.no_grad():
    for name, m in net.named_children():
        if name == 'fft_layers':
            for mm in m:
                fft_layers += [{}]
                fft_layers[-1]["W_L"] = mm.W_lr.weight.detach().numpy()[:, :, 0]
                fft_layers[-1]["W_R"] = mm.W_lr.weight.detach().numpy()[:, :, 1]
                fft_layers[-1]["dilation"] = mm.W_lr.dilation[0]
                fft_layers[-1]["V_L"] = mm.V_lr.weight.detach().numpy()[:, :, 0]
                fft_layers[-1]["V_R"] = mm.V_lr.weight.detach().numpy()[:, :, 1]
                fft_layers[-1]["W_O"] = np.squeeze(mm.W_o.weight.detach().numpy())
        elif name == 'fc_out':
            fc_out = m.weight.detach().numpy()


print(fc_out.shape)