from torch.utils.data import Dataset
import torch
from hparams import hparams
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d


class CMU_Dataset(Dataset):
    def __init__(self,
                 folder,
                 train=True,
                 channels=256,
                 sample_size=5000):
        self.sample_size = sample_size
        self.channels = channels
        self.hopsize = int(hparams.sample_rate * hparams.winstep)
        if train:
            npzfile = os.path.join(folder, "train.npz")
        else:
            npzfile = os.path.join(folder, "test.npz")
        scaler = StandardScaler()
        scaler.mean_ = np.load(os.path.join(folder, 'mean.npy'))
        scaler.scale_ = np.load(os.path.join(folder, 'scale.npy'))
        self.transform_fn = scaler.transform

        self.names_list = []
        data_dict = np.load(npzfile)
        self.data_buffer = {}
        for name, x in data_dict.items():
            if name[-2:] != '_h':
                self.names_list.append(name)
                self.data_buffer[name] = x
            else:
                self.data_buffer[name] = self.transform_fn(x.T).T

    def __len__(self):
        return len(self.names_list)

    def __getitem__(self, index):
        name = self.names_list[index]
        audio = self.data_buffer[name]
        local_condition = self.data_buffer[name + '_h']

        rand_pos = np.random.randint(0, len(audio) - self.sample_size - 1)
        target = audio[rand_pos + 1:rand_pos + 1 + self.sample_size]
        audio = audio[rand_pos:rand_pos + self.sample_size].astype(float) / (self.channels - 1) * 2 - 1

        # interpolation
        if hparams.interp_method == 'interp':
            x = np.arange(local_condition.shape[1]) * self.hopsize
            f = interp1d(x, local_condition, copy=False, axis=1)
            local_condition = f(np.arange(rand_pos + 1, rand_pos + 1 + self.sample_size))
        elif hparams.interp_method == 'repeat':
            local_condition = np.repeat(local_condition, self.hopsize, axis=1)
            local_condition = local_condition[:, rand_pos + 1:rand_pos + 1 + self.sample_size]

        return torch.from_numpy(audio).float().view(1, -1), torch.from_numpy(target).long(), torch.from_numpy(
            local_condition).float()