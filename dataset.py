from torch.utils.data import Dataset
import torch
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.interpolate import interp1d
from utils import zero_padding


class CMU_Dataset(Dataset):
    def __init__(self,
                 folder,
                 sample_size,
                 quantization_channels,
                 hopsize,
                 interp_method,
                 train=True):
        self.train = train
        self.sample_size = sample_size
        self.channels = quantization_channels
        self.hopsize = hopsize
        self.interp_method = interp_method
        if train:
            npzfile = os.path.join(folder, "train.npz")
        else:
            npzfile = os.path.join(folder, "test.npz")
        scaler = StandardScaler()
        scaler_info = np.load(os.path.join(folder, 'scaler.npz'))
        scaler.mean_ = scaler_info['mean']
        scaler.scale_ = scaler_info['scale']
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

        # pad zeros to prevent out range error
        local_condition = zero_padding(local_condition, len(audio) // self.hopsize + 2, 1)

        if self.train:
            rand_pos = np.random.randint(0, len(audio) - self.sample_size - 1)
            target = audio[rand_pos + 1:rand_pos + 1 + self.sample_size]
            audio = audio[rand_pos:rand_pos + self.sample_size].astype(float) / (self.channels - 1) * 2 - 1

            # interpolation
            if self.interp_method == 'linear':
                x = np.arange(local_condition.shape[1]) * self.hopsize
                f = interp1d(x, local_condition, copy=False, axis=1)
                local_condition = f(np.arange(rand_pos + 1, rand_pos + 1 + self.sample_size))
            elif self.interp_method == 'repeat':
                local_condition = np.repeat(local_condition, self.hopsize, axis=1)
                local_condition = local_condition[:, rand_pos + 1:rand_pos + 1 + self.sample_size]
            else:
                print("interpolation method", self.interp_method, "is not implemented.")
                exit(1)

            return torch.from_numpy(audio).float().view(1, -1), torch.from_numpy(target).long(), torch.from_numpy(
                local_condition).float()
        else:
            name_code = [ord(c) for c in name]
            # the batch size should be 1 in test mode
            return torch.LongTensor(name_code), torch.from_numpy(audio).long(), torch.from_numpy(
                local_condition).float()
