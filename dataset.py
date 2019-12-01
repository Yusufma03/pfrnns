import os
import torch
from torch.utils.data.dataset import Dataset
import numpy as np


class LocalizationDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.seq_len = len(self.data['trajs'][0])
        self.seq_num = len(self.data['trajs'])

        self.samp_seq_len = None

        map_temp = self.data['map']
        self.map_size = map_temp.shape[0]
        self.map_mean = np.mean(map_temp)
        self.map_std = np.std(map_temp)

    def __len__(self):
        return self.seq_num

    def set_samp_seq_len(self, seq_len):
        self.samp_seq_len = seq_len

    def __getitem__(self, index):
        seq_idx = index % self.seq_num

        env_map = self.data['map']
        traj = self.data['trajs'][seq_idx]

        env_map = torch.FloatTensor(env_map).unsqueeze(0)
        env_map = (env_map - self.map_mean) / self.map_std
        traj = torch.FloatTensor(traj)

        if self.samp_seq_len is not None and self.samp_seq_len != self.seq_len:
            start = np.random.randint(0, self.seq_len - self.samp_seq_len + 1)
            traj = traj[start:start + self.samp_seq_len]

        obs = traj[:, 6:]
        action = traj[:, 3:6]
        gt_pos = traj[:, :3]

        gt_pos[:, 2] = gt_pos[:, 2] / 360 * 2 * np.pi

        return (env_map, obs, gt_pos, action)
