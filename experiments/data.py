from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


class SkyDataset(Dataset):
    def __init__(self, frame, idx, future, image_path, cap_kw, frames=30, size=128):
        self.frame = frame
        self.idx, self.future = idx, future
        self.image_path, self.cap_kw = image_path, cap_kw
        self.frames, self.size = frames, size
        self.handle = None
        self.power = frame.power_kw.to_numpy(dtype=np.float32) / cap_kw

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        if self.handle is None:
            self.handle = h5py.File(self.image_path, "r")
        issue = self.idx[item]
        rows = self.frame.iloc[issue-self.frames+1:issue+1]
        images = []
        for row in rows.itertuples():
            array = self.handle[row.image_group]["images_log"][row.image_index]
            image = torch.from_numpy(array.copy()).permute(2, 0, 1).float() / 255.
            image = TF.resize(image, [self.size, self.size], antialias=True)
            images.append(TF.normalize(image, [.485, .456, .406], [.229, .224, .225]))
        return torch.stack(images), torch.from_numpy(self.power[issue-self.frames+1:issue+1].copy()), torch.from_numpy(self.power[self.future[issue]].copy())


class StateGridDataset(Dataset):
    def __init__(self, values, power, idx, all_future, cap_kw):
        self.values = np.asarray(values, np.float32)
        self.power = np.asarray(power / cap_kw, np.float32)
        self.idx, self.future = idx, all_future

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, item):
        issue = self.idx[item]
        return torch.from_numpy(self.values[issue-63:issue+1].copy()), torch.from_numpy(self.power[self.future[issue]].copy())
