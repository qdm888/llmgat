import pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class MMDataset(Dataset):
    """Loads MMSA-style feature pickles with train/valid/test splits.

    Each split must provide: text_bert (N, 3, L), audio (N, n_a, d_a),
    vision (N, n_v, d_v), regression_labels (N,), raw_text and id.
    """

    def __init__(self, feature_path, mode='train'):
        with open(feature_path, 'rb') as f:
            data = pickle.load(f)
        split = data[mode]
        self.text = split['text_bert'].astype(np.float32)
        self.audio = split['audio'].astype(np.float32)
        self.vision = split['vision'].astype(np.float32)
        self.labels = np.array(split['regression_labels']).astype(np.float32)
        self.raw_text = list(split['raw_text'])
        self.ids = list(split['id'])
        self.audio[self.audio == -np.inf] = 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return {
            'text': torch.from_numpy(self.text[index]),
            'audio': torch.from_numpy(self.audio[index]),
            'vision': torch.from_numpy(self.vision[index]),
            'label': torch.tensor(self.labels[index].reshape(-1)),
            'id': self.ids[index],
            'raw_text': self.raw_text[index],
        }


def get_dataloaders(feature_path, batch_size, num_workers=4):
    loaders = {}
    for mode in ['train', 'valid', 'test']:
        ds = MMDataset(feature_path, mode)
        loaders[mode] = DataLoader(ds, batch_size=batch_size,
                                   num_workers=num_workers,
                                   shuffle=(mode == 'train'))
    return loaders
