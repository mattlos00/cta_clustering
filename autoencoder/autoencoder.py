import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear
from torch.utils.data import Dataset


class AE(nn.Module):
    """Autoencoder"""

    def __init__(self, n_enc_1, n_dec_1, n_input, n_z):
        super(AE, self).__init__()
        self.enc_1 = Linear(n_input, n_enc_1)
        self.z_layer = Linear(n_enc_1, n_z)

        self.dec_1 = Linear(n_z, n_dec_1)
        self.x_bar_layer = Linear(n_dec_1, n_input)

    def forward(self, x):
        enc_h1 = F.relu(self.enc_1(x))
        z = self.z_layer(enc_h1)

        dec_h1 = F.relu(self.dec_1(z))
        x_bar = self.x_bar_layer(dec_h1)

        return x_bar, z


class LoadDataset(Dataset):
    """Caricamento del dataset per il training"""

    def __init__(self, data):
        self.x = data

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])).float(), torch.from_numpy(
            np.array(idx)
        )
