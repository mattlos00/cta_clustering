import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from sdcn.autoencoder import AE
from sdcn.gnn import GNNLayer


class SDCN(nn.Module):

    def __init__(self, n_enc_1, n_dec_1, n_input, n_z, n_clusters, weights_path, v=1):
        super(SDCN, self).__init__()

        # autoencoder for intra information
        self.ae = AE(n_enc_1=n_enc_1, n_dec_1=n_dec_1, n_input=n_input, n_z=n_z)
        self.ae.load_state_dict(
            torch.load(weights_path, map_location="cpu", weights_only=True)
        )

        # GCN for inter information
        self.gnn_1 = GNNLayer(n_input, n_enc_1)
        self.gnn_2 = GNNLayer(n_enc_1, n_z)
        self.gnn_3 = GNNLayer(n_z, n_clusters)

        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        # degree
        self.v = v

    def forward(self, x, adj):
        # DNN Module
        x_bar, tra1, z = self.ae(x)

        sigma = 0.5

        # GCN Module
        h = self.gnn_1(x, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        h = self.gnn_3((1 - sigma) * h + sigma * z, adj, active=False)
        predict = F.softmax(h, dim=1)

        # Dual Self-supervised Module
        q = 1.0 / (
            1.0
            + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v
        )
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, q, predict, z


    @torch.no_grad()
    def get_gnn_logits(self, data, adj):
        _, tra1, z = self.ae(data) 

        sigma = 0.5

        h = self.gnn_1(data, adj)
        h = self.gnn_2((1 - sigma) * h + sigma * tra1, adj)
        logits = self.gnn_3((1 - sigma) * h + sigma * z, adj, active=False) # active=False for logits
        return logits
    
