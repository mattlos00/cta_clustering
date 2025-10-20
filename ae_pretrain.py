import argparse
import random
from time import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)
from torch.nn import Linear
from torch.nn.parameter import Parameter
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader, Dataset

from evaluation import eva

g = torch.Generator()
g.manual_seed(222)


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


def seed_worker():
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def adjust_learning_rate(optimizer, epoch):
    lr = 0.001 * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def set_seeds(seed=222):
    """Seed per la riproducibilità"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def train_one_epoch(model, train_loader, optimizer, device):
    """Epoca di training per l'autoencoder"""
    model.train()
    total_loss = 0.0
    for x, _ in train_loader:
        x = x.to(device)
        optimizer.zero_grad()
        x_bar, _ = model(x)
        loss = F.mse_loss(x_bar, x)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def evaluate_and_cluster(
    model, full_data, true_labels, device, n_clusters, clus_method, epoch
):
    """
    Estrae le feature latenti, esegue il clustering e calcola le metriche di valutazione
    """
    model.eval()
    with torch.no_grad():
        x_tensor = torch.from_numpy(full_data).float().to(device)
        x_bar, z = model(x_tensor)

        reconstruction_loss = F.mse_loss(x_bar, x_tensor)
        z_cpu = z.cpu().numpy()

    # Clustering
    if clus_method == "kmeans":
        kmeans = KMeans(n_clusters=n_clusters, n_init=20, random_state=222)
        pred_labels = kmeans.fit_predict(z_cpu)
    elif clus_method == "birch":
        brc = Birch(n_clusters=n_clusters)
        pred_labels = brc.fit_predict(z_cpu)
    else:
        raise ValueError(f"Unknown clustering method: {clus_method}")

    # Calcolo delle metriche
    if np.unique(pred_labels).shape[0] > 1:
        sil = silhouette_score(z_cpu, pred_labels)
    else:
        sil = 0.0
    ari, acc, _, n_clusters, sil = eva(true_labels, pred_labels, sil, epoch)

    return reconstruction_loss.item(), ari, acc, n_clusters, sil, pred_labels


def run_pretraining(args):
    """Processo di training e di valutazione"""

    set_seeds()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Caricamento dei dati
    x = np.loadtxt(args.features_path, dtype=float)
    y = np.loadtxt(args.labels_path, dtype=int)
    dataset = LoadDataset(x)
    train_loader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    # Inizializza modello e ottimizzatore
    model = AE(
        n_enc_1=args.hidden_dim,
        n_dec_1=args.hidden_dim,
        n_input=x.shape[1],
        n_z=args.latent_dim,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print(model)

    # Ciclo di training
    best_loss = float("inf")
    best_ari = 0.0
    best_acc = 0.0
    best_k_clusters = 0
    best_ari_epoch = 0
    best_sil = 0.0

    for epoch in range(args.max_epochs):

        _ = train_one_epoch(model, train_loader, optimizer, device)

        loss, ari, acc, n_clusters, sil, pred_labels = evaluate_and_cluster(
            model, x, y, device, args.n_clusters, args.clus_method, epoch
        )

        print(
            f"Epoch {epoch}: Loss={loss:.4f}, ARI={ari:.4f}, ACC={acc:.4f}, Sil={sil:.4f}"
        )

        # Salvataggio e early stopping
        if loss < best_loss:
            best_loss = loss
            print("Loss improved. Saving model.")
            torch.save(model.state_dict(), args.output_model_path)

        # Salva i migliori risultati di clustering
        if ari > best_ari:
            print(f"Ari improved from {best_ari} to {ari}. Saving predicted labels.")
            best_ari = ari
            best_acc = acc
            best_k_clusters = n_clusters
            best_sil = sil
            best_ari_epoch = epoch
            np.savetxt(args.labels_out_path, pred_labels, fmt="%d")

        # Salvataggio del modello per SDCN
        if epoch == 29:
            print(f"Saving pre-trained AE.")
            torch.save(model.state_dict(), args.sdcn_model_path)

        if epoch > args.max_epochs:
            break

    print(f"\nTraining ended: loss: {best_loss}")
    print(
        f"\nBest Acc: {best_acc}, best Ari: {best_ari}, num clusters: {best_k_clusters}, best sil: {best_sil} at epoch {best_ari_epoch}."
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Autoencoder Pre-training")

    # File input/output
    parser.add_argument("--features_path", type=str, default="embeddings.txt")
    parser.add_argument("--labels_path", type=str, default="labels.txt")
    parser.add_argument("--output_model_path", type=str, default="ae.pkl")
    parser.add_argument("--sdcn_model_path", type=str, default="ae_sdcn.pkl")
    parser.add_argument(
        "--labels_out_path", type=str, default="predicted_labels_ae.txt"
    )

    # Architettura AE
    parser.add_argument("--hidden_dim", type=int, default=1000)
    parser.add_argument("--latent_dim", type=int, default=100)

    # Iperparametri
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_epochs", type=int, default=120)
    parser.add_argument(
        "--sdcn_epoch", type=int, default=29, help="Epoch for AE pre-train for SDCN"
    )

    # Argomenti clustering
    parser.add_argument("--n_clusters", type=int, default=5)
    parser.add_argument(
        "--clus_method", type=str, default="birch", choices=["kmeans", "birch"]
    )

    args = parser.parse_args()
    run_pretraining(args)
