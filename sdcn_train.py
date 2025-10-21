from __future__ import division, print_function

import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import Birch, KMeans
from sklearn.metrics import silhouette_score
from torch.optim import Adam

from sdcn.sdcn import SDCN
from utils.evaluation import eva
from utils.utils import load_data, load_graph


def set_seeds(seed=222):
    """Seed per la riproducibilità"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def initialize_cluster_centers(model, data, device, args):
    """Inizializza i centri dei cluster usando KMeans o Birch"""
    print("Initializing cluster centers...")
    with torch.no_grad():
        _, _, z = model.ae(data)

    z_cpu = z.cpu().numpy()
    if args.clus_method == "birch":
        clusterer = Birch(n_clusters=args.n_clusters)
        pred_labels = clusterer.fit_predict(z_cpu)
        cluster_centers = np.zeros((args.n_clusters, z.shape[1]), dtype=np.float32)
        np.add.at(cluster_centers, pred_labels, z_cpu)
        cluster_centers /= np.bincount(pred_labels)[:, None].astype(np.float32)
    elif args.clus_method == "kmeans":
        clusterer = KMeans(n_clusters=args.n_clusters, n_init=20, random_state=222)
        pred_labels = clusterer.fit_predict(z_cpu)
        cluster_centers = clusterer.cluster_centers_
    else:
        raise ValueError(f"Unknown clustering method: {args.clus_method}")

    model.cluster_layer.data = torch.tensor(cluster_centers).to(device)
    return pred_labels


def run_training(args, device):
    """Processo di training del modello SDCN"""
    set_seeds()
    # Caricamento dati e grafo KNN
    dataset = load_data(args.name)
    adj = load_graph(args.name, args.k).to(device)
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y

    # Modello e ottimizzatore
    model = SDCN(
        n_enc_1=args.hidden_dim,
        n_dec_1=args.hidden_dim,
        n_input=args.n_input,
        n_z=args.n_z,
        n_clusters=args.n_clusters,
        weights_path=args.pretrain_path,
    ).to(device)
    optimizer = Adam(model.parameters(), lr=args.lr)
    print(model)

    # Inizializzazione
    initial_preds = initialize_cluster_centers(model, data, device, args)
    print("Initial performance after AE pre-training:")
    eva(y, initial_preds, 0, "Initialization")

    # Training
    p = None
    sil = 0.0
    best_ari = 0.0
    best_acc = 0.0
    best_sil = 0.0
    best_labels = None
    no_improve_count = 0
    min_delta = 1e-2
    patience = 15

    for epoch in range(args.max_epochs):
        # Calcolo distribuzione target
        # Solo DOPO la prima epoca
        # Per non partire da uno spazio latente non ancora ottimale
        if epoch > 0:
            with torch.no_grad():
                _, tmp_q, _, _ = model(data, adj)
                p = target_distribution(tmp_q)

        # Forward e backward pass
        x_bar, q, pred, _ = model(data, adj)

        re_loss = F.mse_loss(x_bar, data)

        if p is not None:
            kl_loss = F.kl_div(q.log(), p, reduction="batchmean")
            ce_loss = F.kl_div(pred.log(), p, reduction="batchmean")
            loss = (
                args.re_weight * re_loss
                + args.kl_weight * kl_loss
                + args.ce_weight * ce_loss
            )
        else:  # Prima epoca
            loss = re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Valutazione
        if p is not None:
            res_q = tmp_q.cpu().numpy().argmax(1)  # Q
            res_p = p.data.cpu().numpy().argmax(1)  # P
            sil = silhouette_score(tmp_q.cpu().numpy(), tmp_q.cpu().numpy().argmax(1))
        res_z = pred.data.cpu().numpy().argmax(1)  # Z
        ari, acc, pred_labels, n_clusters, _ = eva(y, res_z, sil, epoch)
        print(
            f"ACC: {acc:.4f}, ARI: {ari:.4f}, Sil: {sil:.4f}, Num. Clusters: {n_clusters}"
        )

        if (sil - best_sil) > min_delta:
            print(f"\nSil improved from {best_sil} to {sil}. Saving model.")
            best_ari = ari
            best_acc = acc
            best_sil = sil
            best_labels = pred_labels
            no_improve_count = 0
        else:
            no_improve_count += 1

        if no_improve_count > patience:
            break

    print("\nTraining ended. Saving predicted labels.")
    print(f"Best ACC: {best_acc}, Best ARI: {best_ari}, Best Sil: {best_sil}")

    if best_labels is not None:
        np.savetxt(args.output_path, best_labels, fmt="%d")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SDCN Training")
    # Input/output
    parser.add_argument("--name", type=str, default="embedding/embeddings")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--pretrain_path", type=str, default="autoencoder/ae_sdcn.pkl")
    parser.add_argument(
        "--output_path", type=str, default="predictions/predicted_labels_sdcn.txt"
    )
    # Architettura SDCN
    parser.add_argument("--n_input", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=1000)
    parser.add_argument("--n_z", type=int, default=100)
    parser.add_argument("--n_clusters", type=int, default=5)
    # Iperparametri
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--re_weight", type=float, default=1.0, help="Reconstruction loss weight"
    )
    parser.add_argument("--kl_weight", type=float, default=0.1, help="KL loss weight")
    parser.add_argument(
        "--ce_weight", type=float, default=0.01, help="Graph loss weight"
    )
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument(
        "--clus_method", type=str, default="birch", choices=["kmeans", "birch"]
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    run_training(args, device)
