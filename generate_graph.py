import argparse
import random
from time import time

import numpy as np
from sklearn.metrics import pairwise_distances as pair
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize


def set_seeds(seed=222):
    """Seed per la riproducibilità"""
    random.seed(seed)
    np.random.seed(seed)


def calculate_similarity(features, method="cosine"):
    """
    Calcola la matrice di similarità tra i campioni (nodi).

    Args:
        features (np.array): La matrice delle feature.
        method (str): Il metodo da utilizzare.
                      Opzioni: 'cosine', 'heat', 'bin_dot', 'norm_bin_dot'.

    Returns:
        np.array: La matrice di similarità.
    """
    print(f"Calculating similarity matrix using '{method}' method...")
    if method == "heat":
        # Kernel Gaussiano
        dist = -0.5 * pair(features) ** 2
        return np.exp(dist)

    elif method == "bin_dot":
        # Binarizza le feature e calcola il prodotto scalare
        binary_features = features.copy()
        binary_features[binary_features > 0] = 1
        return np.dot(binary_features, binary_features.T)

    elif method == "norm_bin_dot":
        # Binarizza, normalizza L1 e calcola il prodotto scalare
        binary_features = features.copy()
        binary_features[binary_features > 0] = 1
        norm_features = normalize(binary_features, axis=1, norm="l1")
        return np.dot(norm_features, norm_features.T)

    elif method == "cosine":
        # Similarità coseno
        return cosine_similarity(features)

    else:
        raise ValueError(f"Unknown method: {method}")


def construct_and_save_graph(similarity_matrix, k, labels, output_path):
    """
    Costruisce un grafo k-NN, lo salva e calcola l'error rate.

    Args:
        similarity_matrix (np.array): Matrice di similarità.
        k (int): Numero di vicini da considerare.
        labels (np.array): Etichette dei dati per il calcolo dell'errore.
        output_path (str): Percorso del file dove salvare il grafo.

    Returns:
        float: Error rate del grafo.
    """
    print(f"Constructing k-NN graph with k={k}...")
    num_nodes = similarity_matrix.shape[0]
    edges = []
    error_counter = 0

    for i in range(num_nodes):
        # Trova i k+1 vicini più prossimi
        indices = np.argpartition(similarity_matrix[i, :], -(k + 1))[-(k + 1) :]

        for neighbor_idx in indices:
            if neighbor_idx == i:
                continue  # Per l'auto-collegamento

            edges.append(f"{i} {neighbor_idx}\n")

            if labels[neighbor_idx] != labels[i]:
                error_counter += 1

    print(f"Saving graph to {output_path}...")
    with open(output_path, "w") as f:
        f.writelines(edges)

    error_rate = error_counter / (num_nodes * k)
    return error_rate


def main(args):
    """Procedura per la generazione del grafo"""
    set_seeds()

    # Caricamento dati
    print(f"Loading features from {args.features_path}")
    features = np.loadtxt(args.features_path, dtype=float)
    # Caricamento delle label (per verifica)
    print(f"Loading labels from {args.labels_path}")
    labels = np.loadtxt(args.labels_path, dtype=int)

    start_time = time()

    # Calcolo della similarità
    similarity_matrix = calculate_similarity(features, method=args.method)

    # Generazione e salvataggio del grafo
    name = args.features_path.split(".")[0]
    output_path = "graph/{}{}_graph.txt".format(name, args.k)
    error_rate = construct_and_save_graph(
        similarity_matrix, k=args.k, labels=labels, output_path=output_path
    )

    end_time = time()

    print(f"\nProcess ended")
    print(f"Error rate: {error_rate:.4f}")
    print(f"Total time taken: {end_time - start_time:.2f} sec.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Construct a k-NN graph from feature embeddings"
    )

    parser.add_argument(
        "--features_path",
        type=str,
        default="embeddings.txt",
        help="Path to the input features file",
    )
    parser.add_argument(
        "--labels_path",
        type=str,
        default="labels.txt",
        help="Path to the input labels file",
    )

    parser.add_argument(
        "-k",
        "--k",
        type=int,
        default=10,
        help="Number of nearest neighbors to find for each node",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="cosine",
        choices=["cosine", "heat", "bin_dot", "norm_bin_dot"],
        help="Similarity method to use",
    )

    args = parser.parse_args()
    main(args)
