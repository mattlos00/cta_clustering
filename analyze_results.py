import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix


def load_data(embeddings_path, true_labels_path, pred_labels_path, class_map_path):
    """Carica tutti i file necessari per l'analisi"""
    print("Loading data...")
    try:
        X = np.loadtxt(embeddings_path)
        y_true = np.loadtxt(true_labels_path, dtype=int)
        y_pred = np.loadtxt(pred_labels_path, dtype=int)

        with open(class_map_path, "r") as f:
            class_map = json.load(f)
        class_names = [class_map[str(i)] for i in range(len(class_map))]

        print(f"Loaded data: {X.shape[0]} rows")

        return X, y_true, y_pred, class_names
    except FileNotFoundError as e:
        print(f"ERROR: File not found -> {e.filename}")
        exit()
    except Exception as e:
        print(f"Error during data loading: {e}")
        exit()

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir=None):
    """Genera e mostra/salva la matrice di confusione"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=range(len(np.unique(y_pred))),
        yticklabels=class_names,
    )
    plt.title("Confusion Matrix")
    plt.ylabel("Ground Truth Label")
    plt.xlabel("Predicted Label")

    if output_dir:
        output_path = os.path.join(output_dir, "matrix.png")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"Plot saved at: {output_path}")
    else:
        plt.show()
    plt.close()


def plot_tsne_visualization(X, y_true, y_pred, class_names, output_dir=None):
    """Genera e mostra/salva la visualizzazione t-SNE"""
    print("Calculating t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30.0, n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(22, 10))

    # Grafico Ground Truth
    scatter1 = ax1.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y_true, cmap="viridis", alpha=0.7
    )
    ax1.set_title("t-SNE Ground Truth")
    ax1.legend(
        handles=scatter1.legend_elements(num=len(class_names))[0],
        labels=class_names,
        title="Classi",
    )

    # Grafico Cluster Predetti
    scatter2 = ax2.scatter(
        X_tsne[:, 0], X_tsne[:, 1], c=y_pred, cmap="viridis", alpha=0.7
    )
    ax2.set_title("t-SNE Predicted Clusters")
    unique_pred_labels = np.unique(y_pred)
    legend_labels = [f"Cluster {i}" for i in unique_pred_labels]
    ax2.legend(
        handles=scatter2.legend_elements(num=len(unique_pred_labels))[0],
        labels=legend_labels,
        title="Cluster",
    )

    if output_dir:
        output_path = os.path.join(output_dir, "tsne.png")
        plt.savefig(output_path, bbox_inches="tight")
        print(f"t-SNE saved at: {output_path}")
    else:
        plt.show()
    plt.close()


def inspect_cluster(cluster_id, y_true, y_pred, class_names):
    """Stampa una analisi dettagliata della composizione di un singolo cluster"""
    indices = np.where(y_pred == cluster_id)[0]
    if len(indices) == 0:
        print(f"No data for cluster {cluster_id}")
        return

    gt_labels_in_cluster = y_true[indices]
    composition = Counter(gt_labels_in_cluster)

    print(f"Cluster {cluster_id} contains {len(indices)} samples.")
    print("Cluster composition:")
    for label_id, count in sorted(composition.items(), key=lambda item: -item[1]):
        domain_name = class_names[label_id]
        percentage = (count / len(indices)) * 100
        print(
            f" - GT '{domain_name}' (ID {label_id}): {count} samples ({percentage:.2f}%)"
        )


def main(args):
    """Analisi dei risultati di clustering"""
    if "/" in args.pred_labels_file:
        predictions_file = args.pred_labels_file
    else:
        predictions_file = "predictions/" + args.pred_labels_file
    X, y_true, y_pred, class_names = load_data(
        args.embeddings_file,
        args.true_labels_file,
        predictions_file,
        args.class_map_file,
    )

    plot_confusion_matrix(y_true, y_pred, class_names, args.output_dir)

    if not args.run_tsne:
        plot_tsne_visualization(X, y_true, y_pred, class_names, args.output_dir)

    if args.inspect_cluster_id is not None:
        inspect_cluster(args.inspect_cluster_id, y_true, y_pred, class_names)

    elif args.interactive:
        try:
            while True:
                cluster_id_str = input(
                    "\nEnter the cluster ID to analyze ('q' to exit): "
                )
                if cluster_id_str.lower() == "q":
                    break
                inspect_cluster(int(cluster_id_str), y_true, y_pred, class_names)
        except (ValueError, KeyboardInterrupt):
            print("\nAnalysis ended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze and view clustering results")

    # Input
    parser.add_argument(
        "--embeddings_file", type=str, default="embedding/embeddings.txt"
    )
    parser.add_argument("--true_labels_file", type=str, default="embedding/labels.txt")
    parser.add_argument("--pred_labels_file", type=str)  # Obbligatorio
    parser.add_argument(
        "--class_map_file", type=str, default="embedding/label_map.json"
    )

    # Opzionali
    parser.add_argument(
        "--output_dir", type=str, default=None, help="Plots output directory"
    )
    parser.add_argument(
        "--tsne", action="store_false", dest="run_tsne", help="Run t-SNE"
    )

    # Ispezione dei cluster
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--inspect_cluster_id", type=int)
    group.add_argument(
        "--interactive",
        action="store_true",
        help="Start interactive session for multiple cluster analysis",
    )

    args = parser.parse_args()
    main(args)
