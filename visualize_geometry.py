import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os

def visualize_comparison(z1, labels1, name1, z2, labels2, name2, output_path):
    """
    Esegue t-SNE su due spazi latenti e crea una visualizzazione comparativa.
    """
    print("Inizializzazione t-SNE (può richiedere tempo)...")
    tsne_model = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

    print(f"Esecuzione t-SNE per {name1}...")
    z1_2d = tsne_model.fit_transform(z1)

    print(f"Esecuzione t-SNE per {name2}...")
    z2_2d = tsne_model.fit_transform(z2)

    print("Creazione del grafico...")
    plt.style.use('seaborn-whitegrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle('Confronto della Geometria dei Cluster (Visualizzazione t-SNE)', fontsize=20)

    # AE + Birch
    sns.scatterplot(
        x=z1_2d[:, 0],
        y=z1_2d[:, 1],
        hue=labels1,
        palette=sns.color_palette("viridis", as_cmap=True),
        legend="full",
        alpha=0.7,
        ax=ax1
    )
    ax1.set_title(f'Geometria Cluster ({name1})', fontsize=16)
    ax1.set_xlabel('Dimensione 1')
    ax1.set_ylabel('Dimensione 2')
    ax1.legend(title='Cluster Predetti')

    # SDCN
    sns.scatterplot(
        x=z2_2d[:, 0],
        y=z2_2d[:, 1],
        hue=labels2,
        palette=sns.color_palette("viridis", as_cmap=True),
        legend="full",
        alpha=0.7,
        ax=ax2
    )
    ax2.set_title(f'Geometria Cluster ({name2})', fontsize=16)
    ax2.set_xlabel('Dimensione 1')
    ax2.set_ylabel('Dimensione 2')
    ax2.legend(title='Cluster Predetti')
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Grafico comparativo salvato in: {output_path}")
    
    plt.show()


def main(args):
    z1 = np.loadtxt(args.z1_path)
    labels1 = np.loadtxt(args.labels1_path, dtype=int)
    
    z2 = np.loadtxt(args.z2_path)
    labels2 = np.loadtxt(args.labels2_path, dtype=int)
    
    visualize_comparison(z1, labels1, args.name1, z2, labels2, args.name2, args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizzazione comparativa della geometria dei cluster.")
    
    parser.add_argument('--z1_path', type=str, default="models/z_ae_birch.txt", help="Percorso ai vettori latenti del modello AE+Birch")
    parser.add_argument('--labels1_path', type=str, default="predictions/predicted_labels_ae.txt", help="Percorso alle label predette del modello AE + Birch")
    parser.add_argument('--name1', type=str, default="AE + Birch")
    
    parser.add_argument('--z2_path', type=str, default="models/sdcn_best_logits.txt", help="Percorso ai vettori latenti del modello SDCN")
    parser.add_argument('--labels2_path', type=str, default="predictions/predicted_labels_sdcn.txt", help="Percorso alle label predette del modello SDCN")
    parser.add_argument('--name2', type=str, default="SDCN")
    
    parser.add_argument('--output_path', type=str, default="results/geometry.png")

    args = parser.parse_args()
    
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
    main(args)
