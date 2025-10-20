import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def encode_with_batches(texts, model, batch_size=32, device=None):
    """Creazione degli embedding dei dati"""
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        batch_embeddings = model.encode(batch, device=device, convert_to_numpy=True)
        embeddings.extend(batch_embeddings)
    return np.array(embeddings)


def set_seeds(seed=222):
    """Seed per la riproducibilità"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def load_data(file_path):
    """Caricamento e preparazione
    dei dati dal dataset di input"""
    print(f"\nLoading data from {file_path}...")
    df = pd.read_csv(file_path, sep=",")
    return df.applymap(str)  # Conversione dei dati in stringa


def extract_labels(data, output_path):
    """Estrae, codifica, mappa e salva le etichette di dominio"""
    print(f"\nProcessing and saving labels to {output_path}...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(data["domain"])
    pd.DataFrame(y_encoded).to_csv(output_path, header=False, index=False)
    class_map = {i: name for i, name in enumerate(label_encoder.classes_)}
    class_map_filename = "label_map.json"
    with open(class_map_filename, "w") as f:
        json.dump(class_map, f, indent=4)


def calculate_stats_features(data, values):
    """Calcolo delle feature statistiche, z-score e percentile, dai dati numerici"""
    print("\nStats features enabled")
    headers = data.iloc[:, 1]
    numeric_values = pd.to_numeric(values, errors="coerce")
    is_numeric_mask = numeric_values.notna()

    numeric_data_only = pd.DataFrame(
        {
            "header": headers[is_numeric_mask],
            "value": numeric_values[is_numeric_mask],
        }
    )
    column_stats = (
        numeric_data_only.groupby("header")["value"].agg(["mean", "std"]).reset_index()
    )
    # Merge delle feature
    data_with_stats = pd.merge(
        numeric_data_only.reset_index(), column_stats, on="header", how="left"
    ).set_index("index")

    # Calcolo z-score
    data_with_stats["std"] = data_with_stats["std"].fillna(1e-6)
    data_with_stats["std"] = data_with_stats["std"].replace(
        0.0, 1e-9
    )  # Evitare divisioni per zero
    data_with_stats["z_score"] = (
        data_with_stats["value"] - data_with_stats["mean"]
    ) / data_with_stats["std"]
    data_with_stats["z_score"] = data_with_stats["z_score"].fillna(0)

    # Calcolo delle feature solo per valori numerici
    z_score_feature = np.zeros((len(data), 1))
    z_score_feature[is_numeric_mask] = data_with_stats[["z_score"]].to_numpy()
    # Scalatura della feature
    scaler_zscore = MinMaxScaler(feature_range=(-1, 1))
    scaled_z_score_feature = scaler_zscore.fit_transform(z_score_feature)

    # Feature percentili
    data_with_stats["percentile"] = (
        data_with_stats.groupby("header")["value"].rank(pct=True).fillna(0.5)
    )
    percentile_feature = np.zeros((len(data), 1))
    percentile_feature[is_numeric_mask] = data_with_stats[["percentile"]].to_numpy()
    # Scalatura della feature
    scaler_percentile = MinMaxScaler(feature_range=(-1, 1))
    scaled_percentile_feature = scaler_percentile.fit_transform(percentile_feature)

    return scaled_z_score_feature, scaled_percentile_feature


def main():
    """Processo di generazione degli embedding"""
    set_seeds()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Caricamento dei dati
    input_path = args.noisy_input_path if args.noisy_headers else args.input_path
    if args.noisy_headers:
        input_path = args.noisy_input_path
    elif args.ideal_scenario:
        input_path = args.ideal_input_path
    else:
        input_path = args.input_path
    data = load_data(input_path)
    extract_labels(data, args.labels_path)

    headers = data.iloc[:, 1]
    values = data.iloc[:, 2]

    # Caricamento del modello
    model = SentenceTransformer("paraphrase-mpnet-base-v2").to(device)
    batch_size = 32

    print("\nProcessing...")

    header_embeddings = encode_with_batches(headers.tolist(), model, batch_size, device)
    value_embeddings = encode_with_batches(values.tolist(), model, batch_size, device)

    # Combinazione delle feature statistiche
    if args.stats_features:
        print("\nStats features enabled")
        # Calcolo feature statistiche
        z_score, percentile = calculate_stats_features(
            load_data(args.input_path), values
        )
        # Combinazione delle feature
        final_embedding = np.concatenate(
            [header_embeddings, z_score, percentile],
            axis=1,
        )
    else:
        # Text embedding
        # Se la flag 'headers_only' è attivata utilizzo solo 'header_embeddings'
        # Altrimenti, faccio la media tra 'header_embeddings e 'value_embeddings'
        values_weight = 0.0 if args.headers_only else 0.5
        headers_weight = 1.0 if args.headers_only else 0.5
        final_embedding = (
            headers_weight * header_embeddings + values_weight * value_embeddings
        )

    print(f"\nSaving embeddings at {args.embeddings_path}...")
    np.savetxt(args.embeddings_path, final_embedding, delimiter=" ", fmt="%s")
    print(f"\nEmbedding matrix shape: {final_embedding.shape}")
    print("\nProcess ended")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate embeddings from tabular data"
    )
    # File di input
    parser.add_argument(
        "--input_path",
        type=str,
        default="Dataset_Baseline.csv",
        help="Path to the input Dataset file",
    )
    parser.add_argument(
        "--noisy_input_path",
        type=str,
        default="Dataset_Noise.csv",
        help="Path to the noisy Dataset file",
    )
    parser.add_argument(
        "--ideal_input_path",
        type=str,
        default="Dataset_Ideal.csv",
        help="Path to the ideal Dataset file",
    )
    parser.add_argument(
        "--ideal_scenario", action="store_true", help="Use Dataset with ideal headers"
    )
    # File di output
    parser.add_argument(
        "--labels_path", type=str, default="labels.txt", help="Encoded GT labels path"
    )
    parser.add_argument(
        "--embeddings_path",
        type=str,
        default="embeddings.txt",
        help="Final embeddings path",
    )
    # Flag per l'embedding
    parser.add_argument(
        "--stats_features", action="store_true", help="Enable statstical features"
    )
    parser.add_argument(
        "--noisy_headers", action="store_true", help="Use Dataset with noisy headers"
    )
    parser.add_argument(
        "--headers_only", action="store_true", help="Enable schema-level embedding"
    )

    args = parser.parse_args()

    main()
