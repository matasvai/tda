import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out_folder", default="results/plots")
    args = parser.parse_args()

    Path(args.out_folder).mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    print(df.head())
    print()
    print("Label counts:")
    print(df["label"].value_counts())

    label_counts = df.groupby(["layer", "label"]).size().unstack(fill_value=0)

    ax = label_counts.plot(kind="bar", figsize=(12, 6))
    ax.set_title("TDA Class Counts by Layer")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{args.out_folder}/label_counts_by_layer.png", dpi=300)
    plt.close()

    for metric in ["tau", "n0_tau", "n1_tau", "blob_score", "edge_score"]:
        ax = df.boxplot(column=metric, by="layer", figsize=(12, 6))
        ax.set_title(f"{metric} by Layer")
        ax.set_xlabel("Layer")
        ax.set_ylabel(metric)
        plt.suptitle("")
        plt.tight_layout()
        plt.savefig(f"{args.out_folder}/{metric}_by_layer.png", dpi=300)
        plt.close()


if __name__ == "__main__":
    main()