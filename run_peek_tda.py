import argparse
import glob
import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from functions import compute_PEEK, ComputeFeats, ClassFromFeats


def get_image_paths(image_folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []

    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_folder, "**", ext), recursive=True))

    return sorted(paths)


def parse_layers(layer_string):
    if layer_string is None or layer_string.lower() == "all":
        return None
    return [int(x.strip()) for x in layer_string.split(",")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--feature_folder", required=True)
    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--layers", default="all")
    parser.add_argument("--max_images", type=int, default=None)

    parser.add_argument("--tau_mode", default="percentile_lifetime")
    parser.add_argument("--tau_percentile", type=float, default=75)
    parser.add_argument("--tau_frac", type=float, default=0.03)

    args = parser.parse_args()

    Path(os.path.dirname(args.out_csv)).mkdir(parents=True, exist_ok=True)

    selected_layers = parse_layers(args.layers)

    image_paths = get_image_paths(args.image_folder)
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]

    rows = []

    for image_path in tqdm(image_paths):
        base = os.path.splitext(os.path.basename(image_path))[0]
        feature_path = os.path.join(args.feature_folder, f"{base}.pkl")

        if not os.path.exists(feature_path):
            print(f"Missing feature file for {image_path}")
            continue

        image = plt.imread(image_path)
        h, w = image.shape[:2]

        with open(feature_path, "rb") as f:
            loaded_feature_maps = pickle.load(f)

        if selected_layers is None:
            layers = list(range(len(loaded_feature_maps)))
        else:
            layers = selected_layers

        for layer in layers:
            fmap = loaded_feature_maps[layer][0].cpu().numpy()

            # compute_PEEK expects H x W x C
            fmap = np.moveaxis(fmap, 0, -1)

            peek_map = compute_PEEK(fmap, h, w)

            feats, tau, D0, D1 = ComputeFeats(
                peek_map,
                tau_frac=args.tau_frac,
                tau_percentile=args.tau_percentile,
                tau_mode=args.tau_mode,
            )

            label, diag = ClassFromFeats(feats)

            rows.append({
                "image": os.path.basename(image_path),
                "layer": layer,
                "label": label,
                "tau": tau,

                "maxp0": feats[0],
                "n0_tau": feats[1],
                "sump0_tau": feats[2],

                "maxp1": feats[3],
                "n1_tau": feats[4],
                "sump1_tau": feats[5],

                "blob_score": diag["blob_score"],
                "edge_score": diag["edge_score"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)

    print(f"Saved results to {args.out_csv}")
    print(df.head())


if __name__ == "__main__":
    main()