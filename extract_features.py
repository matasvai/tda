import argparse
import glob
import os
import pickle
from pathlib import Path

import torch
from tqdm import tqdm

from functions import VGG16FeatureExtractor


def get_image_paths(image_folder):
    exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
    paths = []

    for ext in exts:
        paths.extend(glob.glob(os.path.join(image_folder, "**", ext), recursive=True))

    return sorted(paths)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_folder", required=True)
    parser.add_argument("--out_folder", required=True)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available()
        else "cpu" if args.device == "auto"
        else args.device
    )

    Path(args.out_folder).mkdir(parents=True, exist_ok=True)

    image_paths = get_image_paths(args.image_folder)
    if args.max_images is not None:
        image_paths = image_paths[:args.max_images]

    model = VGG16FeatureExtractor(weights="DEFAULT").to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Found {len(image_paths)} images.")

    for image_path in tqdm(image_paths):
        base = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(args.out_folder, f"{base}.pkl")

        if os.path.exists(out_path):
            continue

        x = model.load_image(image_path).to(device)

        with torch.no_grad():
            features = model(x)

        features = [feat.cpu() for feat in features]

        with open(out_path, "wb") as f:
            pickle.dump(features, f)


if __name__ == "__main__":
    main()