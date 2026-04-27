import os
import glob
import pickle
from pathlib import Path

import cv2
import gudhi as gd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms

from PIL import Image
from scipy.special import entr


def compute_PEEK(feature_maps, h, w):
    positivized_maps = feature_maps + np.abs(np.min(feature_maps))
    entropy_map = -np.sum(entr(positivized_maps), axis=-1)
    peek_map = cv2.resize(entropy_map, (w, h))
    return peek_map


class VGG16FeatureExtractor(torch.nn.Module):
    def __init__(self, weights="DEFAULT"):
        super().__init__()
        self.vgg16 = models.vgg16(weights=weights).features
        self.conv_layers = [
            i for i, layer in enumerate(self.vgg16)
            if isinstance(layer, torch.nn.Conv2d)
        ]

    def forward(self, x):
        features = []
        for layer_index, layer in enumerate(self.vgg16):
            x = layer(x)
            if layer_index in self.conv_layers:
                features.append(x)
        return features

    def load_image(self, image_path):
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        image = Image.open(image_path).convert("RGB")
        return transform(image).unsqueeze(0)


def ComputeFeats(
    feature_map,
    tau_frac=0.03,
    tau_percentile=75,
    tau_mode="percentile_lifetime",
):
    fm = np.asarray(feature_map)
    cc = gd.CubicalComplex(top_dimensional_cells=fm)
    cc.persistence()

    D0 = np.asarray(cc.persistence_intervals_in_dimension(0), dtype=np.float64)
    D1 = np.asarray(cc.persistence_intervals_in_dimension(1), dtype=np.float64)

    all_pers = []
    for D in [D0, D1]:
        if D.size > 0:
            pers = D[:, 1] - D[:, 0]
            pers = pers[np.isfinite(pers)]
            if pers.size > 0:
                all_pers.append(pers)

    if tau_mode == "range":
        rnge = float(fm.max() - fm.min())
        tau = tau_frac * rnge if rnge > 0 else 0.0

    elif tau_mode == "percentile_lifetime":
        tau = 0.0 if len(all_pers) == 0 else float(
            np.percentile(np.concatenate(all_pers), tau_percentile)
        )

    else:
        raise ValueError("tau_mode must be 'range' or 'percentile_lifetime'")

    def stats(D):
        if D.size == 0:
            return 0.0, 0, 0.0

        pers = D[:, 1] - D[:, 0]
        pers = pers[np.isfinite(pers)]

        if pers.size == 0:
            return 0.0, 0, 0.0

        mask = pers >= tau
        return (
            float(pers.max()),
            int(mask.sum()),
            float(pers[mask].sum()) if mask.any() else 0.0
        )

    maxp0, n0, sump0 = stats(D0)
    maxp1, n1, sump1 = stats(D1)

    feats = np.array(
        [maxp0, n0, sump0, maxp1, n1, sump1],
        dtype=np.float64
    )

    return feats, tau, D0, D1


def ClassFromFeats(
    feats,
    blob_score_min=5.0,
    blob_n0_max=12,
    blob_n1_max=6,
    edge_n1_min=25,
    edge_score_min=1,
):
    maxp0, n0, sump0, maxp1, n1, sump1 = map(float, feats)

    blob_score = maxp0 / (n0 + 1.0)
    edge_score = n1 / (n0 + 1.0)

    if (
        blob_score >= blob_score_min
        and n0 <= blob_n0_max
        and n1 <= blob_n1_max
    ):
        label = "blob"

    elif n1 >= edge_n1_min and edge_score >= edge_score_min:
        label = "edge"

    else:
        label = "mixed"

    diag = {
        "maxp0": maxp0,
        "n0": n0,
        "sump0": sump0,
        "maxp1": maxp1,
        "n1": n1,
        "sump1": sump1,
        "blob_score": blob_score,
        "edge_score": edge_score,
    }

    return label, diag