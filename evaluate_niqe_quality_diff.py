# -*- coding: utf-8 -*-
# BRISQUE + COS evaluator (quiet) with sampling and final English summary

import os
import sys
import csv
import random
import contextlib
import warnings
import cv2
import numpy as np
from tqdm import tqdm
from skimage.util import img_as_float
from imquality import brisque
from numpy.linalg import norm

# ---------------- Quiet noisy logs ----------------
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import tensorflow as tf
    tf.get_logger().setLevel("ERROR")
    try:
        import absl.logging
        absl.logging.set_verbosity(absl.logging.ERROR)
    except Exception:
        pass
except Exception:
    pass

@contextlib.contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# ---------------- Config ----------------
N_SAMPLES = 300          # Max number of files sampled per dataset; None = use all
RANDOM_SEED = 2025       # Reproducibility
CSV_PATH = "results_quality_cos_summary.csv"

DATASETS = {
    "Digiface1M": {
        "img_dir": "./datasets/digiface1m_raw/",
        "pairs_txt": "./datasets/digiface1m_raw/image_pairs.txt",
    },
    "SFHQ": {
        "img_dir": "./datasets/sfhq_subset/",
        "pairs_txt": "./datasets/sfhq_subset/image_pairs.txt",
    },
    "CelebA": {
        "img_dir": "./datasets/celeba/img_align_celebA/",
        "pairs_txt": "./datasets/celeba/img_align_celebA/image_pairs.txt",
    },
}

PERTURB_SUFFIXES = {
    "makeup": "_makeup",
    "illum_blur_noise": "_illum_blur_noise",
    "adversarial": "_adv_Facenet",
}

# ---------------- BRISQUE ----------------
def compute_avg_brisque(img_paths):
    scores = []
    for p in tqdm(img_paths, desc="BRISQUE", leave=False):
        try:
            img = cv2.imread(p)
            if img is None:
                raise ValueError("Image read failed")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img = img_as_float(img)
            score = brisque.score(img)
            scores.append(score)
        except Exception as e:
            print(f"[BRISQUE] Skipped {p}: {e}")
    return (sum(scores) / len(scores)) if scores else None, len(scores)

# ---------------- FaceNet (quiet) ----------------
with suppress_stdout():
    from keras_facenet import FaceNet
    embedder = FaceNet()

def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    return float(np.dot(v1, v2) / (norm(v1) * norm(v2) + 1e-12))

def compute_avg_cosine(original_paths, perturbed_paths):
    assert len(original_paths) == len(perturbed_paths)
    scores = []
    valid_pairs = 0
    for orig, pert in tqdm(zip(original_paths, perturbed_paths),
                           total=len(original_paths), desc="Computing COS", leave=False):
        try:
            im1 = cv2.imread(orig)
            im2 = cv2.imread(pert)
            if im1 is None or im2 is None:
                raise ValueError("Image read failed")
            im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
            im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
            with suppress_stdout():
                emb1 = embedder.embeddings([im1])[0]
                emb2 = embedder.embeddings([im2])[0]
            scores.append(cosine_similarity(emb1, emb2))
            valid_pairs += 1
        except Exception as e:
            print(f"[COS] Skipped pair: {orig} | {pert} -> {e}")
    return (sum(scores) / len(scores)) if scores else None, valid_pairs

def f3(x):
    return "N/A" if x is None else f"{x:.3f}"

# ---------------- Main ----------------
random.seed(RANDOM_SEED)
summary_rows = []  # Collect results

for dataset_name, paths in DATASETS.items():
    print(f"\n=== Dataset: {dataset_name} ===")
    base_dir = paths["img_dir"].rstrip("/") + "/"
    pair_txt = paths["pairs_txt"]

    image_names = set()
    with open(pair_txt, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                image_names.add(parts[0]); image_names.add(parts[1])
            elif len(parts) == 1:
                image_names.add(parts[0])

    image_names = list(image_names)
    total_candidates = len(image_names)
    if (N_SAMPLES is not None) and (total_candidates > N_SAMPLES):
        sample_names = random.sample(image_names, N_SAMPLES)
    else:
        sample_names = image_names
    print(f"Using {len(sample_names)}/{total_candidates} files from pairs list.")

    orig_paths = [os.path.join(base_dir, n) for n in sample_names]

    base_brisque, n_b_base = compute_avg_brisque(orig_paths)

    for pert_name, suffix in PERTURB_SUFFIXES.items():
        pert_dir = base_dir.rstrip("/") + suffix + "/"
        if not os.path.isdir(pert_dir):
            print(f"\n--- Perturbation: {pert_name} ---")
            print(f"[Warning] Folder not found: {pert_dir} -> skipped")
            summary_rows.append([
                dataset_name, pert_name,
                f3(base_brisque), "N/A", "N/A", "N/A", n_b_base, 0
            ])
            continue

        pert_paths = [os.path.join(pert_dir, n) for n in sample_names]

        print(f"\n--- Perturbation: {pert_name} ---")
        print("→ Calculating BRISQUE (quiet)...")
        pert_brisque, n_b_pert = compute_avg_brisque(pert_paths)

        print("→ Calculating COS (quiet)...")
        avg_cos, n_cos = compute_avg_cosine(orig_paths, pert_paths)

        if (base_brisque is None) or (pert_brisque is None):
            delta_str = "N/A"
            print("BRISQUE - not enough valid images.")
        else:
            delta_b = pert_brisque - base_brisque
            sign = "+" if delta_b >= 0 else "-"
            delta_str = f"{sign}{abs(delta_b):.3f}"
            print(f"BRISQUE - Original: {base_brisque:.3f}, {pert_name}: {pert_brisque:.3f}, Δ: {delta_str}")

        if avg_cos is None:
            print("COS - not enough valid pairs.")
        else:
            print(f"COS - Avg similarity (orig vs {pert_name}): {avg_cos:.3f}")

        summary_rows.append([
            dataset_name, pert_name,
            f3(base_brisque), f3(pert_brisque), delta_str,
            f3(avg_cos), n_b_base, n_cos
        ])

# ---------------- Final Summary ----------------
print("\n================== FINAL SUMMARY ==================")
header = ["Dataset", "Perturbation", "BRISQUE_Orig", "BRISQUE_Pert", "Δ(BRISQUE)", "COS_Avg", "N_BRISQUE_Orig", "N_COS_Pairs"]
colw = [12, 16, 14, 14, 12, 10, 14, 12]
print(" ".join([h.ljust(w) for h, w in zip(header, colw)]))
for row in summary_rows:
    print(" ".join([str(v).ljust(w) for v, w in zip(row, colw)]))

with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(header)
    writer.writerows(summary_rows)

print(f"\nSaved summary CSV to: {CSV_PATH}")
print("Interpretation guide:")
print(" - BRISQUE ↑ (Δ>0) means lower visual quality; ↓ means better or similar quality.")
print(" - COS closer to 1 means higher similarity; lower COS means stronger feature disruption (typical in adversarial attacks).")
