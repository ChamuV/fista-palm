# src/utils/data.py

import os
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from skimage import io
from skimage.transform import resize


def generate_low_rank_matrix_problem(m, n, r, sparsity=0.25, seed=42):
    rng = np.random.default_rng(seed)

    A = rng.random((m, r)) @ rng.random((n, r)).T
    X0 = np.abs(rng.standard_normal((m, r)))
    Y0 = np.abs(rng.standard_normal((n, r)))

    s_x = m * r
    s_y = int(sparsity * n * r)

    return A, X0, Y0, s_x, s_y


def svd_initialisation(A, r, nonnegative=True):
    U, S, Vt = np.linalg.svd(A, full_matrices=False)

    sqrt_S = np.sqrt(S[:r])
    X0 = U[:, :r] * sqrt_S
    Y0 = Vt[:r, :].T * sqrt_S

    if nonnegative:
        X0 = np.maximum(X0, 0)
        Y0 = np.maximum(Y0, 0)

    return X0, Y0


def get_milestone_value(history, k):
    if k < len(history):
        return history[k]
    return history[-1]


def create_matrix_with_singular_values(m, n, r, scale, seed=42):
    rng = np.random.default_rng(seed)

    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))

    S = np.diag(np.full(r, scale))

    return U @ S @ V.T


def generate_initial_factors(m, n, r, seed=42):
    rng = np.random.default_rng(seed)

    X0 = np.abs(rng.standard_normal((m, r)))
    Y0 = np.abs(rng.standard_normal((n, r)))

    return X0, Y0


def create_matrix_with_conditioning(m, n, r, condition="well", scale=1.0, seed=42):
    rng = np.random.default_rng(seed)

    U, _ = np.linalg.qr(rng.standard_normal((m, r)))
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))

    if condition == "well":
        singular_values = np.full(r, scale)
    elif condition == "ill":
        singular_values = scale * np.logspace(0, -6, r)
    else:
        raise ValueError("condition must be 'well' or 'ill'")

    S = np.diag(singular_values)

    return U @ S @ V.T


def load_olivetti_matrix(shuffle=True):
    data = fetch_olivetti_faces(shuffle=shuffle)
    A = data.images.reshape((400, -1)).T
    image_shape = (64, 64)

    return A, image_shape


def download_bsds500(root):
    root = Path(root)
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    urls = [
        "https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
        "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz",
    ]

    tgz_path = data_dir / "BSR_bsds500.tgz"
    extract_dir = data_dir / "BSR"

    if not tgz_path.exists():
        last_error = None

        for url in urls:
            try:
                print(f"Downloading BSDS500 from {url}...")
                urllib.request.urlretrieve(url, tgz_path)
                break
            except Exception as err:
                last_error = err
                print(f"Download failed from {url}")

                if tgz_path.exists():
                    tgz_path.unlink()

        if not tgz_path.exists():
            raise RuntimeError(
                "Could not download BSDS500 automatically. "
                "Download BSR_bsds500.tgz manually and place it in data/."
            ) from last_error

    if not extract_dir.exists():
        print("Extracting BSDS500...")
        with tarfile.open(tgz_path, "r:gz") as tar:
            tar.extractall(path=data_dir)

    train_dir = extract_dir / "BSDS500" / "data" / "images" / "train"

    if not train_dir.exists():
        raise FileNotFoundError(f"Could not find BSDS500 train folder: {train_dir}")

    return train_dir


def load_images_as_matrix(folder_path, image_size=(128, 128), max_images=6):
    folder_path = Path(folder_path)

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])[:max_images]

    if len(image_files) == 0:
        raise FileNotFoundError(f"No image files found in {folder_path}")

    images = []

    for file in image_files:
        img = io.imread(folder_path / file, as_gray=True)
        img = resize(img, image_size, anti_aliasing=True)
        images.append(img.flatten())

    return np.stack(images, axis=1)


def add_gaussian_noise(A, sigma=0.3, seed=42):
    rng = np.random.default_rng(seed)
    return np.clip(A + rng.normal(0, sigma, size=A.shape), 0, 1)


def tile_images(A, indices, image_shape):
    return np.hstack([
        A[:, idx].reshape(image_shape)
        for idx in indices
    ])


def stack_reconstruction_rows(A, reconstructions, indices, image_shape):
    rows = [tile_images(A, indices, image_shape)]

    for recon in reconstructions:
        rows.append(tile_images(recon, indices, image_shape))

    return np.vstack(rows)


def stack_image_rows(rows, indices, image_shape):
    return np.vstack([
        tile_images(A, indices, image_shape)
        for A in rows
    ])


def reconstruct(X, Y):
    return X @ Y.T


def load_coil_matrix(folder_path, image_size=(32, 32), max_images=300):
    folder_path = Path(folder_path)

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(".png")
    ])

    if max_images is not None:
        image_files = image_files[:max_images]

    if len(image_files) == 0:
        raise FileNotFoundError(f"No PNG images found in {folder_path}")

    images = []

    for file in image_files:
        img = io.imread(folder_path / file, as_gray=True)
        img = resize(img, image_size, anti_aliasing=True)
        images.append(img.flatten())

    A = np.stack(images, axis=1)

    if A.max() > 0:
        A = A / A.max()

    return A, image_size