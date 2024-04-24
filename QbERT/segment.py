import argparse
from pathlib import Path

from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

import torch
import numba
import numpy as np
import itertools
import scipy.spatial.distance as distance


def segment(sequence, codebook, gamma):
    dists = distance.cdist(sequence, codebook).astype(np.float32)
    alpha, P = _segment(dists, gamma)
    return _backtrack(alpha, P)


@numba.njit()
def _segment(dists, gamma):
    T, K = dists.shape

    alpha = np.zeros(T + 1, dtype=np.float32)
    P = np.zeros((T + 1, 2), dtype=np.int32)
    D = np.zeros((T, T, K), dtype=np.float32)

    for t in range(T):
        for k in range(K):
            D[t, t, k] = dists[t, k]
    for t in range(T):
        for s in range(t + 1, T):
            D[t, s, :] = D[t, s - 1, :] + dists[s, :]

    for t in range(T):
        alpha[t + 1] = np.inf
        for s in range(t + 1):
            k = np.argmin(D[t - s, t, :])
            alpha_min = alpha[t - s] + D[t - s, t, k] - gamma * s
            if alpha_min < alpha[t + 1]:
                P[t + 1, :] = t - s, k
                alpha[t + 1] = alpha_min
    return alpha, P


@numba.njit()
def _backtrack(alpha, P):
    rhs = len(alpha) - 1
    segments = np.zeros(len(alpha) - 1, dtype=np.int32)
    boundaries = [np.int32(rhs)]
    while rhs != 0:
        lhs, code = P[rhs, :]
        boundaries.append(lhs)
        segments[lhs:rhs] = code
        rhs = lhs
    boundaries.reverse()
    return segments, np.array(boundaries)


def segment_file(in_path, out_path, codebook, gamma):
    # print(in_path)
    sequence = np.load(in_path)
    # print(sequence.shape, codebook.shape)
    codes, boundaries = segment(sequence, codebook, gamma)
    np.savez(out_path.with_suffix(".npz"), codes=codes, boundaries=boundaries)
    return sequence.shape[0], np.mean(np.diff(boundaries))


def segment_dataset(args):
    in_paths = list(args.in_dir.rglob("*.npy"))
    out_paths = [args.out_dir / path.relative_to(args.in_dir) for path in in_paths]

    print("Loading HuBERT codebook...")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete")
    codebook = hubert.kmeans.cluster_centers_

    print("Setting up folder structure...")
    for path in tqdm(out_paths):
        path.parent.mkdir(exist_ok=True, parents=True)

    print("Segmenting dataset...")
    with ProcessPoolExecutor(max_workers=1) as executor:
        results = list(
            tqdm(
                executor.map(
                    segment_file,
                    in_paths,
                    out_paths,
                    itertools.repeat(codebook),
                    itertools.repeat(args.gamma),
                ),
                total=len(in_paths),
            )
        )

    frames, boundary_length = zip(*results)
    print(f"Segmented {sum(frames) * 0.02 / 60 / 60} hours of audio")
    print(f"Average segment length: {np.mean(boundary_length) * 0.02} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        type=Path,
        help="path to the HuBERT features.",
    )
    parser.add_argument(
        "out_dir", metavar="out-dir", type=Path, help="path to the output directory."
    )
    parser.add_argument(
        "--gamma", default=2, type=float, help="penalty parameter for segmentation."
    )
    args = parser.parse_args()
    segment_dataset(args)