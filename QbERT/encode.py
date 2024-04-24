import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torchaudio
from torchaudio.functional import resample
import librosa


def encode_dataset(args):
    print(f"Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_discrete").cuda()

    print(f"Encoding dataset at {args.in_dir}")
    for in_path in tqdm(list(args.in_dir.rglob(f"*{args.extension}"))):
        # wav, sr = torchaudio.load(in_path)
        # print(sr)
        wav, sr = librosa.load(in_path, mono=True)
        wav = torch.tensor(wav).unsqueeze(0)

        wav = resample(wav, sr, 16000)
        wav = wav.unsqueeze(0).cuda()

        with torch.inference_mode():
            wav = F.pad(wav, ((400 - 320) // 2, (400 - 320) // 2))
            x, _ = hubert.encode(wav, layer=7)

        out_path = args.out_dir / in_path.relative_to(args.in_dir)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_path.with_suffix(".npy"), x.squeeze().cpu().numpy())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Encode an audio dataset.")
    parser.add_argument(
        "in_dir",
        metavar="in-dir",
        help="path to the dataset directory.",
        type=Path,
    )
    parser.add_argument(
        "out_dir",
        metavar="out-dir",
        help="path to the output directory.",
        type=Path,
    )
    parser.add_argument(
        "--extension",
        help="extension of the audio files (defaults to .flac).",
        default=".flac",
        type=str,
    )
    args = parser.parse_args()
    encode_dataset(args)