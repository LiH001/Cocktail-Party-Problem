"""Evaluate separated sources with SI-SDR and Delta_SDR."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cocktail_separation.audio import read_wav_mono
from cocktail_separation.metrics import best_permutation_si_sdr


def load_many(paths: list[str]) -> np.ndarray:
    audios = [read_wav_mono(path)[0] for path in paths]
    min_len = min(audio.shape[0] for audio in audios)
    return np.stack([audio[:min_len] for audio in audios], axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute SI-SDR improvement")
    parser.add_argument("--mixture", required=True, help="Mixture wav before separation")
    parser.add_argument("--references", nargs="+", required=True, help="Clean source wav files")
    parser.add_argument("--estimates", nargs="+", required=True, help="Separated wav files")
    args = parser.parse_args()

    mixture, _ = read_wav_mono(Path(args.mixture))
    references = load_many(args.references)
    estimates = load_many(args.estimates)
    n = min(mixture.shape[0], references.shape[1], estimates.shape[1])
    mixture = mixture[:n]
    references = references[:, :n]
    estimates = estimates[:, :n]

    mixture_repeated = np.stack([mixture for _ in range(references.shape[0])], axis=0)
    sdr_in, _ = best_permutation_si_sdr(mixture_repeated, references)
    sdr_out, perm = best_permutation_si_sdr(estimates, references)
    delta = sdr_out - sdr_in
    print(f"SDR_in: {sdr_in:.3f} dB")
    print(f"SDR_out: {sdr_out:.3f} dB")
    print(f"Delta_SDR: {delta:.3f} dB")
    print(f"Best permutation: {perm}")


if __name__ == "__main__":
    main()
