"""Evaluate the ideal-ratio-mask SI-SDR upper bound for a manifest."""

from __future__ import annotations

import argparse

import numpy as np

from cocktail_separation.audio import istft
from cocktail_separation.dataset import SpeechMixtureDataset
from cocktail_separation.metrics import best_permutation_si_sdr


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate oracle ideal-ratio-mask SI-SDR")
    parser.add_argument("--manifest", default="data/librispeech_easy_generated/valid.jsonl")
    parser.add_argument("--items", type=int, default=0, help="Number of items; 0 means all")
    args = parser.parse_args()

    dataset = SpeechMixtureDataset(args.manifest)
    count = len(dataset) if args.items <= 0 else min(args.items, len(dataset))
    sdr_in_scores = []
    sdr_out_scores = []
    for idx in range(count):
        item = dataset[idx]
        masks = item["mask_targets"]
        mix_spec = item["mixture_spec"]
        estimates = np.stack(
            [
                istft(
                    mix_spec * masks[spk],
                    n_fft=dataset.n_fft,
                    hop_length=dataset.hop_length,
                    length=item["mixture_audio"].shape[0],
                )
                for spk in range(masks.shape[0])
            ],
            axis=0,
        )
        references = item["sources_audio"]
        mixture_repeated = np.stack([item["mixture_audio"] for _ in range(references.shape[0])], axis=0)
        sdr_in, _ = best_permutation_si_sdr(mixture_repeated, references)
        sdr_out, _ = best_permutation_si_sdr(estimates, references)
        sdr_in_scores.append(sdr_in)
        sdr_out_scores.append(sdr_out)

    sdr_in_mean = float(np.mean(sdr_in_scores))
    sdr_out_mean = float(np.mean(sdr_out_scores))
    print(f"Items: {count}")
    print(f"Oracle SDR_in: {sdr_in_mean:.3f} dB")
    print(f"Oracle SDR_out: {sdr_out_mean:.3f} dB")
    print(f"Oracle Delta_SDR: {sdr_out_mean - sdr_in_mean:.3f} dB")


if __name__ == "__main__":
    main()
