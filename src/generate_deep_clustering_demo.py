"""Generate demo audio and SI-SDR summaries from a trained checkpoint."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from cocktail_separation.audio import istft, normalize_peak, write_wav, read_wav_mono
from cocktail_separation.dataset import SpeechMixtureDataset
from cocktail_separation.deep_clustering import build_model
from cocktail_separation.metrics import best_permutation_si_sdr


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Demo generation requires PyTorch. Install with: pip install -r requirements.txt") from exc
    return torch


def separate_item(model, item: dict, dataset: SpeechMixtureDataset, torch, device: str, num_speakers: int, seed: int):
    features = torch.as_tensor(item["features"][None, :, :], dtype=torch.float32, device=device)
    with torch.no_grad():
        _, predicted_masks = model(features, return_masks=True)

    mix_spec = item["mixture_spec"]
    masks = predicted_masks[0].detach().cpu().numpy()
    estimates = np.stack(
        [
            istft(
                mix_spec * masks[spk],
                n_fft=dataset.n_fft,
                hop_length=dataset.hop_length,
                length=item["mixture_audio"].shape[0],
            )
            for spk in range(num_speakers)
        ],
        axis=0,
    )
    return estimates


def main() -> None:
    parser = argparse.ArgumentParser(description="Create demo audio groups from a trained deep-clustering model")
    parser.add_argument("--checkpoint", default="outputs/deep_clustering_easy_best.pt")
    parser.add_argument("--manifest", default="data/librispeech_easy_generated/valid.jsonl")
    parser.add_argument("--out-dir", default="demo_audio_easy")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--select", default="best", choices=["first", "best"], help="Choose first N or best N by SDR")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    args = parser.parse_args()

    torch = _require_torch()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_speakers = int(ckpt.get("num_speakers", 2))

    dataset = SpeechMixtureDataset(
        args.manifest,
        n_fft=int(ckpt["n_fft"]),
        hop_length=int(ckpt["hop_length"]),
    )
    model = build_model(
        freq_bins=int(ckpt["freq_bins"]),
        embedding_dim=int(ckpt["embedding_dim"]),
        hidden_size=int(ckpt["hidden_size"]),
        layers=int(ckpt["layers"]),
        num_speakers=num_speakers,
        mask_activation=str(ckpt.get("mask_activation", "sigmoid")),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    evaluated = []

    eval_count = len(dataset) if args.select == "best" else min(args.count, len(dataset))
    for idx in range(eval_count):
        item = dataset[idx]
        estimates = separate_item(model, item, dataset, torch, device, num_speakers, seed=idx)
        references = item["sources_audio"][:num_speakers]
        mixture_repeated = np.stack([item["mixture_audio"] for _ in range(num_speakers)], axis=0)
        sdr_in, _ = best_permutation_si_sdr(mixture_repeated, references)
        sdr_out, perm = best_permutation_si_sdr(estimates, references)
        evaluated.append(
            {
                "idx": idx,
                "item": item,
                "estimates": estimates,
                "references": references,
                "sdr_in": float(sdr_in),
                "sdr_out": float(sdr_out),
                "delta_sdr": float(sdr_out - sdr_in),
                "perm": perm,
            }
        )

    summary = {
        "count": len(evaluated),
        "sdr_in": float(np.mean([row["sdr_in"] for row in evaluated])),
        "sdr_out": float(np.mean([row["sdr_out"] for row in evaluated])),
        "delta_sdr": float(np.mean([row["delta_sdr"] for row in evaluated])),
    }

    all_rows = [
        {
            "manifest_index": row["idx"],
            "sdr_in_db": round(row["sdr_in"], 3),
            "sdr_out_db": round(row["sdr_out"], 3),
            "delta_sdr_db": round(row["delta_sdr"], 3),
            "best_permutation": str(row["perm"]),
        }
        for row in evaluated
    ]

    selected = evaluated
    if args.select == "best":
        selected = sorted(evaluated, key=lambda row: row["sdr_out"], reverse=True)

    rows = []
    for rank, row in enumerate(selected[: args.count], start=1):
        item = row["item"]
        estimates = row["estimates"]
        references = row["references"]
        case_dir = out_dir / f"case_{rank:02d}_deep_clustering"
        case_dir.mkdir(parents=True, exist_ok=True)
        write_wav(case_dir / "mixture.wav", normalize_peak(item["mixture_audio"]), 16000)
        for spk in range(num_speakers):
            write_wav(case_dir / f"source_{spk + 1}.wav", normalize_peak(references[spk]), 16000)
            write_wav(case_dir / f"separated_{spk + 1}.wav", normalize_peak(estimates[spk]), 16000)

        raw_item = dataset.items[row["idx"]]
        if "noise" in raw_item:
            try:
                noise_audio, noise_sr = read_wav_mono(raw_item["noise"])
                if noise_sr != 16000:
                    duration = noise_audio.shape[0] / float(noise_sr)
                    target_len = max(1, int(round(duration * 16000)))
                    source_x = np.linspace(0.0, duration, num=noise_audio.shape[0], endpoint=False)
                    target_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
                    noise_audio = np.interp(target_x, source_x, noise_audio).astype(np.float32)
                
                length = item["mixture_audio"].shape[0]
                if noise_audio.shape[0] > length:
                    noise_audio = noise_audio[:length]
                elif noise_audio.shape[0] < length:
                    noise_audio = np.pad(noise_audio, (0, length - noise_audio.shape[0])).astype(np.float32)
                
                write_wav(case_dir / "noise.wav", normalize_peak(noise_audio), 16000)
            except Exception as e:
                print(f"Warning: could not save noise.wav for case {rank:02d}: {e}")

        rows.append(
            {
                "case": case_dir.name,
                "manifest_index": row["idx"],
                "sdr_in_db": round(row["sdr_in"], 3),
                "sdr_out_db": round(row["sdr_out"], 3),
                "delta_sdr_db": round(row["delta_sdr"], 3),
                "best_permutation": str(row["perm"]),
            }
        )

    all_metrics_path = out_dir / "metrics_all.csv"
    with all_metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["manifest_index", "sdr_in_db", "sdr_out_db", "delta_sdr_db", "best_permutation"],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    metrics_path = out_dir / "metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["case", "manifest_index", "sdr_in_db", "sdr_out_db", "delta_sdr_db", "best_permutation"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved deep-clustering demo audio to: {out_dir.resolve()}")
    print(f"Saved metrics to: {metrics_path.resolve()}")
    print(f"Saved full evaluated metrics to: {all_metrics_path.resolve()}")
    print(
        f"Evaluated summary ({summary['count']} items): "
        f"SDR_in={summary['sdr_in']:.3f} dB, "
        f"SDR_out={summary['sdr_out']:.3f} dB, "
        f"Delta_SDR={summary['delta_sdr']:.3f} dB"
    )
    if args.select == "best":
        print(f"Selected best {len(rows)} demo cases below; these are not the full validation average.")
    for row in rows:
        print(
            f"{row['case']}: SDR_in={row['sdr_in_db']:.3f} dB, "
            f"SDR_out={row['sdr_out_db']:.3f} dB, "
            f"Delta_SDR={row['delta_sdr_db']:.3f} dB"
        )


if __name__ == "__main__":
    main()
