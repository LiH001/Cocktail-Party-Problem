"""Run a single-sample overfit check for the separation model."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cocktail_separation.audio import istft, normalize_peak, write_wav
from cocktail_separation.dataset import SpeechMixtureDataset
from cocktail_separation.deep_clustering import build_model, mask_inference_loss
from cocktail_separation.metrics import best_permutation_si_sdr


def _require_torch():
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Single-sample overfit check requires PyTorch") from exc
    return torch


def separate_with_model(model, item, dataset, torch, device, num_speakers):
    model.eval()
    with torch.no_grad():
        features = torch.as_tensor(item["features"][None, :, :], dtype=torch.float32, device=device)
        _, masks = model(features, return_masks=True)
    masks_np = masks[0].detach().cpu().numpy()
    estimates = np.stack(
        [
            istft(
                item["mixture_spec"] * masks_np[spk],
                n_fft=dataset.n_fft,
                hop_length=dataset.hop_length,
                length=item["mixture_audio"].shape[0],
            )
            for spk in range(num_speakers)
        ],
        axis=0,
    )
    return estimates


def sdr_metrics(estimates, item, num_speakers):
    refs = item["sources_audio"][:num_speakers]
    mix_rep = np.stack([item["mixture_audio"] for _ in range(num_speakers)], axis=0)
    sdr_in, _ = best_permutation_si_sdr(mix_rep, refs)
    sdr_out, perm = best_permutation_si_sdr(estimates, refs)
    return sdr_in, sdr_out, sdr_out - sdr_in, perm


def oracle_metrics(item, dataset, num_speakers):
    masks = item["mask_targets"][:num_speakers]
    estimates = np.stack(
        [
            istft(
                item["mixture_spec"] * masks[spk],
                n_fft=dataset.n_fft,
                hop_length=dataset.hop_length,
                length=item["mixture_audio"].shape[0],
            )
            for spk in range(num_speakers)
        ],
        axis=0,
    )
    return sdr_metrics(estimates, item, num_speakers)


def main() -> None:
    parser = argparse.ArgumentParser(description="Overfit one manifest item as a model sanity check")
    parser.add_argument("--manifest", default="data/librispeech_easy_generated/train.jsonl")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=300)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--out-dir", default="outputs/overfit_one")
    args = parser.parse_args()

    torch = _require_torch()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = SpeechMixtureDataset(args.manifest)
    item = dataset[args.index]
    freq_bins = int(item["freq_bins"][0])
    model = build_model(
        freq_bins=freq_bins,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        layers=args.layers,
        num_speakers=args.num_speakers,
        mask_activation="sigmoid",
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    oracle = oracle_metrics(item, dataset, args.num_speakers)
    print(
        f"Oracle: SDR_in={oracle[0]:.3f} dB "
        f"SDR_out={oracle[1]:.3f} dB Delta_SDR={oracle[2]:.3f} dB"
    )

    features = torch.as_tensor(item["features"][None, :, :], dtype=torch.float32, device=device)
    weights = torch.as_tensor(item["weights"][None, :], dtype=torch.float32, device=device)
    mix_mag = torch.as_tensor(item["mix_magnitude"][None, :, :], dtype=torch.float32, device=device)
    src_mag = torch.as_tensor(item["source_magnitudes"][None, :, :, :], dtype=torch.float32, device=device)

    for step in range(1, args.steps + 1):
        model.train()
        optimizer.zero_grad()
        _, masks = model(features, return_masks=True)
        loss = mask_inference_loss(masks, mix_mag, src_mag, weights)
        loss.backward()
        optimizer.step()

        if step == 1 or step % 50 == 0 or step == args.steps:
            estimates = separate_with_model(model, item, dataset, torch, device, args.num_speakers)
            sdr_in, sdr_out, delta, perm = sdr_metrics(estimates, item, args.num_speakers)
            print(
                f"step={step} loss={float(loss.detach().cpu()):.6f} "
                f"SDR_in={sdr_in:.3f} dB SDR_out={sdr_out:.3f} dB "
                f"Delta_SDR={delta:.3f} dB perm={perm}"
            )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    estimates = separate_with_model(model, item, dataset, torch, device, args.num_speakers)
    write_wav(out_dir / "mixture.wav", normalize_peak(item["mixture_audio"]), 16000)
    for spk in range(args.num_speakers):
        write_wav(out_dir / f"source_{spk + 1}.wav", normalize_peak(item["sources_audio"][spk]), 16000)
        write_wav(out_dir / f"separated_{spk + 1}.wav", normalize_peak(estimates[spk]), 16000)
    print(f"Saved overfit audio to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
