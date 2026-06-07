"""Train the BLSTM-based speech separation model."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import numpy as np

from cocktail_separation.audio import istft
from cocktail_separation.dataset import SpeechMixtureDataset, collate_deep_clustering
from cocktail_separation.deep_clustering import build_model, deep_clustering_loss, mask_inference_loss
from cocktail_separation.metrics import best_permutation_si_sdr


def _require_torch_training():
    try:
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as exc:
        raise ImportError(
            "Training requires torch and tqdm. Install with: pip install -r requirements.txt"
        ) from exc
    return torch, DataLoader, tqdm


def checkpoint_payload(model, args, freq_bins: int) -> dict:
    return {
        "model_state": model.state_dict(),
        "freq_bins": freq_bins,
        "embedding_dim": args.embedding_dim,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "n_fft": args.n_fft,
        "hop_length": args.hop_length,
        "num_speakers": args.num_speakers,
        "dc_loss_weight": args.dc_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "mask_activation": args.mask_activation,
    }


def make_log_path(args) -> Path:
    if args.log_file:
        return Path(args.log_file)
    checkpoint_stem = Path(args.checkpoint).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(args.log_dir) / f"{checkpoint_stem}_{timestamp}.csv"


def evaluate_sdr(model, dataset, torch, device: str, num_speakers: int, max_items: int = 3) -> dict[str, float]:
    model.eval()
    sdr_in_scores: list[float] = []
    sdr_out_scores: list[float] = []
    item_count = len(dataset) if max_items <= 0 else min(max_items, len(dataset))
    with torch.no_grad():
        for idx in range(item_count):
            item = dataset[idx]
            features = torch.as_tensor(item["features"][None, :, :], dtype=torch.float32, device=device)
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
            references = item["sources_audio"][:num_speakers]
            mixture_repeated = np.stack([item["mixture_audio"] for _ in range(num_speakers)], axis=0)
            sdr_in, _ = best_permutation_si_sdr(mixture_repeated, references)
            sdr_out, _ = best_permutation_si_sdr(estimates, references)
            sdr_in_scores.append(sdr_in)
            sdr_out_scores.append(sdr_out)

    sdr_in_mean = float(np.mean(sdr_in_scores))
    sdr_out_mean = float(np.mean(sdr_out_scores))
    return {
        "sdr_in": sdr_in_mean,
        "sdr_out": sdr_out_mean,
        "delta_sdr": sdr_out_mean - sdr_in_mean,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train deep clustering model")
    parser.add_argument(
        "--manifest",
        default="data/librispeech_easy_generated/train.jsonl",
        help="JSONL manifest with mixture and sources",
    )
    parser.add_argument("--checkpoint", default="outputs/deep_clustering_easy.pt", help="Output checkpoint")
    parser.add_argument(
        "--best-checkpoint",
        default="outputs/deep_clustering_easy_best.pt",
        help="Best validation checkpoint",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--n-fft", type=int, default=512)
    parser.add_argument("--hop-length", type=int, default=128)
    parser.add_argument("--embedding-dim", type=int, default=20)
    parser.add_argument("--hidden-size", type=int, default=300)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dc-loss-weight", type=float, default=0.0, help="Weight for deep clustering loss")
    parser.add_argument("--mask-loss-weight", type=float, default=1.0, help="Weight for signal-approximation mask loss")
    parser.add_argument("--mask-activation", default="sigmoid", choices=["sigmoid", "softmax"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument(
        "--eval-manifest",
        default=None,
        help="Validation manifest used for SDR evaluation; defaults to valid.jsonl beside --manifest",
    )
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate SDR every N epochs; 0 disables")
    parser.add_argument("--eval-items", type=int, default=3, help="Number of eval items; 0 means all items")
    parser.add_argument("--log-dir", default="outputs/logs", help="Directory for per-run CSV training logs")
    parser.add_argument("--log-file", default=None, help="Optional explicit CSV log file path")
    args = parser.parse_args()

    torch, DataLoader, tqdm = _require_torch_training()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = SpeechMixtureDataset(args.manifest, n_fft=args.n_fft, hop_length=args.hop_length)
    if len(dataset) == 0:
        raise ValueError("Manifest is empty")
    eval_manifest = args.eval_manifest
    if eval_manifest is None:
        candidate = Path(args.manifest).with_name("valid.jsonl")
        eval_manifest = str(candidate) if candidate.exists() else None
    eval_dataset = (
        SpeechMixtureDataset(eval_manifest, n_fft=args.n_fft, hop_length=args.hop_length)
        if eval_manifest
        else dataset
    )
    print(f"Training manifest: {Path(args.manifest).resolve()}")
    print(f"Evaluation manifest: {Path(eval_manifest).resolve() if eval_manifest else 'training manifest'}")
    log_path = make_log_path(args)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Training log: {log_path.resolve()}")
    first = dataset[0]
    freq_bins = int(first["freq_bins"][0])

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_deep_clustering,
    )
    model = build_model(
        freq_bins=freq_bins,
        embedding_dim=args.embedding_dim,
        hidden_size=args.hidden_size,
        layers=args.layers,
        num_speakers=args.num_speakers,
        mask_activation=args.mask_activation,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_delta_sdr = -np.inf
    best_metrics: dict[str, float | int | str] | None = None

    with log_path.open("w", newline="", encoding="utf-8") as log_file:
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "epoch",
                "mean_loss",
                "dc_loss",
                "sa_loss",
                "sdr_in_db",
                "sdr_out_db",
                "delta_sdr_db",
                "is_best",
            ],
        )
        log_writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            losses: list[float] = []
            dc_losses: list[float] = []
            mi_losses: list[float] = []
            bar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
            for batch in bar:
                features = torch.as_tensor(batch["features"], dtype=torch.float32, device=device)
                labels = torch.as_tensor(batch["labels"], dtype=torch.long, device=device)
                weights = torch.as_tensor(batch["weights"], dtype=torch.float32, device=device)
                mix_magnitude = torch.as_tensor(batch["mix_magnitude"], dtype=torch.float32, device=device)
                source_magnitudes = torch.as_tensor(batch["source_magnitudes"], dtype=torch.float32, device=device)

                optimizer.zero_grad()
                embeddings, predicted_masks = model(features, return_masks=True)
                if args.dc_loss_weight > 0:
                    dc_loss = deep_clustering_loss(embeddings, labels, weights, args.num_speakers)
                else:
                    dc_loss = torch.zeros((), dtype=torch.float32, device=device)
                mi_loss = mask_inference_loss(predicted_masks, mix_magnitude, source_magnitudes, weights)
                loss = args.dc_loss_weight * dc_loss + args.mask_loss_weight * mi_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

                losses.append(float(loss.detach().cpu()))
                dc_losses.append(float(dc_loss.detach().cpu()))
                mi_losses.append(float(mi_loss.detach().cpu()))
                bar.set_postfix(loss=np.mean(losses), dc=np.mean(dc_losses), sa=np.mean(mi_losses))

            mean_loss = float(np.mean(losses))
            mean_dc_loss = float(np.mean(dc_losses))
            mean_mi_loss = float(np.mean(mi_losses))
            print(
                f"epoch={epoch} mean_loss={mean_loss:.6f} "
                f"dc_loss={mean_dc_loss:.6f} sa_loss={mean_mi_loss:.6f}"
            )

            metrics = None
            is_best = False
            if args.eval_every > 0 and (epoch == 1 or epoch % args.eval_every == 0 or epoch == args.epochs):
                metrics = evaluate_sdr(
                    model,
                    eval_dataset,
                    torch,
                    device,
                    num_speakers=args.num_speakers,
                    max_items=args.eval_items,
                )
                print(
                    f"epoch={epoch} eval "
                    f"SDR_in={metrics['sdr_in']:.3f} dB "
                    f"SDR_out={metrics['sdr_out']:.3f} dB "
                    f"Delta_SDR={metrics['delta_sdr']:.3f} dB"
                )
                if metrics["delta_sdr"] > best_delta_sdr:
                    is_best = True
                    best_delta_sdr = metrics["delta_sdr"]
                    best_checkpoint = Path(args.best_checkpoint)
                    best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
                    payload = checkpoint_payload(model, args, freq_bins)
                    payload["best_delta_sdr"] = best_delta_sdr
                    payload["best_epoch"] = epoch
                    payload["best_sdr_in"] = metrics["sdr_in"]
                    payload["best_sdr_out"] = metrics["sdr_out"]
                    torch.save(payload, best_checkpoint)
                    best_metrics = {
                        "epoch": epoch,
                        "sdr_in": metrics["sdr_in"],
                        "sdr_out": metrics["sdr_out"],
                        "delta_sdr": metrics["delta_sdr"],
                        "checkpoint": str(best_checkpoint.resolve()),
                    }
                    print(
                        f"Saved best checkpoint: {best_checkpoint.resolve()} "
                        f"(epoch={epoch}, Delta_SDR={best_delta_sdr:.3f} dB)"
                    )

            log_writer.writerow(
                {
                    "epoch": epoch,
                    "mean_loss": f"{mean_loss:.6f}",
                    "dc_loss": f"{mean_dc_loss:.6f}",
                    "sa_loss": f"{mean_mi_loss:.6f}",
                    "sdr_in_db": "" if metrics is None else f"{metrics['sdr_in']:.3f}",
                    "sdr_out_db": "" if metrics is None else f"{metrics['sdr_out']:.3f}",
                    "delta_sdr_db": "" if metrics is None else f"{metrics['delta_sdr']:.3f}",
                    "is_best": int(is_best),
                }
            )
            log_file.flush()

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint_payload(model, args, freq_bins), checkpoint)
    print(f"Saved checkpoint: {checkpoint.resolve()}")
    print(f"Saved training log: {log_path.resolve()}")
    if best_metrics is not None:
        print("Best validation summary:")
        print(f"  epoch: {best_metrics['epoch']}")
        print(f"  SDR_in: {best_metrics['sdr_in']:.3f} dB")
        print(f"  SDR_out: {best_metrics['sdr_out']:.3f} dB")
        print(f"  Delta_SDR: {best_metrics['delta_sdr']:.3f} dB")
        print(f"  checkpoint: {best_metrics['checkpoint']}")
    elif args.eval_every <= 0:
        print("Best validation summary unavailable because evaluation was disabled.")
    else:
        print("Best validation summary unavailable because no evaluation was run.")


if __name__ == "__main__":
    main()
