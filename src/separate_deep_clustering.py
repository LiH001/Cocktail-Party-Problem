"""Separate one mixture WAV with a trained checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cocktail_separation.audio import istft, normalize_peak, stft
from cocktail_separation.dataset import read_audio
from cocktail_separation.deep_clustering import build_model


def _require_torch_and_soundfile():
    try:
        import torch
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Separation requires torch and soundfile. Install with: pip install -r requirements.txt"
        ) from exc
    return torch, sf


def main() -> None:
    parser = argparse.ArgumentParser(description="Deep clustering separation")
    parser.add_argument("--checkpoint", default="outputs/deep_clustering_easy_best.pt")
    parser.add_argument(
        "--mixture",
        default="data/librispeech_easy_generated/mix/mix_000_speaker_2277_speaker_3752.wav",
    )
    parser.add_argument("--out-dir", default="outputs/separated")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch, sf = _require_torch_and_soundfile()
    device = args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.checkpoint, map_location=device)
    num_speakers = args.num_speakers or int(ckpt.get("num_speakers", 2))

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

    mixture, sample_rate = read_audio(args.mixture)
    spec = stft(mixture, n_fft=int(ckpt["n_fft"]), hop_length=int(ckpt["hop_length"]))
    magnitude = np.abs(spec).astype(np.float32)
    features = np.log1p(magnitude)[None, :, :]

    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        _, predicted_masks = model(x, return_masks=True)
        masks = predicted_masks[0].detach().cpu().numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(num_speakers):
        separated_spec = spec * masks[idx]
        audio = istft(
            separated_spec,
            n_fft=int(ckpt["n_fft"]),
            hop_length=int(ckpt["hop_length"]),
            length=mixture.shape[0],
        )
        sf.write(str(out_dir / f"speaker_{idx + 1}.wav"), normalize_peak(audio), sample_rate)

    print(f"Saved separated wav files to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
