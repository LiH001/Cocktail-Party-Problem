"""Create speech-mixture WAV files and JSONL manifests.

Input speaker folders are expected to use this layout:

data/clean/
  speaker_a/*.wav
  speaker_b/*.wav
  speaker_c/*.wav

Each speaker folder must contain enough clean utterances for the selected
experiment. The script writes aligned source contributions, mono mixtures, and
manifest rows that point to the generated files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from cocktail_separation.audio import normalize_peak, read_wav_mono, write_wav


def speaker_wavs(clean_root: Path) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for speaker_dir in sorted(p for p in clean_root.iterdir() if p.is_dir()):
        wavs = sorted(speaker_dir.glob("*.wav"))
        speakers[speaker_dir.name] = wavs
    return speakers


def crop_or_pad(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if audio.shape[0] == length:
        return audio
    if audio.shape[0] > length:
        start = int(rng.integers(0, audio.shape[0] - length + 1))
        return audio[start : start + length]
    return np.pad(audio, (0, length - audio.shape[0])).astype(np.float32)


def collect_noise_files(noise_root: Path | None) -> list[Path]:
    if noise_root is None:
        return []
    exts = {".wav", ".flac"}
    return sorted(path for path in noise_root.rglob("*") if path.suffix.lower() in exts)


def rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def peak_normalize_with_scale(audio: np.ndarray, peak: float = 0.95) -> tuple[np.ndarray, float]:
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current < 1e-8:
        return audio.astype(np.float32), 1.0
    scale = peak / current
    return (audio * scale).astype(np.float32), scale


def resample_linear(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return audio.astype(np.float32)
    if audio.size == 0:
        return audio.astype(np.float32)
    duration = audio.shape[0] / float(source_rate)
    target_len = max(1, int(round(duration * target_rate)))
    source_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    target_x = np.linspace(0.0, duration, num=target_len, endpoint=False)
    return np.interp(target_x, source_x, audio).astype(np.float32)


def add_noise_at_snr(
    clean_mix: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    noise = crop_or_pad(noise, clean_mix.shape[0], rng)
    clean_rms = rms(clean_mix)
    noise_rms = rms(noise)
    target_noise_rms = clean_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / max(noise_rms, 1e-8))
    return clean_mix + scaled_noise


def main() -> None:
    parser = argparse.ArgumentParser(description="Create cocktail-party mixture manifest")
    parser.add_argument("--clean-root", default="data/clean_librispeech_easy", help="Speaker subfolders with clean wavs")
    parser.add_argument("--out-dir", default="data/librispeech_easy_generated", help="Output data directory")
    parser.add_argument("--num-mixtures", type=int, default=250, help="Number of mixtures to create")
    parser.add_argument("--seconds", type=float, default=5.0, help="Length of each generated mixture")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--noise-root", default=None, help="Optional folder with real environmental noise wav/flac")
    parser.add_argument("--snr-db", type=float, default=10.0, help="SNR used when adding noise")
    parser.add_argument("--source1-gain", type=float, default=0.7, help="Mixing gain for source 1")
    parser.add_argument("--source2-gain", type=float, default=0.3, help="Mixing gain for source 2")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    clean_root = Path(args.clean_root)
    out_dir = Path(args.out_dir)
    rng = np.random.default_rng(args.seed)
    speakers = speaker_wavs(clean_root)
    noise_files = collect_noise_files(Path(args.noise_root) if args.noise_root else None)
    if len(speakers) < 2:
        raise ValueError("Need at least two speaker folders under clean-root")

    for speaker, wavs in speakers.items():
        if len(wavs) < 20:
            raise ValueError(f"Speaker {speaker} has {len(wavs)} wavs; at least 20 are required")

    mix_dir = out_dir / "mix"
    source_dir = out_dir / "sources"
    for path in [mix_dir, source_dir]:
        path.mkdir(parents=True, exist_ok=True)

    manifest_path = out_dir / "manifest.jsonl"
    target_len = int(args.seconds * args.sample_rate)
    speaker_names = sorted(speakers)
    rows = []

    for idx in range(args.num_mixtures):
        spk1, spk2 = rng.choice(speaker_names, size=2, replace=False)
        wav1 = rng.choice(speakers[str(spk1)])
        wav2 = rng.choice(speakers[str(spk2)])
        src1, sr1 = read_wav_mono(wav1)
        src2, sr2 = read_wav_mono(wav2)
        if sr1 != args.sample_rate or sr2 != args.sample_rate:
            raise ValueError(
                f"Sample rate mismatch in mixture {idx:03d}: "
                f"{wav1}={sr1}, {wav2}={sr2}, expected {args.sample_rate}"
            )

        src1 = crop_or_pad(src1, target_len, rng)
        src2 = crop_or_pad(src2, target_len, rng)
        sources = np.stack([normalize_peak(src1, 0.8), normalize_peak(src2, 0.8)], axis=0)
        source_contribs = np.stack(
            [args.source1_gain * sources[0], args.source2_gain * sources[1]],
            axis=0,
        )
        mono_mix, speech_scale = peak_normalize_with_scale(np.sum(source_contribs, axis=0))
        source_contribs = source_contribs * speech_scale
        noise_path = None
        if noise_files:
            noise_path = Path(rng.choice(noise_files))
            noise, noise_sr = read_wav_mono(noise_path)
            noise = resample_linear(noise, noise_sr, args.sample_rate)
            mono_mix, noisy_scale = peak_normalize_with_scale(add_noise_at_snr(mono_mix, noise, args.snr_db, rng))
            source_contribs = source_contribs * noisy_scale

        stem = f"mix_{idx:03d}_{spk1}_{spk2}"
        src1_path = source_dir / f"{stem}_s1.wav"
        src2_path = source_dir / f"{stem}_s2.wav"
        mix_path = mix_dir / f"{stem}.wav"
        write_wav(src1_path, source_contribs[0], args.sample_rate)
        write_wav(src2_path, source_contribs[1], args.sample_rate)
        write_wav(mix_path, mono_mix, args.sample_rate)
        row = {"mixture": str(mix_path), "sources": [str(src1_path), str(src2_path)]}
        row["source_gains"] = [args.source1_gain, args.source2_gain]
        if noise_path is not None:
            row["noise"] = str(noise_path)
            row["snr_db"] = args.snr_db
        rows.append(row)

    with manifest_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Created {len(rows)} mixtures")
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Mixtures for deep clustering: {mix_dir.resolve()}")
    if noise_files:
        print(f"Added real noise from: {Path(args.noise_root).resolve()}")
        print(f"SNR: {args.snr_db:.1f} dB")


if __name__ == "__main__":
    main()
