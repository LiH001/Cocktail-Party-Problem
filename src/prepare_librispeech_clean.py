"""Convert LibriSpeech FLAC utterances into clean speaker WAV folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import soundfile as sf

from cocktail_separation.audio import normalize_peak, write_wav


def collect_speakers(librispeech_split: Path) -> dict[str, list[Path]]:
    speakers: dict[str, list[Path]] = {}
    for speaker_dir in sorted(p for p in librispeech_split.iterdir() if p.is_dir()):
        flacs = sorted(speaker_dir.glob("*/*-*.flac"))
        if flacs:
            speakers[speaker_dir.name] = flacs
    return speakers


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert LibriSpeech flac files to clean wav folders")
    parser.add_argument("--librispeech-split", default="data/raw/LibriSpeech/dev-clean")
    parser.add_argument("--out-dir", default="data/clean_librispeech_easy")
    parser.add_argument("--speakers", type=int, default=3)
    parser.add_argument("--utterances-per-speaker", type=int, default=20)
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    split = Path(args.librispeech_split)
    out_dir = Path(args.out_dir)
    speakers = collect_speakers(split)
    eligible = [(speaker, flacs) for speaker, flacs in speakers.items() if len(flacs) >= args.utterances_per_speaker]
    if len(eligible) < args.speakers:
        raise ValueError(
            f"Need {args.speakers} speakers with >= {args.utterances_per_speaker} utterances, "
            f"found {len(eligible)}"
        )

    selected = sorted(eligible, key=lambda item: len(item[1]), reverse=True)[: args.speakers]
    for speaker, flacs in selected:
        speaker_dir = out_dir / f"speaker_{speaker}"
        speaker_dir.mkdir(parents=True, exist_ok=True)
        for idx, flac_path in enumerate(flacs[: args.utterances_per_speaker]):
            audio, sample_rate = sf.read(str(flac_path), dtype="float32", always_2d=False)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)
            if sample_rate != args.sample_rate:
                raise ValueError(f"{flac_path} has sample_rate={sample_rate}, expected {args.sample_rate}")
            wav_path = speaker_dir / f"{speaker}_utt_{idx:03d}.wav"
            write_wav(wav_path, normalize_peak(audio, 0.95), sample_rate)

    print(f"Prepared LibriSpeech clean wavs under: {out_dir.resolve()}")
    for speaker, _ in selected:
        print(f"speaker_{speaker}: {args.utterances_per_speaker} wavs")


if __name__ == "__main__":
    main()
