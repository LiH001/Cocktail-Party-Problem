"""Manifest loading and feature extraction for speech-mixture datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

import numpy as np

from .audio import stft

if TYPE_CHECKING:
    from torch.utils.data import Dataset as _DatasetBase
else:
    class _DatasetBase:
        pass


def _require_soundfile():
    try:
        import soundfile as sf
    except ImportError as exc:
        raise ImportError(
            "Deep clustering data loading needs soundfile. Install with: "
            "pip install -r requirements.txt"
        ) from exc
    return sf


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    items: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "mixture" not in item or "sources" not in item:
                raise ValueError(f"Manifest line {line_no} needs mixture and sources fields")
            items.append(item)
    return items


def read_audio(path: str | Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    sf = _require_soundfile()
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    if sample_rate is not None and sr != sample_rate:
        raise ValueError(f"{path} has sample_rate={sr}, expected {sample_rate}")
    return np.asarray(audio, dtype=np.float32), int(sr)


def dominant_source_labels(
    source_specs: np.ndarray,
    silence_threshold: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Create deep-clustering labels from clean source magnitudes.

    Returns
    -------
    labels:
        Integer source id per time-frequency unit, flattened to [T * F].
    weights:
        Binary activity mask. Silent bins are excluded from the loss.
    """

    mags = np.abs(source_specs)
    labels = np.argmax(mags, axis=0).reshape(-1).astype(np.int64)
    total = np.sum(mags, axis=0).reshape(-1)
    weights = (total > silence_threshold).astype(np.float32)
    return labels, weights


def ideal_ratio_masks(source_specs: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return ideal ratio masks with shape [sources, frames, freq_bins]."""

    mags = np.abs(source_specs).astype(np.float32)
    return mags / (np.sum(mags, axis=0, keepdims=True) + eps)


def pad_1d(values: list[np.ndarray], value: float = 0.0) -> np.ndarray:
    max_len = max(v.shape[0] for v in values)
    out = np.full((len(values), max_len), value, dtype=values[0].dtype)
    for idx, v in enumerate(values):
        out[idx, : v.shape[0]] = v
    return out


class SpeechMixtureDataset(_DatasetBase):
    """Dataset that returns waveform, STFT, mask, and loss-target arrays."""

    def __init__(
        self,
        manifest: str | Path,
        n_fft: int = 512,
        hop_length: int = 128,
        sample_rate: int | None = None,
    ) -> None:
        self.items = load_manifest(manifest)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        item = self.items[idx]
        mixture, sr = read_audio(item["mixture"], self.sample_rate)
        sources = [read_audio(path, sr)[0] for path in item["sources"]]
        min_len = min([mixture.shape[0], *(s.shape[0] for s in sources)])
        mixture = mixture[:min_len]
        sources = [s[:min_len] for s in sources]

        mix_spec = stft(mixture, n_fft=self.n_fft, hop_length=self.hop_length)
        source_specs = np.stack(
            [stft(s, n_fft=self.n_fft, hop_length=self.hop_length) for s in sources],
            axis=0,
        )
        mix_magnitude = np.abs(mix_spec).astype(np.float32)
        source_magnitudes = np.abs(source_specs).astype(np.float32)
        mag = np.log1p(mix_magnitude).astype(np.float32)
        mag = ((mag - mag.mean()) / (mag.std() + 1e-6)).astype(np.float32)
        labels, weights = dominant_source_labels(source_specs)
        mask_targets = ideal_ratio_masks(source_specs).astype(np.float32)

        return {
            "features": mag,
            "labels": labels,
            "weights": weights,
            "mask_targets": mask_targets,
            "mix_magnitude": mix_magnitude,
            "source_magnitudes": source_magnitudes,
            "mixture_audio": mixture,
            "sources_audio": np.stack(sources, axis=0),
            "mixture_spec": mix_spec,
            "frames": np.array([mag.shape[0]], dtype=np.int64),
            "freq_bins": np.array([mag.shape[1]], dtype=np.int64),
        }


def collate_deep_clustering(batch: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    features = [item["features"] for item in batch]
    labels = [item["labels"] for item in batch]
    weights = [item["weights"] for item in batch]
    mask_targets = [item["mask_targets"] for item in batch]
    mix_magnitudes = [item["mix_magnitude"] for item in batch]
    source_magnitudes = [item["source_magnitudes"] for item in batch]
    frames = np.array([int(item["frames"][0]) for item in batch], dtype=np.int64)
    freq_bins = int(batch[0]["freq_bins"][0])
    num_speakers = int(mask_targets[0].shape[0])

    max_frames = max(x.shape[0] for x in features)
    feat_out = np.zeros((len(batch), max_frames, freq_bins), dtype=np.float32)
    label_out = np.zeros((len(batch), max_frames * freq_bins), dtype=np.int64)
    weight_out = np.zeros((len(batch), max_frames * freq_bins), dtype=np.float32)
    mask_out = np.zeros((len(batch), num_speakers, max_frames, freq_bins), dtype=np.float32)
    mix_mag_out = np.zeros((len(batch), max_frames, freq_bins), dtype=np.float32)
    source_mag_out = np.zeros((len(batch), num_speakers, max_frames, freq_bins), dtype=np.float32)

    for idx, feat in enumerate(features):
        n_frames = feat.shape[0]
        feat_out[idx, :n_frames] = feat
        mix_mag_out[idx, :n_frames] = mix_magnitudes[idx]
        flat_len = n_frames * freq_bins
        label_out[idx, :flat_len] = labels[idx]
        weight_out[idx, :flat_len] = weights[idx]
        mask_out[idx, :, :n_frames, :] = mask_targets[idx]
        source_mag_out[idx, :, :n_frames, :] = source_magnitudes[idx]

    return {
        "features": feat_out,
        "labels": label_out,
        "weights": weight_out,
        "mask_targets": mask_out,
        "mix_magnitude": mix_mag_out,
        "source_magnitudes": source_mag_out,
        "frames": frames,
        "freq_bins": np.array([freq_bins], dtype=np.int64),
    }
