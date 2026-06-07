"""Model, losses, and mask-estimation helpers for speech separation."""

from __future__ import annotations

import itertools

import numpy as np


def _require_torch():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
    except ImportError as exc:
        raise ImportError(
            "Deep clustering needs PyTorch. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc
    return torch, nn, F


def build_model(
    freq_bins: int,
    embedding_dim: int = 20,
    hidden_size: int = 300,
    layers: int = 2,
    num_speakers: int = 2,
    mask_activation: str = "sigmoid",
):
    torch, nn, F = _require_torch()

    class DeepClusteringNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.freq_bins = freq_bins
            self.embedding_dim = embedding_dim
            self.num_speakers = num_speakers
            self.mask_activation = mask_activation
            self.blstm = nn.LSTM(
                input_size=freq_bins,
                hidden_size=hidden_size,
                num_layers=layers,
                batch_first=True,
                bidirectional=True,
                dropout=0.0 if layers == 1 else 0.2,
            )
            self.proj = nn.Linear(hidden_size * 2, freq_bins * embedding_dim)
            self.mask_proj = nn.Linear(hidden_size * 2, freq_bins * num_speakers)

        def forward(self, features, return_masks: bool = False):
            batch, frames, _ = features.shape
            out, _ = self.blstm(features)
            emb = self.proj(out).reshape(batch, frames * self.freq_bins, self.embedding_dim)
            emb = F.normalize(emb, p=2, dim=-1)
            if not return_masks:
                return emb
            mask_logits = self.mask_proj(out).reshape(batch, frames, self.freq_bins, self.num_speakers)
            mask_logits = mask_logits.permute(0, 3, 1, 2)
            if self.mask_activation == "softmax":
                masks = F.softmax(mask_logits, dim=1)
            elif self.mask_activation == "sigmoid":
                masks = torch.sigmoid(mask_logits)
            else:
                raise ValueError(f"Unsupported mask_activation: {self.mask_activation}")
            return emb, masks

    return DeepClusteringNet()


def deep_clustering_loss(embeddings, labels, weights, num_speakers: int):
    """Compute the deep-clustering affinity loss without materializing full affinities."""

    torch, _, F = _require_torch()
    y = F.one_hot(labels, num_classes=num_speakers).float()
    weights = weights.float().unsqueeze(-1)
    v = embeddings * weights
    y = y * weights
    vv = torch.bmm(v.transpose(1, 2), v)
    yy = torch.bmm(y.transpose(1, 2), y)
    vy = torch.bmm(v.transpose(1, 2), y)
    active = torch.clamp(weights.sum(dim=(1, 2)), min=1.0)
    loss = (vv.pow(2).sum(dim=(1, 2)) + yy.pow(2).sum(dim=(1, 2)) - 2.0 * vy.pow(2).sum(dim=(1, 2)))
    return (loss / (active**2)).mean()


def mask_inference_loss(predicted_masks, mix_magnitude, source_magnitudes, weights):
    """Permutation-invariant signal-approximation loss.

    The loss compares separated magnitudes against clean-source magnitudes for
    every source permutation and uses the best assignment per sample.
    """

    torch, _, _ = _require_torch()
    batch, num_speakers, frames, freq_bins = predicted_masks.shape
    weights = weights.reshape(batch, 1, frames, freq_bins).float()
    estimated_magnitudes = predicted_masks * mix_magnitude.float().unsqueeze(1)
    source_magnitudes = source_magnitudes.float()
    denom = torch.clamp(weights.sum(dim=(1, 2, 3)) * num_speakers, min=1.0)

    losses = []
    for perm in itertools.permutations(range(num_speakers)):
        target = source_magnitudes[:, perm, :, :]
        per_sample = ((estimated_magnitudes - target).pow(2) * weights).sum(dim=(1, 2, 3)) / denom
        losses.append(per_sample)
    stacked = torch.stack(losses, dim=1)
    return torch.min(stacked, dim=1).values.mean()


def kmeans(
    points: np.ndarray,
    n_clusters: int,
    iterations: int = 30,
    seed: int = 0,
) -> np.ndarray:
    """Cluster time-frequency embeddings with a compact NumPy k-means implementation."""

    points = np.asarray(points, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if points.shape[0] < n_clusters:
        raise ValueError("Need at least as many points as clusters")
    centers = points[rng.choice(points.shape[0], size=n_clusters, replace=False)]
    labels = np.zeros(points.shape[0], dtype=np.int64)
    for _ in range(iterations):
        distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        new_labels = np.argmin(distances, axis=1)
        if np.array_equal(labels, new_labels):
            break
        labels = new_labels
        for cluster in range(n_clusters):
            mask = labels == cluster
            if np.any(mask):
                centers[cluster] = points[mask].mean(axis=0)
    return labels


def estimate_masks(
    embeddings: np.ndarray,
    magnitude: np.ndarray,
    num_speakers: int,
    activity_threshold: float = 1e-3,
    seed: int = 0,
) -> np.ndarray:
    """Cluster active time-frequency embeddings and return binary speaker masks."""

    frames, freq_bins = magnitude.shape
    flat_mag = magnitude.reshape(-1)
    active = flat_mag > activity_threshold
    flat_embeddings = embeddings.reshape(frames * freq_bins, -1)
    masks = np.zeros((num_speakers, frames * freq_bins), dtype=np.float32)
    if active.sum() < num_speakers:
        masks[0, :] = 1.0
        return masks.reshape(num_speakers, frames, freq_bins)

    labels = kmeans(flat_embeddings[active], num_speakers, seed=seed)
    active_indices = np.flatnonzero(active)
    for speaker in range(num_speakers):
        masks[speaker, active_indices[labels == speaker]] = 1.0
    inactive = ~active
    masks[:, inactive] = 1.0 / num_speakers
    return masks.reshape(num_speakers, frames, freq_bins)
