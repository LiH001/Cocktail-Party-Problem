"""Evaluation metrics for speech separation."""

from __future__ import annotations

import itertools

import numpy as np


def si_sdr(estimate: np.ndarray, reference: np.ndarray, eps: float = 1e-8) -> float:
    """Scale-invariant SDR in dB."""

    estimate = np.asarray(estimate, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    n = min(estimate.size, reference.size)
    estimate = estimate[:n] - np.mean(estimate[:n])
    reference = reference[:n] - np.mean(reference[:n])

    scale = np.dot(estimate, reference) / (np.dot(reference, reference) + eps)
    target = scale * reference
    noise = estimate - target
    ratio = (np.sum(target**2) + eps) / (np.sum(noise**2) + eps)
    return float(10.0 * np.log10(ratio + eps))


def best_permutation_si_sdr(
    estimates: np.ndarray,
    references: np.ndarray,
) -> tuple[float, tuple[int, ...]]:
    """Return best average SI-SDR over all source permutations."""

    estimates = np.asarray(estimates)
    references = np.asarray(references)
    n_sources = min(estimates.shape[0], references.shape[0])
    best_score = -np.inf
    best_perm: tuple[int, ...] = tuple(range(n_sources))
    for perm in itertools.permutations(range(n_sources)):
        scores = [si_sdr(estimates[i], references[perm[i]]) for i in range(n_sources)]
        score = float(np.mean(scores))
        if score > best_score:
            best_score = score
            best_perm = perm
    return best_score, best_perm
