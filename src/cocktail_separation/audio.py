"""Audio I/O and time-frequency transforms for speech separation."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Read an audio file as float32 mono samples in [-1, 1]."""

    path = Path(path)
    if path.suffix.lower() != ".wav":
        try:
            import soundfile as sf
        except ImportError as exc:
            raise ImportError(
                f"Reading {path.suffix} audio needs soundfile. Install with: pip install soundfile"
            ) from exc
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = audio.mean(axis=1)
        return np.asarray(audio, dtype=np.float32), int(sample_rate)

    with wave.open(str(path), "rb") as wav:
        sample_rate = wav.getframerate()
        channels = wav.getnchannels()
        sample_width = wav.getsampwidth()
        frames = wav.readframes(wav.getnframes())

    if sample_width == 2:
        data = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    elif sample_width == 4:
        data = np.frombuffer(frames, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported wav sample width: {sample_width} bytes")

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data.astype(np.float32), sample_rate


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """Write float audio as a 16-bit PCM WAV file."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.T.reshape(-1)
        channels = 2
    else:
        channels = 1

    peak = float(np.max(np.abs(audio))) if audio.size else 1.0
    if peak > 1.0:
        audio = audio / peak
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")

    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def hann_window(n_fft: int) -> np.ndarray:
    return np.hanning(n_fft).astype(np.float32)


def stft(
    audio: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    window: np.ndarray | None = None,
) -> np.ndarray:
    """Compute a one-sided complex STFT with shape [frames, freq_bins]."""

    audio = np.asarray(audio, dtype=np.float32)
    window = hann_window(n_fft) if window is None else window.astype(np.float32)
    if audio.size < n_fft:
        audio = np.pad(audio, (0, n_fft - audio.size))

    frames = 1 + int(np.ceil((audio.size - n_fft) / hop_length))
    padded_len = (frames - 1) * hop_length + n_fft
    audio = np.pad(audio, (0, max(0, padded_len - audio.size)))

    spec = np.empty((frames, n_fft // 2 + 1), dtype=np.complex64)
    for idx in range(frames):
        start = idx * hop_length
        frame = audio[start : start + n_fft] * window
        spec[idx] = np.fft.rfft(frame)
    return spec


def istft(
    spec: np.ndarray,
    n_fft: int = 512,
    hop_length: int = 128,
    window: np.ndarray | None = None,
    length: int | None = None,
) -> np.ndarray:
    """Invert a one-sided STFT produced by stft()."""

    spec = np.asarray(spec)
    window = hann_window(n_fft) if window is None else window.astype(np.float32)
    frames = spec.shape[0]
    out_len = (frames - 1) * hop_length + n_fft
    audio = np.zeros(out_len, dtype=np.float32)
    norm = np.zeros(out_len, dtype=np.float32)

    for idx in range(frames):
        start = idx * hop_length
        frame = np.fft.irfft(spec[idx], n=n_fft).astype(np.float32)
        audio[start : start + n_fft] += frame * window
        norm[start : start + n_fft] += window**2

    good = norm > 1e-8
    audio[good] /= norm[good]
    if length is not None:
        audio = audio[:length]
    return audio.astype(np.float32)


def normalize_peak(audio: np.ndarray, peak: float = 0.95) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    current = float(np.max(np.abs(audio))) if audio.size else 0.0
    if current < 1e-8:
        return audio
    return audio * (peak / current)
