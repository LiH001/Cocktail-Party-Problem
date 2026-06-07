from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocktail_separation.audio import read_wav_mono, stft

IMAGES = ROOT / "paper" / "images"


def latest_log(keyword: str) -> Path:
    files = sorted(
        Path(ROOT / "outputs" / "logs").glob(f"*{keyword}*.csv"),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No training log found for {keyword}")
    return files[-1]


def save_training_curves() -> None:
    easy = latest_log("easy")
    hard = latest_log("hard")
    pairs = [("Easy", easy, "#2563eb"), ("Hard", hard, "#ea580c")]

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), sharex="col")
    for row, (name, path, color) in enumerate(pairs):
        df = pd.read_csv(path)
        eval_df = df.dropna(subset=["sdr_out_db", "delta_sdr_db"])

        axes[row, 0].plot(df["epoch"], df["mean_loss"], color="#334155", label="mean loss")
        axes[row, 0].plot(df["epoch"], df["sa_loss"], color=color, label="SA loss", alpha=0.9)
        axes[row, 0].set_title(f"{name} training loss")
        axes[row, 0].set_ylabel("Loss")
        axes[row, 0].grid(alpha=0.25)
        axes[row, 0].legend(fontsize=8)

        axes[row, 1].plot(eval_df["epoch"], eval_df["sdr_out_db"], marker="o", color=color, label="SDR out")
        axes[row, 1].plot(eval_df["epoch"], eval_df["delta_sdr_db"], marker="o", color="#16a34a", label="Delta SDR")
        axes[row, 1].axhline(3.0, color="#64748b", linestyle="--", linewidth=1.0)
        axes[row, 1].set_title(f"{name} validation metrics")
        axes[row, 1].set_ylabel("dB")
        axes[row, 1].grid(alpha=0.25)
        axes[row, 1].legend(fontsize=8)

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 1].set_xlabel("Epoch")
    fig.tight_layout()
    fig.savefig(IMAGES / "results_training_curves_easy_hard.png", dpi=220)
    plt.close(fig)


def save_metrics_summary() -> None:
    rows = []
    files = [
        ("Easy selected", ROOT / "demo_audio_easy" / "metrics.csv"),
        ("Easy valid all", ROOT / "demo_audio_easy" / "metrics_all.csv"),
        ("Hard selected", ROOT / "demo_audio_hard" / "metrics.csv"),
        ("Hard valid all", ROOT / "demo_audio_hard" / "metrics_all.csv"),
    ]
    for name, path in files:
        df = pd.read_csv(path)
        rows.append(
            {
                "name": name,
                "SDR_in": df["sdr_in_db"].mean(),
                "SDR_out": df["sdr_out_db"].mean(),
                "Delta_SDR": df["delta_sdr_db"].mean(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(ROOT / "paper" / "results_metrics_summary.csv", index=False)

    matrix = summary[["SDR_in", "SDR_out", "Delta_SDR"]].to_numpy()
    norm = TwoSlopeNorm(vmin=-1.5, vcenter=3.0, vmax=max(14.0, float(np.nanmax(matrix))))

    base_cmap = plt.get_cmap("RdYlBu_r")
    base_colors = base_cmap(np.linspace(0.0, 1.0, 256))
    white = np.array([1.0, 1.0, 1.0, 1.0])
    softened_colors = base_colors * 0.68 + white * 0.32
    softened_colors[:, 3] = 1.0
    soft_cmap = LinearSegmentedColormap.from_list("soft_RdYlBu_r", softened_colors)

    fig, ax = plt.subplots(figsize=(9.6, 4.9))
    im = ax.imshow(matrix, cmap=soft_cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["Input SDR", "Output SDR", "Improvement"], fontsize=10)
    ax.set_yticks(np.arange(len(summary)))
    ax.set_yticklabels(summary["name"], fontsize=10)
    ax.set_title("SI-SDR Metric Matrix", fontsize=13, pad=14)

    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, f"{value:.2f} dB", ha="center", va="center", color="#0f172a", fontsize=10)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, 3, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(summary), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)

    cbar = fig.colorbar(im, ax=ax, fraction=0.036, pad=0.035)
    cbar.set_label("dB", rotation=0, labelpad=12)
    cbar.ax.axhline(norm(3.0), color="white", linewidth=1.4)
    cbar.ax.text(1.7, norm(3.0), "3 dB", va="center", fontsize=8, color="#111827")

    fig.tight_layout()
    fig.savefig(IMAGES / "results_metrics_summary.png", dpi=220)
    plt.close(fig)


def save_train_valid_comparison() -> None:
    rows = []
    files = [
        ("Easy train", ROOT / "outputs" / "eval_easy_train" / "metrics_all.csv"),
        ("Easy valid", ROOT / "demo_audio_easy" / "metrics_all.csv"),
        ("Hard train", ROOT / "outputs" / "eval_hard_train" / "metrics_all.csv"),
        ("Hard valid", ROOT / "demo_audio_hard" / "metrics_all.csv"),
    ]
    for name, path in files:
        df = pd.read_csv(path)
        rows.append(
            {
                "name": name,
                "SDR_in": df["sdr_in_db"].mean(),
                "SDR_out": df["sdr_out_db"].mean(),
                "Delta_SDR": df["delta_sdr_db"].mean(),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "paper" / "results_train_valid_summary.csv", index=False)

    comparison = pd.DataFrame(
        {
            "condition": ["Easy", "Hard"],
            "train": [
                float(df.loc[df["name"] == "Easy train", "Delta_SDR"].iloc[0]),
                float(df.loc[df["name"] == "Hard train", "Delta_SDR"].iloc[0]),
            ],
            "valid": [
                float(df.loc[df["name"] == "Easy valid", "Delta_SDR"].iloc[0]),
                float(df.loc[df["name"] == "Hard valid", "Delta_SDR"].iloc[0]),
            ],
        }
    )

    fig, ax = plt.subplots(figsize=(8.9, 4.8))
    colors = {"Easy": "#2563eb", "Hard": "#ea580c"}
    y_positions = np.arange(len(comparison))[::-1]

    ax.axvspan(0, 3, color="#f1f5f9", zorder=0)
    ax.axvline(3.0, color="#64748b", linestyle="--", linewidth=1.15, zorder=1)
    ax.text(3.08, -0.35, "3 dB target", color="#475569", fontsize=9, va="center")

    for idx, row in comparison.iterrows():
        y = y_positions[idx]
        color = colors[row["condition"]]
        ax.plot([row["train"], row["valid"]], [y, y], color=color, linewidth=3.2, alpha=0.35, zorder=2)
        ax.scatter(row["train"], y, s=160, color=color, edgecolor="white", linewidth=1.8, zorder=3)
        ax.scatter(row["valid"], y, s=160, color="white", edgecolor=color, linewidth=2.2, zorder=4)

        gap = row["train"] - row["valid"]
        ax.annotate(
            f"gap {gap:.2f} dB",
            xy=((row["train"] + row["valid"]) / 2, y),
            xytext=(0, 16),
            textcoords="offset points",
            ha="center",
            va="bottom",
            color="#334155",
            fontsize=9,
        )
        ax.text(row["train"] + 0.18, y - 0.17, f"train {row['train']:.2f}", color=color, fontsize=9)
        ax.text(row["valid"] + 0.18, y + 0.12, f"valid {row['valid']:.2f}", color=color, fontsize=9)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(comparison["condition"], fontsize=11)
    ax.set_xlabel("Delta SDR (dB)")
    ax.set_title("Train-to-Validation Generalization Gap", fontsize=13, pad=12)
    ax.set_xlim(0, max(float(comparison["train"].max()), float(comparison["valid"].max())) + 2.0)
    ax.set_ylim(-0.55, len(comparison) - 0.45)
    ax.grid(axis="x", alpha=0.25)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)

    train_handle = ax.scatter([], [], s=120, color="#334155", edgecolor="white", linewidth=1.5, label="Train")
    valid_handle = ax.scatter([], [], s=120, color="white", edgecolor="#334155", linewidth=2.0, label="Validation")
    ax.legend(handles=[train_handle, valid_handle], loc="lower right", frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(IMAGES / "results_train_valid_comparison.png", dpi=220)
    plt.close(fig)


def plot_wave_and_spec(path: Path, ax_wave, ax_spec, title: str) -> None:
    audio, sr = read_wav_mono(path)
    time = np.arange(audio.shape[0]) / sr
    ax_wave.plot(time, audio, linewidth=0.55, color="#2563eb")
    ax_wave.set_title(f"{title} waveform", fontsize=10)
    ax_wave.set_ylim(-1.05, 1.05)
    ax_wave.set_xlabel("Time (s)")
    ax_wave.set_ylabel("Amplitude")
    ax_wave.grid(alpha=0.2)

    spec = stft(audio, n_fft=512, hop_length=128)
    mag_db = 20 * np.log10(np.abs(spec).T + 1e-6)
    ax_spec.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        cmap="magma",
        extent=[0, audio.shape[0] / sr, 0, sr / 2],
        vmin=np.percentile(mag_db, 5),
        vmax=np.percentile(mag_db, 99),
    )
    ax_spec.set_title(f"{title} spectrogram", fontsize=10)
    ax_spec.set_xlabel("Time (s)")
    ax_spec.set_ylabel("Frequency (Hz)")


def save_case_visual(case_dir: Path, title: str, filename: str) -> None:
    files = [
        ("Mixture", case_dir / "mixture.wav"),
        ("Separated 1", case_dir / "separated_1.wav"),
        ("Separated 2", case_dir / "separated_2.wav"),
    ]
    fig, axes = plt.subplots(len(files), 2, figsize=(10, 7.5))
    for row, (name, path) in enumerate(files):
        plot_wave_and_spec(path, axes[row, 0], axes[row, 1], name)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(IMAGES / filename, dpi=220)
    plt.close(fig)


def main() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)
    save_training_curves()
    save_metrics_summary()
    save_train_valid_comparison()
    save_case_visual(
        ROOT / "demo_audio_easy" / "case_01_deep_clustering",
        "Easy Case 01 Audio Visualization",
        "results_easy_case_visualization.png",
    )
    save_case_visual(
        ROOT / "demo_audio_hard" / "case_01_deep_clustering",
        "Hard Case 01 Audio Visualization",
        "results_hard_case_visualization.png",
    )


if __name__ == "__main__":
    main()
