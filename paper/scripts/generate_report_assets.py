from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocktail_separation.audio import read_wav_mono, stft


PAPER = ROOT / "paper"
IMAGES = PAPER / "images"
OVERWRITE = False


def should_write(path: Path) -> bool:
    return OVERWRITE or not path.exists()


def ensure_dirs() -> None:
    IMAGES.mkdir(parents=True, exist_ok=True)


def save_summary_chart() -> None:
    rows = []
    for name, path in [
        ("Easy selected", ROOT / "demo_audio_easy" / "metrics.csv"),
        ("Easy all", ROOT / "demo_audio_easy" / "metrics_all.csv"),
        ("Hard selected", ROOT / "demo_audio_hard" / "metrics.csv"),
        ("Hard all", ROOT / "demo_audio_hard" / "metrics_all.csv"),
    ]:
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
    if should_write(PAPER / "metrics_summary.csv"):
        summary.to_csv(PAPER / "metrics_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(summary))
    width = 0.25
    for offset, col, color in [
        (-width, "SDR_in", "#5b8fd9"),
        (0.0, "SDR_out", "#2a9d8f"),
        (width, "Delta_SDR", "#e76f51"),
    ]:
        ax.bar(x + offset, summary[col], width=width, label=col.replace("_", " "), color=color)
    ax.axhline(3.0, color="#555555", linestyle="--", linewidth=1.1, label="3 dB improvement")
    ax.set_xticks(x)
    ax.set_xticklabels(summary["name"], rotation=15, ha="right")
    ax.set_ylabel("SI-SDR / Delta SDR (dB)")
    ax.set_title("Easy and Hard Separation Metrics")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    if should_write(IMAGES / "metrics_summary.png"):
        fig.savefig(IMAGES / "metrics_summary.png", dpi=220)
    plt.close(fig)


def save_case_chart() -> None:
    frames = []
    for group, path in [
        ("Easy", ROOT / "demo_audio_easy" / "metrics.csv"),
        ("Hard", ROOT / "demo_audio_hard" / "metrics.csv"),
    ]:
        df = pd.read_csv(path)
        df["label"] = group + " " + df["case"].str.extract(r"case_(\d+)")[0]
        frames.append(df)
    cases = pd.concat(frames, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8.2, 4.2))
    colors = ["#2a9d8f" if label.startswith("Easy") else "#e76f51" for label in cases["label"]]
    ax.bar(cases["label"], cases["delta_sdr_db"], color=colors)
    ax.axhline(3.0, color="#555555", linestyle="--", linewidth=1.1)
    ax.set_ylabel("Delta SDR (dB)")
    ax.set_title("Selected Demo Case Improvements")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    if should_write(IMAGES / "case_delta_sdr.png"):
        fig.savefig(IMAGES / "case_delta_sdr.png", dpi=220)
    plt.close(fig)


def save_train_valid_chart() -> None:
    rows = []
    for name, path in [
        ("Easy train", ROOT / "outputs" / "eval_easy_train" / "metrics_all.csv"),
        ("Easy valid", ROOT / "outputs" / "eval_easy_valid" / "metrics_all.csv"),
        ("Hard train", ROOT / "outputs" / "eval_hard_train" / "metrics_all.csv"),
        ("Hard valid", ROOT / "outputs" / "eval_hard_valid" / "metrics_all.csv"),
    ]:
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
    if should_write(PAPER / "train_valid_summary.csv"):
        df.to_csv(PAPER / "train_valid_summary.csv", index=False)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    colors = ["#2a9d8f", "#8fd1c7", "#e76f51", "#f1a488"]
    ax.bar(df["name"], df["Delta_SDR"], color=colors)
    ax.axhline(3.0, color="#555555", linestyle="--", linewidth=1.1, label="3 dB improvement")
    ax.set_ylabel("Delta SDR (dB)")
    ax.set_title("Train and Validation Generalization Comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    if should_write(IMAGES / "train_valid_comparison.png"):
        fig.savefig(IMAGES / "train_valid_comparison.png", dpi=220)
    plt.close(fig)


def save_code_structure_diagram() -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axis("off")
    blocks = [
        (0.08, 0.72, "Data scripts\nprepare / manifest / split", "#eef4fb"),
        (0.39, 0.72, "Core package\naudio / dataset / model / metrics", "#f2f7ee"),
        (0.70, 0.72, "Experiment scripts\ntrain / evaluate / demo", "#fff4e8"),
        (0.39, 0.35, "GUI dashboard\nplayback / visualization / separation", "#f8eefb"),
    ]
    for x, y, text, color in blocks:
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=11,
            bbox=dict(boxstyle="round,pad=0.55", facecolor=color, edgecolor="#6b7280"),
            transform=ax.transAxes,
        )
    arrows = [((0.18, 0.72), (0.29, 0.72)), ((0.49, 0.72), (0.60, 0.72)), ((0.70, 0.64), (0.48, 0.43)), ((0.39, 0.64), (0.39, 0.44))]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", linewidth=1.4, color="#555555"),
        )
    ax.text(0.39, 0.08, "Outputs: checkpoints, logs, demo_audio, metrics.csv, GUI-separated WAV", ha="center", fontsize=10)
    fig.tight_layout()
    if should_write(IMAGES / "code_structure.png"):
        fig.savefig(IMAGES / "code_structure.png", dpi=220)
    plt.close(fig)


def save_training_curve() -> None:
    logs = sorted((ROOT / "outputs" / "logs").glob("*.csv"), key=lambda path: path.stat().st_mtime)
    if not logs:
        return
    df = pd.read_csv(logs[-1])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    axes[0].plot(df["epoch"], df["mean_loss"], color="#264653", label="mean loss")
    axes[0].plot(df["epoch"], df["sa_loss"], color="#2a9d8f", label="SA loss", alpha=0.9)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    eval_df = df.dropna(subset=["sdr_out_db", "delta_sdr_db"])
    axes[1].plot(eval_df["epoch"], eval_df["sdr_out_db"], marker="o", color="#e76f51", label="SDR out")
    axes[1].plot(eval_df["epoch"], eval_df["delta_sdr_db"], marker="o", color="#5b8fd9", label="Delta SDR")
    axes[1].axhline(3.0, color="#555555", linestyle="--", linewidth=1.1)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("dB")
    axes[1].set_title("Validation Metrics")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle(logs[-1].name, fontsize=11)
    fig.tight_layout()
    if should_write(IMAGES / "training_curve.png"):
        fig.savefig(IMAGES / "training_curve.png", dpi=220)
    plt.close(fig)


def plot_wave_and_spec(path: Path, ax_wave, ax_spec, title: str) -> None:
    audio, sr = read_wav_mono(path)
    time = np.arange(audio.shape[0]) / sr
    ax_wave.plot(time, audio, linewidth=0.55, color="#0b6bcb")
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


def save_audio_visuals() -> None:
    case = ROOT / "demo_audio_easy" / "case_01_deep_clustering"
    files = [
        ("Mixture", case / "mixture.wav"),
        ("Separated 1", case / "separated_1.wav"),
        ("Separated 2", case / "separated_2.wav"),
    ]
    fig, axes = plt.subplots(len(files), 2, figsize=(10, 7.5))
    for row, (title, path) in enumerate(files):
        plot_wave_and_spec(path, axes[row, 0], axes[row, 1], title)
    fig.suptitle("Easy Case 01 Audio Visualization", fontsize=13)
    fig.tight_layout()
    if should_write(IMAGES / "easy_case_visualization.png"):
        fig.savefig(IMAGES / "easy_case_visualization.png", dpi=220)
    plt.close(fig)

    case = ROOT / "demo_audio_hard" / "case_01_deep_clustering"
    files = [
        ("Mixture", case / "mixture.wav"),
        ("Separated 1", case / "separated_1.wav"),
        ("Separated 2", case / "separated_2.wav"),
    ]
    fig, axes = plt.subplots(len(files), 2, figsize=(10, 7.5))
    for row, (title, path) in enumerate(files):
        plot_wave_and_spec(path, axes[row, 0], axes[row, 1], title)
    fig.suptitle("Hard Case 01 Audio Visualization", fontsize=13)
    fig.tight_layout()
    if should_write(IMAGES / "hard_case_visualization.png"):
        fig.savefig(IMAGES / "hard_case_visualization.png", dpi=220)
    plt.close(fig)


def save_pipeline_diagram() -> None:
    fig, ax = plt.subplots(figsize=(10, 2.7))
    ax.axis("off")
    labels = [
        "LibriSpeech\nclean speech",
        "Mixture\nmanifest",
        "STFT\nfeatures",
        "BLSTM mask\nmodel",
        "ISTFT\nseparation",
        "SI-SDR\nevaluation",
    ]
    x = np.linspace(0.08, 0.92, len(labels))
    for i, (xi, label) in enumerate(zip(x, labels)):
        ax.text(
            xi,
            0.52,
            label,
            ha="center",
            va="center",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.45", facecolor="#eef4fb", edgecolor="#5b8fd9"),
        )
        if i < len(labels) - 1:
            ax.annotate(
                "",
                xy=(x[i + 1] - 0.07, 0.52),
                xytext=(xi + 0.07, 0.52),
                arrowprops=dict(arrowstyle="->", color="#555555", linewidth=1.5),
            )
    fig.tight_layout()
    if should_write(IMAGES / "pipeline_diagram.png"):
        fig.savefig(IMAGES / "pipeline_diagram.png", dpi=220)
    plt.close(fig)


def save_gui_mockup() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 20, 60, color="#f0f3f7"))
    ax.text(3, 53, "Cocktail GUI", fontsize=14, weight="bold", color="#202124")
    for i, label in enumerate(["Demo Compare", "Waveform", "Custom Separation", "Metrics"]):
        y = 46 - i * 5
        ax.text(5, y, label, fontsize=9, color="#202124")
        ax.plot(3.2, y + 0.5, marker="o", color="#ff4b4b" if i == 0 else "#c7ced8")
    ax.text(27, 52, "Cocktail-Party Speech Separation", fontsize=20, weight="bold", color="#202124")
    ax.text(27, 42, "Demo Compare", fontsize=15, weight="bold", color="#202124")
    ax.add_patch(plt.Rectangle((27, 35), 64, 4.5, color="#eef1f5"))
    ax.text(29, 36.3, "case_01_deep_clustering", fontsize=9, color="#202124")
    for i, (label, value) in enumerate([("SDR in", "0.060 dB"), ("SDR out", "13.153 dB"), ("Delta SDR", "13.093 dB")]):
        x0 = 27 + i * 21.5
        ax.add_patch(plt.Rectangle((x0, 24), 19, 8, facecolor="#f7f9fb", edgecolor="#d9dee5"))
        ax.text(x0 + 1.2, 29.5, label, fontsize=8, color="#243044")
        ax.text(x0 + 1.2, 25.5, value, fontsize=13, color="#050b16")
    ax.add_patch(plt.Rectangle((27, 10), 31, 6, color="#edf0f2"))
    ax.text(27, 18, "Mixture", fontsize=10, weight="bold", color="#202124")
    ax.text(61, 18, "Noise Reference / Sources / Separated Outputs", fontsize=10, color="#526070")
    fig.tight_layout()
    if should_write(IMAGES / "gui_design_mockup.png"):
        fig.savefig(IMAGES / "gui_design_mockup.png", dpi=220)
    plt.close(fig)


def save_table_snapshot() -> None:
    summary = pd.read_csv(PAPER / "metrics_summary.csv")
    fig, ax = plt.subplots(figsize=(8.8, 2.4))
    ax.axis("off")
    display = summary.copy()
    for col in ["SDR_in", "SDR_out", "Delta_SDR"]:
        display[col] = display[col].map(lambda x: f"{x:.3f}")
    table = ax.table(cellText=display.values, colLabels=display.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#dfeaf7")
            cell.set_text_props(weight="bold")
        else:
            cell.set_facecolor("#f8fafc")
    fig.tight_layout()
    if should_write(IMAGES / "metrics_table.png"):
        fig.savefig(IMAGES / "metrics_table.png", dpi=220)
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    save_summary_chart()
    save_case_chart()
    save_train_valid_chart()
    save_code_structure_diagram()
    save_training_curve()
    save_audio_visuals()
    save_pipeline_diagram()
    save_gui_mockup()
    save_table_snapshot()


if __name__ == "__main__":
    main()
