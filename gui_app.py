"""Streamlit dashboard for the cocktail-party speech separation experiment."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from cocktail_separation.audio import istft, normalize_peak, read_wav_mono, stft
from cocktail_separation.dataset import read_audio
from cocktail_separation.deep_clustering import build_model


DEMO_GROUPS = {
    "Easy": ROOT / "demo_audio_easy",
    "Hard": ROOT / "demo_audio_hard",
}


st.set_page_config(
    page_title="Cocktail Speech Separation",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_css() -> None:
    st.markdown(
        """
        <style>
        :root {
            --text: #202124;
            --muted: #64707d;
            --line: #d9dee5;
            --panel: #f7f9fb;
            --accent: #0b6bcb;
            --good: #1b806a;
        }
        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1280px;
        }
        h1, h2, h3 {
            letter-spacing: 0;
        }
        div[data-testid="stMetric"] {
            background: var(--panel);
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 0.75rem 0.85rem;
            color: var(--text);
        }
        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricDelta"] {
            color: var(--muted);
        }
        div[data-testid="stMetricValue"] {
            color: var(--text);
        }
        .metric-card {
            background: #f7f9fb;
            border: 1px solid #d9dee5;
            border-radius: 8px;
            padding: 0.9rem 1rem;
            min-height: 7.2rem;
            display: flex;
            flex-direction: column;
            justify-content: center;
        }
        .metric-card .metric-label {
            color: #243044;
            font-size: 0.94rem;
            line-height: 1.25;
            margin-bottom: 0.45rem;
        }
        .metric-card .metric-value {
            color: #050b16;
            font-size: 2.05rem;
            line-height: 1.15;
            font-weight: 500;
            word-break: break-word;
        }
        .case-title {
            font-size: 0.95rem;
            color: var(--muted);
            margin-bottom: 0.25rem;
        }
        .section-note {
            color: var(--muted);
            font-size: 0.95rem;
            margin-top: -0.35rem;
            margin-bottom: 1rem;
        }
        .audio-label {
            font-weight: 600;
            margin-bottom: 0.25rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_data(show_spinner=False)
def read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_audio(path: str) -> tuple[np.ndarray, int]:
    audio, sr = read_wav_mono(path)
    return audio.astype(np.float32), int(sr)


def audio_player(path: Path, label: str) -> None:
    st.markdown(f'<div class="audio-label">{label}</div>', unsafe_allow_html=True)
    if path.exists():
        st.audio(str(path))
        st.caption(path.name)
    else:
        st.info(f"Missing: {path.name}")


def list_cases(demo_dir: Path) -> list[Path]:
    if not demo_dir.exists():
        return []
    return sorted(path for path in demo_dir.iterdir() if path.is_dir() and path.name.startswith("case_"))


def metrics_for_case(demo_dir: Path, case_name: str) -> pd.Series | None:
    metrics_path = demo_dir / "metrics.csv"
    if not metrics_path.exists():
        return None
    df = read_csv(str(metrics_path))
    rows = df[df["case"] == case_name]
    if rows.empty:
        return None
    return rows.iloc[0]


def manifest_for_group(group: str) -> Path:
    if group == "Hard":
        return ROOT / "data" / "librispeech_hard_generated" / "valid.jsonl"
    return ROOT / "data" / "librispeech_easy_generated" / "valid.jsonl"


@st.cache_data(show_spinner=False)
def load_manifest_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    manifest = Path(path)
    if not manifest.exists():
        return rows
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def manifest_row_for_case(group: str, row: pd.Series | None) -> dict | None:
    if row is None or "manifest_index" not in row:
        return None
    rows = load_manifest_rows(str(manifest_for_group(group)))
    idx = int(row["manifest_index"])
    if idx < 0 or idx >= len(rows):
        return None
    return rows[idx]


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def noise_reference_path(group: str, case: Path, row: pd.Series | None) -> Path | None:
    saved_noise = case / "noise.wav"
    if saved_noise.exists():
        return saved_noise
    manifest_row = manifest_row_for_case(group, row)
    if manifest_row and manifest_row.get("noise"):
        candidate = resolve_project_path(manifest_row["noise"])
        if candidate.exists():
            return candidate
    return None


def exact_noise_in_mixture_path(group: str, case: Path, row: pd.Series | None) -> Path | None:
    manifest_row = manifest_row_for_case(group, row)
    if not manifest_row or "noise" not in manifest_row:
        return None

    out_dir = ROOT / "outputs" / "gui_noise_estimates"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{group.lower()}_{case.name}_noise_in_mixture.wav"
    if out_path.exists():
        return out_path

    try:
        import soundfile as sf

        mixture, sr = read_audio(resolve_project_path(manifest_row["mixture"]))
        sources = [read_audio(resolve_project_path(path), sr)[0] for path in manifest_row["sources"]]
        n = min([mixture.shape[0], *(source.shape[0] for source in sources)])
        residual = mixture[:n] - np.sum(np.stack([source[:n] for source in sources], axis=0), axis=0)
        sf.write(str(out_path), normalize_peak(residual), sr)
        return out_path
    except Exception:
        return None


def plot_waveform(audio: np.ndarray, sr: int, title: str) -> plt.Figure:
    duration = np.arange(audio.shape[0], dtype=np.float32) / float(sr)
    fig, ax = plt.subplots(figsize=(8, 2.1))
    ax.plot(duration, audio, linewidth=0.7, color="#0b6bcb")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-1.05, 1.05)
    fig.tight_layout()
    return fig


def plot_spectrogram(audio: np.ndarray, sr: int, title: str) -> plt.Figure:
    spec = stft(audio, n_fft=512, hop_length=128)
    mag_db = 20.0 * np.log10(np.abs(spec).T + 1e-6)
    fig, ax = plt.subplots(figsize=(8, 2.8))
    extent = [0, audio.shape[0] / float(sr), 0, sr / 2]
    image = ax.imshow(
        mag_db,
        origin="lower",
        aspect="auto",
        extent=extent,
        cmap="magma",
        vmin=np.percentile(mag_db, 5),
        vmax=np.percentile(mag_db, 99),
    )
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, label="dB")
    fig.tight_layout()
    return fig


def render_case_metrics(row: pd.Series | None) -> None:
    if row is None:
        st.info("No metrics found for this case.")
        return
    metrics = [
        ("SDR in", f"{float(row['sdr_in_db']):.3f} dB"),
        ("SDR out", f"{float(row['sdr_out_db']):.3f} dB"),
        ("Delta SDR", f"{float(row['delta_sdr_db']):.3f} dB"),
        ("Permutation", str(row["best_permutation"])),
    ]
    for col, (label, value) in zip(st.columns(4), metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def demo_compare_page() -> None:
    st.header("Demo Compare")
    st.markdown(
        '<div class="section-note">Listen to the mixture, clean references, and separated outputs side by side.</div>',
        unsafe_allow_html=True,
    )
    group = st.segmented_control("Experiment", list(DEMO_GROUPS), default="Easy", key="demo_group")
    demo_dir = DEMO_GROUPS[group]
    cases = list_cases(demo_dir)
    if not cases:
        st.warning(f"No demo cases found in {demo_dir}")
        return

    case = st.selectbox("Case", cases, format_func=lambda path: path.name)
    row = metrics_for_case(demo_dir, case.name)
    render_case_metrics(row)

    st.divider()
    cols = st.columns(2)
    with cols[0]:
        audio_player(case / "mixture.wav", "Mixture")
    with cols[1]:
        if group == "Hard":
            noise_path = noise_reference_path(group, case, row)
            if noise_path is not None:
                audio_player(noise_path, "Noise Reference")
            else:
                st.info("No noise reference found for this case.")
        else:
            st.markdown('<div class="case-title">Clean two-speaker mixture without added environmental noise.</div>', unsafe_allow_html=True)

    if group == "Hard":
        estimated_noise = exact_noise_in_mixture_path(group, case, row)
        if estimated_noise is not None:
            st.markdown('<div class="case-title">Estimated noise actually present in the mixture, computed as mixture minus the two aligned source contributions.</div>', unsafe_allow_html=True)
            audio_player(estimated_noise, "Estimated Noise In Mixture")

    st.subheader("Reference Sources")
    src_cols = st.columns(2)
    with src_cols[0]:
        audio_player(case / "source_1.wav", "Source 1")
    with src_cols[1]:
        audio_player(case / "source_2.wav", "Source 2")

    st.subheader("Separated Outputs")
    sep_cols = st.columns(2)
    with sep_cols[0]:
        audio_player(case / "separated_1.wav", "Separated 1")
    with sep_cols[1]:
        audio_player(case / "separated_2.wav", "Separated 2")


def visualization_page() -> None:
    st.header("Waveform And Spectrogram")
    st.markdown(
        '<div class="section-note">Compare time-domain waveforms and time-frequency structure for one demo case.</div>',
        unsafe_allow_html=True,
    )
    group = st.segmented_control("Experiment", list(DEMO_GROUPS), default="Easy", key="viz_group")
    cases = list_cases(DEMO_GROUPS[group])
    if not cases:
        st.warning("No demo cases found.")
        return

    case = st.selectbox("Case", cases, format_func=lambda path: path.name, key="viz_case")
    choices = [
        ("Mixture", case / "mixture.wav"),
        ("Source 1", case / "source_1.wav"),
        ("Source 2", case / "source_2.wav"),
        ("Separated 1", case / "separated_1.wav"),
        ("Separated 2", case / "separated_2.wav"),
    ]
    row = metrics_for_case(DEMO_GROUPS[group], case.name)
    noise_path = noise_reference_path(group, case, row)
    if noise_path is not None:
        choices.append(("Noise Reference", noise_path))
    estimated_noise = exact_noise_in_mixture_path(group, case, row)
    if estimated_noise is not None:
        choices.append(("Estimated Noise In Mixture", estimated_noise))

    selected = st.multiselect(
        "Signals",
        options=[name for name, _ in choices],
        default=["Mixture", "Separated 1", "Separated 2"],
    )
    selected_paths = [(name, path) for name, path in choices if name in selected]
    if not selected_paths:
        st.info("Select at least one signal to visualize.")
        return

    for name, path in selected_paths:
        if not path.exists():
            st.warning(f"Missing {path.name}")
            continue
        audio, sr = load_audio(str(path))
        col1, col2 = st.columns([1, 1])
        with col1:
            st.pyplot(plot_waveform(audio, sr, f"{name} waveform"), clear_figure=True)
        with col2:
            st.pyplot(plot_spectrogram(audio, sr, f"{name} spectrogram"), clear_figure=True)


def checkpoint_options() -> list[Path]:
    return sorted((ROOT / "outputs").glob("*.pt"))


def mixture_options() -> list[Path]:
    paths: list[Path] = []
    for demo_dir in DEMO_GROUPS.values():
        for case in list_cases(demo_dir):
            mix = case / "mixture.wav"
            if mix.exists():
                paths.append(mix)
    paths.extend(sorted((ROOT / "data").glob("**/mix/*.wav")))
    unique: dict[str, Path] = {}
    for path in paths:
        unique[str(path)] = path
    return list(unique.values())


@st.cache_resource(show_spinner=False)
def load_model(checkpoint: str, device_preference: str):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for separation.") from exc

    device = device_preference if device_preference == "cpu" or torch.cuda.is_available() else "cpu"
    ckpt = torch.load(checkpoint, map_location=device)
    num_speakers = int(ckpt.get("num_speakers", 2))
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
    return torch, model, ckpt, device, num_speakers


def separate_wav(checkpoint: Path, mixture_path: Path, out_dir: Path, device_preference: str) -> list[Path]:
    torch, model, ckpt, device, num_speakers = load_model(str(checkpoint), device_preference)
    mixture, sample_rate = read_audio(mixture_path)
    spec = stft(mixture, n_fft=int(ckpt["n_fft"]), hop_length=int(ckpt["hop_length"]))
    magnitude = np.abs(spec).astype(np.float32)
    features = np.log1p(magnitude)[None, :, :]

    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=device)
        _, predicted_masks = model(x, return_masks=True)
        masks = predicted_masks[0].detach().cpu().numpy()

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError("soundfile is required to write separated audio.") from exc

    outputs = []
    for idx in range(num_speakers):
        separated_spec = spec * masks[idx]
        audio = istft(
            separated_spec,
            n_fft=int(ckpt["n_fft"]),
            hop_length=int(ckpt["hop_length"]),
            length=mixture.shape[0],
        )
        out_path = out_dir / f"speaker_{idx + 1}.wav"
        sf.write(str(out_path), normalize_peak(audio), sample_rate)
        outputs.append(out_path)
    return outputs


def separation_page() -> None:
    st.header("Custom Separation")
    st.markdown(
        '<div class="section-note">Run a trained checkpoint on one mixture WAV and listen to the separated speakers.</div>',
        unsafe_allow_html=True,
    )

    checkpoints = checkpoint_options()
    if not checkpoints:
        st.warning("No checkpoints found under outputs/*.pt")
        return

    left, right = st.columns([1, 1])
    with left:
        checkpoint = st.selectbox("Checkpoint", checkpoints, format_func=lambda path: path.name)
        device = st.radio("Device", ["cuda", "cpu"], horizontal=True, help="CUDA falls back to CPU if unavailable.")
    with right:
        source_mode = st.radio("Mixture source", ["Project file", "Upload WAV"], horizontal=True)
        selected_mixture: Path | None = None
        uploaded_path: Path | None = None
        if source_mode == "Project file":
            mixtures = mixture_options()
            if mixtures:
                selected_mixture = st.selectbox(
                    "Mixture WAV",
                    mixtures,
                    format_func=lambda path: str(path.relative_to(ROOT)),
                )
            else:
                st.info("No mixture WAV files found.")
        else:
            uploaded = st.file_uploader("Upload mixture WAV", type=["wav"])
            if uploaded is not None:
                digest = hashlib.sha1(uploaded.getvalue()).hexdigest()[:10]
                upload_dir = ROOT / "outputs" / "gui_uploads"
                upload_dir.mkdir(parents=True, exist_ok=True)
                uploaded_path = upload_dir / f"{Path(uploaded.name).stem}_{digest}.wav"
                uploaded_path.write_bytes(uploaded.getvalue())
                selected_mixture = uploaded_path

    if selected_mixture:
        st.subheader("Input")
        audio_player(selected_mixture, "Mixture")

    run = st.button("Separate Audio", type="primary", disabled=selected_mixture is None)
    if run and selected_mixture is not None:
        stem = selected_mixture.stem
        out_dir = ROOT / "outputs" / "gui_separated" / f"{stem}_{checkpoint.stem}"
        with st.spinner("Separating audio..."):
            try:
                outputs = separate_wav(checkpoint, selected_mixture, out_dir, device)
            except Exception as exc:
                st.error(f"Separation failed: {exc}")
                return
        st.success(f"Saved separated WAV files to {out_dir}")
        cols = st.columns(len(outputs))
        for col, path in zip(cols, outputs):
            with col:
                audio_player(path, path.stem.replace("_", " ").title())


def summarize_demo_metrics() -> pd.DataFrame:
    rows = []
    for group, demo_dir in DEMO_GROUPS.items():
        for filename, scope in [("metrics.csv", "Selected demo"), ("metrics_all.csv", "Full evaluated")]:
            path = demo_dir / filename
            if not path.exists():
                continue
            df = read_csv(str(path))
            rows.append(
                {
                    "experiment": group,
                    "scope": scope,
                    "items": len(df),
                    "sdr_in_db": df["sdr_in_db"].mean(),
                    "sdr_out_db": df["sdr_out_db"].mean(),
                    "delta_sdr_db": df["delta_sdr_db"].mean(),
                }
            )
    return pd.DataFrame(rows)


def training_log_options() -> list[Path]:
    return sorted((ROOT / "outputs" / "logs").glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)


def metrics_overview_page() -> None:
    st.header("Metrics Overview")
    st.markdown(
        '<div class="section-note">Review Easy versus Hard metrics, selected cases, full demo evaluation, and training logs.</div>',
        unsafe_allow_html=True,
    )

    summary = summarize_demo_metrics()
    if summary.empty:
        st.warning("No demo metrics found.")
    else:
        st.subheader("Experiment Summary")
        st.dataframe(
            summary.style.format(
                {
                    "sdr_in_db": "{:.3f}",
                    "sdr_out_db": "{:.3f}",
                    "delta_sdr_db": "{:.3f}",
                }
            ),
            use_container_width=True,
            hide_index=True,
        )
        chart = summary[summary["scope"] == "Selected demo"].set_index("experiment")
        if not chart.empty:
            st.bar_chart(chart[["sdr_in_db", "sdr_out_db", "delta_sdr_db"]])

    st.subheader("Case-Level Delta SDR")
    case_frames = []
    for group, demo_dir in DEMO_GROUPS.items():
        path = demo_dir / "metrics.csv"
        if path.exists():
            df = read_csv(str(path)).copy()
            df["experiment"] = group
            df["case_label"] = df["experiment"] + " " + df["case"].str.replace("case_", "", regex=False)
            case_frames.append(df)
    if case_frames:
        cases = pd.concat(case_frames, ignore_index=True)
        st.bar_chart(cases.set_index("case_label")["delta_sdr_db"])
        st.dataframe(cases, use_container_width=True, hide_index=True)
    else:
        st.info("No selected case metrics found.")

    st.subheader("Training Logs")
    logs = training_log_options()
    if not logs:
        st.info("No training logs found under outputs/logs.")
        return
    log_path = st.selectbox("Log file", logs, format_func=lambda path: path.name)
    log_df = read_csv(str(log_path))
    left, right = st.columns(2)
    with left:
        st.line_chart(log_df.set_index("epoch")[["mean_loss", "sa_loss"]])
    with right:
        eval_cols = [col for col in ["sdr_in_db", "sdr_out_db", "delta_sdr_db"] if col in log_df.columns]
        eval_df = log_df[["epoch", *eval_cols]].dropna()
        if not eval_df.empty:
            st.line_chart(eval_df.set_index("epoch")[eval_cols])
        else:
            st.info("This log does not contain SDR evaluation rows.")
    st.dataframe(log_df, use_container_width=True, hide_index=True)


def main() -> None:
    add_css()
    st.sidebar.title("Cocktail GUI")
    page = st.sidebar.radio(
        "Page",
        [
            "Demo Compare",
            "Waveform And Spectrogram",
            "Custom Separation",
            "Metrics Overview",
        ],
    )
    st.sidebar.caption("Audio demos, separation, metrics, and training logs.")

    st.title("Cocktail-Party Speech Separation")
    if page == "Demo Compare":
        demo_compare_page()
    elif page == "Waveform And Spectrogram":
        visualization_page()
    elif page == "Custom Separation":
        separation_page()
    elif page == "Metrics Overview":
        metrics_overview_page()


if __name__ == "__main__":
    main()
