from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from dsp_engine import (
    RECONSTRUCTION_METHODS,
    WAVEFORMS,
    apparent_frequency,
    fold_back_curve,
    nyquist_state,
    reconstruct_signal,
    sample_signal,
    signal_waveform,
    spectrum_curves,
    synthesize_audio,
)

st.set_page_config(
    page_title="Sampling & Aliasing Visualizer",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODE_VISUAL = "Visual Inspection (1 Hz – 50 Hz)"
MODE_AUDIO = "Audio Synthesis (200 Hz – 4000 Hz)"

MODE_CONFIG = {
    MODE_VISUAL: {
        "f0_min": 1.0,
        "f0_max": 50.0,
        "f0_step": 0.1,
        "f0_default": 5.0,
        "fs_step": 0.1,
        "f0_format": "%.1f",
        "fs_format": "%.1f",
        "default_fs": 50.0,
    },
    MODE_AUDIO: {
        "f0_min": 200.0,
        "f0_max": 4000.0,
        "f0_step": 1.0,
        "f0_default": 1000.0,
        "fs_step": 1.0,
        "f0_format": "%.0f",
        "fs_format": "%.0f",
        "default_fs": 10000.0,
    },
}

BLUE = "#1D4ED8"
CRIMSON = "#DC2626"
EMERALD = "#059669"
INK = "#0F172A"
SLATE = "#475569"
MUTED = "#64748B"
GRID = "#E2E8F0"
AXIS = "#CBD5E1"
WHITE = "#FFFFFF"
AMBER = "#D97706"

MONO = '"JetBrains Mono", "Cascadia Mono", "SFMono-Regular", Consolas, monospace'
SANS = '"Inter", "Segoe UI", "Helvetica Neue", Arial, sans-serif'


def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --bg: #F8FAFC;
            --card: #FFFFFF;
            --ink: #0F172A;
            --muted: #64748B;
            --border: #E2E8F0;
            --blue: #1D4ED8;
        }}

        html, body, [class*="css"] {{
            font-family: {SANS};
        }}

        .stApp {{
            background: #F8FAFC;
        }}

        .block-container {{
            max-width: 1560px;
            padding-top: 2.0rem;
            padding-bottom: 2.5rem;
        }}

        [data-testid="stSidebar"] {{
            background: #FFFFFF;
            border-right: 1px solid #E2E8F0;
        }}

        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
        }}

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {{
            letter-spacing: -0.01em;
        }}

        .app-header {{
            padding: 0 0 14px 0;
        }}

        .app-kicker {{
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.13em;
            color: #64748B;
            text-transform: uppercase;
            margin-bottom: 6px;
        }}

        .app-title {{
            font-size: 31px;
            line-height: 1.1;
            font-weight: 650;
            letter-spacing: -0.035em;
            color: #0F172A;
            margin: 0;
        }}

        .app-subtitle {{
            font-size: 12.5px;
            color: #64748B;
            margin-top: 6px;
        }}

        .status-bar {{
            display: flex;
            align-items: center;
            flex-wrap: wrap;
            gap: 4px 10px;
            min-height: 40px;
            padding: 8px 14px;
            border: 1px solid #E2E8F0;
            border-left-width: 3px;
            border-radius: 5px;
            background: #FFFFFF;
            margin: 2px 0 12px 0;
        }}

        .status-bar .dot {{
            width: 7px;
            height: 7px;
            border-radius: 50%;
            flex: 0 0 auto;
        }}

        .status-bar .status-text {{
            font-size: 11.5px;
            font-weight: 700;
            letter-spacing: 0.055em;
            text-transform: uppercase;
        }}

        .status-bar .status-meta {{
            margin-left: auto;
            font-family: {MONO};
            font-size: 10.5px;
            color: #64748B;
            white-space: nowrap;
        }}

        .status-ok {{
            border-left-color: #059669;
            background: #F0FDF4;
        }}
        .status-ok .dot {{ background: #059669; }}
        .status-ok .status-text {{ color: #047857; }}

        .status-warn {{
            border-left-color: #D97706;
            background: #FFFBEB;
        }}
        .status-warn .dot {{ background: #D97706; }}
        .status-warn .status-text {{ color: #B45309; }}

        .status-alert {{
            border-left-color: #DC2626;
            background: #FEF2F2;
        }}
        .status-alert .dot {{ background: #DC2626; }}
        .status-alert .status-text {{ color: #B91C1C; }}

        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 2px;
        }}

        @media (max-width: 1200px) {{
            .metric-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
        }}

        @media (max-width: 620px) {{
            .metric-grid {{ grid-template-columns: 1fr; }}
        }}

        .metric-card {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 6px;
            padding: 13px 15px 12px 15px;
            min-height: 92px;
            min-width: 0;
        }}

        .metric-label {{
            font-size: 9.5px;
            font-weight: 700;
            letter-spacing: 0.105em;
            text-transform: uppercase;
            color: #64748B;
            white-space: normal;
        }}

        .metric-label sub {{
            font-size: 0.75em;
        }}

        .metric-value {{
            font-family: {MONO};
            font-size: 25px;
            font-weight: 600;
            line-height: 1.25;
            color: #0F172A;
            font-variant-numeric: tabular-nums;
            margin-top: 5px;
            white-space: nowrap;
        }}

        .metric-unit {{
            font-family: {MONO};
            font-size: 10px;
            font-weight: 500;
            color: #64748B;
            margin-left: 3px;
        }}

        .metric-sub {{
            font-family: {MONO};
            font-size: 9.5px;
            color: #94A3B8;
            margin-top: 2px;
            white-space: normal;
        }}

        .section-title {{
            font-size: 10.5px;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #334155;
            margin: 1px 0 3px 0;
        }}

        .side-section {{
            font-size: 9.5px;
            font-weight: 700;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: #64748B;
            margin: 14px 0 2px 0;
        }}

        .ctrl-label {{
            font-size: 10.5px;
            font-weight: 600;
            color: #334155;
            margin: 4px 0 6px 0;
        }}

        .ctrl-label .sym {{
            font-family: {MONO};
            color: #1D4ED8;
        }}

        .ctrl-label .unit {{
            font-family: {MONO};
            color: #94A3B8;
            font-size: 9px;
        }}

        .quick-note {{
            font-size: 9px;
            letter-spacing: 0.045em;
            color: #94A3B8;
            text-align: center;
            margin-top: -4px;
            font-family: {MONO};
        }}

        .side-footer {{
            font-size: 9.5px;
            line-height: 1.55;
            color: #94A3B8;
            border-top: 1px solid #E2E8F0;
            padding-top: 10px;
            margin-top: 18px;
        }}

        .audio-label {{
            font-size: 10px;
            font-weight: 700;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: #334155;
            margin-bottom: 1px;
        }}

        .audio-meta {{
            font-family: {MONO};
            font-size: 9.5px;
            color: #94A3B8;
            margin-bottom: 7px;
        }}

        .footer {{
            border-top: 1px solid #E2E8F0;
            margin-top: 20px;
            padding-top: 10px;
            font-size: 9.5px;
            letter-spacing: 0.05em;
            color: #94A3B8;
            text-transform: uppercase;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: #FFFFFF;
            border: 1px solid #E2E8F0;
            border-radius: 7px;
        }}

        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding: 0.55rem 0.65rem 0.35rem 0.65rem;
        }}

        .stSlider {{
            padding-top: 0;
        }}

        .stNumberInput input {{
            font-family: {MONO};
            font-size: 11px;
            font-variant-numeric: tabular-nums;
        }}

        .stButton > button {{
            font-size: 10.5px;
            font-weight: 600;
            padding: 0.22rem 0.3rem;
            border: 1px solid #CBD5E1;
            border-radius: 4px;
            color: #334155;
            background: #FFFFFF;
            min-height: 28px;
        }}

        .stButton > button:hover {{
            border-color: #1D4ED8;
            color: #1D4ED8;
        }}

        .stAudio {{
            margin-top: 2px;
        }}

        details {{
            border: 1px solid #E2E8F0 !important;
            border-radius: 6px !important;
            background: #FFFFFF !important;
        }}

        summary {{
            font-size: 11px !important;
            font-weight: 600 !important;
            color: #334155 !important;
        }}

        .theory h4 {{
            font-size: 12px;
            margin: 12px 0 4px 0;
            color: #0F172A;
        }}

        .theory p, .theory li {{
            font-size: 11.5px;
            line-height: 1.6;
            color: #475569;
        }}

        .theory code {{
            font-family: {MONO};
            font-size: 10.5px;
            background: #F1F5F9;
            padding: 1px 4px;
            border-radius: 3px;
        }}

        #MainMenu, footer {{ visibility: hidden; }}
        [data-testid="stDecoration"] {{ display: none; }}
        [data-testid="stToolbar"] {{ visibility: hidden; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _defaults_for_mode(mode: str) -> dict[str, float]:
    cfg = MODE_CONFIG[mode]
    return {
        "f0_slider": cfg["f0_default"],
        "f0_number": cfg["f0_default"],
        "fs_slider": cfg["default_fs"],
        "fs_number": cfg["default_fs"],
    }


def init_state() -> None:
    if "mode" not in st.session_state:
        st.session_state["mode"] = MODE_VISUAL
        for key, value in _defaults_for_mode(MODE_VISUAL).items():
            st.session_state[key] = value
    else:
        defaults = _defaults_for_mode(st.session_state["mode"])
        for key, value in defaults.items():
            st.session_state.setdefault(key, value)

    st.session_state.setdefault("waveform", "Sine")
    st.session_state.setdefault("reconstruction", RECONSTRUCTION_METHODS[0])
    st.session_state.setdefault("prefilter", False)


def clamp_fs_to_f0() -> None:
    f0 = float(st.session_state["f0_slider"])
    low, high = 0.5 * f0, 20.0 * f0
    for key in ("fs_slider", "fs_number"):
        if key in st.session_state:
            st.session_state[key] = float(
                min(max(float(st.session_state[key]), low), high)
            )


def on_f0_slider() -> None:
    st.session_state["f0_number"] = float(st.session_state["f0_slider"])
    clamp_fs_to_f0()


def on_f0_number() -> None:
    st.session_state["f0_slider"] = float(st.session_state["f0_number"])
    clamp_fs_to_f0()


def on_fs_slider() -> None:
    st.session_state["fs_number"] = float(st.session_state["fs_slider"])


def on_fs_number() -> None:
    st.session_state["fs_slider"] = float(st.session_state["fs_number"])


def on_mode_change() -> None:
    for key, value in _defaults_for_mode(st.session_state["mode"]).items():
        st.session_state[key] = value


def set_fs_ratio(ratio: float) -> None:
    f0 = float(st.session_state["f0_slider"])
    fs = min(max(ratio * f0, 0.5 * f0), 20.0 * f0)
    st.session_state["fs_slider"] = fs
    st.session_state["fs_number"] = fs


@st.cache_data(show_spinner=False, max_entries=24)
def cached_audio(
    f0: float,
    fs: float,
    waveform: str,
    method: str,
    prefilter: bool,
):
    return synthesize_audio(
        f0, fs, waveform, method, prefilter=prefilter
    )


def style_figure(
    fig: go.Figure,
    *,
    height: int,
    x_title: str,
    y_title: str,
    x_range: tuple[float, float] | None = None,
    y_range: tuple[float, float] | None = None,
    hovermode: str = "x unified",
) -> None:
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=8, t=30, b=8),
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font=dict(family=SANS, size=11, color=SLATE),
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor=WHITE,
            bordercolor=GRID,
            font=dict(family=MONO, size=10.5, color=INK),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=9.5, color=SLATE),
        ),
        xaxis=dict(
            title=dict(text=x_title, font=dict(size=10.5, color=SLATE)),
            gridcolor=GRID,
            gridwidth=1,
            zeroline=False,
            linecolor=AXIS,
            showline=True,
            ticks="outside",
            tickcolor=AXIS,
            tickfont=dict(size=9.5, color=MUTED),
        ),
        yaxis=dict(
            title=dict(text=y_title, font=dict(size=10.5, color=SLATE)),
            gridcolor=GRID,
            gridwidth=1,
            zeroline=False,
            linecolor=AXIS,
            showline=True,
            ticks="outside",
            tickcolor=AXIS,
            tickfont=dict(size=9.5, color=MUTED),
        ),
    )
    if x_range is not None:
        fig.update_xaxes(range=list(x_range))
    if y_range is not None:
        fig.update_yaxes(range=list(y_range))


def build_time_figure(
    f0: float,
    fs: float,
    waveform: str,
    method: str,
    prefilter: bool,
    time_unit: str,
) -> go.Figure:
    window = 5.0 / f0
    scale = 1000.0 if time_unit == "ms" else 1.0

    continuous_count = int(np.ceil(200.0 * f0 * window)) + 1
    t_continuous = np.linspace(0.0, window, continuous_count)
    x_continuous = signal_waveform(t_continuous, f0, waveform)

    guard = 2.0 / f0
    samples = sample_signal(
        f0,
        fs,
        waveform,
        window + 2.0 * guard,
        prefilter=prefilter,
        start_time=-guard,
    )
    visible = (samples.times >= -1e-12) & (samples.times <= window + 1e-12)
    stem_times = samples.times[visible]
    stem_values = samples.values[visible]

    x_reconstructed = reconstruct_signal(
        t_continuous, samples, method, sinc_half_width=64
    )

    stem_x = np.full(stem_times.size * 3, np.nan)
    stem_y = np.full(stem_times.size * 3, np.nan)
    stem_x[0::3] = stem_times * scale
    stem_x[1::3] = stem_times * scale
    stem_y[0::3] = 0.0
    stem_y[1::3] = stem_values

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_continuous * scale,
            y=x_continuous,
            mode="lines",
            name="Continuous x(t)",
            legendrank=1,
            line=dict(color=BLUE, width=2),
            hovertemplate=(
                "t = %{x:.4f} " + time_unit + "<br>x(t) = %{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_continuous * scale,
            y=x_reconstructed,
            mode="lines",
            name="Reconstructed x̂(t)",
            legendrank=3,
            line=dict(color=EMERALD, width=2, dash="dash"),
            hovertemplate=(
                "t = %{x:.4f} "
                + time_unit
                + "<br>x̂(t) = %{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stem_x,
            y=stem_y,
            mode="lines",
            line=dict(color=CRIMSON, width=1),
            hoverinfo="skip",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=stem_times * scale,
            y=stem_values,
            mode="markers",
            name="Samples x[n]",
            legendrank=2,
            marker=dict(
                color=CRIMSON,
                size=8,
                line=dict(color=WHITE, width=1),
            ),
            hovertemplate=(
                "t = %{x:.4f} "
                + time_unit
                + "<br>x[n] = %{y:.3f}<extra></extra>"
            ),
        )
    )

    fig.add_hline(y=0.0, line_color=AXIS, line_width=1)
    style_figure(
        fig,
        height=430,
        x_title=f"Time ({time_unit})",
        y_title="Amplitude",
        y_range=(-1.35, 1.35),
    )
    fig.update_xaxes(
        tickformat=".2f" if time_unit == "ms" else ".3f",
        range=[0.0, window * scale],
    )
    return fig


def build_spectrum_figure(
    f0: float,
    fs: float,
    waveform: str,
    prefilter: bool,
) -> go.Figure:
    data = spectrum_curves(f0, fs, waveform, prefilter=prefilter)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.original_freqs,
            y=data.original_magnitude,
            mode="lines",
            name="Original x(t)",
            line=dict(color=BLUE, width=1.7),
            hovertemplate=(
                "%{x:.1f} Hz<br>|X(f)| = %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=data.sampled_freqs,
            y=data.sampled_magnitude,
            mode="lines",
            name="Sampled x[n]",
            line=dict(color=CRIMSON, width=1.4),
            hovertemplate=(
                "%{x:.1f} Hz<br>|X_s(f)| = %{y:.3f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=data.nyquist,
        line_dash="dash",
        line_color="#64748B",
        line_width=1.1,
        annotation_text="f<sub>s</sub>/2",
        annotation_position="top right",
        annotation_font=dict(size=9.5, color=MUTED),
    )
    style_figure(
        fig,
        height=340,
        x_title="Frequency (Hz)",
        y_title="Normalized magnitude",
        x_range=(0.0, data.f_max),
        y_range=(0.0, data.y_max),
    )
    fig.update_xaxes(tickformat="~s")
    return fig


def build_fold_figure(f0: float, fs: float, alias: float) -> go.Figure:
    fs_max = 4.0 * f0 if fs <= 4.0 * f0 else 1.05 * fs
    data = fold_back_curve(f0, fs_max)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=data.sampling_freqs,
            y=data.alias_freqs,
            mode="lines",
            name="Fold-back locus",
            line=dict(color=BLUE, width=2),
            hovertemplate=(
                "f<sub>s</sub> = %{x:.1f} Hz<br>"
                "f<sub>alias</sub> = %{y:.2f} Hz<extra></extra>"
            ),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[fs],
            y=[alias],
            mode="markers",
            name="Operating point",
            marker=dict(
                color=CRIMSON,
                size=10,
                line=dict(color=WHITE, width=1.5),
            ),
            hovertemplate=(
                "f<sub>s</sub> = %{x:.1f} Hz<br>"
                "f<sub>alias</sub> = %{y:.2f} Hz<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=fs, line_dash="dot", line_color=CRIMSON, line_width=1, opacity=0.45
    )
    fig.add_hline(
        y=alias,
        line_dash="dot",
        line_color=CRIMSON,
        line_width=1,
        opacity=0.45,
    )
    fig.add_vline(
        x=2.0 * f0,
        line_dash="dash",
        line_color="#64748B",
        line_width=1.1,
        annotation_text="2f<sub>0</sub>",
        annotation_position="top right",
        annotation_font=dict(size=9.5, color=MUTED),
    )
    if fs > 4.0 * f0:
        fig.add_vline(
            x=4.0 * f0,
            line_dash="dot",
            line_color="#94A3B8",
            line_width=1,
            annotation_text="4f<sub>0</sub>",
            annotation_position="top left",
            annotation_font=dict(size=9, color=MUTED),
        )

    style_figure(
        fig,
        height=340,
        x_title="Sampling frequency f<sub>s</sub> (Hz)",
        y_title="Apparent frequency f<sub>alias</sub> (Hz)",
        x_range=(0.5 * f0, fs_max),
        y_range=(0.0, 1.08 * f0),
        hovermode="closest",
    )
    fig.update_xaxes(tickformat="~s")
    return fig


def metric_card(
    label: str,
    value: float,
    sublabel: str,
    *,
    unit: str = "Hz",
) -> str:
    return (
        '<div class="metric-card">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value:,.2f}'
        f'<span class="metric-unit">{unit}</span></div>'
        f'<div class="metric-sub">{sublabel}</div>'
        "</div>"
    )


inject_css()
init_state()

with st.sidebar:
    st.markdown(
        '<div class="app-kicker">CS-419 · DSP · OPTION 2</div>'
        '<div style="font-size:15px;font-weight:650;letter-spacing:-0.02em;'
        'color:#0F172A;">Sampling & Aliasing</div>'
        '<div style="font-size:10px;color:#94A3B8;margin-top:2px;">'
        "Interactive laboratory instrument</div>",
        unsafe_allow_html=True,
    )

    st.markdown('<div class="side-section">Operation mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Operation mode",
        [MODE_VISUAL, MODE_AUDIO],
        key="mode",
        on_change=on_mode_change,
        label_visibility="collapsed",
    )

    st.markdown('<div class="side-section">Waveform</div>', unsafe_allow_html=True)
    waveform = st.radio(
        "Waveform",
        WAVEFORMS,
        key="waveform",
        horizontal=True,
        label_visibility="collapsed",
    )

    cfg = MODE_CONFIG[mode]

    st.markdown(
        '<div class="ctrl-label">Signal frequency '
        '<span class="sym">f<sub>0</sub></span> '
        '<span class="unit">Hz</span></div>',
        unsafe_allow_html=True,
    )
    f0_col, f0_num_col = st.columns([3.1, 1.25], gap="small")
    with f0_col:
        st.slider(
            "Signal frequency f0",
            min_value=cfg["f0_min"],
            max_value=cfg["f0_max"],
            step=cfg["f0_step"],
            key="f0_slider",
            on_change=on_f0_slider,
            label_visibility="collapsed",
            format=cfg["f0_format"],
        )
    with f0_num_col:
        st.number_input(
            "Exact signal frequency",
            min_value=cfg["f0_min"],
            max_value=cfg["f0_max"],
            step=cfg["f0_step"],
            key="f0_number",
            on_change=on_f0_number,
            label_visibility="collapsed",
            format=cfg["f0_format"],
        )

    f0_value = float(st.session_state["f0_slider"])

    st.markdown(
        '<div class="ctrl-label">Sampling frequency '
        '<span class="sym">f<sub>s</sub></span> '
        '<span class="unit">Hz</span></div>',
        unsafe_allow_html=True,
    )
    fs_col, fs_num_col = st.columns([3.1, 1.25], gap="small")
    with fs_col:
        st.slider(
            "Sampling frequency fs",
            min_value=0.5 * f0_value,
            max_value=20.0 * f0_value,
            step=cfg["fs_step"],
            key="fs_slider",
            on_change=on_fs_slider,
            label_visibility="collapsed",
            format=cfg["fs_format"],
        )
    with fs_num_col:
        st.number_input(
            "Exact sampling frequency",
            min_value=0.5 * f0_value,
            max_value=20.0 * f0_value,
            step=cfg["fs_step"],
            key="fs_number",
            on_change=on_fs_number,
            label_visibility="collapsed",
            format=cfg["fs_format"],
        )

    quick_10, quick_2, quick_alias = st.columns(3, gap="small")
    with quick_10:
        st.button(
            "10x",
            key="quick_10x",
            on_click=set_fs_ratio,
            args=(10.0,),
            help="Oversample: fs = 10 f0",
            width="stretch",
        )
    with quick_2:
        st.button(
            "2x",
            key="quick_2x",
            on_click=set_fs_ratio,
            args=(2.0,),
            help="Nyquist boundary: fs = 2 f0",
            width="stretch",
        )
    with quick_alias:
        st.button(
            "1.3x",
            key="quick_alias",
            on_click=set_fs_ratio,
            args=(1.3,),
            help="Aliasing region: fs = 1.3 f0",
            width="stretch",
        )
    st.markdown(
        '<div class="quick-note">Oversample · Nyquist · Alias</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="side-section">Reconstruction filter</div>',
        unsafe_allow_html=True,
    )
    reconstruction = st.radio(
        "Reconstruction filter",
        RECONSTRUCTION_METHODS,
        key="reconstruction",
        label_visibility="collapsed",
    )

    st.markdown(
        '<div class="side-section">Anti-aliasing pre-filter</div>',
        unsafe_allow_html=True,
    )
    prefilter = st.checkbox(
        "Apply ideal low-pass filter at fs/2 before sampling",
        key="prefilter",
    )

    st.markdown(
        '<div class="side-footer">CS-419 Digital Signal Processing<br>'
        "Open-Ended Lab · Option 2<br>"
        "Sampling theorem · spectral folding · reconstruction</div>",
        unsafe_allow_html=True,
    )

f0 = float(st.session_state["f0_slider"])
fs = float(st.session_state["fs_slider"])
alias = apparent_frequency(f0, fs)
state = nyquist_state(f0, fs)

st.markdown(
    '<div class="app-header">'
    '<div class="app-kicker">CS-419 · Digital Signal Processing · Open-Ended Lab</div>'
    '<div class="app-title">Sampling & Aliasing Visualizer</div>'
    '<div class="app-subtitle">Time-domain sampling, spectral folding, and '
    "reconstruction of periodic signals</div>"
    "</div>",
    unsafe_allow_html=True,
)

if state == "satisfied":
    status_class = "status-ok"
    status_text = "NYQUIST CRITERION SATISFIED (fs &gt; 2f0)"
elif state == "boundary":
    status_class = "status-warn"
    status_text = "CRITICAL NYQUIST BOUNDARY (fs = 2f0)"
else:
    status_class = "status-alert"
    status_text = f"ALIASING ACTIVE — FOLDING TO {alias:.2f} Hz (fs &lt; 2f0)"

st.markdown(
    f'<div class="status-bar {status_class}">'
    '<span class="dot"></span>'
    f'<span class="status-text">{status_text}</span>'
    f'<span class="status-meta">fs / f0 = {fs / f0:.2f} · '
    f"fs / 2 = {fs / 2.0:,.2f} Hz</span>"
    "</div>",
    unsafe_allow_html=True,
)

metric_html = (
    '<div class="metric-grid">'
    + metric_card(
        "Signal frequency f<sub>0</sub>",
        f0,
        "Fundamental component",
    )
    + metric_card(
        "Nyquist rate 2f<sub>0</sub>",
        2.0 * f0,
        "Minimum alias-free sampling rate",
    )
    + metric_card(
        "Sampling rate f<sub>s</sub>",
        fs,
        f"Folding frequency fs/2 = {fs / 2.0:,.2f} Hz",
    )
    + metric_card(
        "Apparent frequency f<sub>alias</sub>",
        alias,
        "Observed frequency after sampling",
    )
    + "</div>"
)
st.markdown(metric_html, unsafe_allow_html=True)

time_unit = "ms" if mode == MODE_AUDIO else "s"
window = 5.0 / f0

with st.container(border=True):
    st.markdown(
        '<div class="section-title">Time-domain sampling and reconstruction</div>',
        unsafe_allow_html=True,
    )
    prefilter_note = ""
    if prefilter and fs / 2.0 < f0:
        prefilter_note = " · pre-filter cutoff is below f0; the fundamental is suppressed"
    elif prefilter:
        prefilter_note = f" · pre-filter cutoff {fs / 2.0:,.2f} Hz"
    st.caption(
        f"Display window: 5 periods of f0 ({window * (1000.0 if time_unit == 'ms' else 1.0):.2f} "
        f"{time_unit}) · continuous reference simulated at 200 f0"
        f"{prefilter_note}"
    )
    st.plotly_chart(
        build_time_figure(f0, fs, waveform, reconstruction, prefilter, time_unit),
        width="stretch",
        config={"displayModeBar": False, "responsive": True},
    )

spectrum_col, fold_col = st.columns(2, gap="medium")

with spectrum_col:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Single-sided magnitude spectrum</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Original components and sampled spectrum · dashed line marks fs/2 · "
            "harmonics above it fold"
        )
        st.plotly_chart(
            build_spectrum_figure(f0, fs, waveform, prefilter),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

with fold_col:
    with st.container(border=True):
        st.markdown(
            '<div class="section-title">Aliasing fold-back characteristic</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            "Apparent frequency versus sampling rate · crosshair marks the operating point"
        )
        st.plotly_chart(
            build_fold_figure(f0, fs, alias),
            width="stretch",
            config={"displayModeBar": False, "responsive": True},
        )

st.markdown(
    '<div class="section-title" style="margin-top:16px;">Audio comparison</div>',
    unsafe_allow_html=True,
)
if mode == MODE_VISUAL:
    st.caption(
        "Visual-mode frequencies are largely below the audible range. "
        "Switch to Audio Synthesis mode for listening tests."
    )

audio = cached_audio(f0, fs, waveform, reconstruction, prefilter)
audio_left, audio_right = st.columns(2, gap="medium")

with audio_left:
    with st.container(border=True):
        st.markdown(
            '<div class="audio-label">Original continuous signal</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="audio-meta">{f0:,.1f} Hz · 2.00 s · 44.1 kHz · 16-bit PCM</div>',
            unsafe_allow_html=True,
        )
        st.audio(audio.original_wav, format="audio/wav")

with audio_right:
    with st.container(border=True):
        st.markdown(
            '<div class="audio-label">Sampled and reconstructed signal</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="audio-meta">fs = {fs:,.1f} Hz · {reconstruction} · '
            "2.00 s · 44.1 kHz · 16-bit PCM</div>",
            unsafe_allow_html=True,
        )
        st.audio(audio.reconstructed_wav, format="audio/wav")

with st.expander("Mathematical Reference & Theory"):
    st.markdown("#### Nyquist-Shannon sampling theorem")
    st.markdown(
        "A band-limited signal with highest component $f_{\\max}$ is uniquely "
        "recoverable from its uniform samples when the sampling rate exceeds "
        "twice that component."
    )
    st.latex(r"f_s > 2 f_{\max}")
    st.markdown(
        "For a sine wave $f_{\\max}=f_0$. Square and triangular waves contain "
        "odd harmonics $(2k+1)f_0$, so their highest significant component "
        "depends on how many harmonics are retained."
    )

    st.markdown("#### Whittaker-Shannon sinc interpolation")
    st.latex(
        r"\hat{x}(t)=\sum_{n=-\infty}^{\infty}"
        r"x[n]\,\operatorname{sinc}\!\left(\frac{t-nT_s}{T_s}\right),"
        r"\qquad \operatorname{sinc}(u)=\frac{\sin(\pi u)}{\pi u}"
    )
    st.markdown(
        "Ideal reconstruction is a sum of shifted sinc kernels, one per sample. "
        "Linear interpolation and zero-order hold are practical approximations "
        "with different high-frequency behavior."
    )

    st.markdown("#### Aliasing and harmonic folding")
    st.latex(
        r"f_{\text{alias}}="
        r"\left|f_0-\operatorname{round}\!\left(\frac{f_0}{f_s}\right)f_s\right|"
    )
    st.latex(
        r"f_{\text{alias},k}="
        r"\left|(2k+1)f_0-"
        r"\operatorname{round}\!\left(\frac{(2k+1)f_0}{f_s}\right)f_s\right|"
    )
    st.markdown(
        "The sampled spectrum repeats every $f_s$. Any component above the "
        "folding frequency $f_s/2$ reappears below it at the folded location."
    )

    st.markdown("#### Ideal anti-aliasing pre-filter")
    st.latex(
        r"H(f)=\begin{cases}1, & |f|<f_s/2\\[2pt]0, & |f|\geq f_s/2\end{cases}"
    )
    st.markdown(
        "Applying this brick-wall filter before sampling removes harmonics "
        "that would otherwise fold into the baseband. At exactly $f_s=2f_0$ "
        "the result depends on sample phase: two samples per period do not "
        "reliably preserve amplitude and phase."
    )

st.markdown(
    '<div class="footer">CS-419 Digital Signal Processing · Open-Ended Lab · '
    "Option 2 · Sampling and Aliasing Visualizer</div>",
    unsafe_allow_html=True,
)
