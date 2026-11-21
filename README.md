# Sampling & Aliasing Visualizer

Interactive DSP laboratory application for **CS-419 Digital Signal Processing, Open-Ended Lab, Option 2**. The app demonstrates uniform sampling, the Nyquist criterion, spectral folding, anti-alias pre-filtering, and three reconstruction methods.

Built with Python, Streamlit, NumPy, SciPy, and Plotly.

## Features

- Visual Inspection mode: 1 Hz to 50 Hz
- Audio Synthesis mode: 200 Hz to 4000 Hz
- Sine, square, and triangular waveforms
- Synchronized slider and numeric input for signal frequency `f0`
- Synchronized slider and numeric input for sampling frequency `fs`
- Quick sampling presets: 10x oversampling, 2x Nyquist boundary, 1.3x aliasing
- Whittaker-Shannon sinc, linear interpolation, and zero-order hold reconstruction
- Optional ideal low-pass pre-filter at `fs/2` before sampling
- Live Nyquist status banner and four monospace telemetry cards
- Time-domain sampling and reconstruction plot
- Single-sided magnitude spectrum with the `fs/2` folding limit
- Theoretical aliasing fold-back curve with a live operating-point crosshair
- Two 2-second, 16-bit, 44.1 kHz WAV players for audible comparison
- Collapsible mathematical reference section

## Project structure

```
sampling-aliasing-visualizer/
├── app.py                  Streamlit interface and Plotly figures
├── dsp_engine.py           Pure DSP routines and WAV generation
├── requirements.txt        Python dependencies
├── README.md               This file
└── .streamlit/
    └── config.toml         Light engineering theme and server settings
```

## Installation

Python 3.10 or newer is recommended. Python 3.12 is the tested version.

### Windows PowerShell

```powershell
cd sampling-aliasing-visualizer
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, either use the interpreter directly:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py
```

or allow local scripts for the current user:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

### macOS / Linux

```bash
cd sampling-aliasing-visualizer
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Streamlit opens `http://localhost:8501`. If it does not open automatically, visit that URL manually.

## How to use

1. Select **Visual Inspection** to see individual sample stems clearly, or **Audio Synthesis** for listening tests.
2. Choose a waveform. Square and triangular waves expose harmonic folding; a sine wave isolates the fundamental.
3. Set `f0` and `fs` with either the slider or the numeric box. The two controls stay synchronized.
4. Use **10x**, **2x**, and **1.3x** for fast comparisons of oversampling, the Nyquist boundary, and active aliasing.
5. Compare the blue continuous reference, crimson samples, and dashed green reconstruction.
6. Switch reconstruction between sinc, linear, and zero-order hold.
7. Enable the anti-alias pre-filter to remove harmonics above `fs/2` before sampling.
8. Use the two audio players to compare the original and reconstructed sound.

Suggested demonstrations:

- Sine, `f0 = 5 Hz`, `fs = 50 Hz`: clean reconstruction.
- Sine, `f0 = 5 Hz`, `fs = 6.5 Hz` (1.3x): visible low-frequency alias.
- Square, `f0 = 5 Hz`, `fs = 12 Hz`: several harmonics fold back.
- Same square case with the pre-filter enabled: high harmonics are removed before sampling.
- Audio mode, `f0 = 1000 Hz`, then step through 10x, 2x, and 1.3x while listening.

## DSP notes

- The continuous reference is evaluated at 200 samples per period of `f0`.
- The primary plot displays five periods of `f0`.
- The apparent frequency is computed as `f_alias = |f0 - round(f0/fs) * fs|`.
- The sampled spectrum is the baseband DTFT magnitude repeated every `fs` hertz.
- The pre-filter is an ideal brick-wall low-pass at `fs/2`, implemented as a Fourier partial sum for square and triangular waves.
- Audio is rendered at 44.1 kHz as 16-bit PCM. Both streams are band-limited to 20 kHz before conversion so the WAV container does not introduce an unrelated second alias.
- The sinc reconstruction uses a finite 64-sample kernel for interactive speed and a 48-sample kernel for audio rendering. The visible result matches the ideal interpolation inside the display window.

## Verification

Compile and smoke-test the modules:

```bash
python -m compileall app.py dsp_engine.py
python -c "import dsp_engine; print(dsp_engine.apparent_frequency(5.0, 6.5))"
```

Run the app headlessly to check startup:

```bash
streamlit run app.py --server.headless true --server.port 8501
```

Then open `http://localhost:8501`.

## Troubleshooting

- **`python` opens the Microsoft Store:** install Python 3.12 from python.org and disable the Windows Store `python.exe` app alias.
- **Port already in use:** run `streamlit run app.py --server.port 8502`.
- **No audio in Visual mode:** frequencies from 1 Hz to 50 Hz are mostly below the audible range. Use Audio Synthesis mode.
- **Plots look compressed on a small screen:** collapse the sidebar or use a wider browser window. The layout is designed for a desktop demonstration.
- **First launch is slow:** Streamlit and Plotly import time dominates; subsequent launches are faster.
