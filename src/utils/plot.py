from turtle import title
import plotly.graph_objs as go
import numpy as np
from typing import List, Optional
from src.utils.models import Signals, AudioFile


def plot_waveforms(
    signals: List[Signals],
    sampling_rates: List[int],
    signal_names: Optional[List[str]] = [],
) -> List[go.Figure]:
    """
    Plot the waveforms of the given signals.

    Args:
        signals (List[Signals]): A list of signals (either AudioFile instances or lists of floats).
        sampling_rates (List[int]): A list of sampling rates corresponding to the signals.

    Returns:
        List[go.Figure]: The Plotly figures containing the waveform plots.
    """
    figures = [go.Figure() for _ in signals]
    for i, (signal, sampling_rate, fig) in enumerate(
        zip(signals, sampling_rates, figures)
    ):
        if isinstance(signal, AudioFile):
            x = [i / sampling_rate for i in range(len(signal.data))]
            y = signal.data
            name = signal.name
        else:
            x = [i / sampling_rate for i in range(len(signal))]
            y = signal
            name = signal_names[i]
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))
        fig.update_layout(
            title=f"Waveform ({name})",
            xaxis_title="Time (s)",
            yaxis_title="Amplitude",
        )

    return figures


def plot_fourier_transforms(
    signals: List[Signals],
    sampling_rates: List[int],
    signal_names: Optional[List[str]] = [],
) -> List[go.Figure]:
    """
    Plot the Fourier transforms of the given signals.

    Args:
        signals (List[Signals]): A list of signals (either AudioFile instances or lists of floats).
        sampling_rates (List[int]): A list of sampling rates corresponding to the signals.

    Returns:
        List[go.Figure]: The Plotly figure containing the Fourier transform plots.
    """
    figures = []
    for i, (signal, sampling_rate) in enumerate(zip(signals, sampling_rates)):
        fig = go.Figure()
        if isinstance(signal, AudioFile):
            signal_data = signal.data
            signal_name = signal.name
        else:
            signal_data = signal
            signal_name = signal_names[i]

        fft_signal, freqs = np.fft.fft(signal_data), np.fft.fftfreq(
            len(signal_data), 1 / sampling_rate
        )
        fig.add_trace(
            go.Scatter(
                x=freqs,
                y=np.abs(fft_signal),
                mode="lines",
                name=f"Signal {len(fig.data) + 1}",
            )
        )
        fig.update_layout(
            title=f"FFT ({signal_name})",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude",
        )
        figures.append(fig)

    return figures
