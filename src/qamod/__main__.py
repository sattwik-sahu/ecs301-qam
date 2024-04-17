from typing import List
import numpy as np
import streamlit as st
from src.utils.models import AudioFile, CarrierFrequency
from src.utils.qam import upsample_signal, qam_modulation, demodulate_signal
from src.utils.plot import plot_waveforms, plot_fourier_transforms
from src.utils.audio import convert_to_audio_bytes
from io import BytesIO
import plotly.graph_objects as go
from scipy.signal import resample


def display_plots(title: str, plots: List[go.Figure]) -> None:
    st.write(f"## {title}")
    for fig in plots:
        st.plotly_chart(figure_or_data=fig)


def main():
    st.title("Quadrature Amplitude Modulation")

    st.markdown(
        """
This is part of the project for the course ECS301 - Principles of Communication
taught by Dr. Ankur Raina at Indian Institute of Science Education and Research Bhopal
in Spring, 2024.
                
**Project Author:** Sattwik Kumar Sahu
"""
    )

    # File upload section
    audio_file1 = st.file_uploader("Upload first audio file", type=["wav", "mp3"])
    audio_file2 = st.file_uploader("Upload second audio file", type=["wav", "mp3"])

    if audio_file1 and audio_file2:
        # Load audio files
        audio_file1 = AudioFile.from_file(BytesIO(audio_file1.read()), audio_file1.name)
        audio_file2 = AudioFile.from_file(BytesIO(audio_file2.read()), audio_file2.name)

        # Upsample the shorter signal
        if len(audio_file1.data) < len(audio_file2.data):
            audio_file1.data = upsample_signal(audio_file1, len(audio_file2.data))
        elif len(audio_file2.data) < len(audio_file1.data):
            audio_file2.data = upsample_signal(audio_file2, len(audio_file1.data))

        # Display waveforms
        # display_plots(
        #     title="Input Waveforms",
        #     plots=plot_waveforms(
        #         [audio_file1, audio_file2],
        #         [audio_file1.sampling_rate, audio_file2.sampling_rate],
        #     ),
        # )

        # Display Fourier transforms
        fourier_in = plot_fourier_transforms(
            signals=[audio_file1, audio_file2],
            sampling_rates=[audio_file1.sampling_rate, audio_file2.sampling_rate],
        )
        # display_plots(title="Input Fourier Transforms", plots=fourier_in)

        # Get carrier frequency and unit
        carrier_freq = CarrierFrequency(
            value=st.number_input("Carrier Frequency", value=100.0, step=0.1),
            unit=st.selectbox("Carrier Frequency Unit", ["MHz"]),
        )

        # Perform QAM modulation
        modulated_signal = qam_modulation(audio_file1, audio_file2, carrier_freq)

        # Display Fourier transform of the modulated signal
        display_plots(
            title="Modulated Signal",
            plots=plot_fourier_transforms(
                [np.abs(modulated_signal)],
                [audio_file1.sampling_rate],
                ["Modulated Signal"],
            ),
        )

        # Demodulate the modulated signal
        demodulated_signal1, demodulated_signal2 = demodulate_signal(
            modulated_signal, carrier_freq, audio_file1.sampling_rate
        )

        display_plots(
            title="Demodulated Waveforms",
            plots=plot_waveforms(
                signals=[demodulated_signal1, demodulated_signal2],
                sampling_rates=[audio_file1.sampling_rate, audio_file2.sampling_rate],
                signal_names=[audio_file1.name, audio_file2.name],
            ),
        )

        # Allow the user to play the demodulated audio files
        # Convert the demodulated signals to a playable format
        demodulated_signal1_bytes = convert_to_audio_bytes(
            demodulated_signal1, audio_file1.sampling_rate
        )
        demodulated_signal2_bytes = convert_to_audio_bytes(
            demodulated_signal2, audio_file2.sampling_rate
        )

        # Normalize the demodulated signals
        demodulated_signal1 = np.clip(demodulated_signal1, -1, 1)
        demodulated_signal2 = np.clip(demodulated_signal2, -1, 1)

        # Display Fourier Transforms of Demodulated
        fourier_out = plot_fourier_transforms(
            signals=[demodulated_signal1, demodulated_signal2],
            sampling_rates=[audio_file1.sampling_rate, audio_file2.sampling_rate],
            signal_names=["Demodulated 1", "Demodulated 2"],
        )
        display_plots(
            title="Fourier Transforms of Demodulated Signals", plots=fourier_out
        )

        # Resample the demodulated signals to the expected sample rate
        expected_sample_rate = 44100  # Sample rate expected by st.audio()
        # demodulated_signal1 = resample(
        #     demodulated_signal1,
        #     int(
        #         len(demodulated_signal1)
        #         * expected_sample_rate
        #         / audio_file1.sampling_rate
        #     ),
        # )
        # demodulated_signal2 = resample(
        #     demodulated_signal2,
        #     int(
        #         len(demodulated_signal2)
        #         * expected_sample_rate
        #         / audio_file2.sampling_rate
        #     ),
        # )

        # Convert the demodulated signals to a playable format
        # demodulated_signal1_bytes = convert_to_audio_bytes(
        #     demodulated_signal1, expected_sample_rate
        # )
        # demodulated_signal2_bytes = convert_to_audio_bytes(
        #     demodulated_signal2, expected_sample_rate
        # )

        st.write("Play Demodulated Audio Files:")
        # st.audio(demodulated_signal1_bytes, sample_rate=expected_sample_rate)
        # st.audio(demodulated_signal2_bytes, sample_rate=expected_sample_rate)

        st.audio(np.array(demodulated_signal1), sample_rate=expected_sample_rate)
        st.audio(np.array(demodulated_signal2), sample_rate=expected_sample_rate)


if __name__ == "__main__":
    main()
