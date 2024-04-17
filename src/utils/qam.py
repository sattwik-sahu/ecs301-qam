import numpy as np
from typing import Tuple
from src.utils.models import Signals, CarrierFrequency
from scipy.signal import butter, lfilter


def upsample_signal(signal: Signals, target_length: int) -> list[float]:
    """
    Upsample the signal to the target length using zero padding.

    Args:
        signal (Signals): The input signal (either an AudioFile instance or a list of floats).
        target_length (int): The desired length of the upsampled signal.

    Returns:
        list[float]: The upsampled signal.
    """
    if isinstance(signal, list):
        padded_signal = np.pad(
            signal, (0, target_length - len(signal)), mode="constant"
        )
    else:
        padded_signal = np.pad(
            signal.data, (0, target_length - len(signal.data)), mode="constant"
        )
    return padded_signal.tolist()


def qam_modulation(
    signal1: Signals, signal2: Signals, carrier_freq: CarrierFrequency
) -> list[float]:
    """
    Perform quadrature amplitude modulation (QAM) on two signals.

    Args:
        signal1 (Signals): The first input signal (either an AudioFile instance or a list of floats).
        signal2 (Signals): The second input signal (either an AudioFile instance or a list of floats).
        carrier_freq (CarrierFrequency): The carrier frequency and its unit.

    Returns:
        list[float]: The modulated signal.
    """
    # Convert the carrier frequency to the appropriate unit
    if carrier_freq.unit == "khz":
        freq = carrier_freq.value * 1e3
    elif carrier_freq.unit == "mhz":
        freq = carrier_freq.value * 1e6

    # Extract the signal data
    if isinstance(signal1, list):
        signal1_data = signal1
    else:
        signal1_data = signal1.data

    if isinstance(signal2, list):
        signal2_data = signal2
    else:
        signal2_data = signal2.data

    # Calculate the time vector
    t = np.arange(len(signal1_data)) / len(signal1_data)

    # Generate the carrier signals
    carrier_signal_1 = np.cos(2 * np.pi * freq * t, dtype=np.float32)
    carrier_signal_2 = -1j * np.sin(2 * np.pi * freq * t, dtype=np.float32)

    # Perform QAM modulation
    modulated_signal_1 = np.array(signal1_data, dtype=np.float32) * carrier_signal_1
    modulated_signal_2 = np.array(signal2_data, dtype=np.float32) * carrier_signal_2

    modulated_signal = modulated_signal_1 + modulated_signal_2

    return modulated_signal


def demodulate_signal(
    modulated_signal: list[float], carrier_freq: CarrierFrequency, sampling_rate: int
) -> Tuple[list[float], list[float]]:
    """
    Demodulate the modulated signal.

    Args:
        modulated_signal (list[float]): The modulated signal.
        carrier_freq (CarrierFrequency): The carrier frequency and its unit.
        sampling_rate (int): The sampling rate of the modulated signal.

    Returns:
        Tuple[list[float], list[float]]: The demodulated signals.
    """
    # Convert the carrier frequency to the appropriate unit
    if carrier_freq.unit == "khz":
        freq = carrier_freq.value * 1e3
    elif carrier_freq.unit == "mhz":
        freq = carrier_freq.value * 1e6

    # Calculate the time vector
    t = np.arange(len(modulated_signal)) / sampling_rate

    # Generate the carrier signals
    carrier_signal_1 = np.cos(2 * np.pi * freq * t, dtype=np.float32)
    carrier_signal_2 = -1j * np.sin(2 * np.pi * freq * t, dtype=np.float32)

    # Demodulate the signal
    demodulated_signal1 = np.array(modulated_signal) * carrier_signal_1
    demodulated_signal2 = np.array(modulated_signal) * carrier_signal_2

    b, a = butter(1, 2 * carrier_freq.value / sampling_rate, btype="low", analog=False)
    demodulated_signal1 = lfilter(b, a, demodulated_signal1)
    demodulated_signal2 = lfilter(b, a, demodulated_signal2)

    return np.real(demodulated_signal1).tolist(), np.real(demodulated_signal2).tolist()
