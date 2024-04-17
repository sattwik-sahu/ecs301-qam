from io import BytesIO
import numpy as np
from scipy.io import wavfile


def convert_to_audio_bytes(signal, sampling_rate):
    # Convert the signal to a NumPy array
    signal_np = np.array(signal, dtype=np.float32)

    # Write the signal to a BytesIO object
    with BytesIO() as buffer:
        wavfile.write(buffer, sampling_rate, signal_np)
        return buffer.getvalue()
