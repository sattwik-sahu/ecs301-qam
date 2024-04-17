from typing import Union
import numpy as np
from pydantic import BaseModel, field_validator
from pydub import AudioSegment


class AudioFile(BaseModel):
    """
    Model representing an audio file.
    """

    data: list[float]
    sampling_rate: int
    name: str

    @field_validator("data")
    def check_data_length(cls, v):
        """
        Validate that the data list is not empty.
        """
        if not v:
            raise ValueError("Data list cannot be empty.")
        return v

    @field_validator("sampling_rate")
    def check_sampling_rate(cls, v):
        """
        Validate that the sampling rate is a positive integer.
        """
        if v <= 0:
            raise ValueError("Sampling rate must be a positive integer.")
        return v

    @classmethod
    def from_file(cls, file, name):
        """
        Create an AudioFile instance from a file.

        Args:
            file (BytesIO): The file object containing the audio data.
            name (str): The name of the audio file.

        Returns:
            AudioFile: The AudioFile instance.
        """
        # Load the audio data
        audio = AudioSegment.from_file(file)

        # Convert to mono if stereo
        if audio.channels > 1:
            audio = audio.set_channels(1)

        # Convert to float32 numpy array
        data = np.array(audio.get_array_of_samples(), dtype=np.float32) / (
            2 ** (8 * audio.sample_width - 1)
        )

        return cls(data=data.tolist(), sampling_rate=audio.frame_rate, name=name)


class CarrierFrequency(BaseModel):
    """
    Model representing the carrier frequency and its unit.
    """

    value: float
    unit: str

    @field_validator("unit")
    def check_unit(cls, v):
        """
        Validate that the unit is either 'kHz' or 'MHz'.
        """
        if v.lower() not in ["khz", "mhz"]:
            raise ValueError("Unit must be either 'kHz' or 'MHz'.")
        return v.lower()


Signals = Union[AudioFile, list[float]]
