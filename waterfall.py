from typing import Union, Optional
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import os


def convert_all_mp3_to_wav() -> None:
    """
    Convert all files in sources/mp3 to wav files in sources/wav.
    :return: None
    """

    # File directories
    source_dir = os.path.join(os.getcwd(), 'sources', 'mp3')
    target_dir = os.path.join(os.getcwd(), 'sources', 'wav')

    # Walk folder
    for file in os.listdir(source_dir):

        # Sense the mp3 and wav files
        is_mp3 = file.endswith('.mp3')
        filename = file.split('.')[0]
        source_path = os.path.join(source_dir, file)
        target_path = os.path.join(target_dir, f'{filename}.wav')
        wave_exists = os.path.exists(target_path)

        # If mp3 not yet converted
        if is_mp3 and not wave_exists:

            sound = AudioSegment.from_mp3(source_path)
            sound.export(target_path, format='wav')


def wav_to_waterfall_data(
        path: Union[os.PathLike, str],
        min_t: float = 0,
        max_t: float = np.inf,
        min_f: float = 0,
        max_f: float = np.inf

) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Convert a wav file to a spectrum
    :param path: Path to wav file
    :param min_t: Start t crop (sec)
    :param max_t: Stop t crop (sec)
    :param min_f: Start f crop (sec)
    :param max_f: Stop f crop (sec)
    :return: spectrum (shape f x t), frequency (shape f), and time (shape t)
    """

    # Read the data
    sampling_frequency, signal_data = wavfile.read(path)

    # Convert to frequency data, and close plot.
    spec, freq, time, im = plt.specgram(
        x=signal_data[:, 0],
        Fs=sampling_frequency,
    )
    plt.close()

    # Create the cropping mask
    mask_t = np.logical_and(time > min_t, time < max_t)
    mask_f = np.logical_and(freq > min_f, freq < max_f)

    # Crop the spectrum, time stamps
    cropped_spec = spec[mask_f][:, mask_t]
    cropped_time = time[mask_t]
    cropped_freq = freq[mask_f]

    # Return the frequencies
    return cropped_spec, cropped_freq, cropped_time


if __name__ == '__main__':
    convert_all_mp3_to_wav()
    b_spec, b_freq, b_time = wav_to_waterfall_data(
        path='sources/wav/bird-song.wav',
        min_t=0.1,
        max_t=1,
        min_f=1000,
        max_f=10000,
    )
