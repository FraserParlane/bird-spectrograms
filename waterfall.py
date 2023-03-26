from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from typing import Union, Optional
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
import numpy as np
import logging
import math
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


def wav_to_spec(
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
        # NFFT=512,
    )
    plt.close()

    # Create the cropping mask
    mask_t = np.logical_and(time > min_t, time < max_t)
    mask_f = np.logical_and(freq > min_f, freq < max_f)

    # Crop the spectrum, time stamps
    cropped_spec = spec[mask_f][:, mask_t]
    cropped_time = time[mask_t]
    cropped_freq = freq[mask_f]

    # Log
    logging.info(f'Freq res: {len(freq)}')
    logging.info(f'Time res: {len(time)}')
    logging.info(f'Cropped freq res: {len(cropped_freq)}')
    logging.info(f'Cropped time res: {len(cropped_time)}')

    # Return the frequencies
    return cropped_spec, cropped_freq, cropped_time


def waterfall_plot(
        path: Union[os.PathLike, str],
        filename: str,
        min_t: float = 0,
        max_t: float = np.inf,
        min_f: float = 0,
        max_f: float = np.inf,
        background_color: str = '#000000',
        tick_color: str = '#FFFFFF',
) -> None:
    """
    Convert a wav file to a plot
    :param path: Path to wav file
    :param filename: Plot filename in /plots/
    :param min_t: Start t crop (sec)
    :param max_t: Stop t crop (sec)
    :param min_f: Start f crop (sec)
    :param max_f: Stop f crop (sec)
    :param background_color: Color of plot background
    :param tick_color: Color of ticks
    :return: None
    """

    # Generate the spectrum data.
    spec, freq, time = wav_to_spec(
        path=path,
        min_t=min_t,
        max_t=max_t,
        min_f=min_f,
        max_f=max_f,
    )

    # Determine the extent
    left = 0
    right = time[-1] - time[0]
    bottom = freq[0] / 1000
    top = freq[-1] / 1000
    extent = (left, right, bottom, top)
    norm = LogNorm(
        vmin=spec.min(),
        vmax=spec.max(),
    )

    # Make the plot.
    figure: plt.Figure = plt.figure(
        dpi=300,
        figsize=(8, 3),
        facecolor=background_color,
    )
    ax: plt.Axes = figure.add_subplot()

    # Plot the data
    ax.imshow(
        X=spec,
        extent=extent,
        aspect='auto',
        origin='lower',
        cmap='viridis',
        norm=norm,
        # interpolation='gaussian',
    )

    # Determine the y ticks
    y_ticks = np.arange(math.ceil(bottom), math.ceil(top))

    # Format
    ax.set_yscale('log')
    ax.set_xlabel('time (sec)')
    ax.set_ylabel('frequency (kHz)')
    ax.xaxis.label.set_color(tick_color)
    ax.yaxis.label.set_color(tick_color)
    ax.yaxis.set_minor_formatter(mticker.ScalarFormatter())
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticks)
    ax.tick_params(which='both', colors=tick_color)
    figure.subplots_adjust(
        top=0.98,
        right=0.98,
        bottom=0.17,
        left=0.07,
    )

    # Save
    figure.savefig(f'plots/{filename}.png')


if __name__ == '__main__':
    convert_all_mp3_to_wav()
    b_spec, b_freq, b_time = wav_to_spec(
        path='sources/wav/bird-song.wav',
        min_t=0.1,
        max_t=1,
        min_f=1000,
        max_f=10000,
    )
