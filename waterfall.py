from typing import Union, Optional, Tuple
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io import wavfile
from scipy import signal
from tqdm import tqdm
import numpy as np
import logging
import shutil
import math
import cv2
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
        max_f: float = np.inf,
        window: int = 256,

) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Convert a wav file to a spectrum
    :param path: Path to wav file
    :param min_t: Start t crop (sec)
    :param max_t: Stop t crop (sec)
    :param min_f: Start f crop (sec)
    :param max_f: Stop f crop (sec)
    :param window: window of the FFT
    :return: spectrum (shape f x t), frequency (shape f), and time (shape t)
    """

    # Read the data
    sampling_frequency, signal_data = wavfile.read(path)

    # Convert to frequency data, and close plot.
    spec, freq, time, im = plt.specgram(
        x=signal_data[:, 0],
        Fs=sampling_frequency,
        window=signal.windows.bartlett(window),
        NFFT=window,
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
        path: Union[os.PathLike, str, None],
        filename: str,
        min_t: float = 0,
        max_t: float = np.inf,
        min_f: float = 0,
        max_f: float = np.inf,
        background_color: str = '#000000',
        tick_color: str = '#FFFFFF',
        window: int = 256,
        exp: float = 0.3,
        yscale: str = 'log',
        cmap: str = 'inferno',
        save_fig: bool = True,
        return_fig: bool = False,
) -> Union[Tuple[plt.Figure, plt.Axes], None]:
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
    :param window: The window of the FFT
    :param exp: The exponent to scale the amplitude by
    :param yscale: The scale for the y-axis.
    :param cmap: Color scale for plot.
    :param save_fig: Should fig be saved.
    :param return_fig: Should fig be returned.
    :return: None
    """

    # Generate the spectrum data.
    spec, freq, time = wav_to_spec(
        path=path,
        min_t=min_t,
        max_t=max_t,
        min_f=min_f,
        max_f=max_f,
        window=window,
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

    spec = spec ** exp

    # Plot the data
    plot = ax.imshow(
        X=spec,
        extent=extent,
        aspect='auto',
        origin='lower',
        cmap=cmap,
        # norm=norm,
    )

    # Add the color bar
    cbar = figure.colorbar(
        plot,
        ax=ax,
    )
    cbar.set_label('Amplitude', color=tick_color)
    cbar.ax.yaxis.set_tick_params(color=tick_color)
    cbar.outline.set_edgecolor(background_color)
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color=tick_color)

    # Determine the y ticks
    y_ticks = np.arange(math.ceil(bottom), math.ceil(top))

    # Format
    ax.set_facecolor(background_color)
    ax.set_yscale(yscale)
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
    if save_fig:
        figure.savefig(f'plots/{filename}.png')

    if return_fig:
        return figure, ax

    plt.close(figure)


def waterfall_animation(
        fps: int = 25,
        video_name: str = 'animation',
        window_s: float = 5,
        regenerate_frames: bool = True,
        **kwargs,
) -> None:

    # Folder for frames
    path = os.path.join(os.getcwd(), 'frames')

    # Regenerate the png files.
    if regenerate_frames:

        # Clear the folder
        if os.path.exists(path):
            shutil.rmtree(path)
        os.mkdir(path)

        # Make the figure
        figure, ax = waterfall_plot(
            **kwargs,
        )

        # Get the plot bounds
        min_p, max_p = ax.get_xlim()

        # Make ints
        min_p = int(min_p)
        max_p = int(max_p)

        # Get an iterator for the center of the frame
        n_frames = (max_p - min_p) * fps + 1
        ts = np.linspace(min_p, max_p, n_frames)

        # A place to store the paths
        paths = []

        # For each frame
        for i, t in tqdm(enumerate(ts), total=len(ts)):

            # Center the frame on the current time stamp
            frame_min = t - window_s / 2
            frame_max = t + window_s / 2
            ax.set_xlim(frame_min, frame_max)

            # Add vertical line
            line = ax.axvline(
                t,
                lw=1,
                c=(1, 1, 1, 0.5),
            )

            # Reformat the x-axis
            ax.set_xticks([t - 1, t, t + 1])
            ax.set_xticklabels(['-1', '0', '1'])

            # Save
            filepath = os.path.join(path, f'frame_{i:05}.png')
            paths.append(filepath)
            figure.savefig(filepath)

            # Remove the line
            line.remove()

    # If not regenerating, get the paths of the images
    else:
        paths = [os.path.join(os.getcwd(), 'frames', fname) for fname in sorted(os.listdir(path)) if fname.endswith(".png")]

    # If video folder doesn't exist, create it
    if not os.path.exists('videos'):
        os.mkdir('videos')

    # Convert the frames into an mp4
    first_frame = cv2.imread(paths[0])
    height, width, layers = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'H264')
    video = cv2.VideoWriter(
        f'videos/{video_name}.avi',
        fourcc,
        fps=fps,
        frameSize=(width, height),
    )
    cv2.VideoWriter()
    for path in tqdm(paths, total=len(paths)):
        video.write(cv2.imread(path))
    cv2.destroyAllWindows()
    video.release()



if __name__ == '__main__':
    convert_all_mp3_to_wav()
    b_spec, b_freq, b_time = wav_to_spec(
        path='sources/wav/bird-song.wav',
        min_t=0.1,
        max_t=1,
        min_f=1000,
        max_f=10000,
    )
