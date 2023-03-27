"""
Generate the mp4 animation of the spectrograms for the README.md file.
Fraser Parlane 20220327
"""
from waterfall import convert_all_mp3_to_wav, waterfall_animation
import logging
import os


def bird_chorus():

    # Generate the spectrum data
    path = os.path.join(os.getcwd(), 'sources', 'wav', 'bird-chorus.wav')
    bird_chorus_kwargs = {
        'path': path,
        'filename': 'bird-chorus',
        'min_f': -1,
        'max_f': 6010,
        'window': 2 ** 11,
        'exp': 0.2,
        'yscale': 'linear',
        'cmap': 'inferno',
        'save_fig': True,
        'return_fig': False,
    }
    waterfall_plot(**bird_chorus_kwargs)

    bird_chorus_kwargs['save_fig'] = False
    bird_chorus_kwargs['return_fig'] = True
    bird_chorus_kwargs['video_name'] = 'bird_chorus'

    waterfall_animation(
        **bird_chorus_kwargs,
        clear_frames=False,
        regenerate_frames=False,
        window_s=10,
        fps=25,
    )


def overtone_singing():

    # Generate the spectrum data
    path = os.path.join(os.getcwd(), 'sources', 'wav', 'overtone-singing.wav')
    overtone_kwargs = {
        'path': path,
        'filename': 'overtone-singing',
        'min_f': 0,
        'max_f': 5000,
        'window': 2 ** 12,
        'exp': 0.1,
        'yscale': 'linear',
        'cmap': 'inferno',
        'save_fig': False,
        'return_fig': True,
    }

    waterfall_animation(
        **overtone_kwargs,
        clear_frames=False,
        regenerate_frames=False,
        window_s=60,
        fps=25,
    )


def generate_readme():
    convert_all_mp3_to_wav()
    bird_chorus()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_readme()
