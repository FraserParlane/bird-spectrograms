from waterfall import convert_all_mp3_to_wav, wav_to_spec, waterfall_plot, waterfall_animation
import logging
import os


def bird_song():
    """
    Visualize the bird song data.
    :return: None
    """

    # Generate the spectrum data
    path = os.path.join(os.getcwd(), 'sources', 'wav', 'bird-song.wav')
    waterfall_plot(
        path=path,
        filename='bird_song',
        min_t=1,
        max_t=3.5,
        min_f=2000,
        max_f=15000,
        # background_color='white',
        # tick_color='black'
    )


def overtone_singing():
    """
    Visualize the bird song data.
    :return: None
    """

    # Generate the spectrum data
    path = os.path.join(os.getcwd(), 'sources', 'wav', 'overtone-singing.wav')
    waterfall_plot(
        path=path,
        filename='overtone-singing',
        min_t=11,
        max_t=80,
        min_f=0,
        max_f=5000,
        window=2 ** 12,
        exp=0.1,
        yscale='linear',
        cmap='inferno',
    )


def overtone_singing_animation():

    # Generate the spectrum data
    path = os.path.join(os.getcwd(), 'sources', 'wav', 'overtone-singing.wav')
    overtone_kwargs = {
        'path': path,
        'filename': 'overtone-singing',
        # 'min_t': 30,
        # 'max_t': 35,
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
        fps=12,
    )

    # Generate the spectrum data
    overtone_kwargs['save_fig'] = True
    overtone_kwargs['return_fig'] = False
    # waterfall_plot(**overtone_kwargs)


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

    waterfall_animation(
        **bird_chorus_kwargs,
        clear_frames=False,
        window_s=10,
        fps=25,
    )


def generate_readme():
    convert_all_mp3_to_wav()
    # bird_song()
    # overtone_singing()
    # overtone_singing_animation()
    bird_chorus()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_readme()
