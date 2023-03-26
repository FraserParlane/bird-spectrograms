from waterfall import convert_all_mp3_to_wav, wav_to_spec, waterfall_plot
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


def generate_readme():
    # bird_song()
    overtone_singing()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_readme()
