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
        min_f=2500,
        max_f=10000,
    )


def generate_readme():
    bird_song()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    generate_readme()
