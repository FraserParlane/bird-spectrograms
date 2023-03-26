from waterfall import convert_all_mp3_to_wav, wav_to_spec, waterfall_plot
import os


def bird_song():
    """
    Visualize the bird song data.
    :return: None
    """

    # Generate the spectrum data
    path = os.path.join(os.getcwd(), 'sources', 'wav', 'bird-song-short.wav')
    waterfall_plot(
        path=path,
        filename='bird_song',
        min_t=0.1,
        min_f=2500,
        max_f=10000,
    )


def generate_readme():
    bird_song()


if __name__ == '__main__':
    generate_readme()
