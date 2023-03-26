from waterfall import convert_all_mp3_to_wav, wav_to_waterfall_data
import os


def bird_song():
    """
    Visualize the bird song data.
    :return: None
    """

    # Generate the spectrum data
    path = os.path.join(os.getcwd(), 'sources', 'wav', 'bird-song-short.wav')
    spect, freqs, time = wav_to_waterfall_data(path)

    print('a')



def generate_readme():
    bird_song()


if __name__ == '__main__':
    generate_readme()
