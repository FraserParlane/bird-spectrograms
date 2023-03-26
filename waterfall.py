import matplotlib.pyplot as plt
from pydub import AudioSegment
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


if __name__ == '__main__':
    convert_all_mp3_to_wav()
