import os
import csv
from pydub import AudioSegment

# Function to split .wav files into 10-second chunks and store them in the specified directory structure
def split_wav_file(file_path, output_base_dir, chunk_length=10000):
    audio = AudioSegment.from_wav(file_path)
    duration = len(audio)  # Duration in milliseconds
    duration = (duration//chunk_length)*chunk_length
    relative_path = os.path.relpath(file_path, start=input_directory)
    base_name = os.path.basename(file_path).replace('.wav', '')

    # Extract directory structure without the file name
    file_dir = os.path.dirname(relative_path)
    output_dir = os.path.join(output_base_dir, file_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i in range(0, duration, chunk_length):
        chunk = audio[i:i + chunk_length]
        chunk_name = f"{base_name}_chunk{i // chunk_length + 1}.wav"
        chunk.export(os.path.join(output_dir, chunk_name), format="wav")

# Function to filter out the files that donot contain mosquito sounds.
def dataset_generation(root, output_base_dir):
    species_number = {}
    os.makedirs(output_base_dir, exist_ok=True)
    for paths in os.listdir(root):
        count = 0
        for root_dir, cur_dir, files in os.walk(root + paths):
            index = paths.split('. ')
            filename = index[0] + '.csv'
            for name in files:
                fname = os.path.join(root_dir, name)
                if fname[-3:] != 'wav' or fname[-6:-4] == 'bg' or fname[-14:-4] == 'Background':
                    continue
                print(fname)
                split_wav_file(fname, output_base_dir)  # Split and save the .wav file chunks

if __name__ == "__main__":
    input_directory = '/work/pi_rozhin_hajian_uml_edu/mwb_pred/data/Stanford/'
    output_directory = '/work/pi_rozhin_hajian_uml_edu/mwb_pred/data/Stanford_sampled/'
    dataset_generation(input_directory, output_directory)
