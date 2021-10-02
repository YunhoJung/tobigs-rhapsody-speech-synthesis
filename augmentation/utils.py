import librosa


def read_file(file_name):
    sample_file = file_name
    sample_path = sample_file

    y, sr = librosa.load(sample_path, None)

    return y, sr


def output_file(destination, file_name, y, sr, ext=""):
    destination = destination + file_name.split("/")[-1][:-4] + ext + '.wav'
    librosa.output.write_wav(destination, y, sr)
