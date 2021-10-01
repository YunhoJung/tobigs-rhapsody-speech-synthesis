import os

import librosa
import numpy as np
import matplotlib.pyplot as plt


def get_mel_spectrogram(chunk_path, file_name):
    y, sr = librosa.load(os.path.join(chunk_path, file_name))

    mel_spectrogram = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
    log_mel_spectrogram = librosa.amplitude_to_db(mel_spectrogram, ref=np.max)

    plt.figure(figsize=(12, 4))
    librosa.display.specshow(log_mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.title('mel power spectrogram')
    plt.colorbar(format='%+02.0f dB')
    plt.tight_layout()
    plt.show()
