import librosa
from pysndfx import AudioEffectsChain
import numpy as np
import math
import scipy


def read_file(file_name):
    sample_file = file_name
    sample_path = sample_file

    y, sr = librosa.load(sample_path, None)

    return y, sr


def reduce_noise_power(y, sr):
    """
    :param y: audio matrix
    :param sr:
    :return: audio matrix after gain reduction on noise
    """
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = round(np.median(cent)) * 1.5
    threshold_l = round(np.median(cent)) * 0.1

    less_noise = AudioEffectsChain()\
        .lowshelf(gain=-30.0, frequency=threshold_l, slope=0.8)\
        .highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)
    y_clean = less_noise(y)

    return y_clean


def reduce_noise_centroid_s(y, sr):
    """
    :param y: audio matrix
    :param sr:
    :return: audio matrix after gain reduction on noise
    """
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain()\
        .lowshelf(gain=-12.0, frequency=threshold_l, slope=0.5)\
        .highshelf(gain=-12.0, frequency=threshold_h, slope=0.5)\
        .limiter(gain=6.0)

    y_cleaned = less_noise(y)

    return y_cleaned


def reduce_noise_centroid_mb(y, sr):
    """
    :param y: audio matrix
    :param sr:
    :return: audio matrix after gain reduction on noise
    """
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)

    threshold_h = np.max(cent)
    threshold_l = np.min(cent)

    less_noise = AudioEffectsChain()\
        .lowshelf(gain=-30.0, frequency=threshold_l, slope=0.5)\
        .highshelf(gain=-30.0, frequency=threshold_h, slope=0.5)\
        .limiter(gain=10.0)
    y_cleaned = less_noise(y)

    cent_cleaned = librosa.feature.spectral_centroid(y=y_cleaned, sr=sr)
    columns, rows = cent_cleaned.shape
    boost_h = math.floor(rows / 3 * 2)

    boost_bass = AudioEffectsChain().lowshelf(gain=16.0, frequency=boost_h, slope=0.5)
    y_clean_boosted = boost_bass(y_cleaned)

    return y_clean_boosted


def reduce_noise_median(y):
    """
    :param y: audio matrix
    :return: audio matrix after gain reduction on noise
    """
    y = scipy.signal.medfilt(y, 3)
    return y


def trim_silence(y):
    """
    :param y:
    :return: audio matrix with less silence and the amount of time that was trimmed
    """
    y_trimmed, index = librosa.effects.trim(y, top_db=20, frame_length=2, hop_length=10)
    trimmed_length = librosa.get_duration(y) - librosa.get_duration(y_trimmed)

    return y_trimmed, trimmed_length


def enhance(y):
    """
    :param y: audio matrix
    :return: audio matrix after audio manipulation
    """
    apply_audio_effects = AudioEffectsChain()\
        .lowshelf(gain=10.0, frequency=260, slope=0.1)\
        .reverb(reverberance=25, hf_damping=5, room_scale=5, stereo_depth=50,  pre_delay=20, wet_gain=0, wet_only=False)
    y_enhanced = apply_audio_effects(y)

    return y_enhanced


def output_file(destination, file_name, y, sr, ext=""):
    """
    generates a wav file
    :param destination:
    :param file_name:
    :param y:
    :param sr:
    :param ext:
    :return: None
    """
    destination = destination + file_name.split("/")[-1][:-4] + ext + '.wav'
    librosa.output.write_wav(destination, y, sr)
