import os

import ProgressBar

from augmentation.expand_data import pitch_up, pitch_down, speed_up, speed_down
from augmentation.reduct_noise import reduce_noise_power, reduce_noise_centroid_s, reduce_noise_centroid_mb, \
    reduce_noise_median, enhance, trim_silence
from augmentation.utils import read_file, output_file


if __name__ == "__main__":
    bar = ProgressBar()
    chunk_path = "../../data/chunk"
    for file_name in bar(sorted(os.listdir(chunk_path))):
        y, sr = read_file(os.path.join(chunk_path, file_name))

        # expand data
        y_pitch_up = pitch_up(y)
        y_pitch_down = pitch_down(y)
        y_speed_up = speed_up(y)
        y_speed_down = speed_down(y)

        # reduct noise
        y_reduced_power = reduce_noise_power(y, sr)
        y_reduced_centroid_s = reduce_noise_centroid_s(y, sr)
        y_reduced_centroid_mb = reduce_noise_centroid_mb(y, sr)
        y_reduced_median = reduce_noise_median(y)
        y_enhanced = enhance(y)

        y_reduced_power_trim_silence, _ = trim_silence(y_reduced_power)
        y_reduced_centroid_s_trim_silence, _ = trim_silence(y_reduced_centroid_s)
        y_reduced_centroid_mb_trim_silence, _ = trim_silence(y_reduced_centroid_mb)
        y_reduced_median_trim_silence, _ = trim_silence(y_reduced_median)
        y_enhanced_trim_silence, _ = trim_silence(y_enhanced)

        # save
        output_file(chunk_path, file_name, y_pitch_up, sr, "pch_up")
        output_file(chunk_path, file_name, y_pitch_down, sr, "pch_down")
        output_file(chunk_path, file_name, y_speed_up, sr, "spd_up")
        output_file(chunk_path, file_name, y_speed_down, sr, "spd_down")

        output_file(chunk_path, file_name, y_reduced_power_trim_silence, sr, "_pwr")
        output_file(chunk_path, file_name, y_reduced_centroid_s_trim_silence, sr, "_ctr_s")
        output_file(chunk_path, file_name, y_reduced_centroid_mb_trim_silence, sr, "_ctr_mb")
        output_file(chunk_path, file_name, y_reduced_median_trim_silence, sr, "_median")
        output_file(chunk_path, file_name, y_enhanced_trim_silence, sr, "_enhanced")
