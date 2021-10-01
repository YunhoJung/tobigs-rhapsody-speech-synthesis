import json
import os
from os.path import splitext

from progressbar import ProgressBar
from pydub import AudioSegment

from preprocess.speech_slice.get_chunk_text import get_speech_text
from preprocess.speech_slice.get_mel_spectrogram import get_mel_spectrogram
from preprocess.speech_slice.get_speech_chunk import get_speech_chunk


if __name__ == "__main__":
    data_path = "../../data"
    speech_path = os.path.join(data_path, "speech")
    chunk_path = os.path.join(data_path, "chunk")

    # get speech chunk
    wav_files = [file_name for file_name in os.listdir(speech_path) if splitext(file_name)[1] == ".wav"]
    total_length = 0.
    for file_name in wav_files:
        get_speech_chunk(speech_path, chunk_path, file_name, "wav")
        total_length += AudioSegment.from_file(os.path.join(speech_path, file_name), format="wav").duration_seconds
    print("Audio Files Total Length :", total_length, "sec")

    # get speech text
    bar = ProgressBar()
    data = dict()
    for file_name in bar(sorted(os.listdir(chunk_path))):
        data[file_name] = get_speech_text(chunk_path, file_name)

    with open("justtest-3c032791a663.json", "w") as f:
        f.write(json.dumps(data, ensure_ascii=False))

    # get speech spectrum
    chunk_wav_files = [file_name for file_name in os.listdir(chunk_path) if splitext(file_name)[1] == ".wav"]
    for file_name in chunk_wav_files:
        get_mel_spectrogram(chunk_path, file_name)
