import io
import os
import json

from progressbar import ProgressBar

from google.cloud import speech
from google.cloud.speech import types


def get_speech_text(chunk_path, file_name):
    client = speech.SpeechClient()

    file_path = os.path.join(chunk_path, file_name)
    with io.open(file_path, 'rb') as audio_file:
        content = audio_file.read()
        audio = types.RecognitionAudio(content=content)

    config = types.RecognitionConfig(language_code='ko-KR')
    response = client.recognize(config, audio)
    try:
        result = response.results[0].alternatives[0].transcript
    except Exception as e:
        print(e)
        result = ""
    return result


if __name__ == "__main__":
    bar = ProgressBar()
    data = dict()
    chunk_path = "../../data/chunk"
    for file_name in bar(sorted(os.listdir(chunk_path))):
        data[file_name] = get_speech_text(chunk_path, file_name)

    with open("justtest-3c032791a663.json", "w") as f:
        f.write(json.dumps(data, ensure_ascii=False, indent="\t"))
