import os
from pydub import AudioSegment
from pydub.silence import split_on_silence


def get_speech_chunk(speech_path, chunk_path, file_name, file_format):

    try:
        sound = AudioSegment.from_file(os.path.join(speech_path, file_name), format=file_format)
        sound = sound.set_channels(1)

        # 0.2초(500ms)보다 긴 무음과 -30 dBFS 미만의 소리는 무음으로 간주
        chunks = split_on_silence(
            sound,
            min_silence_len=200,
            silence_thresh=-30,
        )

        sound_count = 0
        for chunk in chunks:
            # 잘려진 음성의 길이가 2초 이상, 12초 이하인 것만 학습 데이터로 취급
            if (chunk.duration_seconds >= 2) and (chunk.duration_seconds <= 12):
                chunk.export(os.path.join(chunk_path, os.path.splitext(file_name)[0] + "{0}.wav".format(sound_count)),
                             format=file_format)
                sound_count += 1
    except Exception as e:
        print(e)
