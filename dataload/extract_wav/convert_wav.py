import os

from moviepy.editor import VideoFileClip


if __name__ == "__main__":
    data_path = "../../data"
    video_path = os.path.join(data_path, "video")
    speech_path = os.path.join(data_path, "speech")

    files = list(filter(lambda x: os.path.splitext(x)[1] == ".mp4", os.listdir(video_path)))
    for filename in files:
        try:
            video_clip = VideoFileClip(os.path.join(video_path, filename))
            audio_clip = video_clip.audio
            audio_clip.write_audiofile(os.path.join(speech_path, os.path.splitext(filename)[0] + ".wav"))
            audio_clip.close()
        except Exception as e:
            print(filename, e)
