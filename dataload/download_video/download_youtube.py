import os
import re
import sys
from builtins import OSError

import pytube


def download_youtube(url):
    download_dir = "../../data/video"
    try:
        if not os.path.isdir(download_dir):
            os.mkdir(download_dir)

        yt = pytube.YouTube(url)
        videos = yt.streams.all()
        videos[0].download(dir)

        new_filename = re.findall("watch\?v=(\S+)", url)[0]+".mp4"
        print(new_filename)
        default_filename = videos[0].default_filename

        try:
            os.rename(os.path.join(download_dir, default_filename), os.path.join(download_dir, new_filename))
        except OSError as e:
            print(new_filename, "file already exists")
            os.remove(os.path.join(download_dir, default_filename))
    except Exception as e:
        print(e)
