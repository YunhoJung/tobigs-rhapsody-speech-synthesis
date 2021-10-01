from dataload.download_video.download_youtube import download_youtube


if __name__ == "__main__":
    try:
        with open("url_list.txt", "r") as f:
            urls = f.readlines()

        for url in urls:
            download_youtube(url)
    except Exception as e:
        print(e)
