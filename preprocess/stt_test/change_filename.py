import os
import re


def change_filename():
    file_list = []
    for i in range(len(os.listdir("."))):
        filename = os.listdir(".")[i]
        if filename.endswith("wav"):
            file_list.append(filename)
    sorted_file_list = sorted(file_list, key=lambda integer: int(re.findall('\d+', integer)[0]))

    j = 1
    t = 1
    for file_name in sorted_file_list:
        if (j >= 1) and (j < 10):
            new_filename = file_name.replace(file_name.split('.')[0], "LJ00"+str(t)+"-000"+str(j))
        elif (j >= 10) and (j < 100):
            new_filename = file_name.replace(file_name.split('.')[0], "LJ00"+str(t)+"-00"+str(j))
        else:
            new_filename = file_name.replace(file_name.split('.')[0], "LJ00"+str(t)+"-0"+str(j))
        os.rename(file_name, new_filename)
        j = j + 1

