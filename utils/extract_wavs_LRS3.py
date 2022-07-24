import os
import cv2
import time
import numpy as np
import torch
from argparse import ArgumentParser

import sys
sys.path.append("face_parsing")


def extract_wav(videopath):
    print(videopath)

    os.system("ffmpeg -i {} {} -y".format(videopath, videopath.replace(".mp4",".wav")))

from multiprocessing import Pool
from tqdm import tqdm

def main():
    # Parse command-line arguments
    parser = ArgumentParser()

    root = "/raid/gretsinas/LRS3/test"

    p = Pool(12)

    l = list(os.listdir("/raid/gretsinas/LRS3/test"))
    test_list = []
    for folder in l:
        for file in os.listdir(os.path.join("/raid/gretsinas/LRS3/test",folder)):

            if file.endswith(".txt"):
                test_list.append(os.path.join("/raid/gretsinas/LRS3/test",folder,file.replace(".txt",".mp4")))

    # print(test_list)
    # extract_wav(test_list[0])
    for _ in tqdm(p.imap_unordered(extract_wav, test_list), total=len(test_list)):
        pass


main()

# import os
# import cv2
# import time
# import numpy as np
# import torch
# from argparse import ArgumentParser
#
# import sys
# sys.path.append("face_parsing")
#
#
# def extract_wav(videopath):
#     # print(videopath)
#
#     os.system("ffmpeg -i {} {} -y".format(videopath, videopath.replace("/videos/","/wavs/").replace(".mp4",".wav")))
#
# from multiprocessing import Pool
# from tqdm import tqdm
#
# def main():
#     # Parse command-line arguments
#     parser = ArgumentParser()
#
#     root = "/gpu-data3/filby/MEAD/rendered/train/MEAD/videos"
#
#     p = Pool(20)
#
#     test_list = []
#     for file in os.listdir(root):
#         test_list.append(os.path.join(root,file))
#
#     # print(test_list)
#     # extract_wav(test_list[0])
#     for _ in tqdm(p.imap_unordered(extract_wav, test_list), total=len(test_list)):
#         pass
#
#
# main()