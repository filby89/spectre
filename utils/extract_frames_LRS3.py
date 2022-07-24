import os
import cv2
import time
import numpy as np
import torch
from argparse import ArgumentParser

import sys



def extract(video, tmpl='%06d.jpg'):
    os.makedirs(video.replace(".mp4", ""),exist_ok=True)
    cmd = 'ffmpeg -i \"{}\" -threads 1 -q:v 0 \"{}/%06d.jpg\"'.format(video,
                                                                                       video.replace(".mp4", ""))
    os.system(cmd)

    # os.system("ffmpeg -i {} {} -y".format(videopath, videopath.replace(".mp4",".wav")))


# -*- coding: utf-8 -*-

import os, sys
import cv2
import numpy as np
from time import time
from scipy.io import savemat
import argparse
from tqdm import tqdm
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from decalib.deca import DECA
from decalib.datasets import datasets
from decalib.utils import util
from decalib.utils.config import cfg as deca_cfg
import pickle



def video2sequence(video_path, videofolder):
    os.makedirs(videofolder, exist_ok=True)
    video_name = os.path.splitext(os.path.split(video_path)[-1])[0]
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = os.path.join(videofolder, f'{video_name}_frame{count:05d}.jpg')
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return imagepath_list


from multiprocessing import Pool
from tqdm import tqdm

def main():
    # Parse command-line arguments
    parser = ArgumentParser()

    root = "/gpu-data3/filby/LRS3/pretrain"


    l = list(os.listdir("/gpu-data3/filby/LRS3/pretrain"))
    test_list = []
    for folder in l:
        for file in os.listdir(os.path.join("/gpu-data3/filby/LRS3/pretrain",folder)):

            if file.endswith(".txt"):
                test_list.append([os.path.join("/gpu-data3/filby/LRS3/pretrain",folder,file.replace(".txt",".mp4")),os.path.join("/gpu-data3/filby/LRS3/pretrain",folder,file.replace(".txt",".mp4"))])

    # print(test_list[0])
    extract(test_list[0])
    raise
    p = Pool(12)

    for _ in tqdm(p.imap_unordered(video2sequence, test_list), total=len(test_list)):
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