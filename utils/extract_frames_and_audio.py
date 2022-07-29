# -*- coding: utf-8 -*-

import os, sys
import cv2
import argparse
from tqdm import tqdm
from multiprocessing import Pool

def video2sequence(video_path):
    videofolder = os.path.splitext(video_path)[0]
    os.makedirs(videofolder, exist_ok=True)
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    count = 0
    imagepath_list = []
    while success:
        imagepath = os.path.join(videofolder, f'%06d.jpg'%count)
        cv2.imwrite(imagepath, image)     # save frame as JPEG file
        success,image = vidcap.read()
        count += 1
        imagepath_list.append(imagepath)
    print('video frames are stored in {}'.format(videofolder))
    return videofolder


def extract_audio(video_path):
    os.system("ffmpeg -i {} {} -y".format(video_path, video_path.replace(".mp4",".wav")))


def main(args):
    video_list = []

    for mode in ["trainval","test"]:
        for folder in os.listdir(os.path.join(args.dataset_path,mode)):
            for file in os.listdir(os.path.join(args.dataset_path,mode,folder)):
                if file.endswith(".mp4"):
                    video_list.append(os.path.join(args.dataset_path,mode,folder,file))

    p = Pool(12)

    for _ in tqdm(p.imap_unordered(video2sequence, video_list), total=len(video_list)):
        pass

    for _ in tqdm(p.imap_unordered(extract_audio, video_list), total=len(video_list)):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset_path', default='./data/LRS3', type=str, help='path to dataset')
    main(parser.parse_args())