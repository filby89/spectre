#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import os
import torch
from lipreading.subroutines import LipreadingPipeline
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
separator = Separator(phone='-', word=' ')
backend = EspeakBackend('en-us', words_mismatch='ignore', with_stress=False)
import jiwer

# phonemes to visemes map. this was created using Amazon Polly
# https://docs.aws.amazon.com/polly/latest/dg/polly-dg.pdf

def get_phoneme_to_viseme_map():
    pho2vi = {}
    # pho2vi_counts = {}
    all_vis = []

    p2v = "data/phonemes2visemes.csv"

    with open(p2v) as file:
        lines = file.readlines()
        # for line in lines[2:29]+lines[30:50]:
        for line in lines:
            if line.split(",")[0] in pho2vi:
                if line.split(",")[4].strip() != pho2vi[line.split(",")[0]]:
                    print('error')
            pho2vi[line.split(",")[0]] = line.split(",")[4].strip()

            all_vis.append(line.split(",")[4].strip())
            # pho2vi_counts[line.split(",")[0]] = 0
    return pho2vi, all_vis

pho2vi, all_vis = get_phoneme_to_viseme_map()

def convert_text_to_visemes(text):
    phonemized = backend.phonemize([text], separator=separator)[0]

    text = ""
    for word in phonemized.split(" "):
        visemized = []
        for phoneme in word.split("-"):
            if phoneme == "":
                continue
            try:
                visemized.append(pho2vi[phoneme.strip()])
                if pho2vi[phoneme.strip()] not in all_vis:
                    all_vis.append(pho2vi[phoneme.strip()])
                # pho2vi_counts[phoneme.strip()] += 1
            except:
                print('Count not find', phoneme)
                continue
        text += " " + "".join(visemized)
    return text


def get_video_lipread_metrics(lipreader, video_path, landmarks, groundtruth_text):
    """

    :param lipreader: a lipreader
    :param data: the path of the video, landmarks, and the groundtruth text
    :return: mouths and lipread metrics
    """


    with torch.no_grad():
        output = lipreader(video_path, landmarks)

    wer = jiwer.wer(groundtruth_text, output)
    cer = jiwer.cer(groundtruth_text, output)

    # ---------- convert to visemes -------- #
    vg = convert_text_to_visemes(groundtruth_text)
    v = convert_text_to_visemes(output)
    # -------------------------------------- #
    werv = jiwer.wer(vg, v)
    cerv = jiwer.cer(vg, v)

    print(f"hyp: {output}")
    print(f"hypv: {v}")
    print(f"ref: {groundtruth_text}")
    print(f"refv: {vg}")


    return wer, cer, werv, cerv

