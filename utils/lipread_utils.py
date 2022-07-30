#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
from phonemizer.backend import EspeakBackend
from phonemizer.separator import Separator
separator = Separator(phone='-', word=' ')
backend = EspeakBackend('en-us', words_mismatch='ignore', with_stress=False)
import cv2

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



def save2avi(filename, data=None, fps=25):
    """save2avi. - function taken from Visual Speech Recognition repository

    :param filename: str, the filename to save the video (.avi).
    :param data: numpy.ndarray, the data to be saved.
    :param fps: the chosen frames per second.
    """
    assert data is not None, "data is {}".format(data)
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc("F", "F", "V", "1")
    writer = cv2.VideoWriter(filename, fourcc, fps, (data[0].shape[1], data[0].shape[0]), 0)
    for frame in data:
        writer.write(frame)
    writer.release()


def predict_text(lipreader, mouth_sequence):
    from external.Visual_Speech_Recognition_for_Multiple_Languages.espnet.asr.asr_utils import add_results_to_json
    lipreader.model.eval()
    with torch.no_grad():
        enc_feats, _ = lipreader.model.encoder(mouth_sequence, None)
        enc_feats = enc_feats.squeeze(0)

        nbest_hyps = lipreader.beam_search(
            x=enc_feats,
            maxlenratio=lipreader.maxlenratio,
            minlenratio=lipreader.minlenratio
        )
        nbest_hyps = [
            h.asdict() for h in nbest_hyps[: min(len(nbest_hyps), lipreader.nbest)]
        ]

        transcription = add_results_to_json(nbest_hyps, lipreader.char_list)

    return transcription.replace("<eos>", "")

def predict_text_deca(lipreader, mouth_sequence):
    from external.Visual_Speech_Recognition_for_Multiple_Languages.espnet.asr.asr_utils import add_results_to_json
    lipreader.model.eval()
    with torch.no_grad():
        enc_feats, _ = lipreader.model.encoder(mouth_sequence, None)
        enc_feats = enc_feats.squeeze(0)

        ys_hat = lipreader.model.ctc.ctc_lo(enc_feats)
        # print(ys_hat)
        ys_hat = ys_hat.argmax(1)
        ys_hat = torch.unique_consecutive(ys_hat, dim=-1)

        ys = [lipreader.model.args.char_list[x] for x in ys_hat if x != 0]

        ys = "".join(ys)
        ys = ys.replace("<space>", " ")

    return ys.replace("<eos>", "")


