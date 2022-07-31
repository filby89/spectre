import os
from .datasets import SpectreDataset


def get_datasets_MEAD(config=None):
    import pandas as pd
    questionnaire_list = pd.read_csv("../utils/MEAD_test_set_final.csv")
    test_list = [(x[0],x[1]) for x in zip(questionnaire_list.name,questionnaire_list.text)]
    landmarks_path = "../Visual_Speech_Recognition_for_Multiple_Languages/landmarks/MEAD_images_25fps"

    return None, None, SpectreDataset(test_list, landmarks_path, cfg=config, test=True)


def get_datasets_TCDTIMIT(config=None):
    tcd_root = "/gpu-data3/filby/EAVTTS"

    landmarks_path = "../Visual_Speech_Recognition_for_Multiple_Languages/landmarks/TCDTIMIT_images_25fps"

    root = f"{tcd_root}/TCDTIMIT_preprocessed/TCDSpkrIndepTrainSet.scp"
    files = open(root).readlines()
    train_list = []
    for file in files:
        f = file.strip().split("/")
        new_name = f"{f[0]}_{f[-1]}"

        ff = "/".join([f[0],f[1],f[2]])

        text = open(os.path.join(f"{tcd_root}/TCDTIMITprocessing/downloadTCDTIMIT/volunteers",ff,f[-1].upper().replace(".MP4",".txt"))).readlines()

        text = " ".join([x.split()[2].strip() for x in text])

        train_list.append((new_name.split(".")[0],text))


    root = f"{tcd_root}/TCDTIMIT_preprocessed/TCDSpkrIndepTestSet.scp"
    files = open(root).readlines()
    test_list = []
    for file in files:
        f = file.strip().split("/")
        new_name = f"{f[0]}_{f[-1]}"

        ff = "/".join([f[0],f[1],f[2]])

        text = open(os.path.join(f"{tcd_root}/TCDTIMITprocessing/downloadTCDTIMIT/volunteers",ff,f[-1].upper().replace(".MP4",".txt"))).readlines()

        text = " ".join([x.split()[2].strip() for x in text])

        test_list.append((new_name.split(".")[0],text.upper()))


    return SpectreDataset(train_list, landmarks_path, cfg=config), SpectreDataset(test_list, landmarks_path, cfg=config), SpectreDataset(test_list, landmarks_path, cfg=config, test=True)