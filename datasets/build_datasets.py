import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from skimage.transform import estimate_transform, warp
import random
import pickle
from .data_utils import landmarks_interpolate, linear_interpolate
import torch.nn.functional as F

class LRS3Dataset(Dataset):
    def __init__(self, data_list, cfg, test=False):
        self.data_list = data_list
        self.dataset_name = cfg.name
        self.image_size = 224
        self.K = cfg.K
        self.test = test
        self.cfg=cfg

        if not self.test:
            self.scale = [1.4, 1.8]
        else:
            self.scale = 1.6

        if self.dataset_name == 'LRS3':
            labels = open("../Visual_Speech_Recognition_for_Multiple_Languages/labels/LRS3/test.ref")
            self.labels_dict = {}
            for line in labels.readlines():
                basename, groundtruth = line.split()[0], " ".join(line.split()[1:])
                self.labels_dict[basename+".mp4"] = groundtruth


    def crop_face(self, frame, landmarks, scale=1.0):
        left = np.min(landmarks[:, 0]);
        right = np.max(landmarks[:, 0]);
        top = np.min(landmarks[:, 1]);
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])

        size = int(old_size * scale)
        # size = old_size

        # crop image
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        DST_PTS = np.array([[0, 0], [0, self.image_size - 1], [self.image_size - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)

        return tform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        images_list = []; kpt_list = [];

        if host == 'glaros':
            data_root = "/gpu-data3/filby/MEAD/rendered/train/MEAD"
            tcd_root = "/gpu-data3/filby/EAVTTS/TCDTIMIT_preprocessed"
        elif host == "halki" or host=="kalymnos":
            data_root = "/gpu-data3/filby/MEAD/rendered/train/MEAD"
            tcd_root = "/gpu-data3/filby/EAVTTS/TCDTIMIT_preprocessed"
        else:
            data_root = "/raid/gretsinas/MEAD"
            tcd_root = "/raid/gretsinas/TCDTIMIT_preprocessed"


        sample = self.data_list[index]

        if self.dataset_name == "LRS3":
            landmarks_filename = os.path.join("../Visual_Speech_Recognition_for_Multiple_Languages","landmarks", "LRS3", "LRS3_landmarks", sample[0],
                                              sample[1].replace(".txt", ".pkl"))
            if host == "glaros" or host == "kalymnos":
                data_filename = os.path.join("/gpu-data3/filby/LRS3", sample[0], sample[1].replace(".txt", ""))
            else:
                data_filename = os.path.join("/raid/gretsinas/LRS3", sample[0], sample[1].replace(".txt", "_v2"))

        elif self.dataset_name == "M003":
            data_filename = os.path.join("/gpu-data3/filby/MEAD/rendered/train/M003_full/images",
                                         sample[0].replace(".mp4", ""))

            landmarks_filename = os.path.join(
                "../Visual_Speech_Recognition_for_Multiple_Languages",
                "landmarks", "M003_images2vid", sample[0].replace(".mp4", ".pkl"))
        elif "MEAD" in self.dataset_name:
            data_filename = os.path.join(f"{data_root}/images",
                                         sample[0])

            landmarks_filename = os.path.join(
                "../Visual_Speech_Recognition_for_Multiple_Languages",
                "landmarks", "MEAD_images_25fps", sample[0]+".pkl")
        elif "TCDTIMIT" in self.dataset_name:
            data_filename = os.path.join(f"{tcd_root}/images",
                                         sample[0].replace(".mp4","").replace("59F","59M"))

            landmarks_filename = os.path.join(
                "../Visual_Speech_Recognition_for_Multiple_Languages",
                "landmarks", "TCDTIMIT_images_25fps", sample[0].replace(".mp4",".pkl").replace("59F","59M"))

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                return None

        if self.test:
            start_idx = 0
            end_idx = len(landmarks)
            frame_indices = list(range(len(landmarks)))
        else:
            if len(landmarks) < self.K:
                start_idx = 0
                end_idx = len(landmarks)
            else:
                start_idx = random.randint(0, len(landmarks) - self.K)
                end_idx = start_idx + self.K

            frame_indices = list(range(start_idx,end_idx))

        if isinstance(self.scale, list):
            scale = np.random.rand() * (self.scale[1] - self.scale[0]) + self.scale[0]
        else:
            scale = self.scale

        for frame_idx in frame_indices:
            if self.dataset_name == "LRS3":
                frame = cv2.imread(os.path.join(data_filename,"%06d.jpg"%(frame_idx)))
            else:
                frame = cv2.imread(os.path.join(data_filename,"%06d.png"%frame_idx))

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            kpt = preprocessed_landmarks[frame_idx]
            tform = self.crop_face(frame,kpt,scale)
            cropped_image = warp(frame, tform.inverse, output_shape=(self.image_size, self.image_size))

            cropped_kpt = np.dot(tform.params, np.hstack([kpt, np.ones([kpt.shape[0],1])]).T).T

            cropped_kpt[:,:2] = cropped_kpt[:,:2]/self.image_size * 2  - 1

            images_list.append(cropped_image.transpose(2,0,1))
            kpt_list.append(cropped_kpt)


        images_array = torch.from_numpy(np.array(images_list)).type(dtype = torch.float32) #K,224,224,3
        kpt_array = torch.from_numpy(np.array(kpt_list)).type(dtype = torch.float32) #K,224,224,3

        if "MEAD" in self.dataset_name:
            label = sample[1]
            vid_name = sample[0]+".mp4"
            wav_file = os.path.join(f"{data_root}/wavs",sample[0]+".wav")
        elif "TCDTIMIT" in self.dataset_name:
            label = sample[1]
            vid_name = sample[0]
            wav_file = os.path.join(f"{tcd_root}/wavs",sample[0].replace(".mp4",".wav"))
        else:
            vid_name = os.path.join(sample[0], sample[1].replace(".txt", ".mp4"))  # .repeat(self.K)
            try:
                label = self.labels_dict[vid_name]
            except:
                label = "UNK"

            wav_file = os.path.join("/gpu-data3/filby/LRS3", vid_name.replace(".mp4",".wav"))

        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'vid_name': vid_name,
            'label': label,
            'wav_file': wav_file,
        }

        return data_dict


def build_train_val(config, is_train=True):
    root_path = "/gpu-data3/filby/LRS3/trainval"
    from sklearn.model_selection import train_test_split

    # pickle.dump([train_list,test_list],open("lists.pkl","wb"))
    # lists = pickle.load(open("../Visual_Speech_Recognition_for_Multiple_Languages/lists.pkl", "rb"))
    lists = pickle.load(open("LRS3_lists.pkl", "rb"))
    #
    # train_list = lists[0]
    # val_list = lists[1]
    # test_list = lists[2]
    # #
    # print(len(train_list))
    # print(len(val_list))
    #
    #
    # l = list(os.listdir("/gpu-data3/filby/LRS3/pretrain"))
    # pretrain_list = []
    # for folder in l:
    #     for file in os.listdir(os.path.join("/gpu-data3/filby/LRS3/pretrain",folder)):
    #         if file.endswith(".txt"):
    #             pretrain_list.append(["pretrain/"+folder,file])
    #
    #
    # l = list(os.listdir("/gpu-data3/filby/LRS3/trainval"))
    # train_list = []
    # for folder in l:
    #     for file in os.listdir(os.path.join("/gpu-data3/filby/LRS3/trainval",folder)):
    #         if file.endswith(".txt"):
    #             train_list.append(["trainval/"+folder,file])
    #
    #
    # l = list(os.listdir("/gpu-data3/filby/LRS3/test"))
    # test_list = []
    # for folder in l:
    #     for file in os.listdir(os.path.join("/gpu-data3/filby/LRS3/test",folder)):
    #         if file.endswith(".txt"):
    #             test_list.append(["test/"+folder,file])
    # #
    # # print(len(pretrain_list), len(train_list), len(test_list))
    # # # print(test_list)
    # # #
    # pickle.dump([pretrain_list, train_list, test_list],open("LRS3_lists_official.pkl","wb"))
    # # raise
    # lists = pickle.load(open("lists.pkl", "rb"))
    # raise
    # lists = pickle.load(open("LRS3_lists_official.pkl", "rb"))
    #
    # pretrain = lists[0]
    # train_list = lists[1]
    # test_list = lists[2]
    # print(len(pretrain), len(train_list), len(test_list))
    # train_list = pretrain+train_list
    # print(len(train_list),len(test_list))

    train_list = lists[0]
    val_list = lists[1]
    test_list = lists[2]

    train_list = [("trainval/"+x[0],x[1]) for x in train_list]
    val_list = [("trainval/"+x[0],x[1]) for x in val_list]
    test_list = [("test/"+x[0],x[1]) for x in test_list]

    return LRS3Dataset(train_list, cfg=config), LRS3Dataset(val_list, cfg=config), LRS3Dataset(test_list, cfg=config, test=True)



def build_train_val_MEAD_txt(config=None):
    train_subjects_male = ["M009", "M011", "M013", "M019", "M023", "M024", "M025", "M026", "M027", "M028", "M030", "M032", "M033", "M034", "M035", "M037", "M039", "M041", "M042"]
    test_subjects_male = ["M003", "M007", "M005", "M012", "M022", "M029", "M031", "M037", "M040"]

    train_subjects_female = ["W014", "W015", "W016", "W021", "W023", "W019", "W024", "W026", "W028", "W029", "W033", "W036", "W038"]
    test_subjects_female = ["W009", "W011", "W018", "W025", "W037", "W035", "W040"]

    train_subjects = train_subjects_male + train_subjects_female
    test_subjects = test_subjects_male + test_subjects_female

    test_sentence_keywords = ["TODD PLACED", "DOG A BISCUIT", "SCOOP",
                              "PREACHED", "PLAINTIFF", "RADAR", "TALE WAS", "DRUNK HE", "NO PRICE", "THE REVOLUTION"]

    # gt = open("/home/filby/workspace/EAVTTS/Visual_Speech_Recognition_for_Multiple_Languages/list_full_mead_annotated.txt").readlines()
    # gt_dic = {}
    #
    # def is_test_sentence(text):
    #     for t in test_sentence_keywords:
    #         if t in text:
    #             return True
    #     return False
    #
    # train_list = []
    # test_list = []
    # for line in gt:
    #     # gt_dic[line.split()[0]] = " ".join(line.split()[1:])
    #     vid_name = line.split()[0]
    #     subject = vid_name.split("_")[0]
    #     text = " ".join(line.split()[1:])
    #
    #     if subject in test_subjects and is_test_sentence(text):
    #         test_list.append((vid_name, text))
    #     elif subject in train_subjects and not is_test_sentence(text):
    #         train_list.append((vid_name,text))
    #
    # torch.save(train_list, 'decalib/datasets/MEAD_train_text.pth')
    # torch.save(test_list, 'decalib/datasets/MEAD_test_text.pth')

    list_train = torch.load('decalib/datasets/MEAD_train_text.pth')
    list_test = torch.load('decalib/datasets/MEAD_test_text.pth')
    # print(len(list_test))
    # raise
    # print([x for x in list_test if "M003" in x[0]])
    # raise
    return LRS3Dataset(list_train, cfg=config), LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config, test=True)

# build_train_val_MEAD_txt()

import glob
def build_train_val_MEAD(config):
    # l = list(os.listdir(root_path))

    # labels = open("../Visual_Speech_Recognition_for_Multiple_Languages/list.txt")
    # labels_dict = {}
    # for line in labels.readlines():
    #     basename, groundtruth = line.split()[0], " ".join(line.split()[1:])
    #     labels_dict[basename] = groundtruth
    # print(len(labels_dict))

    # l_with_labels =[(x, labels_dict[x]) for x in l if x in labels_dict]

    train_subjects_male = ["M009", "M011", "M013", "M019", "M023", "M024", "M025", "M026", "M027", "M028", "M030", "M032", "M033", "M034", "M035", "M037", "M039", "M041", "M042"]
    test_subjects_male = ["M003", "M007", "M005",  "M012", "M022", "M029", "M031", "M037", "M040"]

    train_subjects_female = ["W014", "W015", "W016", "W021", "W023", "W019", "W024", "W026", "W028", "W029", "W033", "W036", "W038"]
    test_subjects_female = ["W009", "W011", "W018", "W025", "W037", "W035", "W040"]

    train_subjects = train_subjects_male + train_subjects_female
    test_subjects = test_subjects_male + test_subjects_female


    train_subjects = ["M031", "M030", "M027", "M023", "M009", "M022", "W009", "W014", "W021", "W028", "W036", "W040"]
    test_subjects = ["M003", "M007", "M012", "W011", "W015", "W016"]


    root_path = "/gpu-data3/filby/MEAD/rendered/train/MEAD/images"
    emotions = ['neutral', 'angry', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'contempt']
    levels = ["1", "2", "3"]

    # list_train = []
    # for subject in train_subjects:
    #     for emotion in emotions:
    #         for level in levels:
    #             f = sorted(glob.glob(os.path.join(root_path, f"{subject}_level_{level}_{emotion}_*")))
    #             list_train.extend(f[:len(f)-10])
    #
    # torch.save(list_train, 'MEAD_train_subset.pth')
    #
    # list_test = []
    # for subject in test_subjects:
    #     for emotion in emotions:
    #         for level in levels:
    #             f = sorted(glob.glob(os.path.join(root_path, f"{subject}_level_{level}_{emotion}_*")))
    #             list_test.extend(f[len(f)-10:])
    #
    # torch.save(list_test, 'MEAD_test_subset.pth')
    if "subset" in config.name:
        list_train = torch.load('decalib/datasets/MEAD_train_subset.pth')
        list_test = torch.load('decalib/datasets/MEAD_test_subset.pth')
    else:
        list_train = torch.load('decalib/datasets/MEAD_train.pth')
        list_test = torch.load('decalib/datasets/MEAD_test.pth')
    print(len(list_train), len(list_test))

    list_train = [(x.split("/")[-1],'UNK') for x in list_train]
    list_test = [(x.split("/")[-1],'UNK') for x in list_test]



    return LRS3Dataset(list_train, cfg=config), LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config, test=True)
# #
# def build_questionnaire(config):
#     import pandas as pd
#     questionnaire_list = pd.read_csv("../utils/MEAD_test_set_final.csv")
#
#     list_test = [(x,'UNK') for x in questionnaire_list.name]
#     # print(list_test)
#     return LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config, test=True)

def build_questionnaire_TCDTIMIT(config):
    clips = ["28M_si1869.mp4",
            "49F_sx409.mp4",
            "25M_sx204.mp4",
            "33F_sx124.mp4",
             "56M_sx251.mp4",
             "54M_sx450.mp4",
             "08F_si1171.mp4",
             "58F_sx345.mp4",
             "41M_sx403.mp4",
             "44F_sx76.mp4",
             ]

    clips2 = ["28M_si2038.mp4",
        "45F_sx181.mp4",
        "08F_si546.mp4",
        "44F_sa1.mp4",
        "41M_sx366.mp4",
        "56M_sx424.mp4",
        "55F_sx277.mp4",
        "54M_sx185.mp4",
        "55F_sa2.mp4",
        "09F_si1998.mp4"
        ]


    import pandas as pd
    # questionnaire_list = pd.read_csv("../utils/mouth_questionnaire.csv")

    list_test = [(x,'UNK') for x in clips2]
    # print(list_test)
    return LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config, test=True)


def build_questionnaire(config):
    import pandas as pd
    # questionnaire_list = pd.read_csv("../utils/MEAD_test_set_final.csv")
    questionnaire_list = pd.read_csv("../utils/MEAD_transcription_quiz.csv")
    clips = [
        "W016_level_2_surprised_013",
        "W019_level_3_disgusted_014",
        "M030_level_2_disgusted_018",
        "M025_level_1_angry_013",

        "W028_level_2_sad_009",
        "W021_level_3_surprised_043",
        "M009_level_2_fear_021",
        "M003_level_2_contempt_029",
        "M013_level_1_neutral_037",
        "M027_level_2_angry_011",
        "W023_level_1_surprised_023",
        "M034_level_1_neutral_019",
        "W016_level_1_neutral_018",
        "W019_level_1_neutral_030",
        "W009_level_1_neutral_028",
        "W019_level_1_neutral_030",
        "M035_level_1_neutral_040"

    ]

    # list_test = [(x,'UNK') for x in clips]
    list_test = [(x,'UNK') for x in questionnaire_list.name]
    return LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config, test=True)



import glob
def build_train_val_TCD(config=None):
    if host == 'bessarion':
        tcd_root = "/raid/gretsinas"
    else:
        tcd_root = "/gpu-data3/filby/EAVTTS"

    root = f"{tcd_root}/TCDTIMIT_preprocessed/TCDSpkrIndepTrainSet.scp"
    files = open(root).readlines()
    list_train = []
    for file in files:
        f = file.strip().split("/")
        new_name = f"{f[0]}_{f[-1]}"

        ff = "/".join([f[0],f[1],f[2]])

        text = open(os.path.join(f"{tcd_root}/TCDTIMITprocessing/downloadTCDTIMIT/volunteers",ff,f[-1].upper().replace(".MP4",".txt"))).readlines()

        text = " ".join([x.split()[2].strip() for x in text])

        list_train.append((new_name,text))


    root = f"{tcd_root}/TCDTIMIT_preprocessed/TCDSpkrIndepTestSet.scp"
    files = open(root).readlines()
    list_test = []
    for file in files:
        f = file.strip().split("/")
        new_name = f"{f[0]}_{f[-1]}"

        ff = "/".join([f[0],f[1],f[2]])

        text = open(os.path.join(f"{tcd_root}/TCDTIMITprocessing/downloadTCDTIMIT/volunteers",ff,f[-1].upper().replace(".MP4",".txt"))).readlines()

        text = " ".join([x.split()[2].strip() for x in text])

        list_test.append((new_name,text.upper()))



    return LRS3Dataset(list_train, cfg=config), LRS3Dataset(list_test, cfg=config), LRS3Dataset(list_test, cfg=config, test=True)

