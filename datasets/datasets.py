import os
import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from skimage.transform import estimate_transform, warp
import random
import pickle
from .data_utils import landmarks_interpolate

class SpectreDataset(Dataset):
    def __init__(self, data_list, landmarks_path, cfg, test=False):
        self.data_list = data_list
        self.image_size = 224
        self.K = cfg.K
        self.test = test
        self.cfg=cfg
        self.landmarks_path = landmarks_path

        if not self.test:
            self.scale = [1.4, 1.8]
        else:
            self.scale = 1.6

    def crop_face(self, frame, landmarks, scale=1.0):
        left = np.min(landmarks[:, 0])
        right = np.max(landmarks[:, 0])
        top = np.min(landmarks[:, 1])
        bottom = np.max(landmarks[:, 1])

        h, w, _ = frame.shape
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])  # + old_size*0.1])

        size = int(old_size * scale)

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

        sample = self.data_list[index]

        landmarks_filename = os.path.join(self.landmarks_path, sample[0]+".pkl")
        folder_path = os.path.join(self.cfg.LRS3_path, sample[0])

        with open(landmarks_filename, "rb") as pkl_file:
            landmarks = pickle.load(pkl_file)
            preprocessed_landmarks = landmarks_interpolate(landmarks)
            if preprocessed_landmarks is None:
                return None

        if self.test:
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
            if "LRS3" in self.landmarks_path:
                frame = cv2.imread(os.path.join(folder_path,"%06d.jpg"%(frame_idx)))
                folder_path = os.path.join(self.cfg.LRS3_path, sample[0])
                wav = folder_path + ".wav"
            else: # during test mode for other datasets
                if 'MEAD' in self.landmarks_path:
                    folder_path = os.path.join("/gpu-data3/filby/MEAD/rendered/train/MEAD/images", sample[0])
                    frame = cv2.imread(os.path.join(folder_path,"%06d.png"%(frame_idx)))
                    wav = folder_path.replace("images","wavs") + ".wav"
                else:
                    folder_path = os.path.join("/gpu-data3/filby/EAVTTS/TCDTIMIT_preprocessed/images", sample[0])
                    frame = cv2.imread(os.path.join(folder_path,"%06d.png"%(frame_idx)))
                    wav = folder_path.replace("images","wavs") + ".wav"

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

        # text = open(folder_path+".txt").readlines()[0].replace("Text:","").strip()
        text = sample[1] # open(folder_path+".txt").readlines()[0].replace("Text:","").strip()

        data_dict = {
            'image': images_array,
            'landmark': kpt_array,
            'vid_name': sample[0],
            'wav_path': wav, # this is only used for evaluation - you can remove this key from the dictionary if you don't need it
            'text': text,  # this is only used for evaluation - you can remove this key from the dictionary if you don't need it
        }

        return data_dict


def get_datasets_LRS3(config=None):
    if not os.path.exists('data/LRS3_lists.pkl'):
        print('Creating train, validation, and test lists for LRS3... (This only happens once)')

        from .data_utils import create_LRS3_lists
        create_LRS3_lists(config.LRS3_path)


    lists = pickle.load(open("data/LRS3_lists.pkl", "rb"))
    train_list = lists[0]
    val_list = lists[1]
    test_list = lists[2]
    landmarks_path = config.LRS3_landmarks_path
    return SpectreDataset(train_list, landmarks_path, cfg=config), SpectreDataset(val_list, landmarks_path, cfg=config), SpectreDataset(test_list, landmarks_path,
                                                                                               cfg=config,
                                                                                               test=True)
