import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F


def l2_distance(verts1, verts2):
    return torch.sqrt(((verts1 - verts2)**2).sum(2)).mean(1).mean()

### ------------------------------------- Losses/Regularizations for vertices
def batch_kp_2d_l1_loss(real_2d_kp, predicted_2d_kp, weights=None):
    """
    Computes the l1 loss between the ground truth keypoints and the predicted keypoints
    Inputs:
    kp_gt  : N x K x 3
    kp_pred: N x K x 2
    """
    if weights is not None:
        real_2d_kp[:,:,2] = weights[None,:]*real_2d_kp[:,:,2]
    kp_gt = real_2d_kp.view(-1, 3)
    kp_pred = predicted_2d_kp.contiguous().view(-1, 2)
    vis = kp_gt[:, 2]
    k = torch.sum(vis) * 2.0 + 1e-8

    dif_abs = torch.abs(kp_gt[:, :2] - kp_pred).sum(1)

    return torch.matmul(dif_abs, vis) * 1.0 / k

def landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt).cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).cuda()], dim=-1)

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks)
    return loss_lmk_2d * weight


def weighted_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    #smaller inner landmark weights
    # (predicted_theta, predicted_verts, predicted_landmarks) = ringnet_outputs[-1]
    # import ipdb; ipdb.set_trace()
    real_2d = landmarks_gt
    weights = torch.ones((68,)).cuda()
    weights[5:7] = 2
    weights[10:12] = 2
    # nose points
    weights[27:36] = 1.5
    weights[30] = 3
    weights[31] = 3
    weights[35] = 3

    # set mouth to zero
    weights[60:68] = 0
    weights[48:60] = 0
    weights[48] = 0
    weights[54] = 0


    # weights[36:48] = 0 # these are eyes

    loss_lmk_2d = batch_kp_2d_l1_loss(real_2d, predicted_landmarks, weights)
    return loss_lmk_2d * weight


def rel_dis(landmarks):

    lip_right = landmarks[:, [57, 51, 48, 60, 61, 62, 63], :]
    lip_left = landmarks[:, [8, 33, 54, 64, 67, 66, 65], :]

    # lip_right = landmarks[:, [61, 62, 63], :]
    # lip_left = landmarks[:, [67, 66, 65], :]

    dis = torch.sqrt(((lip_right - lip_left) ** 2).sum(2))  # [bz, 4]

    return dis

def relative_landmark_loss(predicted_landmarks, landmarks_gt, weight=1.):
    if torch.is_tensor(landmarks_gt) is not True:
        real_2d = torch.cat(landmarks_gt)#.cuda()
    else:
        real_2d = torch.cat([landmarks_gt, torch.ones((landmarks_gt.shape[0], 68, 1)).to(device=predicted_landmarks.device) #.cuda()
                             ], dim=-1)
    pred_lipd = rel_dis(predicted_landmarks[:, :, :2])
    gt_lipd = rel_dis(real_2d[:, :, :2])

    loss = (pred_lipd - gt_lipd).abs().mean()
    # loss = F.mse_loss(pred_lipd, gt_lipd)

    return loss.mean()

