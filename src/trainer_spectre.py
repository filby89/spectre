# -*- coding: utf-8 -*-
#

import os
import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pickle
from loguru import logger
from datetime import datetime
from tqdm import tqdm
import torchaudio
from .utils import util
torch.backends.cudnn.benchmark = True
from .utils import lossfunc
from .models.expression_loss import ExpressionLossNet
import torchvision.transforms.functional as F_v
import sys
sys.path.append("external/Visual_Speech_Recognition_for_Multiple_Languages")



class Trainer(object):
    def __init__(self, model, config=None, device='cuda:0'):
        self.cfg = config
        self.device = device


        # deca model
        self.spectre = model.to(self.device)

        self.global_step = 0

        logger.add(os.path.join(self.cfg.output_dir, self.cfg.train.log_dir, 'train.log'))
        if self.cfg.train.write_summary:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=os.path.join(self.cfg.output_dir, self.cfg.train.log_dir))


        self.prepare_training_losses()

    def prepare_training_losses(self):

        # ----- initialize resnet trained from EMOCA https://github.com/radekd91/emoca for expression loss ----- #
        self.expression_net = ExpressionLossNet().to(self.device)
        self.emotion_checkpoint = torch.load("data/ResNet50/checkpoints/deca-epoch=01-val_loss_total/dataloader_idx_0=1.27607644.ckpt")['state_dict']
        self.emotion_checkpoint['linear.0.weight'] = self.emotion_checkpoint['linear.weight']
        self.emotion_checkpoint['linear.0.bias'] = self.emotion_checkpoint['linear.bias']

        m, u = self.expression_net.load_state_dict(self.emotion_checkpoint, strict=False)
        self.expression_net.eval()

        # ----- initialize lipreader network for lipread loss ----- #
        from external.Visual_Speech_Recognition_for_Multiple_Languages.lipreading.model import Lipreading

        from external.Visual_Speech_Recognition_for_Multiple_Languages.dataloader.transform import Compose, Normalize, CenterCrop, SpeedRate, Identity
        from configparser import ConfigParser
        config = ConfigParser()

        config.read('configs/lipread_config.ini')
        self.lip_reader = Lipreading(
            config,
            device=self.device
        )

        """ this lipreader is used during evaluation to obtain an estimate of some lip reading metrics
        Note that the lipreader used for evaluation in the paper is different:

        https://github.com/facebookresearch/av_hubert/
        
        to obtain unbiased results
        """

        # ---- initialize values for cropping the face around the mouth for lipread loss ---- #
        # ---- this code is borrowed from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages ---- #
        self._crop_width = 96
        self._crop_height = 96
        self._window_margin = 12
        self._start_idx = 48
        self._stop_idx = 68
        crop_size = (88, 88)
        (mean, std) = (0.421, 0.165)

        # ---- transform mouths before going into the lipread network for loss ---- #
        self.mouth_transform = Compose([
            Normalize(0.0, 1.0),
            CenterCrop(crop_size),
            Normalize(mean, std),
            Identity()]
        )

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("E flame parameters: ", count_parameters(self.spectre.E_flame))
        print("flame parameters: ", count_parameters(self.spectre.flame))



    def step(self, batch, phase='train'):
        if phase!='train':
            self.spectre.eval()
        else:
            self.spectre.train()

        # [B, K, 3, size, size] ==> [BxK, 3, size, size]
        images = batch['image'].to(self.device)
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        lmk = batch['landmark'].to(self.device)
        lmk = lmk.view(-1, lmk.shape[-2], lmk.shape[-1])
        batch_size = images.shape[0]

        # ---- forward pass - encoder ---- #
        codedict, deca_exp, deca_jaw = self.spectre.encode(images)

        # ---- we calculate residual from DECA and add it to initial estimate of jaw and expression ---- #
        codedict['exp'] = codedict['exp'] + deca_exp
        codedict['pose'][...,3:] = codedict['pose'][...,3:] + deca_jaw

        rendering = True


        # ---- forward pass - decode using FLAME ---- #
        if phase == 'test':
            opdict, visdict = self.spectre.decode(codedict, rendering = rendering, vis_lmk=False, return_vis=True)
            opdict['shape_images'] = visdict['shape_images']
        else:
            opdict = self.spectre.decode(codedict, rendering = rendering, vis_lmk=False, return_vis=False)

        opdict['lmk'] = lmk


        # ---- calculate rendered images (shapes + texture) ---- #
        mask_face_eye = F.grid_sample(self.spectre.uv_face_eye_mask.expand(batch_size,-1,-1,-1), opdict['grid'].detach(), align_corners=False)
        predicted_images = opdict['rendered_images']*mask_face_eye*opdict['alpha_images']
        opdict['predicted_images'] = predicted_images


        # ---- start calculating losses ---- #

        losses = {}


        # ---- geometric losses ---- #
        predicted_landmarks = opdict['landmarks2d']


        # ---- we calculate losses only for [2:-2] indices of the sequence because of the temporal kernel of size 5 (2 left - 2 right) (we don't learn for padded frames) ---- #
        loss_indices = list(range(2,batch_size-2))

        # ---- various geometric losses - only landmark and relative_landmark are used in SPECTRE ---- #
        losses['landmark'] = lossfunc.weighted_landmark_loss(predicted_landmarks[loss_indices], lmk[loss_indices])
        losses['relative_landmark'] = lossfunc.relative_landmark_loss(predicted_landmarks[loss_indices], lmk[loss_indices])
        losses['lip_landmarks'] = F.mse_loss(predicted_landmarks[loss_indices,48:68,:2],lmk[loss_indices,48:68,:2])


        # ------------ photometric loss - not used in SPECTRE as well  ---------------- #
        masks = mask_face_eye * opdict['alpha_images']
        losses['photometric_texture'] = (masks*(predicted_images - images).abs()).mean()


        # ------------- emotion loss ------------------ #
        faces_gt = images
        faces_pred = opdict['rendered_images']

        self.expression_net.eval()

        emotion_features_pred = self.expression_net(faces_pred)

        with torch.no_grad():
            emotion_features_gt = self.expression_net(faces_gt)

        losses['expression'] = F.mse_loss(emotion_features_pred[loss_indices],emotion_features_gt[loss_indices])


        faces_gt = images
        faces_pred = opdict['rendered_images']
        opdict['faces_gt'] = faces_gt
        opdict['faces_pred'] = faces_pred


        """ lipread loss - first crop the mouths of the input and rendered faces
        and then calculate the cosine distance of features 
        """

        mouths_gt = self.cut_mouth(images, lmk[...,:2])
        mouths_pred = self.cut_mouth(opdict['rendered_images'], predicted_landmarks[...,:2])
        opdict['mouths_gt'] = mouths_gt
        opdict['mouths_pred'] = mouths_pred
        mouths_gt = self.mouth_transform(mouths_gt)
        mouths_pred = self.mouth_transform(mouths_pred)


        # ---- resize back to BxKx1xHxW (grayscale input for lipread net) ---- #
        mouths_gt = mouths_gt.view(-1,batch['image'].shape[1],mouths_gt.shape[-2], mouths_gt.shape[-1])
        mouths_pred = mouths_pred.view(-1,batch['image'].shape[1],mouths_gt.shape[-2], mouths_gt.shape[-1])



        self.lip_reader.eval()
        self.lip_reader.model.eval()

        lip_features_gt = self.lip_reader.model.encoder(
            mouths_gt,
            None,
            extract_resnet_feats=True
        )

        lip_features_pred = self.lip_reader.model.encoder(
            mouths_pred,
            None,
            extract_resnet_feats=True
        )

        lip_features_gt = lip_features_gt.view(-1, lip_features_gt.shape[-1])
        lip_features_pred = lip_features_pred.view(-1, lip_features_pred.shape[-1])
        # print(images.shape, lip_features_pred.shape)
        lr = (lip_features_gt*lip_features_pred).sum(1)/torch.linalg.norm(lip_features_pred,dim=1)/torch.linalg.norm(lip_features_gt,dim=1)

        losses['lipread'] = 1-torch.mean(lr[loss_indices])

        # nonlinear regularization of expression parameters
        reg = torch.sum((codedict['exp'][loss_indices] - deca_exp[loss_indices]) ** 2,dim=-1) / 2

        weight_vector = torch.ones_like(reg).cuda()
        weight_vector[reg > 40] = 2e-3
        weight_vector[reg < 40] = 1e-3

        losses['expression_reg'] = torch.mean(weight_vector * reg)

        loss_to_log = torch.mean(torch.sum(codedict['exp'][loss_indices] ** 2,dim=-1) / 2) # keep this for logging purposes
        losses['jaw_reg'] = torch.mean((codedict['pose'][loss_indices,3:]-deca_jaw[loss_indices])**2)/2

        # ---- now multiply each loss with the corresponding weight in config.py and add them ---- #

        all_loss = 0.
        losses_key = losses.keys()
        for key in losses_key:
            if key == 'expression_reg' and self.cfg.model.regularization_type=='nonlinear': # weight has been added for this specific case
                all_loss = all_loss + losses[key]
            else:
                all_loss = all_loss + losses[key] * self.cfg.loss.train[key]

        losses['all_loss'] = all_loss

        if self.cfg.model.regularization_type=='nonlinear':
            losses['expression_reg'] = loss_to_log


        return losses, opdict, codedict

    def cut_mouth(self, images, landmarks, convert_grayscale=True):
        """ function adapted from https://github.com/mpc001/Visual_Speech_Recognition_for_Multiple_Languages"""

        mouth_sequence = []

        landmarks = landmarks * 112 + 112
        for frame_idx,frame in enumerate(images):
            window_margin = min(self._window_margin // 2, frame_idx, len(landmarks) - 1 - frame_idx)
            smoothed_landmarks = landmarks[frame_idx-window_margin:frame_idx + window_margin + 1].mean(dim=0)
            smoothed_landmarks += landmarks[frame_idx].mean(dim=0) - smoothed_landmarks.mean(dim=0)

            center_x, center_y = torch.mean(smoothed_landmarks[self._start_idx:self._stop_idx], dim=0)

            center_x = center_x.round()
            center_y = center_y.round()

            height = self._crop_height//2
            width = self._crop_width//2

            threshold = 5

            if convert_grayscale:
                img = F_v.rgb_to_grayscale(frame).squeeze()
            else:
                img = frame

            if center_y - height < 0:
                center_y = height
            if center_y - height < 0 - threshold:
                raise Exception('too much bias in height')
            if center_x - width < 0:
                center_x = width
            if center_x - width < 0 - threshold:
                raise Exception('too much bias in width')

            if center_y + height > img.shape[-2]:
                center_y = img.shape[-2] - height
            if center_y + height > img.shape[-2] + threshold:
                raise Exception('too much bias in height')
            if center_x + width > img.shape[-1]:
                center_x = img.shape[-1] - width
            if center_x + width > img.shape[-1] + threshold:
                raise Exception('too much bias in width')

            mouth = img[...,int(center_y - height): int(center_y + height),
                                 int(center_x - width): int(center_x + round(width))]

            mouth_sequence.append(mouth)

        mouth_sequence = torch.stack(mouth_sequence,dim=0)
        return mouth_sequence

    def fit(self):
        self.prepare_data()
        start_epoch = 0
        self.global_step = 0

        # initialize outputs close to DECA result (since we find residual from coarse DECA estimate)
        self.spectre.E_expression.layers[0].weight.data *= 0.001
        self.spectre.E_expression.layers[0].bias.data *= 0.001

        self.opt = torch.optim.Adam(
                                self.spectre.E_expression.parameters(),
                                lr=self.cfg.train.lr)

        scheduler = torch.optim.lr_scheduler.MultiStepLR(self.opt,[50000],gamma=0.2)

        for epoch in range(start_epoch, self.cfg.train.max_epochs):
            self.epoch = epoch

            all_loss_mean = {}
            for i, batch in enumerate(tqdm(self.train_dataloader, total=len(self.train_dataloader))):
                if batch is None: continue

                losses, opdict, _ = self.step(batch)

                all_loss = losses['all_loss']

                for key in opdict.keys():
                    opdict[key] = opdict[key].cpu()
                all_loss.backward()

                self.opt.step()
                self.opt.zero_grad()
                # ---- we log the average train loss every 10 steps to obtain a smoother visual curve ---- #
                for key in losses.keys():
                    if key in all_loss_mean:
                        all_loss_mean[key] += losses[key].cpu().item()
                    else:
                        all_loss_mean[key] = losses[key].cpu().item()

                if self.global_step % self.cfg.train.log_steps == 0 and self.global_step > 0:
                    loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {epoch}, Global Iter: {self.global_step}, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                    for k, v in all_loss_mean.items():
                        v = v / self.cfg.train.log_steps
                        loss_info = loss_info + f'{k}: {v:.6f}, '
                        if self.cfg.train.write_summary:
                            self.writer.add_scalar('train_loss/'+k, v, global_step=self.global_step)
                    logger.info(loss_info)
                    all_loss_mean = {}

                # ---- visualize several stuff during training ---- #
                if self.global_step % self.cfg.train.vis_steps == 0 and self.global_step > 0:
                    visdict = self.create_grid(opdict)
                    savepath = os.path.join(self.cfg.output_dir, self.cfg.train.vis_dir, f'{self.global_step:06}.jpg')
                    util.visualize_grid(visdict, savepath, return_gird=True)

                if self.global_step>0 and self.global_step % self.cfg.train.checkpoint_steps == 0:
                    model_dict = self.spectre.model_dict()
                    model_dict['opt'] = self.opt.state_dict()
                    model_dict['global_step'] = self.global_step
                    model_dict['batch_size'] = self.cfg.dataset.batch_size
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'model' + '.tar'))
                    os.makedirs(os.path.join(self.cfg.output_dir, 'models'), exist_ok=True)
                    torch.save(model_dict, os.path.join(self.cfg.output_dir, 'models', f'{self.global_step:08}.tar'))

                # ---- take one random sample of validation and visualize it ---- #
                if self.global_step % self.cfg.train.vis_steps == 0 and self.global_step > 0:
                    for i, eval_batch in enumerate(tqdm(self.val_dataloader)):
                        if eval_batch is None: continue

                        with torch.no_grad():
                            losses, opdict, _ = self.step(eval_batch, phase='val')

                        visdict = self.create_grid(opdict)

                        savepath = os.path.join(self.cfg.output_dir, self.cfg.train.val_vis_dir, f'{self.global_step:08}.jpg')

                        util.visualize_grid(visdict, savepath, return_gird=True)
                        break

                # ---- evaluate the model on the test set every 10k iters ---- #
                if self.global_step % self.cfg.train.evaluation_steps == 0 and self.global_step > 0:
                    self.evaluate(self.test_datasets)

                self.global_step+=1

                scheduler.step()

        # evaluate one last time after training completes
        self.evaluate(self.test_datasets)

    def evaluate(self, datasets=None):
        save_dir = os.path.join(self.cfg.output_dir, 'test_videos_%06d'%self.global_step)
        os.makedirs(save_dir, exist_ok=True)
        print('Starting Evaluation ...')
        all_losses = {
            'wer': 0,
            'cer': 0,
            'vwer': 0,
            'ver': 0
        }

        count_frames = 0
        count_vids = 0


        for _, dataset in enumerate(datasets):
            indices = np.arange(0,len(dataset),1)

            with torch.no_grad():
                for idx in tqdm(indices):
                    batch = dataset[idx]

                    if batch is None:
                        continue

                    vid_name = batch['vid_name']
                    vid_name = "_".join(vid_name.split("/"))

                    os.makedirs(os.path.dirname(os.path.join(save_dir,vid_name)), exist_ok=True)
                    out_vid_path = os.path.join(save_dir,vid_name+".mp4")

                    vid_landmarks = []
                    all_images = []
                    all_shape_images = []
                    all_orig_images = []

                    """ SPECTRE uses a temporal convolution of size 5. 
                    Thus, in order to predict the parameters for a contiguous video with need to 
                    process the video in chunks of overlap 2, dropping values which were computed from the 
                    temporal kernel which uses pad 'same'. For the start and end of the video we drop 
                    the first two and last frames (look at the demo for a version with padding).
                    e.g., consider a video of size 48 frames and we want to predict it in chunks of 20 frames 
                    (due to memory limitations). 
                    We process independently the following chunks:
                    [[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
                     [16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35]
                     [32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47]]

                     In the first chunk, after computing the 3DMM params we drop 0,1 and 18,19, since they were computed 
                     from the temporal kernel with padding (we followed the same procedure in training and computed loss 
                     only from valid outputs of the temporal kernel) In the second chunk, we drop 16,17 and 34,35, and in 
                     the last chunk we drop 32,33 and 46,47. As a result we get results for frames:
                     [2..17], [18..33], [34..47] (end included).     
                    """
                    L = 50  # chunk size

                    images = batch['image']
                    landmarks = batch['landmark']

                    # we do this ugly workaround because mode replicate does not handle 4D inputs
                    # doing this padding results in slight under-evaluation of wer,cer, ver, vwer from our lipread
                    # however av_hubert is not affected
                    # images = F.pad(images,(0,0,0,0,0,0,2,2),mode='constant', value=0)
                    # images[0] = images[2]
                    # images[1] = images[2]
                    # images[-1] = images[-3]
                    # images[-2] = images[-3]
                    #
                    # landmarks = F.pad(landmarks,(0,0,0,0,2,2),mode='constant', value=0)
                    # landmarks[0] = landmarks[2]
                    # landmarks[1] = landmarks[2]
                    # landmarks[-1] = landmarks[-3]
                    # landmarks[-2] = landmarks[-3]

                    codedicts = []

                    indices = list(range(images.size(0)))

                    overlapping_indices = [indices[i: i + L] for i in range(0, len(indices), L-4)]

                    if len(overlapping_indices[-1]) < 5:
                        overlapping_indices[-2] = overlapping_indices[-2] + overlapping_indices[-1]
                        overlapping_indices[-2] = np.unique(overlapping_indices[-2]).tolist()
                        overlapping_indices = overlapping_indices[:-1]

                    sanity = [] # this is a sanity check that calculated indices are correct
                    for chunk_id in range(len(overlapping_indices)):

                        chunk_dict = {}
                        chunk_dict['image'] = images[overlapping_indices[chunk_id]].unsqueeze(0)
                        chunk_dict['landmark'] = landmarks[overlapping_indices[chunk_id]].unsqueeze(0)

                        losses, opdict, codedict = self.step(chunk_dict, phase='test')

                        """ filter out invalid indices in the dictionaries - see explanation at the top of the function """

                        for key in codedict.keys():
                            if chunk_id == 0 and chunk_id == len(overlapping_indices)-1:
                                pass
                            elif chunk_id == 0:
                                codedict[key] = codedict[key][:-2]
                            elif chunk_id == len(overlapping_indices)-1:
                                codedict[key] = codedict[key][2:]
                            else:
                                codedict[key] = codedict[key][2:-2]

                        for key in opdict.keys():
                            if chunk_id == 0 and chunk_id == len(overlapping_indices)-1:
                                pass
                            elif chunk_id == 0:
                                opdict[key] = opdict[key][:-2]
                            elif chunk_id == len(overlapping_indices)-1:
                                opdict[key] = opdict[key][2:]
                            else:
                                opdict[key] = opdict[key][2:-2]

                        for key in losses.keys():
                            if key not in all_losses:
                                all_losses[key] = 0

                            all_losses[key] += losses[key]*opdict['rendered_images'].shape[0]

                        count_frames += opdict['rendered_images'].shape[0]

                        # ---- accumulate images and landmarks for the video ---- #
                        all_images.append(opdict['rendered_images'])
                        all_shape_images.append(opdict['shape_images'])
                        vid_landmarks.append(opdict['landmarks2d'])

                        # ---- accumulate input images as well for visualization purposes ---- #
                        if chunk_id == 0 and chunk_id == len(overlapping_indices) - 1:
                            all_orig_images.append(chunk_dict['image'].squeeze(0))
                            sanity.append(overlapping_indices[chunk_id])
                        elif chunk_id == 0:
                            all_orig_images.append(chunk_dict['image'].squeeze(0)[:-2])
                            sanity.append(overlapping_indices[chunk_id][:-2])
                        elif chunk_id == len(overlapping_indices) - 1:
                            all_orig_images.append(chunk_dict['image'].squeeze(0)[2:])
                            sanity.append(overlapping_indices[chunk_id][2:])
                        else:
                            all_orig_images.append(chunk_dict['image'].squeeze(0)[2:-2])
                            sanity.append(overlapping_indices[chunk_id][2:-2])

                        codedict.pop('images', None)
                        codedict['verts'] = opdict['verts']
                        for key in codedict.keys():
                            codedict[key] = codedict[key].detach().cpu()
                        codedicts.append(codedict)

                    # --- stack all codedicts and save - uncomment if you want --- #
                    # codedict = {}
                    # for key in codedicts[0].keys():
                    #     codedict[key] = torch.cat([x[key] for x in codedicts], dim=0)
                    # torch.save(codedict, out_vid_path.replace(".mp4",".pth"))


                    # ---- some assertions that verify we decoded the video correctly ---- #
                    sanity = [x for xs in sanity for x in xs]
                    assert len(sanity) == images.size(0)
                    assert len(set(sanity)) == len(sanity)
                    assert max(sanity) == images.size(0)-1


                    # ---- stack landmarks and video images ---- #
                    vid = torch.cat(all_images, dim=0)[2:-2]; vid_rendered = util.tensor2video(vid) # video of rendered images (shape + texture)
                    vid_shape = util.tensor2video(torch.cat(all_shape_images, dim=0))[2:-2]  # video of shape - output of SPECTRE
                    vid_orig = util.tensor2video(torch.cat(all_orig_images, dim=0))[2:-2] # video of original images
                    vid_landmarks = torch.cat(vid_landmarks, dim=0)[2:-2] # landmarks of video

                    # assert vid_rendered.shape[0] == images.size(0)# - 4

                    grid_vid = np.concatenate((vid_shape, vid_orig), axis=2)

                    # ---- load wav file as well to put it in the output video ---- #
                    if 'wav_path' in batch:
                        wav, sr = torchaudio.load(batch['wav_path'])
                        wav = wav[:,1280:-1280]

                        # ---- save rendered, shape, and original videos removing pads---- #
                        torchvision.io.write_video(out_vid_path, vid_rendered, fps=self.cfg.dataset.fps, audio_codec='aac', audio_array=wav, audio_fps=16000)
                        torchvision.io.write_video(out_vid_path.replace(".mp4","_shape.mp4"), vid_shape, fps=self.cfg.dataset.fps, audio_codec='aac', audio_array=wav, audio_fps=16000)
                        # torchvision.io.write_video(out_vid_path.replace(".mp4","_orig.mp4"), vid_orig, fps=self.cfg.dataset.fps, audio_codec='aac', audio_array=wav, audio_fps=16000)
                        torchvision.io.write_video(out_vid_path.replace(".mp4","_grid.mp4"), grid_vid, fps=self.cfg.dataset.fps, audio_codec='aac', audio_array=wav, audio_fps=16000)
                    else:
                        torchvision.io.write_video(out_vid_path, vid_rendered, fps=self.cfg.dataset.fps)
                        torchvision.io.write_video(out_vid_path.replace(".mp4","_shape.mp4"), vid_shape, fps=self.cfg.dataset.fps)
                        # torchvision.io.write_video(out_vid_path.replace(".mp4","_orig.mp4"), vid_orig, fps=self.cfg.dataset.fps)
                        torchvision.io.write_video(out_vid_path.replace(".mp4","_grid.mp4"), grid_vid, fps=self.cfg.dataset.fps)


                    # you can uncomment the following and change ffmpeg paths if you want to create a grid to compare with other methods
                    # if enn == 0:
                    #     st = "ffmpeg -i {} -i {} -i {}  -filter_complex hstack=inputs=3 {} -y ".format(
                    #         os.path.join("/gpu-data3/filby/EAVTTS_experiments/audiovisual_DECA_results/cross/LRS3_test/DECA/", vid_name + "_shape.mp4"),
                    #         os.path.join("/gpu-data3/filby/EAVTTS_experiments/audiovisual_DECA_results/cross/LRS3_test/EMOCA/", vid_name + "_shape.mp4"),
                    #         os.path.join(out_vid_path.replace(".mp4", "_grid.mp4")),
                    #         os.path.join(out_vid_path.replace(".mp4", "_shape_grid.mp4"))
                    #     )

                    # ---- extract and save the mouth as well - useful for evaluation with av hubert afterwards ---- #
                    mouth_sequence = self.cut_mouth(vid, vid_landmarks, convert_grayscale=True)

                    if 'text' in batch:
                        from utils.lipread_utils import convert_text_to_visemes, save2avi, predict_text
                        import jiwer

                        with torch.no_grad():
                            mouth_sequence_transformed = self.mouth_transform(mouth_sequence)
                            predicted_text = predict_text(self.lip_reader, mouth_sequence_transformed.unsqueeze(0))

                        wer = jiwer.wer(batch['text'], predicted_text)
                        cer = jiwer.cer(batch['text'], predicted_text)

                        # ---------- convert to visemes -------- #
                        vg = convert_text_to_visemes(batch['text'])
                        v = convert_text_to_visemes(predicted_text)
                        # -------------------------------------- #
                        vwer = jiwer.wer(vg, v)
                        ver = jiwer.cer(vg, v)

                        print(f"hyp: {predicted_text}")
                        print(f"hypv: {v}")
                        print(f"ref: {batch['text']}")
                        print(f"refv: {vg}")


                        all_losses['wer'] += wer
                        all_losses['cer'] += cer
                        all_losses['vwer'] += vwer
                        all_losses['ver'] += ver

                        count_vids += 1

                        print(
                            "cur WER: {:.2f}\tcur CER: {:.2f}\t avg WER: {:.2f}\tavg CER: {:.2f}\n".format(100*wer, 100*cer, 100*all_losses['wer']/count_vids, 100*all_losses['cer']/count_vids)+
                            "cur VWER: {:.2f}\tcur VER: {:.2f}\t avg VWER: {:.2f}\tavg VER: {:.2f}".format(100*vwer, 100*ver, 100*all_losses['vwer']/count_vids, 100*all_losses['ver']/count_vids)
                        )

                    from utils.lipread_utils import save2avi
                    save2avi(out_vid_path.replace(".mp4", "_mouth.avi"), data=util.tensor2video(mouth_sequence,gray=True),
                             fps=self.cfg.dataset.fps)

                loss_info = f"ExpName: {self.cfg.exp_name} \nEpoch: {self.epoch}, Testing, Time: {datetime.now().strftime('%Y-%m-%d-%H:%M:%S')} \n"
                for k, v in all_losses.items():
                    if k in ['wer', 'cer', 'vwer', 'ver']:
                        loss_info += f"{k}: {100*v/count_vids:.2f}%\n"
                        self.writer.add_scalar('val_loss/' + k, 100*v/count_vids, global_step=self.global_step)
                    else:
                        loss_info = loss_info + f'{k}: {v/count_frames:.6f}, '
                        self.writer.add_scalar('val_loss/' + k, v/count_frames, global_step=self.global_step)
                logger.info(loss_info)

    def create_grid(self, opdict):
        # Visualize some stuff during training
        shape_images = self.spectre.render.render_shape(opdict['verts'].cuda(), opdict['trans_verts'].cuda())
        input_with_gt_landmarks = util.tensor_vis_landmarks(opdict['images'], opdict['lmk'], isScale=True)

        visdict = {
            'inputs': input_with_gt_landmarks,
            'mouths_gt': opdict['mouths_gt'].unsqueeze(1),
            'mouths_pred': opdict['mouths_pred'].unsqueeze(1),
            'faces_gt': opdict['faces_gt'],
            'faces_pred': opdict['faces_pred'],
            'shape_images': shape_images
        }

        return visdict


    def prepare_data(self):
        from datasets.datasets import get_datasets_LRS3
        self.train_dataset, self.val_dataset, self.test_dataset = get_datasets_LRS3(self.cfg.dataset)

        self.test_datasets = []
        if 'LRS3' in self.cfg.test_datasets:
            self.test_datasets.append(self.test_dataset)

        if 'TCDTIMIT' in self.cfg.test_datasets:
            from datasets.extra_datasets import get_datasets_TCDTIMIT
            _, _, test_dataset_TCDTIMIT = get_datasets_TCDTIMIT(self.cfg.dataset)
            self.test_datasets.append(test_dataset_TCDTIMIT)

        if 'MEAD' in self.cfg.test_datasets:
            from datasets.extra_datasets import get_datasets_MEAD
            _, _, test_dataset_MEAD = get_datasets_MEAD(self.cfg.dataset)
            self.test_datasets.append(test_dataset_MEAD)

        def collate_fn(batch):
            batch = list(filter(lambda x: x is not None, batch))
            if not batch:  # edge case
                return None
            return torch.utils.data.dataloader.default_collate(batch)

        logger.info('---- training data numbers: ', len(self.train_dataset), len(self.val_dataset))

        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=True,
                            num_workers=self.cfg.dataset.num_workers,
                            pin_memory=True,
                            drop_last=True, collate_fn=collate_fn)

        self.val_dataloader = DataLoader(self.val_dataset, batch_size=self.cfg.dataset.batch_size, shuffle=True,
                            num_workers=4,
                            pin_memory=True,
                            drop_last=False, collate_fn=collate_fn)

