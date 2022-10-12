# -*- coding: utf-8 -*-
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms
# in the LICENSE file included with this software distribution.
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at deca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from .models.encoders import PerceptualEncoder
from .utils.renderer import SRenderY, set_rasterizer
from .models.encoders import ResnetEncoder
from .models.FLAME import FLAME, FLAMETex
from .utils import util
from .utils.tensor_cropper import transform_points
from skimage.io import imread
torch.backends.cudnn.benchmark = True
import numpy as np

class SPECTRE(nn.Module):
    def __init__(self, config=None, device='cuda'):
        super(SPECTRE, self).__init__()
        self.cfg = config
        self.device = device
        self.image_size = self.cfg.dataset.image_size
        self.uv_size = self.cfg.model.uv_size
        self._create_model(self.cfg.model)
        self._setup_renderer(self.cfg.model)


    def _setup_renderer(self, model_cfg):
        set_rasterizer(self.cfg.rasterizer_type)
        self.render = SRenderY(self.image_size, obj_filename=model_cfg.topology_path, uv_size=model_cfg.uv_size, rasterizer_type=self.cfg.rasterizer_type).to(self.device)
        # face mask for rendering details
        mask = imread(model_cfg.face_eye_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_eye_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        mask = imread(model_cfg.face_mask_path).astype(np.float32)/255.; mask = torch.from_numpy(mask[:,:,0])[None,None,:,:].contiguous()
        self.uv_face_mask = F.interpolate(mask, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # displacement correction
        fixed_dis = np.load(model_cfg.fixed_displacement_path)
        self.fixed_uv_dis = torch.tensor(fixed_dis).float().to(self.device)
        # mean texture
        mean_texture = imread(model_cfg.mean_tex_path).astype(np.float32)/255.; mean_texture = torch.from_numpy(mean_texture.transpose(2,0,1))[None,:,:,:].contiguous()
        self.mean_texture = F.interpolate(mean_texture, [model_cfg.uv_size, model_cfg.uv_size]).to(self.device)
        # dense mesh template, for save detail mesh
        self.dense_template = np.load(model_cfg.dense_template_path, allow_pickle=True, encoding='latin1').item()

    def _create_model(self, model_cfg):
        # set up parameters
        self.n_param = model_cfg.n_shape + model_cfg.n_tex + model_cfg.n_exp + model_cfg.n_pose + model_cfg.n_cam + model_cfg.n_light
        self.n_cond = model_cfg.n_exp + 3  # exp + jaw pose
        self.num_list = [model_cfg.n_shape, model_cfg.n_tex, model_cfg.n_exp, model_cfg.n_pose, model_cfg.n_cam,
                         model_cfg.n_light]
        self.param_dict = {i: model_cfg.get('n_' + i) for i in model_cfg.param_list}

        # encoders
        self.E_flame = ResnetEncoder(outsize=self.n_param).to(self.device)

        self.E_expression = PerceptualEncoder(model_cfg.n_exp, model_cfg).to(self.device)

        # decoders
        self.flame = FLAME(model_cfg).to(self.device)
        if model_cfg.use_tex:
            self.flametex = FLAMETex(model_cfg).to(self.device)

        # resume model
        model_path = self.cfg.pretrained_modelpath
        if os.path.exists(model_path):
            # print(f'trained model found. load {model_path}')
            checkpoint = torch.load(model_path)

            if 'state_dict' in checkpoint.keys():
                self.checkpoint = checkpoint['state_dict']
            else:
                self.checkpoint = checkpoint

            processed_checkpoint = {}
            processed_checkpoint["E_flame"] = {}
            processed_checkpoint["E_expression"] = {}
            if 'deca' in list(self.checkpoint.keys())[0]:
                for key in self.checkpoint.keys():
                    # print(key)
                    k = key.replace("deca.","")
                    if "E_flame" in key:
                        processed_checkpoint["E_flame"][k.replace("E_flame.","")] = self.checkpoint[key]#.replace("E_flame","")
                    elif "E_expression" in key:
                        processed_checkpoint["E_expression"][k.replace("E_expression.","")] = self.checkpoint[key]#.replace("E_flame","")
                    else:
                        pass

            else:
                processed_checkpoint = self.checkpoint


            self.E_flame.load_state_dict(processed_checkpoint['E_flame'], strict=True)
            try:
                m,u = self.E_expression.load_state_dict(processed_checkpoint['E_expression'], strict=True)
                # print('Missing keys', m)
                # print('Unexpected keys', u)
                # pass
            except Exception as e:
                print(f'Missing keys {e} in expression encoder weights. If starting training from scratch this is normal.')
        else:
            raise(f'please check model path: {model_path}')

        # eval mode
        self.E_flame.eval()

        self.E_expression.eval()

        self.E_flame.requires_grad_(False)


    def decompose_code(self, code, num_dict):
        ''' Convert a flattened parameter vector to a dictionary of parameters
        code_dict.keys() = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
        '''
        code_dict = {}
        start = 0

        for key in num_dict:
            end = start + int(num_dict[key])
            code_dict[key] = code[..., start:end]
            start = end
            if key == 'light':
                dims_ = code_dict[key].ndim -1 # (to be able to handle batches of videos)
                code_dict[key] = code_dict[key].reshape(*code_dict[key].shape[:dims_], 9, 3)
        return code_dict

    def encode(self, images):
        with torch.no_grad():
            parameters = self.E_flame(images)

        codedict = self.decompose_code(parameters, self.param_dict)
        deca_exp = codedict['exp'].clone()
        deca_jaw = codedict['pose'][...,3:].clone()

        codedict['images'] = images

        codedict['exp'], jaw = self.E_expression(images)
        codedict['pose'][..., 3:] = jaw

        return codedict, deca_exp, deca_jaw


    def decode(self, codedict, rendering=True, vis_lmk=True, return_vis=True,
               render_orig=False, original_image=None, tform=None):
        images = codedict['images']

        is_video_batch = images.ndim == 5
        if is_video_batch:
            B, T, C, H, W = images.shape
            images = images.view(B*T, C, H, W)
            codedict_ = codedict 
            codedict = {}
            for key in codedict_.keys():
                # if key != 'images':
                codedict[key] = codedict_[key].view(B*T, *codedict_[key].shape[2:])


        batch_size = images.shape[0]

        ## decode
        verts, landmarks2d, landmarks3d = self.flame(shape_params=codedict['shape'], expression_params=codedict['exp'],
                                                     pose_params=codedict['pose'])
        if self.cfg.model.use_tex:
            albedo = self.flametex(codedict['tex']).detach()
        else:
            albedo = torch.zeros([batch_size, 3, self.uv_size, self.uv_size], device=images.device)
        landmarks3d_world = landmarks3d.clone()

        ## projection
        landmarks2d = util.batch_orth_proj(landmarks2d, codedict['cam'])[:, :, :2];
        landmarks2d[:, :, 1:] = -landmarks2d[:, :,
                                 1:]
        landmarks3d = util.batch_orth_proj(landmarks3d, codedict['cam']);
        landmarks3d[:, :, 1:] = -landmarks3d[:, :,
                                 1:]
        trans_verts = util.batch_orth_proj(verts, codedict['cam']);
        trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        opdict = {
            'verts': verts,
            'trans_verts': trans_verts,
            'landmarks2d': landmarks2d,
            'landmarks3d': landmarks3d,
            'landmarks3d_world': landmarks3d_world,
        }

        if rendering and render_orig and original_image is not None and tform is not None:
            points_scale = [self.image_size, self.image_size]
            _, _, h, w = original_image.shape
            trans_verts = transform_points(trans_verts, tform, points_scale, [h, w])
            landmarks2d = transform_points(landmarks2d, tform, points_scale, [h, w])
            landmarks3d = transform_points(landmarks3d, tform, points_scale, [h, w])
            background = images
        else:
            h, w = self.image_size, self.image_size
            background = None


        if rendering:
            if self.cfg.model.use_tex:
                ops = self.render(verts, trans_verts, albedo, codedict['light'])
                ## output
                opdict['predicted_inner_mouth'] = ops['predicted_inner_mouth']
                opdict['grid'] = ops['grid']
                opdict['rendered_images'] = ops['images']
                opdict['alpha_images'] = ops['alpha_images']
                opdict['normal_images'] = ops['normal_images']
                opdict['images'] = images

            else:
                shape_images, _, grid, alpha_images, pos_mask = self.render.render_shape(verts, trans_verts, h=h, w=w,
                                                                                         images=background,
                                                                                         return_grid=True,
                                                                                         return_pos=True)

                opdict['rendered_images'] = shape_images

        if self.cfg.model.use_tex:
            opdict['albedo'] = albedo

        if vis_lmk:
            landmarks3d_vis = self.visofp(ops['transformed_normals'])  # /self.image_size
            landmarks3d = torch.cat([landmarks3d, landmarks3d_vis], dim=2)
            opdict['landmarks3d'] = landmarks3d

        if is_video_batch:
            for key in opdict.keys():
                opdict[key] = opdict[key].view(B, T, *opdict[key].shape[1:])

        if return_vis:
            ## render shape
            shape_images, _, grid, alpha_images, pos_mask = self.render.render_shape(verts, trans_verts, h=h, w=w,
                                                                           images=background, return_grid=True, return_pos=True)

            # opdict['uv_texture_gt'] = uv_texture_gt
            visdict = {
                # 'inputs': images,
                'landmarks2d': util.tensor_vis_landmarks(images, landmarks2d),
                'landmarks3d': util.tensor_vis_landmarks(images, landmarks3d),
                'shape_images': shape_images,
                # 'rendered_images': ops['images']
            }

            if is_video_batch:
                for key in visdict.keys():
                    visdict[key] = visdict[key].view(B, T, *visdict[key].shape[1:])

            return opdict, visdict

        else:
            return opdict

    def train(self):
        self.E_expression.train()

        self.E_flame.eval()


    def eval(self):
        self.E_expression.eval()
        self.E_flame.eval()


    def model_dict(self):
        return {
            'E_flame': self.E_flame.state_dict(),
            'E_expression': self.E_expression.state_dict(),
        }
