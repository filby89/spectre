'''
Default config for SPECTRE - adapted from DECA
'''
from yacs.config import CfgNode as CN
import argparse
import yaml
import os

cfg = CN()

cfg.project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src', '..'))
cfg.device = 'cuda'
cfg.device_ids = '0'

cfg.pretrained_modelpath = os.path.join(cfg.project_dir, 'data', 'deca_model.tar')
cfg.output_dir = ''
cfg.rasterizer_type = 'pytorch3d'
# ---------------------------------------------------------------------------- #
# Options for FLAME and from original DECA
# ---------------------------------------------------------------------------- #
cfg.model = CN()
cfg.model.topology_path = os.path.join(cfg.project_dir, 'data' , 'head_template.obj')
# texture data original from http://files.is.tue.mpg.de/tbolkart/FLAME/FLAME_texture_data.zip
cfg.model.dense_template_path = os.path.join(cfg.project_dir, 'data', 'texture_data_256.npy')
cfg.model.fixed_displacement_path = os.path.join(cfg.project_dir, 'data', 'fixed_displacement_256.npy')
cfg.model.flame_model_path = os.path.join(cfg.project_dir, 'data', 'FLAME2020', 'generic_model.pkl')
cfg.model.flame_lmk_embedding_path = os.path.join(cfg.project_dir, 'data', 'landmark_embedding.npy')
cfg.model.face_mask_path = os.path.join(cfg.project_dir, 'data', 'uv_face_mask.png')
cfg.model.face_eye_mask_path = os.path.join(cfg.project_dir, 'data', 'uv_face_eye_mask.png')
cfg.model.mean_tex_path = os.path.join(cfg.project_dir, 'data', 'mean_texture.jpg')
cfg.model.tex_path = os.path.join(cfg.project_dir, 'data', 'FLAME_albedo_from_BFM.npz')
cfg.model.tex_type = 'BFM' # BFM, FLAME, albedoMM
cfg.model.uv_size = 256
cfg.model.param_list = ['shape', 'tex', 'exp', 'pose', 'cam', 'light']
cfg.model.n_shape = 100
cfg.model.n_tex = 50
cfg.model.n_exp = 50
cfg.model.n_cam = 3
cfg.model.n_pose = 6
cfg.model.n_light = 27
cfg.model.jaw_type = 'aa' # default use axis angle, another option: euler. Note that: aa is not stable in the beginning



cfg.model.model_type = "SPECTRE"

cfg.model.temporal = True


# ---------------------------------------------------------------------------- #
# Options for Dataset
# ---------------------------------------------------------------------------- #
cfg.dataset = CN()
cfg.dataset.LRS3_path = "/gpu-data3/filby/LRS3"
cfg.dataset.LRS3_landmarks_path = "../Visual_Speech_Recognition_for_Multiple_Languages/landmarks/LRS3/LRS3_landmarks"

cfg.dataset.LRS3_path = "/gpu-data3/filby/LRS3"
cfg.dataset.LRS3_landmarks_path = "../Visual_Speech_Recognition_for_Multiple_Languages/landmarks/LRS3/LRS3_landmarks"

cfg.dataset.LRS3_path = "/gpu-data3/filby/LRS3"
cfg.dataset.LRS3_landmarks_path = "../Visual_Speech_Recognition_for_Multiple_Languages/landmarks/LRS3/LRS3_landmarks"

cfg.dataset.batch_size = 1
cfg.dataset.K = 20
cfg.dataset.num_workers = 8
cfg.dataset.image_size = 224
cfg.dataset.scale_min = 1.4
cfg.dataset.scale_max = 1.8
cfg.dataset.trans_scale = 0.
cfg.dataset.fps = 25
cfg.dataset.test_datasets = ['LRS3']

# ---------------------------------------------------------------------------- #
# Options for training
# ---------------------------------------------------------------------------- #
cfg.train = CN()
cfg.train.max_epochs = 6
cfg.train.log_dir = 'logs'
cfg.train.log_steps = 10
cfg.train.vis_dir = 'train_images'
cfg.train.vis_steps = 500
cfg.train.write_summary = True
cfg.train.checkpoint_steps = 10000
cfg.train.val_vis_dir = 'val_images'

cfg.train.evaluation_steps = 10000

# ---------------------------------------------------------------------------- #
# Options for Losses
# ---------------------------------------------------------------------------- #
cfg.loss = CN()
cfg.loss.train = CN()

cfg.model.use_tex = True
cfg.model.regularization_type = 'nonlinear'
cfg.model.backbone = 'mobilenetv2' # perceptual encoder backbone

cfg.loss.train.landmark = 50
cfg.loss.train.lip_landmarks = 0
cfg.loss.train.relative_landmark = 50# 50
cfg.loss.train.photometric_texture = 0
cfg.loss.train.lipread = 2
cfg.loss.train.jaw_reg = 200
cfg.train.lr = 5e-5
cfg.loss.train.expression = 0.5

cfg.test_mode = False

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return cfg.clone()

def update_cfg(cfg, cfg_file):
    cfg.merge_from_file(cfg_file)
    return cfg.clone()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, help='output path')
    parser.add_argument('--LRS3_path', default=None, type=str, help='path to LRS3 dataset')
    parser.add_argument('--LRS3_landmarks_path', default=None, type=str, help='path to LRS3 landmarks')
    parser.add_argument('--model_path', default=None, help='path to pretrained model')
    parser.add_argument('--batch-size', type=int, default=1, help='the batch size')
    parser.add_argument('--epochs', type=int, default=6, help='number of epochs to train for')
    parser.add_argument('--K', type=int, default=20, help='length of sampled frame sequence')
    parser.add_argument('--lipread', type=float, default=None, help='lipread loss weight')
    parser.add_argument('--expression', type=float, default=None, help='expression loss weight')
    parser.add_argument('--lr', type=float, default=None, help='learning rate')
    parser.add_argument('--landmark', type=float, default=None, help='landmark loss weight')
    parser.add_argument('--relative_landmark', type=float, default=None, help='relative landmark loss weight')
    parser.add_argument('--backbone', type=str, default='mobilenetv2', choices=['mobilenetv2', 'resnet50'])

    parser.add_argument('--test', action='store_true', help='test mode')
    parser.add_argument('--test_datasets', type=str, nargs='+', default=['LRS3'], help='test datasets')

    args = parser.parse_args()

    cfg = get_cfg_defaults()

    cfg.output_dir = args.output_dir

    if args.model_path is not None:
        cfg.pretrained_modelpath = args.model_path

    if args.batch_size is not None:
        cfg.dataset.batch_size = args.batch_size

    cfg.dataset.K = args.K

    if args.landmark is not None:
        cfg.loss.train.landmark = args.landmark

    if args.relative_landmark is not None:
        cfg.loss.train.relative_landmark = args.relative_landmark

    if args.lipread is not None:
        cfg.loss.train.lipread = args.lipread

    if args.expression is not None:
        cfg.loss.train.expression = args.expression

    if args.lr is not None:
        cfg.train.lr = args.lr

    if args.epochs is not None:
        cfg.train.max_epochs = args.epochs

    if args.LRS3_path is not None:
        cfg.dataset.LRS3_path = args.LRS3_path

    if args.LRS3_landmarks_path is not None:
        cfg.dataset.LRS3_landmarks_path = args.LRS3_landmarks_path

    cfg.model.backbone = args.backbone

    cfg.test_mode = args.test

    cfg.test_datasets = args.test_datasets

    return cfg
