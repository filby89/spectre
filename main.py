import os, sys
import numpy as np
import yaml
import torch.backends.cudnn as cudnn
import torch
import shutil


def main(cfg):
    # creat folders
    os.makedirs(os.path.join(cfg.output_dir, cfg.train.log_dir), exist_ok=True)

    if cfg.test_mode is False:
        os.makedirs(os.path.join(cfg.output_dir, cfg.train.vis_dir), exist_ok=True)
        os.makedirs(os.path.join(cfg.output_dir, cfg.train.val_vis_dir), exist_ok=True)
        with open(os.path.join(cfg.output_dir, 'full_config.yaml'), 'w') as f:
            yaml.dump(cfg, f, default_flow_style=False)

        # cudnn related setting
        cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.enabled = True

    # start training
    from src.trainer_spectre import Trainer
    from src.spectre import SPECTRE
    spectre = SPECTRE(cfg)

    trainer = Trainer(model=spectre, config=cfg)

    if cfg.test_mode:
        trainer.prepare_data()
        trainer.evaluate(trainer.test_datasets)
    else:
        trainer.fit()

if __name__ == '__main__':
    from config import parse_args
    cfg = parse_args()
    cfg.exp_name = cfg.output_dir

    main(cfg)
