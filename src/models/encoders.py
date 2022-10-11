# -*- coding: utf-8 -*-
import torch.nn as nn
import torch
import torch.nn.functional as F
from . import resnet


class PerceptualEncoder(nn.Module):
    def __init__(self, outsize, cfg):
        super(PerceptualEncoder, self).__init__()
        if cfg.backbone == "mobilenetv2":
            self.encoder = torch.hub.load('pytorch/vision:v0.8.1', 'mobilenet_v2', pretrained=True)
            feature_size = 1280
        elif cfg.backbone == "resnet50":
            self.encoder = resnet.load_ResNet50Model() #out: 2048
            feature_size = 2048

        ### regressor
        self.temporal = nn.Sequential(
            nn.Conv1d(in_channels=feature_size, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU()
        )

        self.layers = nn.Sequential(
            nn.Linear(256, 53),
        )

        self.backbone = cfg.backbone

    def forward(self, inputs):
        is_video_batch = inputs.ndim == 5

        if self.backbone == 'resnet50':
            features = self.encoder(inputs).squeeze(-1).squeeze(-1)
        else:
            inputs_ = inputs
            if is_video_batch:
                B, T, C, H, W = inputs.shape
                inputs_ = inputs.view(B * T, C, H, W)
            features = self.encoder.features(inputs_)
            features = nn.functional.adaptive_avg_pool2d(features, (1, 1)).squeeze(-1).squeeze(-1)
            if is_video_batch:
                features = features.view(B, T, -1)

        features = features
        if is_video_batch:
            features = features.permute(0, 2, 1)
        else:
            features = features.permute(1,0).unsqueeze(0)

        features = self.temporal(features)

        if is_video_batch:
            features = features.permute(0, 2, 1)
        else:
            features = features.squeeze(0).permute(1,0)

        parameters = self.layers(features)

        parameters[...,50] = F.relu(parameters[...,50]) # jaw x is highly improbably negative and can introduce artifacts

        return parameters[...,:50], parameters[...,50:]


class ResnetEncoder(nn.Module):
    def __init__(self, outsize):
        super(ResnetEncoder, self).__init__()

        feature_size = 2048

        self.encoder = resnet.load_ResNet50Model() #out: 2048
        ### regressor
        self.layers = nn.Sequential(
            nn.Linear(feature_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, outsize)
        )

    def forward(self, inputs):
        inputs_ = inputs
        if inputs.ndim == 5: # batch of videos
            B, T, C, H, W = inputs.shape
            inputs_ = inputs.view(B * T, C, H, W)
        features = self.encoder(inputs_)
        parameters = self.layers(features)
        if inputs.ndim == 5: # batch of videos
            parameters = parameters.view(B, T, -1)
        return parameters

