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

import torch.nn as nn
from torchvision import models
from . import resnet



class ExpressionLossNet(nn.Module):
    """ Code borrowed from EMOCA https://github.com/radekd91/emoca """
    def __init__(self):
        super(ExpressionLossNet, self).__init__()

        self.backbone = resnet.load_ResNet50Model() #out: 2048

        self.linear = nn.Sequential(
            nn.Linear(2048, 10))

    def forward2(self, inputs):
        features = self.backbone(inputs)
        out = self.linear(features)
        return features, out

    def forward(self, inputs):
        features = self.backbone(inputs)
        return features
