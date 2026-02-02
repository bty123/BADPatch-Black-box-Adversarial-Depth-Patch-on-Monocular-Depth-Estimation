# Copyright Niantic 2019. Patent Pending. All rights reserved.
#
# This software is licensed under the terms of the Monodepth2 licence
# which allows for non-commercial use only, the full terms of which are made
# available in the LICENSE file.

from __future__ import absolute_import, division, print_function

import numpy as np

import torch
import torch.nn as nn
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F


class ResNetMultiImageInput(models.ResNet):
    """Constructs a resnet model with varying number of input images.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """
    def __init__(self, block, layers, num_classes=1000, num_input_images=1):
        super(ResNetMultiImageInput, self).__init__(block, layers)
        self.inplanes = 64
        self.conv1 = nn.Conv2d(
            num_input_images * 3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def resnet_multiimage_input(num_layers, pretrained=False, num_input_images=1):
    """Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_input_images (int): Number of frames stacked as input
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock, 50: models.resnet.Bottleneck}[num_layers]
    model = ResNetMultiImageInput(block_type, blocks, num_input_images=num_input_images)

    if pretrained:
        loaded = model_zoo.load_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        loaded['conv1.weight'] = torch.cat(
            [loaded['conv1.weight']] * num_input_images, 1) / num_input_images
        model.load_state_dict(loaded)
    return model


class ResnetEncoder(nn.Module):
    """Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained, num_input_images=1):
        super(ResnetEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        if num_input_images > 1:
            self.encoder = resnet_multiimage_input(num_layers, pretrained, num_input_images)
        else:
            self.encoder = resnets[num_layers](pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = (input_image - 0.45) / 0.225
        x = self.encoder.conv1(x)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

class BackprojectDepth(nn.Module):
    """Layer to transform a depth image into a point cloud
    """
    def __init__(self, batch_size, height, width):
        super(BackprojectDepth, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords),
                                      requires_grad=False)

        self.ones = nn.Parameter(torch.ones(self.batch_size, 1, self.height * self.width),
                                 requires_grad=False)

        self.pix_coords = torch.unsqueeze(torch.stack(
            [self.id_coords[0].view(-1), self.id_coords[1].view(-1)], 0), 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1)
        self.pix_coords = nn.Parameter(torch.cat([self.pix_coords, self.ones], 1),
                                       requires_grad=False)

    def forward(self, depth, inv_K):
        cam_points = torch.matmul(inv_K[:, :3, :3], self.pix_coords)
        cam_points = depth.view(self.batch_size, 1, -1) * cam_points
        cam_points = torch.cat([cam_points, self.ones], 1)

        return cam_points


class Project3D(nn.Module):
    """Layer which projects 3D points into a camera with intrinsics K and at position T
    """
    def __init__(self, batch_size, height, width, eps=1e-7):
        super(Project3D, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width
        self.eps = eps

    def forward(self, points, K, T):
        P = torch.matmul(K, T)[:, :3, :]

        cam_points = torch.matmul(P, points)

        pix_coords = cam_points[:, :2, :] / (cam_points[:, 2, :].unsqueeze(1) + self.eps)
        pix_coords = pix_coords.view(self.batch_size, 2, self.height, self.width)
        pix_coords = pix_coords.permute(0, 2, 3, 1)
        pix_coords[..., 0] /= self.width - 1
        pix_coords[..., 1] /= self.height - 1
        pix_coords = (pix_coords - 0.5) * 2
        return pix_coords

class ResnetEncoderMatching(nn.Module):
    """Resnet encoder adapted to include a cost volume after the 2nd block.

    Setting adaptive_bins=True will recompute the depth bins used for matching upon each
    forward pass - this is required for training from monocular video as there is an unknown scale.
    """

    def __init__(self, num_layers, pretrained, input_height, input_width,
                 min_depth_bin=0.1, max_depth_bin=20.0, num_depth_bins=96,
                 adaptive_bins=False, depth_binning='linear'):

        super(ResnetEncoderMatching, self).__init__()

        self.adaptive_bins = adaptive_bins
        self.depth_binning = depth_binning
        self.set_missing_to_max = True

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])
        self.num_depth_bins = num_depth_bins
        # we build the cost volume at 1/4 resolution
        self.matching_height, self.matching_width = input_height // 4, input_width // 4

        self.is_cuda = False
        self.warp_depths = None
        self.depth_bins = None

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError("{} is not a valid number of resnet layers".format(num_layers))

        encoder = resnets[num_layers](pretrained)
        self.layer0 = nn.Sequential(encoder.conv1,  encoder.bn1, encoder.relu)
        self.layer1 = nn.Sequential(encoder.maxpool,  encoder.layer1)
        self.layer2 = encoder.layer2
        self.layer3 = encoder.layer3
        self.layer4 = encoder.layer4

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

        self.backprojector = BackprojectDepth(batch_size=self.num_depth_bins,
                                              height=self.matching_height,
                                              width=self.matching_width)
        self.projector = Project3D(batch_size=self.num_depth_bins,
                                   height=self.matching_height,
                                   width=self.matching_width)

        self.compute_depth_bins(min_depth_bin, max_depth_bin)

        self.prematching_conv = nn.Sequential(nn.Conv2d(64, out_channels=16,
                                                        kernel_size=1, stride=1, padding=0),
                                              nn.ReLU(inplace=True)
                                              )

        self.reduce_conv = nn.Sequential(nn.Conv2d(self.num_ch_enc[1] + self.num_depth_bins,
                                                   out_channels=self.num_ch_enc[1],
                                                   kernel_size=3, stride=1, padding=1),
                                         nn.ReLU(inplace=True)
                                         )


class UpSampleBN(nn.Module):
    def __init__(self, skip_input, output_features):
        super(UpSampleBN, self).__init__()

        self._net = nn.Sequential(nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU(),
                                  nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1),
                                  nn.BatchNorm2d(output_features),
                                  nn.LeakyReLU())

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        f = torch.cat([up_x, concat_with], dim=1)
        return self._net(f)

class DecoderBN(nn.Module):
    def __init__(self, num_features=2048, num_classes=1, bottleneck_features=512): 
        super(DecoderBN, self).__init__()
        features = int(num_features)

        self.conv2 = nn.Conv2d(bottleneck_features, features, kernel_size=1, stride=1, padding=1)
        # for res50
        self.up1 = UpSampleBN(skip_input=features // 1 + 1024, output_features=features // 2)
        self.up2 = UpSampleBN(skip_input=features // 2 + 512, output_features=features // 4)
        self.up3 = UpSampleBN(skip_input=features // 4 + 256, output_features=features // 8)
        self.up4 = UpSampleBN(skip_input=features // 8 + 64, output_features=features // 16)

        # self.up5 = UpSampleBN(skip_input=features // 16 + 3, output_features=features//16)
        self.conv3 = nn.Conv2d(features // 16, num_classes, kernel_size=3, stride=1, padding=1)
        # self.act_out = nn.Softmax(dim=1) if output_activation == 'softmax' else nn.Identity()

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[0], features[1], features[2], features[3], features[4]
        x_d0 = self.conv2(x_block4)

        x_d1 = self.up1(x_d0, x_block3)
        x_d2 = self.up2(x_d1, x_block2)
        x_d3 = self.up3(x_d2, x_block1)
        x_d4 = self.up4(x_d3, x_block0)
        # x_d5 = self.up5(x_d4, features[0])
        # out = self.conv3(x_d5)
        out = self.conv3(x_d4)
        return out


class ResnetEncoderDecoder(nn.Module):
    def __init__(self, num_layers=50, num_features=512, model_dim=32):
        super(ResnetEncoderDecoder, self).__init__()
        self.encoder = ResnetEncoder(num_layers = num_layers, pretrained = True, num_input_images=1)
        self.decoder = DecoderBN(num_features=num_features, num_classes=model_dim, bottleneck_features=2048)
    def forward(self, x, **kwargs):
        x = self.encoder(x)
        return self.decoder(x, **kwargs)