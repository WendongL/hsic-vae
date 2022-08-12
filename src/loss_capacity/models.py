
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

class AbstractModel(nn.Module):
    def __init__(self, number_of_classes, number_of_channels=1):
        super().__init__()
        self.latent_dim = -1
        # self.predictor = nn.Linear(self.latent_dim, number_of_classes)

    def encode(self, x):
        flat = torch.flatten(x, 1)
        return flat

    def forward(self, x):
        return self.encode(x)

class ConvNet(nn.Module):
    def __init__(self, number_of_classes, number_of_channels=1, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(number_of_channels, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, self.latent_dim, 4, 1),  # B, 256,  1,  1
        )
        self.predictor = nn.Linear(self.latent_dim, number_of_classes)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x
    def forward(self, x):
        x = self.encode(x)
        # TODO: shoud we add ReLU.. seems that we should't since 
        # we want the latent encoding to actually be a disentangled representation 
        # x = F.relu(x)
        pred = self.predictor(x)
        return pred

import timm
class ResNet18(nn.Module):
    def __init__(self, number_of_classes, number_of_channels=1, latent_dim=256, pretrained=True):
        super().__init__()
        
        self.encoder = timm.create_model('resnet18', 
            pretrained=pretrained, 
            in_chans=number_of_channels,
            num_classes=0)# the last layer of resnet would be identity
        self.latent_dim = self.encoder.feature_info[-1]['num_chs']
        self.predictor = nn.Linear(self.latent_dim, number_of_classes)

    def encode(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return x
    def forward(self, x):
        x = self.encode(x)
        # TODO: shoud we add ReLU.. seems that we should't since 
        # we want the latent encoding to actually be a disentangled representation 
        # x = F.relu(x)
        pred = self.predictor(x)
        return pred

class RawData(nn.Module):
    def __init__(self, latent_dim=10):
        super().__init__()
        self.latent_dim = latent_dim
        # self.predictor = nn.Linear(self.latent_dim, number_of_classes)

    def encode(self, x):
        flat = torch.flatten(x, 1)
        return flat

    def forward(self, x):
        return self.encode(x)

# import rff
# class RandomFourierFeatures(nn.Module):
#     def __init__(self, latent_dim=10):
#         super().__init__()
#         self.latent_dim = latent_dim
#         self.rff_encoding = rff.layers.GaussianEncoding(sigma=10.0, input_size=2, encoded_size=256)
#         # self.predictor = nn.Linear(self.latent_dim, number_of_classes)

#     def encode(self, x):
#         x = torch.flatten(x, 1)
#         feats = self.rff_encoding(x)
#         return feats

#     def forward(self, x):
#         return self.encode(x)

class NoisyLabels(nn.Module):
    def __init__(self, number_of_classes, number_of_channels=1, noise_std=1.0):
        super().__init__()
        self.latent_dim = number_of_classes
        self.noise_std = noise_std

    def encode(self, labels):
        noise = torch.normal(mean=torch.zeros_like(labels), std=self.noise_std * torch.ones_like(labels))
        noisy_labels = labels + noise
        return noisy_labels

    def forward(self, x):
        noisy_labels = self.encode(x)
        return noisy_labels


class LinearMixLabels(nn.Module):
    def __init__(self, number_of_classes, number_of_channels=1, noise_std=1.0, mix_type='random_linear'):
        super().__init__()
        self.latent_dim = number_of_classes
        self.noise_std = noise_std

        self.mix_type = mix_type
        if mix_type == 'random_linear':
            self.mix_ = nn.Linear(number_of_classes, number_of_classes, bias=False)
            self.mix = self.mix_ 
        elif mix_type == 'almost_uniform_mix_labels':
            self.mix_matrix_ = self.get_mix_matrix()
            self.register_buffer('mix_matrix', self.mix_matrix_)
            self.mix = self.linear_mix

        # elif mix_type == 'almost_uniform_mix_labels_diag':
        #     self.mix_matrix_ = self.get_mix_matrix_diag()
        #     self.register_buffer('mix_matrix', self.mix_matrix_)
        #     self.mix = self.linear_mix

    # def init_mix(self):
    #     self.mix.bias.fill_(0.0)
    #     noise = torch.normal(mean=torch.zeros_like(self.mix.weight), 
    #         std=self.noise_std * torch.ones_like(self.mix.weight))
    #     self.mix.weight.fill_(torch.ones_like(self.mix.weight) + noise)

    def get_mix_matrix(self):
        noise = torch.normal(mean=torch.zeros(size=(self.latent_dim,self.latent_dim)), 
            std=self.noise_std * torch.ones(size=(self.latent_dim,self.latent_dim)))
        mix_matrix = 1.0 / self.latent_dim * torch.ones(size=(self.latent_dim,self.latent_dim)) + noise # + torch.diag(noise)
        return mix_matrix

    # def get_mix_matrix_diag(self):
    #     noise = torch.normal(mean=torch.zeros(self.latent_dim), 
    #         std=self.noise_std * torch.ones(self.latent_dim))
    #     mix_matrix = 1.0 / self.latent_dim * torch.ones(size=(self.latent_dim,self.latent_dim)) + torch.diag(noise)
    #     return mix_matrix

    def linear_mix(self, labels):
        mixed_labels = labels @ self.mix_matrix.T
        return mixed_labels

    def encode(self, labels):
        labels_mix = self.mix(labels)
        # labels_mix = labels_mix - labels_mix.mean(0, keepdims=True)
        # labels_mix = labels_mix / labels_mix.std(0, keepdims=True)

        return labels_mix

    def forward(self, x):
        return self.encode(x)