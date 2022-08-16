from itertools import zip_longest
import torch
import matplotlib.pyplot as plt
from glob import glob
import os
from loss_capacity.utils import from_pickle
import sys
import numpy as np
from matplotlib.pyplot import cm
import torchvision.utils as vutils
import sys
sys.path.insert(0, './../')
import yaml
import argparse
import numpy as np
from pathlib import Path
from PyTorchVAE.models.beta_vae import BetaVAE, SmallBetaVAE
from PyTorchVAE.experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from PyTorchVAE.dataset import  DisentDatasets
from pytorch_lightning.plugins import DDPPlugin
from liftoff import parse_opts
from pathlib import Path

import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
# models
from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
# from loss_capacity.train_model import train_test_model
from loss_capacity.utils import list2tuple_, config_to_string, save_representation_dataset, get_representations_data_split
# from loss_capacity.probing import Probe
from loss_capacity.functions import HSIC

import timm

import torch
import pickle
import json
import pdb


# folder = 'results/2022Aug11-170838_unsup_vae_dsprites/0000_vae.trainer_params.max_epochs_10__vae.exp_params.LR_0.002__vae.exp_params.scheduler_gamma_0.9999__vae.exp_params.kld_weight_0.1__vae.model_params.model_type_big__vae.model_params.recons_type_bce__dataset_dsprites/'

def plot_reconstructed(experiment, path, r0=(-5, 10), r1=(-10, 5), n=12):
    # n: number of images per row/column
    
    latent_dim = experiment.model.latent_dim
    x = torch.linspace(*r0,n)
    y = torch.linspace(*r1,n)
    xx, yy = torch.meshgrid(x,y, indexing='xy')
    xx = torch.unsqueeze(xx, 2)
    yy = torch.unsqueeze(yy, 2)
    z = torch.cat((xx,yy), 2)
    t = torch.zeros(n, n, latent_dim)
    t[:,:,0:2] = z
    x_hat = experiment.model.decode(t)
    vutils.save_image(x_hat.data,
                          os.path.join(path , 
                                       'interpolate_'+ 
                                       f"Epoch_{experiment.current_epoch}.png"),
                          normalize=True,
                          nrow=n)

def run(params):
    params = list2tuple_(params)
    params.num_workers = 2
    print(config_to_string(params))

    config = params.vae
    if 'none' in  params.probe.max_leaf_nodes or 'None'  in  params.probe.max_leaf_nodes:
        params.probe.max_leaf_nodes = None
    print(f'out dir: {params.out_dir}')
        
    tb_logger =  TensorBoardLogger(save_dir=params.out_dir,
                                name=config.model_params.name)

    # For reproducibility
    seed_everything(config.exp_params.manual_seed * params.run_id, True)

    # TODO: maybe do smt more generic
    # model = vae_models[config.model_params.name](**configmodel_params)


    device = 'cuda'

    path = params.result_path # './results/2022Aug12-134358_unsup_vae_dsprites_bottle_kl100'
    
    # if restore:
    #     ckpt = params.out_dir + '/BetaVAE/version_0/checkpoints/last.ckpt'
    #     print(f'loading from: {ckpt}')
    #     lightning_ckpt = torch.load(ckpt)
    #     experiment.load_state_dict(lightning_ckpt['state_dict'])


    # number_of_channels = 1 if params.dataset == 'dsprites' else 3
    # print(f'number_of_channels: {number_of_channels}')
    imagenet_normalise = True if 'resnet' in params.model_type else False
    for folder in glob(path+"/*/", recursive = True):
        experiment = VAEXperiment.load_from_checkpoint(folder+"0/BetaVAE/version_0/checkpoints/last.ckpt")
        plot_reconstructed(experiment, path = folder+"0/BetaVAE/", r0=(-5, 10), r1=(-10, 5), n=12)

if __name__ == "__main__":
    run(parse_opts())
