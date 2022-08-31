from itertools import zip_longest
import torch
import matplotlib.pyplot as plt
from glob import glob
import os
from loss_capacity.utils import from_pickle
import sys
# import numpy as np
# from matplotlib.pyplot import cm
import torchvision.utils as vutils
import sys
sys.path.insert(0, './../')
# import yaml
# import argparse
# import numpy as np
# from pathlib import Path
# from PyTorchVAE.models.beta_vae import BetaVAE, SmallBetaVAE
from PyTorchVAE.experiment import VAEXperiment
from PyTorchVAE.experiment_hsicbeta import VAEXperiment_hsicbeta
# import torch.backends.cudnn as cudnn
# from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
# from PyTorchVAE.dataset import  DisentDatasets
# from pytorch_lightning.plugins import DDPPlugin
from liftoff import parse_opts
from pathlib import Path

import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
# models
# from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
# # from loss_capacity.train_model import train_test_model
from loss_capacity.utils import list2tuple_, config_to_string, save_representation_dataset, get_representations_data_split
# from loss_capacity.probing import Probe
# from loss_capacity.functions import HSIC

# import timm

import torch
# import pickle
# import json
import pdb
from tqdm import tqdm

# folder = 'results/2022Aug11-170838_unsup_vae_dsprites/0000_vae.trainer_params.max_epochs_10__vae.exp_params.LR_0.002__vae.exp_params.scheduler_gamma_0.9999__vae.exp_params.kld_weight_0.1__vae.model_params.model_type_big__vae.model_params.recons_type_bce__dataset_dsprites/'

def plot_reconstructed(experiment, x, path, n=12):
    # n: number of images per row/column
    # x: only choose one input image as a reproducible benchmark
    latent_dim = experiment.model.latent_dim
    t = torch.zeros(n, n, latent_dim)
    mu, log_var = experiment.model.encode_(x)
    mu = torch.squeeze(mu)
    log_var = torch.squeeze(log_var)
    std = torch.sqrt(torch.exp(log_var))
    for i_latent in range(latent_dim -1):
        t[:,:,i_latent] = torch.full((n,n),mu[i_latent].item())
    for i_latent in range(latent_dim -1):
        x = torch.linspace(mu[i_latent].item() - 5, mu[i_latent].item() + 5, n)
        y = torch.linspace(mu[i_latent+1].item() - 5, mu[i_latent+1].item() + 5, n)
        xx, yy = torch.meshgrid(x,y, indexing='xy')
        xx = torch.unsqueeze(xx, 2)
        yy = torch.unsqueeze(yy, 2)
        z = torch.cat((xx,yy), 2)
        
        t[:,:,i_latent:i_latent+2] = z
        # pdb.set_trace()
        x_hat = experiment.model.decode(t)
        vutils.save_image(x_hat.data,
                            os.path.join(path , 
                                        'interpolate_'+ str(i_latent)
                                        +".png"),
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

    path = params.result_path
    # params.result_path
    # if restore:
    #     ckpt = params.out_dir + '/BetaVAE/version_0/checkpoints/last.ckpt'
    #     print(f'loading from: {ckpt}')
    #     lightning_ckpt = torch.load(ckpt)
    #     experiment.load_state_dict(lightning_ckpt['state_dict'])


    # number_of_channels = 1 if params.dataset == 'dsprites' else 3
    # print(f'number_of_channels: {number_of_channels}')
    imagenet_normalise = True if 'resnet' in params.model_type else False
    dataloader_val = load_dataset.load_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant='random',  # splittrainer_params types are random, composition, interpolation, extrapolation
            mode='test',
            dataset_path=params.dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            imagenet_normalise=imagenet_normalise,
            shuffle=False
        )
    x, _ = next(iter(dataloader_val))
    x = x[0]
    x = torch.unsqueeze(x,0)
    for folder in tqdm(glob(path+"/*/*/", recursive = True)):
        # try:
            print(folder)
            if 'hsic' in folder:
                experiment = VAEXperiment_hsicbeta.load_from_checkpoint(folder+"HsicBetaVAE/version_0/checkpoints/last.ckpt")
                plot_reconstructed(experiment, path = folder+"HsicBetaVAE/", x=x, n=12)
            else:
                experiment = VAEXperiment.load_from_checkpoint(folder+"BetaVAE/version_0/checkpoints/last.ckpt")
                plot_reconstructed(experiment, path = folder+"BetaVAE/", x=x, n=12)
            
        # except:
        #     continue
if __name__ == "__main__":
    run(parse_opts())
