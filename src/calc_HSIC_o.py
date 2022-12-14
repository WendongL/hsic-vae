import sys
sys.path.insert(0, './../')
import os
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


# -------
# import sys
# sys.path.insert(0, './../InDomainGeneralizationBenchmark/src/')
# import lablet_generalization_benchmark.evaluate_model as evaluate_model
# import lablet_generalization_benchmark.load_dataset as load_dataset
# import lablet_generalization_benchmark.model as models
# from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
# from loss_capacity.train_model import train_test_model
# from loss_capacity.probing import Probe
# datasets and evals



import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
# models
from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
# from loss_capacity.train_model import train_test_model
from loss_capacity.utils import list2tuple_, config_to_string, save_representation_dataset
# from loss_capacity.probing import Probe
from loss_capacity.functions import HSIC
from  loss_capacity.train_model_random_forest import train_test_random_forest
import timm

import torch
import pickle
import json
import pdb

from loss_capacity.utils import hsic_batch, from_pickle, to_pickle
from glob import glob


def run(params):
    """ Entry point for liftoff. """
    path = "./results/2022Aug11-170838_unsup_vae_dsprites"
    sigma_hsic = [0.01, 0.1, 1, 10, 100, 1000]
    results_hsic = dict([(str(x),0) for x in sigma_hsic])

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

    if config.model_params.model_type == 'small':
        model = SmallBetaVAE(**config.model_params.__dict__)
    elif config.model_params.model_type == 'big':
        model = BetaVAE(**config.model_params.__dict__)


    device = 'cuda'
    model = model.to(device)
    experiment = VAEXperiment(model,
                            config.exp_params)

    
    # if restore:
    #     ckpt = params.out_dir + '/BetaVAE/version_0/checkpoints/last.ckpt'
    #     print(f'loading from: {ckpt}')
    #     lightning_ckpt = torch.load(ckpt)
    #     experiment.load_state_dict(lightning_ckpt['state_dict'])


    # number_of_channels = 1 if params.dataset == 'dsprites' else 3
    # print(f'number_of_channels: {number_of_channels}')
    imagenet_normalise = True if 'resnet' in params.model_type else False
    # dataloader_train = load_dataset.load_dataset(
    #     dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
    #     variant='random',  # split types are random, composition, interpolation, extrapolation
    #     mode='train',
    #     dataset_path=params.dataset_path, 
    #     batch_size=params.probe.batch_size, 
    #     num_workers=params.num_workers,
    #     standardise=True,
    #     imagenet_normalise=imagenet_normalise,
    #     data_fraction=1.0
    # )
    # dataloader_val = load_dataset.load_dataset(
    #     dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
    #     variant='random',  # split types are random, composition, interpolation, extrapolation
    #     mode='test',
    #     dataset_path=params.dataset_path, 
    #     batch_size=params.probe.batch_size, 
    #     num_workers=params.num_workers,
    #     standardise=True,
    #     imagenet_normalise=imagenet_normalise,
    #     shuffle=False
    # )
    dataloader_test = load_dataset.load_dataset(
        dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
        variant='random',  # split types are random, composition, interpolation, extrapolation
        mode='test',
        dataset_path=params.dataset_path, 
        batch_size=params.probe.batch_size, 
        num_workers=params.num_workers,
        standardise=True,
        imagenet_normalise=imagenet_normalise,
        shuffle=False
    )

    for folder in glob(path+"/*/", recursive = True):
        # ckpt = torch.load(folder+"0/BetaVAE/version_0/checkpoints/last.ckpt")
        # print(ckpt.keys())
        # model.load_state_dict(ckpt['state_dict'])
        
        experiment = VAEXperiment.load_from_checkpoint(folder+"0/BetaVAE/version_0/checkpoints/last.ckpt")

        # num_probe_params, probe_train_score, probe_val_score, probe_test_score, dci_scores_trees = train_test_random_forest(
        #         model=model, 
        #         dataloader_train=dataloader_train, 
        #         dataloader_val=dataloader_val, 
        #         dataloader_test=dataloader_test, 
        #         method='rf',#params.probe.type,
        #         max_leaf_nodes=params.probe.max_leaf_nodes,
        #         max_depth=params.probe.max_depth,
        #         num_trees=params.probe.num_trees,
        #         seed = params.seed * params.run_id,
        #         lr = params.probe.rf_lr,
        #         epochs = params.probe.epochs, 
        #         log_interval =  params.log_interval, save_model = params.save_model,
        #         train_name='probe_trees',
        #         tb_writer=None, eval_model=True, savefolder=params.out_dir,
        #         device=device,
        #         data_fraction=params.probe.data_fraction,
        #         use_sage=False
        #         )
        for sigma in sigma_hsic:
            hsic_score = 0
            for images, labels in dataloader_test:
                images.to('cuda')
                hsic_score += hsic_batch(images, experiment, s_x=sigma, s_y=sigma, device='cuda', batch_size=512).item()
            hsic_score /= len(dataloader_test)
            results_hsic[str(sigma)] = hsic_score
            print(sigma, 'hsic_score', hsic_score)
            to_pickle(results_hsic,folder+'0/hsic_sigma_o.pickle')


    # dci_scores_trees['hsic'] = hsic_score
if __name__ == "__main__":
    run(parse_opts())
