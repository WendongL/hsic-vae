import sys
# sys.path.insert(0, './../src')
sys.path.insert(0, './../')
from liftoff import parse_opts
import numpy as np
import argparse
from torch.utils.tensorboard import SummaryWriter
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.model as models
from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
from loss_capacity.train_model import train_test_model, train_test_probe, evaluate_probe
from loss_capacity.train_model_random_forest import train_test_random_forest
from loss_capacity.analitic_solution import get_linear_least_squares, get_linear_least_squares_search
from loss_capacity.probing import Probe, ProbeContDiscrete, ProbeIndividual, RFFProbeIndividual
from loss_capacity.utils import list2tuple_, config_to_string
# from metrics.get_metric import get_probe_dci, get_dci

# from metrics.compute_metrics import compute_disentanglement_metric
import timm

import torch
import pickle
import json
import os
import pdb


sys.path.insert(1, './../src')

import numpy as np

import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.model as models
from PyTorchVAE import experiment_hsicbeta
from PyTorchVAE.models.hsicbeta_vae import SmallHsicBetaVAE
from PyTorchVAE.models.beta_vae import BetaVAE, SmallBetaVAE
from PyTorchVAE.models.hsicbeta_vae import HsicBetaVAE, SmallHsicBetaVAE
from PyTorchVAE.experiment import VAEXperiment
from PyTorchVAE.experiment_hsicbeta import VAEXperiment_hsicbeta

def run(params):
    params = list2tuple_(params)
    params.num_workers = 2
    print(config_to_string(params))

    config = params.vae
    dataloader = load_dataset.load_dataset(
        dataset_name='dsprites',  # datasets are dsprites, shapes3d and mpi3d
        variant='extrapolation',  # split types are random, composition, interpolation, extrapolation
        mode='test',
        dataset_path='./data/', 
        batch_size=256, 
        num_workers=4
    )

    device = 'cuda'  # cuda or cpu
    # model = experiment_hsicbeta.VAEXperiment_hsicbeta(vae_model = hsicbeta_vae, params = params.vae.exp_params).to(device)
    model = SmallHsicBetaVAE(**config.model_params.__dict__)
    experiment = VAEXperiment_hsicbeta(model, config.exp_params)
    checkpoint = torch.load('/home/wliang/Github/loss_capacity/src/results/2022Oct18-100404_dsprites_hsicbetavae_3s/0010_variant_extrapolation__vae.model_params.alpha_10000.0/1/HsicBetaVAE/version_0/checkpoints/last.ckpt', 
                            map_location=torch.device(device))
    # pdb.set_trace()
    experiment.load_state_dict(checkpoint['state_dict'])
    experiment.to(device)
    epoch = checkpoint['epoch']

    experiment.eval()

    def model_fn(images):
        representation = experiment.model(torch.tensor(images).to(device)) 
        pdb.set_trace()
        return representation.detach().cpu().numpy()

    scores = evaluate_model.evaluate_model(model_fn, dataloader)
    print(scores)



if __name__ == "__main__":
    run(parse_opts())