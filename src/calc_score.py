import sys
sys.path.insert(0, './../')
import os
import numpy as np
from pathlib import Path
from PyTorchVAE.experiment import VAEXperiment
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from liftoff import parse_opts
from pathlib import Path
from tqdm import tqdm
from data.ground_truth import dsprites
from evaluation.metrics.beta_vae import compute_beta_vae_sklearn
from evaluation.metrics.factor_vae import compute_factor_vae

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

from loss_capacity.utils import hsic_batch, hsic_batch_v2, hsic_v3, from_pickle, to_pickle
from glob import glob

from numpy.random import RandomState

def run(params):
    """ Entry point for liftoff. """
    params = list2tuple_(params)
    params.num_workers = 2
    print(config_to_string(params))
    path = params.result_path
    num_sample_reparam = params.hsic.num_sample_reparam
    div_subsample = params.hsic.div_subsample # the divisor of the test dataset for subsampling
    choose_dataloader = params.hsic.choose_dataloader
    seed_folder = params.seed_folder # the seed of the folder of results
    job_folder = params.job_folder
    config = params.vae
    rs = RandomState(config.exp_params.manual_seed)
    
    print(f'out dir: {params.out_dir}')
        
    tb_logger =  TensorBoardLogger(save_dir=params.out_dir,
                                name=config.model_params.name)

    # For reproducibility
    seed_everything(config.exp_params.manual_seed * params.run_id, True) # this makes the sampling of dataset afterward the same

    # TODO: maybe do smt more generic
    # model = vae_models[config.model_params.name](**configmodel_params)
    device = 'cuda'
    
    
    for folder in tqdm(sorted(glob(path+"/"+job_folder+"*/", recursive = True))):
        
        subfolder = os.path.join(folder, str(seed_folder))
        print(subfolder)
        experiment = VAEXperiment.load_from_checkpoint(subfolder+"/"+config.model_params.name+"/version_0/checkpoints/last.ckpt")
        latent_dim = experiment.model.latent_dim
        if params.score_name == 'beta':
            score = compute_beta_vae_sklearn(ground_truth_data=dsprites.DSprites(),
                        representation_function = experiment.model.encode,
                        random_state = rs,
                        batch_size = params.batch_size,
                        num_train = params.probe.num_train,
                        num_eval = params.probe.num_eval,
                        artifact_dir=None,
                        )
            to_pickle(score,subfolder+'/beta_score.pickle')
            print('beta score', score)
        elif params.score_name == 'factor':
            score = compute_factor_vae(ground_truth_data=dsprites.DSprites(),
                    representation_function= experiment.model.encode,
                    random_state =rs,
                    batch_size= params.batch_size,
                    num_train = params.probe.num_train,
                    num_eval = params.probe.num_eval,
                    num_variance_estimate = params.probe.num_variance_estimate,
                    artifact_dir=None
                    )
            to_pickle(score,subfolder+'/factor_score.pickle')
            print('factor score', score)
        
        
        

    # dci_scores_trees['hsic'] = hsic_score
if __name__ == "__main__":
    run(parse_opts())
