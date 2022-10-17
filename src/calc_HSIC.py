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


def run(params):
    """ Entry point for liftoff. """
    params = list2tuple_(params)
    params.num_workers = 2
    print(config_to_string(params))

    path = params.result_path
    sigma_hsic = params.hsic.sigma_hsic   ##[0.1, 1, 10, 100, 1000, 10000, 100000]
    results_hsic = dict([(str(x),0) for x in sigma_hsic])
    num_sample_reparam = params.hsic.num_sample_reparam
    div_subsample = params.hsic.div_subsample # the divisor of the test dataset for subsampling
    choose_dataloader = params.hsic.choose_dataloader
    seed_folder = params.seed_folder # the seed of the folder of results
    job_folder = params.job_folder

    config = params.vae
    if 'none' in params.probe.max_leaf_nodes or 'None'  in  params.probe.max_leaf_nodes:
        params.probe.max_leaf_nodes = None
    print(f'out dir: {params.out_dir}')
        
    tb_logger =  TensorBoardLogger(save_dir=params.out_dir,
                                name=config.model_params.name)

    # For reproducibility
    seed_everything(config.exp_params.manual_seed * params.run_id, True) # this makes the sampling of dataset afterward the same

    # TODO: maybe do smt more generic
    # model = vae_models[config.model_params.name](**configmodel_params)


    device = 'cuda'

    
    # if restore:
    #     ckpt = params.out_dir + '/HsicHsicBetaVAE/version_0/checkpoints/last.ckpt'
    #     print(f'loading from: {ckpt}')
    #     lightning_ckpt = torch.load(ckpt)
    #     experiment.load_state_dict(lightning_ckpt['state_dict'])


    # number_of_channels = 1 if params.dataset == 'dsprites' else 3
    # print(f'number_of_channels: {number_of_channels}')
    imagenet_normalise = True if 'resnet' in params.model_type else False

    if choose_dataloader == 'train':
        dataloader = load_dataset.load_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant='random',  # split types are random, composition, interpolation, extrapolation
            mode='train',
            dataset_path=params.dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            imagenet_normalise=imagenet_normalise,
            data_fraction=1.0
        )
    elif choose_dataloader == 'val':
        dataloader = load_dataset.load_dataset(
        dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
        variant='random',  # splittrainer_params types are random, composition, interpolation, extrapolation
        mode='test',
        dataset_path=params.dataset_path, 
        batch_size=params.probe.batch_size, 
        num_workers=params.num_workers,
        standardise=True,
        imagenet_normalise=imagenet_normalise,
        shuffle=True
    )
    elif choose_dataloader == 'test':
        dataloader = load_dataset.load_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant='random',  # split types are random, composition, interpolation, extrapolation
            mode='test',
            dataset_path=params.dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            imagenet_normalise=imagenet_normalise,
            shuffle=True
        )
    if params.hsic.xlatent:
        title_text_x =  'xlatent'
    else:
        title_text_x = ''
    if params.hsic.ylatent:
        title_text_y =  'ylatent'
    else:
        title_text_y = ''

    for folder in sorted(glob(path+"/"+job_folder+"*/", recursive = True)):
            for subfolder in sorted(glob(os.path.join(folder,str(seed_folder)), recursive = True)):
                # ckpt = torch.load(subfolder+"0/HsicBetaVAE/version_0/checkpoints/last.ckpt")
                # print(ckpt.keys())
                # model.load_state_dict(ckpt['state_dict'])

                experiment = VAEXperiment.load_from_checkpoint(subfolder+"/"+config.model_params.name+"/version_0/checkpoints/last.ckpt")
                latent_dim = experiment.model.latent_dim

                if params.hsic.xlatent:
                    x_multi = latent_dim
                else:
                    x_multi = 1
                if params.hsic.ylatent:
                    y_multi = latent_dim
                else:
                    y_multi = 1
                if params.vae.exp_params.hsic_reg_version == 'v1':
                    hsic_func = hsic_batch
                    for sigma in tqdm(sigma_hsic):
                        hsic_score = 0
                        i = 0
                        for images, labels in dataloader: # dataloader is shuffled, we take a subsampling because it is too big.
                            while i < len(dataloader)/div_subsample:
                                images.to('cuda')
                                # print(images.shape)
                                
                                hsic_score += hsic_func(images, experiment, s_x=sigma*x_multi, s_y=sigma*y_multi, 
                                                device='cuda', num_sample_reparam=num_sample_reparam).item()
                                i += 1
                        hsic_score /= i
                        results_hsic[str(sigma)] = hsic_score
                        print(sigma, 'hsic_score testset', hsic_score)
                    to_pickle(results_hsic,subfolder+'/hsic_num_reparam_'+str(num_sample_reparam)+'_'+ 'div_'+ str(div_subsample) + title_text_x +'_'+ title_text_y + '_'+ choose_dataloader +'_v1.pickle')
                
                elif params.vae.exp_params.hsic_reg_version == 'v2':
                    hsic_func = hsic_batch_v2 # v2 was a wrong version in history. Just pass it.
                elif params.vae.exp_params.hsic_reg_version == 'v3':
                    hsic_func = hsic_batch_v2
                    # v3: hsic_batch_v2     (batchwise HSIC)
                    for sigma in tqdm(sigma_hsic):
                        hsic_score = 0
                        i = 0
                        for images, labels in dataloader: # dataloader is shuffled, we take a subsampling because it is too big.
                            while i < len(dataloader)/div_subsample:
                                images.to('cuda')
                                # print(images.shape)
                                
                                hsic_score += hsic_func(images, experiment, s_x=sigma*x_multi, s_y=sigma*y_multi, 
                                                device='cuda', num_sample_reparam=num_sample_reparam).item()
                                i += 1
                        hsic_score /= i
                        results_hsic[str(sigma)] = hsic_score
                        print(sigma, 'hsic_score testset', hsic_score)
                    to_pickle(results_hsic,subfolder+'/hsic_num_reparam_'+str(num_sample_reparam)+'_'+ 'div_'+ str(div_subsample) + title_text_x +'_'+ title_text_y + '_'+ choose_dataloader +'_v3.pickle')
                elif params.vae.exp_params.hsic_reg_version == 'v4':
                    hsic_func = hsic_v3
                    # v4: for every x in loop, calculate HSIC separately
                    for sigma in tqdm(sigma_hsic):
                        hsic_score = 0
                        i = 0
                        for images, labels in dataloader: # dataloader is shuffled, we take a subsampling because it is too big.
                            hsic_score_inbatch = 0
                            while i < len(dataloader)/div_subsample:
                                images.to('cuda')
                                # print(images.shape)
                                for j in range(images.shape[0]):
                                    single_image = images[j]
                                    hsic_score_inbatch += hsic_func(single_image, experiment, s_x=sigma*x_multi, s_y=sigma*y_multi, 
                                                    device='cuda', num_sample_reparam=num_sample_reparam).item()
                                hsic_score_inbatch /= j
                                i += 1
                                hsic_score += hsic_score_inbatch
                        hsic_score /= i
                        results_hsic[str(sigma)] = hsic_score
                        print(sigma, 'hsic_score testset', hsic_score)
                    to_pickle(results_hsic,subfolder+'/hsic_num_reparam_'+str(num_sample_reparam)+'_'+ 'div_'+ str(div_subsample) + title_text_x +'_'+ title_text_y + '_'+ choose_dataloader +'_v4.pickle')
                elif params.vae.exp_params.hsic_reg_version == 'v5':
                    hsic_func = hsic_v3
                    # v5: HSIC test and threshold as in the code of Gretton 2005
                    for sigma in tqdm(sigma_hsic):
                        hsic_score = 0
                        i = 0
                        for images, labels in dataloader: # dataloader is shuffled, we take a subsampling because it is too big.
                            hsic_score_inbatch = 0
                            while i < len(dataloader)/div_subsample:
                                images.to('cuda')
                                # print(images.shape)
                                for j in range(images.shape[0]):
                                    single_image = images[j]
                                    hsic_score_inbatch += hsic_func(single_image, experiment, s_x=sigma*x_multi, s_y=sigma*y_multi, 
                                                    device='cuda', num_sample_reparam=num_sample_reparam, threshold = True)
                                hsic_score_inbatch /= j
                                i += 1
                                hsic_score += hsic_score_inbatch
                        hsic_score /= i
                        results_hsic[str(sigma)] = hsic_score
                        print(sigma, 'hsic_score testset', hsic_score)
                    to_pickle(results_hsic,subfolder+'/hsic_num_reparam_'+str(num_sample_reparam)+'_'+ 'div_'+ str(div_subsample) + title_text_x +'_'+ title_text_y + '_'+ choose_dataloader +'_v1.pickle')


    # dci_scores_trees['hsic'] = hsic_score
if __name__ == "__main__":
    run(parse_opts())
