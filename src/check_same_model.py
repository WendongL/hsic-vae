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
from loss_capacity.utils import list2tuple_, config_to_string, save_representation_dataset, get_representations_data_split
# from loss_capacity.probing import Probe
from loss_capacity.functions import HSIC
import timm

import torch
import pickle
import json
import pdb
from glob import glob


def compare_models(model_1, model_2):
    models_differ = 0
    for key_item_1, key_item_2 in zip(model_1.items(), model_2.items()):
        if torch.equal(key_item_1[1], key_item_2[1]):
            pass
        else:
            models_differ += 1
            if (key_item_1[0] == key_item_2[0]):
                print('Mismtach found at', key_item_1[0])
            else:
                raise Exception
    if models_differ == 0:
        print('Models match perfectly! :)')


# def run(params):
#     params = list2tuple_(params)
#     params.num_workers = 2
#     config = params.vae
#     # For reproducibility
#     seed_everything(config.exp_params.manual_seed * params.run_id, True)

#     if config.model_params.model_type == 'small':
#         model = SmallBetaVAE(**config.model_params.__dict__)
#     elif config.model_params.model_type == 'big':
#         model = BetaVAE(**config.model_params.__dict__)
#     savename = params.out_dir + f'/{params.dataset}_{params.model_type}_{params.name}'
#     checkpoint_path = savename + '.pt'
#     Path(params.out_dir).mkdir(parents=True, exist_ok=True)  
#     torch.save(model.state_dict(), savename +'.pt')

# if __name__ == "__main__":
#     run(parse_opts())

path = './results/2022Aug18-154612_unsup_vae_dsprites_check'
folders = glob(path+"/*/*/")
model0 = torch.load(folders[0]+'dsprites_vae_pretrained.pt')
model1 = torch.load(folders[1]+'dsprites_vae_pretrained.pt')
model2 = torch.load(folders[2]+'dsprites_vae_pretrained.pt')
model3 = torch.load(folders[3]+'dsprites_vae_pretrained.pt')
model4 = torch.load(folders[4]+'dsprites_vae_pretrained.pt')
model5 = torch.load(folders[5]+'dsprites_vae_pretrained.pt')
compare_models(model0, model1)
compare_models(model1, model2)
compare_models(model2, model3)
compare_models(model3, model4)
compare_models(model4, model5)