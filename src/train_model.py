import sys
# sys.path.insert(0, './../src')
sys.path.insert(0, './../')
from liftoff import parse_opts
import numpy as np
import argparse
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.evaluate_model as evaluate_model
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.load_dataset as load_dataset
import InDomainGeneralizationBenchmark.src.lablet_generalization_benchmark.model as models
from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
from loss_capacity.train_model import train_test_model
from loss_capacity.probing import Probe
import timm

import torch
import pickle
import json
import pdb

def run(params):
    print(f'model dir: {params.model_dir}')
    # print(f'starting model: {params.model_type}')
    # return 0
    # load appropiate dataset depending on model type
    # models based on noisy (mix of) labels loads only the labels dataset
    number_of_channels = 1 if params.dataset == 'dsprites' else 3
    print(f'number_of_channels: {number_of_channels}')
    if 'labels' in params.model_type:

        dataloader_train = load_dataset.load_labels_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant='random',  # split types are random, composition, interpolation, extrapolation
            mode='train',
            dataset_path='/home/anico/data/disent_indommain/', 
            batch_size=256, 
            num_workers=10,
            standardise=True
        )

        dataloader_test = load_dataset.load_labels_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant='random',  # split types are random, composition, interpolation, extrapolation
            mode='test',
            dataset_path='/home/anico/data/disent_indommain/', 
            batch_size=256, 
            num_workers=10,
            standardise=True
        )
    else:
        dataloader_train = load_dataset.load_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant='random',  # split types are random, composition, interpolation, extrapolation
            mode='train',
            dataset_path='/home/anico/data/disent_indommain/', 
            batch_size=256, 
            num_workers=10,
            standardise=True
        )

        dataloader_test = load_dataset.load_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant='random',  # split types are random, composition, interpolation, extrapolation
            mode='test',
            dataset_path='/home/anico/data/disent_indommain/', 
            batch_size=256, 
            num_workers=10,
            standardise=True
        )

    number_targets = len(dataloader_train.dataset._factor_sizes)
    device = 'cuda'  # cuda or cpu

    # load model
    if params.model_type == 'raw_data':
        # TODO: compute the dim of the raw data
        model = RawData(latent_dim=number_of_channels * 64 * 64)

    elif params.model_type == 'noisy_labels':
        model = NoisyLabels(
                    number_of_classes=number_targets, 
                    noise_std=params.noise_std
        )
    elif params.model_type == 'random_linear_mix_labels':
        model = LinearMixLabels(
                    number_of_classes=number_targets, 
                    noise_std=params.noise_std,
                    mix_type='random_linear'
        )
    elif params.model_type == 'noisy_uniform_mix_labels':
        model = LinearMixLabels(
                    number_of_classes=number_targets, 
                    noise_std=params.noise_std,
                    mix_type='noisy_uniform'
        )    
    elif params.model_type == 'conv_net':
        model = ConvNet(
            number_of_classes=number_targets, 
            number_of_channels=number_of_channels
        )
    elif params.model_type == 'resnet18':
        # model = timm.create_model('resnet18', 
        #     pretrained=True, 
        #     in_chans=1,
        #     num_classes=number_targets)
        pretrained = True if params.pretrained == 'yes' else False
        model = ResNet18(
            number_of_classes=number_targets, 
            number_of_channels=number_of_channels,
            pretrained=pretrained
        )

    model = model.to(device)
    
    # TODO
    # add pretrained models!!!
    # TODO: change params to training instead of supervision
    if params.supervision == 'supervised':
        model, test_loss = train_test_model(params, model=model, 
            dataloader_train=dataloader_train, 
            dataloader_test=dataloader_test
        )

    model.eval()

    # def model_fn1(images):
    #     representation = model(torch.tensor(images).to(device))
    #     return representation.detach().cpu().numpy()

    # scores1 = evaluate_model.evaluate_model(model_fn1, dataloader_test)
    # print(f'scores before training: {scores1}')

    # create probe
    probe = Probe(model, 
        num_factors=number_targets,
        num_hidden_layers=params.probe_hidden_layers,
        multiplier=params.probe_hidden_multiplier
    )
    probe = probe.to(device)

    print(f'Initialising probe with {probe.count_parameters()} parameters')
    # train the probe
    probe, probe_test_loss = train_test_model(params, model=probe, 
        dataloader_train=dataloader_train, dataloader_test=dataloader_test)
    
    probe.eval()
    def probe_fn(images):
        representation = probe(torch.tensor(images).to(device))
        return representation.detach().cpu().numpy()

    # TODO: idealy we would want to use dev split for model selection and test for the final evaluation
    # right now we use the model at the end of the training...
    scores_probe = evaluate_model.evaluate_model(probe_fn, dataloader_train)
    print(f'Scores Train probe: {scores_probe}')

    scores_probe = evaluate_model.evaluate_model(probe_fn, dataloader_test)
    print(f'Scores Test probe: {scores_probe}')

    results = {}
    num_probe_params = probe.count_parameters()
    id=f'model_{params.name}_type_{params.model_type}_probe_params_{num_probe_params}'
    
    results['id'] = id
    results['model_type'] = params.model_type
    results['num_params'] = num_probe_params
    results['mse'] = scores_probe['mse']
    results['rsquared'] = scores_probe['rsquared']
    results['params'] = params
    with open(f'{params.model_dir}/results.pkl', 'wb') as fp:
        pickle.dump(results, fp)

    json_results = { key : str(val) for key, val in results.items()}

    with open(f'{params.model_dir}/results.json', 'w') as fp:
        json.dump(json_results, fp)

# def main():
#     parser = argparse.ArgumentParser(description="Loss capacity project")
#     parser.add_argument("--name", type=str, default='model')
#     parser.add_argument("--model_dir", type=str, default='model_dir')
#     parser.add_argument("--dataset", type=str, default='dsprites')


#     parser.add_argument(
#         "--batch-size", type=int, default=64, metavar="N", help="input batch size for training (default: 64)"
#     )
#     parser.add_argument("--epochs", type=int, default=10, metavar="N", help="number of epochs to train (default: 14)")
#     parser.add_argument("--lr", type=float, default=0.001, metavar="LR", help="learning rate (default: 0.001)")
#     # parser.add_argument("--gamma", type=float, default=0.7, metavar="M", help="Learning rate step gamma (default: 0.7)")
#     parser.add_argument("--dry-run", action="store_true", default=False, help="quickly check a single pass")
#     parser.add_argument("--seed", type=int, default=1, metavar="S", help="random seed (default: 1)")
#     parser.add_argument(
#         "--log-interval",
#         type=int,
#         default=10,
#         metavar="N",
#         help="how many batches to wait before logging training status",
#     )
#     parser.add_argument("--save-model", action="store_true", default=False, help="For Saving the current Model")
#     # model parameters
#     parser.add_argument("--model_type", type=str, default='noisy_labels',
#          help='model used for obtaining the representations chose from [raw_data/ noisy_labels / random_linear_mix_labels / noisy_uniform_mix_labels / conv_net]')
#     # model type: raw_data/ noisy_labels / random_linear_mix_labels / noisy_uniform_mix_labels / conv_net
#     # parser.add_argument("--supervised_model", action="store_true", default=False, help="train the model in a supervised way")
#     parser.add_argument("--supervision", type=str, default='none',
#          help='supervision of the model [none / supervised]')
#     parser.add_argument("--pretrained", type=str, default='no',
#          help='supervision of the model [no / yes]')

#     parser.add_argument("--noise_std", type=float, default=0.1, help="noise in models that use noisy labels as input")
   
   

#     # probe parameters
#     parser.add_argument("--probe_hidden_layers", type=int, default=2, help="number of hidden layers in the probe MLP (default: 1)")
#     parser.add_argument("--probe_hidden_multiplier", type=int, default=16, help="size of the hidden layer (multiplier x num_factors) (default: 16)")

#     hparams = parser.parse_args()
#     run(hparams)



if __name__ == "__main__":
    run(parse_opts())