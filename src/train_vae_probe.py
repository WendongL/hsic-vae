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
from loss_capacity.train_model import train_test_model, train_test_probe
from loss_capacity.probing import Probe, ProbeContDiscrete
from loss_capacity.utils import list2tuple_, config_to_string
import timm

import torch
import pickle
import json
import os
import pdb

def run(params):
    """ Entry point for liftoff. """
    params = list2tuple_(params)
    print(config_to_string(params))

    import glob
    folders = glob.glob(params.representation_dataset_path + '/*')
    for folder in folders:
        # print(f'[{str(params.vae.exp_params.kld_weight)}] in [{folder}] -- {str(params.vae.exp_params.kld_weight) in folder}')
        if str(params.vae.exp_params.kld_weight) in folder and params.vae.model_params.recons_type in folder :
            params.representation_dataset_path = folder + '/0/'+params.dataset+'_vae_pretrained_dataset'
            print(f'load data from: {params.representation_dataset_path}')
    # pdb.set_trace()
    device = 'cuda'  # cuda or cpu
    # TODO: does this decrease performance???
    torch.multiprocessing.set_sharing_strategy('file_system')
    number_of_channels = 1 if params.dataset == 'dsprites' else 3
    if params.cached_representations:
        # we iterate through saved representations
        dataloader_train = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            mode='train_without_val',
            dataset_path=params.representation_dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True
        )

        dataloader_val = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            mode='val',
            dataset_path=params.representation_dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            shuffle=False
        )

        dataloader_test = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            mode='test',
            dataset_path=params.representation_dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            shuffle=False
        )
        inputs, targets = next(iter(dataloader_test))
        model = RawData(latent_dim=inputs.shape[-1])
        # TODO: is there an error here??
        # model = RawData(latent_dim=512)
        model = model.to(device)
        number_targets = len(dataloader_train.dataset._factor_sizes)

    else:
        print(f'model dir: {params.out_dir}')
        # load appropiate dataset depending on model type
        # models based on noisy (mix of) labels loads only the labels dataset
        print(f'number_of_channels: {number_of_channels}')
        imagenet_normalise = True if 'resnet' in params.model_type else False

        if 'labels' in params.model_type:

            dataloader_train = load_dataset.load_labels_dataset(
                dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
                variant='random',  # split types are random, composition, interpolation, extrapolation
                mode='train_without_val',
                dataset_path=params.dataset_path, 
                batch_size=params.probe.batch_size, 
                num_workers=params.num_workers,
                standardise=True
            )

            dataloader_val = load_dataset.load_labels_dataset(
                dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
                variant='random',  # split types are random, composition, interpolation, extrapolation
                mode='val',
                dataset_path=params.dataset_path, 
                batch_size=params.probe.batch_size, 
                num_workers=params.num_workers,
                standardise=True,
                shuffle=False
            )


            dataloader_test = load_dataset.load_labels_dataset(
                dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
                variant='random',  # split types are random, composition, interpolation, extrapolation
                mode='test',
                dataset_path=params.dataset_path, 
                batch_size=params.probe.batch_size, 
                num_workers=params.num_workers,
                standardise=True,
                shuffle=False
            )
        else:
            dataloader_train = load_dataset.load_dataset(
                dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
                variant='random',  # split types are random, composition, interpolation, extrapolation
                mode='train_without_val',
                dataset_path=params.dataset_path, 
                batch_size=params.probe.batch_size, 
                num_workers=params.num_workers,
                standardise=True,
                imagenet_normalise=imagenet_normalise
            )
            
            dataloader_val = load_dataset.load_dataset(
                dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
                variant='random',  # split types are random, composition, interpolation, extrapolation
                mode='val',
                dataset_path=params.dataset_path, 
                batch_size=params.probe.batch_size, 
                num_workers=params.num_workers,
                standardise=True,
                imagenet_normalise=imagenet_normalise,
                shuffle=False
            )

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

        number_targets = len(dataloader_train.dataset._factor_sizes)
        

        # load model
        if params.model_type == 'raw_data':
            # TODO: compute the dim of the raw data
            model = RawData(latent_dim=number_of_channels * 64 * 64)

        elif params.model_type == 'noisy_labels':
            noise_std = 0.2 * params.probe.noise_std_mult
            model = NoisyLabels(
                        number_of_classes=number_targets, 
                        noise_std=noise_std
            )
        elif params.model_type == 'random_linear_mix_labels':
            noise_std = 0.003 * params.probe.noise_std_mult
            model = LinearMixLabels(
                        number_of_classes=number_targets, 
                        noise_std=noise_std,
                        mix_type='random_linear'
            )
        elif params.model_type == 'noisy_uniform_mix_labels':
            noise_std = 0.003 * params.probe.noise_std_mult
            model = LinearMixLabels(
                        number_of_classes=number_targets, 
                        noise_std=noise_std,
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
            model = ResNet18(
                number_of_classes=number_targets, 
                number_of_channels=number_of_channels,
                pretrained=params.pretrained
            )

        model = model.to(device)
        
        if params.supervision:
            
            ckeckpoint_path = 'ceva'
            if os.exists(ckeckpoint_path):
                ckpt = torch.load(ckeckpoint_path)
                model.load_state_dict(ckpt)
            else:
                print('checkpoint not found: Start training the model.')
                model, test_loss = train_test_model(model=model, 
                            dataloader_train=dataloader_train, 
                            dataloader_test=dataloader_test,
                            seed = params.seed,
                            lr = params.model.lr, optim_steps=params.model.optim_steps,
                            epochs = params.model.epochs, 
                            log_interval =  params.log_interval, save_model = params.save_model
                        )

    model.eval()

    # img, target = next(iter(dataloader_test))
    # feats = model.encode(img.to(device))
    # itera = iter(dataloader_test)
    
    # model.eval()
    # for idx, (data, target) in enumerate(dataloader_test):
    #     data
    #     feats = model.encode(data.to(device))
    #     feats, target = next(itera)
        
    

    # def model_fn1(images):
    #     representation = model(torch.tensor(images).to(device))
    #     return representation.detach().cpu().numpy()

    # scores1 = evaluate_model.evaluate_model(model_fn1, dataloader_test)
    # print(f'scores before training: {scores1}')

    # crate tensorboard writter
    writer = SummaryWriter(log_dir=params.out_dir)
    # create probe
    use_cross_entropy = True
    if use_cross_entropy == False:
        probe = Probe(model, 
            num_factors=number_targets,
            num_hidden_layers=params.probe.hidden_layers,
            multiplier=params.probe.hidden_multiplier
        )
    else:
        probe = ProbeContDiscrete(model, 
            num_factors=number_targets,
            num_hidden_layers=params.probe.hidden_layers,
            multiplier=params.probe.hidden_multiplier,
            factor_sizes=dataloader_train.dataset._factor_sizes,
            factor_discrete=dataloader_train.dataset._factor_discrete
        )
    probe = probe.to(device)

    print(f'Initialising probe with {probe.count_parameters()} parameters')
    # train the probe
    probe.train()
    probe.model.eval() # the model is freezed, so it should stay in eval mode
    if use_cross_entropy:
        probe, probe_test_loss, probe_test_score, probe_val_score = train_test_probe(model=probe, 
            dataloader_train=dataloader_train, 
            dataloader_val=dataloader_val, 
            dataloader_test=dataloader_test, 
            seed = params.seed,
            lr = params.probe.lr, optim_steps=params.model.optim_steps,
            epochs = params.probe.epochs, 
            log_interval =  params.log_interval, save_model = params.save_model,
            train_name='probe',
            tb_writer=writer, eval_model=True, savefolder=params.out_dir,
            cross_ent_mult=params.probe.cross_ent_mult)
    else:
        probe, probe_test_loss = train_test_model(model=probe, 
            dataloader_train=dataloader_train, dataloader_test=dataloader_test,
            seed = params.seed,
            lr = params.probe.lr, optim_steps=params.model.optim_steps,
            epochs = params.probe.epochs, 
            log_interval =  params.log_interval, save_model = params.save_model,
            train_name='probe',
            tb_writer=writer, eval_model=True, savefolder=params.out_dir)
    
    probe.eval()
    # def probe_fn(images):
    #     representation = probe(torch.tensor(images).to(device))
    #     return representation.detach().cpu().numpy()

    # # TODO: idealy we would want to use dev split for model selection and test for the final evaluation
    # # right now we use the model at the end of the training...
    # scores_probe = evaluate_model.evaluate_model(probe_fn, dataloader_train)
    # print(f'Scores Train probe: {scores_probe}')

    # scores_probe = evaluate_model.evaluate_model(probe_fn, dataloader_test)
    # print(f'Scores Test probe: {scores_probe}')



    from pathlib import Path
    Path(params.out_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    num_probe_params = probe.count_parameters()
    id=f'model_{params.name}_type_{params.model_type}_probe_params_{num_probe_params}'
    
    results['id'] = id
    results['model_type'] = params.model_type
    results['num_params'] = num_probe_params
    results['mse'] = probe_test_loss
    results['val_rsquared_acc'] = probe_val_score
    results['test_rsquared_acc'] = probe_test_score
    results['params'] = params
    with open(f'{params.out_dir}/results.pkl', 'wb') as fp:
        pickle.dump(results, fp)

    json_results = { key : str(val) for key, val in results.items()}

    with open(f'{params.out_dir}/results.json', 'w') as fp:
        json.dump(json_results, fp)

if __name__ == "__main__":
    run(parse_opts())