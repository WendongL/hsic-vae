import sys
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

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

import pdb

def run(params):
    """ Entry point for liftoff. """
    # params.probe.data_fraction = 0.0001
    if 'hidden_dim' not in params.probe:
        print('WARNING: converting from hidden_multiplier to hidden_dim')
        # map from multiplier to hidden dim
        mult = [0, 2 , 4 , 8 , 12 , 16 , 32 , 64 , 128, 256 , 512 ]
        hdim = [0, 33, 47, 75, 103, 131, 243, 467, 915, 1811, 3603]
        
        m = params.probe.hidden_multiplier
        if m in mult:
            params.probe.hidden_dim = hdim[ mult.index(m) ]
    DEBUG_FLAG = False
    if DEBUG_FLAG:
        print('-'*120)
        print(f'WARNING: DEBUG_FLAG={DEBUG_FLAG}')
        print('-'*120)

    if True:
        params.probe.use_norm = False
        params.probe.use_dropout =  False
        params.probe.weight_decay = 0.0
        print(f'WARNING: hardcodding use_norm, dropout, decay')
    seed_bias = 0
    print(f'WARNING: setting seed_bias = {seed_bias} !!!!')

    params = list2tuple_(params)
    print(config_to_string(params))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  # cuda or cpu
    print('WARNING: setting num_trees = 100 !!!!')
    params.probe.num_trees = 100
    print('WARNING: hard-codding some parameters')
    
    # TODO: remove hard codding
    sage_group_pixels = True if 'raw' in params.model_type else False
    print(f'WARNING: hardcodding sage_group pixels={sage_group_pixels}')

    # if 'almost_uniform_mix_labels' in params.model_type:
    #     if params.probe.hidden_multiplier == 0: # or params.probe.hidden_dim == 0:
    #         params.probe.lr = 2.0

    if 'none' in  params.probe.max_leaf_nodes or 'None'  in  params.probe.max_leaf_nodes:
        params.probe.max_leaf_nodes = None
    # TODO: does this decrease reading performance???
    torch.multiprocessing.set_sharing_strategy('file_system')
    # find path of the VAEs encoring corresponding to the current hiper parameters (e.g. for the current beta)
    if 'vae' in params.model_type:
        import glob
        folders = glob.glob(params.vae_representation_dataset_path + '/*')
        for folder in folders:
            # print(f'[{str(params.vae.exp_params.kld_weight)}] in [{folder}] -- {str(params.vae.exp_params.kld_weight) in folder}')
            # if ( ('kld_weight_' + str(params.vae.exp_params.kld_weight) + '_' in folder 
            #             or (int(params.vae.exp_params.kld_weight) >= 1 
            #             and 'kld_weight_' + str(int(params.vae.exp_params.kld_weight)) + '_' in folder) )
            #         and params.vae.model_params.model_type in folder
            #         and params.vae.model_params.recons_type in folder
            #         and 'LR_'+str(params.vae.exp_params.LR) + '_'  in folder) :
            #     # params.representation_dataset_path = folder + f'/{params.run_id}/mpi3d_vae_pretrained_dataset'
            #     # TODO: multiple probes runs use the same model
            if "alpha_"+str(params.vae.exp_params.alpha) in folder and params.variant in folder:
                params.representation_dataset_path = folder + f'/{params.train_run_id}/{params.dataset}_{params.variant}_vae_pretrained_dataset'
                print(f'load data from: {params.representation_dataset_path}')
    
    number_of_channels = 1 if params.dataset == 'dsprites' else 3
    if params.cached_representations and ('resnet' in params.model_type or 'vae' in params.model_type ):
        # load the representations / ecodings / embdenddings from the path
        # we iterate through saved representations
        dataloader_train = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant=params.variant,
            # mode='train_without_val',
            mode = 'train',
            dataset_path=params.representation_dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            data_fraction=params.probe.data_fraction if DEBUG_FLAG == False else 256
        )

        dataloader_val = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant=params.variant,
            mode='val',
            dataset_path=params.representation_dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            shuffle=False,
            data_fraction=1.0 if DEBUG_FLAG == False else 256
        )

        dataloader_test = load_dataset.load_representation_dataset(
            dataset_name=params.dataset,  # datasets are dsprites, shapes3d and mpi3d
            variant=params.variant,
            mode='test' if DEBUG_FLAG == False else 'val',
            dataset_path=params.representation_dataset_path, 
            batch_size=params.probe.batch_size, 
            num_workers=params.num_workers,
            standardise=True,
            shuffle=False,
            data_fraction=1.0 if DEBUG_FLAG == False else 256
        )
        inputs, targets = next(iter(dataloader_test))
        # identity function. lor compatibility with other code
        model = RawData(latent_dim=inputs.shape[-1])
        model = model.to(device)
        number_targets = len(dataloader_train.dataset._factor_sizes)

        model = model.to(device)
        

    print(f'Loaded train dataset with size: {len(dataloader_train.dataset)}')
    print(f'Loaded val dataset with size: {len(dataloader_val.dataset)}')
    print(f'Loaded test dataset with size: {len(dataloader_test.dataset)}')
    
    model.eval()
    # crate tensorboard writter
    writer = SummaryWriter(log_dir=params.out_dir)
    # create probe
    probe = None
    # if params.probe.type == 'MLP_reg':
    #     probe = Probe(model, 
    #         num_factors=number_targets,
    #         num_hidden_layers=params.probe.hidden_layers,
    #         multiplier=params.probe.hidden_multiplier
    #     )
    #     probe = probe.to(device)

    # elif params.probe.type == 'MLP_reg_cl':
    #     probe = ProbeContDiscrete(model, 
    #         num_factors=number_targets,
    #         num_hidden_layers=params.probe.hidden_layers,
    #         # multiplier=params.probe.hidden_multiplier,
    #         hidden_dim=params.probe.hidden_dim,
    #         factor_sizes=dataloader_train.dataset._factor_sizes,
    #         factor_discrete=dataloader_train.dataset._factor_discrete
    #     )
    #     probe = probe.to(device)
    
    if params.probe.type == 'MLP_reg_cl_ind':
        # probe represents the fuctions that receives the representation and predicts the factors
        # each factor is predicted by an independent MLP
        # the set of MLPs constitutes the probe
        probe = ProbeIndividual(model, 
            num_factors=number_targets,
            num_hidden_layers=params.probe.hidden_layers,
            # multiplier=params.probe.hidden_multiplier,
            hidden_dim=params.probe.hidden_dim,
            factor_sizes=dataloader_train.dataset._factor_sizes,
            factor_discrete=dataloader_train.dataset._factor_discrete,
            use_norm=params.probe.use_norm,
            use_dropout=params.probe.use_dropout
        )
        probe = probe.to(device)
    

    # train the probe
    if probe is not None:
        probe.train()
        probe.model.eval() # the model is freezed, so it should stay in eval mode

    resume = True
    dci_scores_trees = None
    dci_scores_val = None
    dci_scores = None
    probe_test_loss=0
    probe_val_score = probe_test_score =  -1
    ckeckpoint_path = f'{params.out_dir}/best_probe_model.pt'

    ckeckpoint_path = f'{params.out_dir}/best_probe_model.pt'
    if not resume or os.path.exists(ckeckpoint_path) == False:
        # train the probes 
        if 'MLP_reg_cl' in params.probe.type: # MLP_reg_cl or MLP_reg_cl_ind
            probe, probe_test_loss, probe_test_score, probe_val_score = train_test_probe(model=probe, 
                dataloader_train=dataloader_train, 
                dataloader_val=dataloader_val, 
                dataloader_test=dataloader_test, 
                seed = params.seed * params.run_id + seed_bias,
                lr = params.probe.lr, 
                weight_decay = params.probe.weight_decay,
                optim_steps=params.model.optim_steps,
                epochs = params.probe.epochs, 
                log_interval =  params.log_interval, save_model = params.save_model,
                train_name='probe_mlp',
                tb_writer=writer, eval_model=True, savefolder=params.out_dir,
                cross_ent_mult=params.probe.cross_ent_mult)
            num_probe_params = probe.count_parameters()
            
        # elif params.probe.type == 'MLP_reg':
        #     probe, probe_test_loss = train_test_model(model=probe, 
        #         dataloader_train=dataloader_train, dataloader_val=dataloader_val,
        #         seed = params.seed * params.run_id + seed_bias,
        #         lr = params.probe.lr, 
        #         weight_decay = params.probe.weight_decay,
        #         optim_steps=params.model.optim_steps,
        #         epochs = params.probe.epochs, 
        #         log_interval =  params.log_interval, save_model = params.save_model,
        #         train_name='probe_mlp',
        #         tb_writer=writer, eval_model=True, savefolder=params.out_dir)
        #     num_probe_params = probe.count_parameters()
        # elif 'tree_ens' in params.probe.type:
        # # if False:
        #     num_probe_params, probe_train_score, probe_val_score, probe_test_score, dci_scores_trees = train_test_random_forest(
        #             model=model, 
        #             dataloader_train=dataloader_train, 
        #             dataloader_val=dataloader_val, 
        #             dataloader_test=dataloader_test, 
        #             method='rf',#params.probe.type,
        #             max_leaf_nodes=params.probe.max_leaf_nodes,
        #             max_depth=params.probe.max_depth,
        #             num_trees=params.probe.num_trees,
        #             seed = params.seed * params.run_id + seed_bias,
        #             lr = params.probe.rf_lr,
        #             epochs = params.probe.epochs, 
        #             log_interval =  params.log_interval, save_model = params.save_model,
        #             train_name='probe_trees',
        #             tb_writer=writer, eval_model=True, savefolder=params.out_dir,
        #             device=device,
        #             data_fraction=params.probe.data_fraction,
        #             use_sage=False
        #             )
            
        #     probe_test_loss = 0
    else:
        if os.path.exists(ckeckpoint_path):
            ckpt = torch.load(ckeckpoint_path)
            probe.load_state_dict(ckpt)
        num_probe_params = probe.count_parameters()

    if probe is not None:
        probe.eval()


    from pathlib import Path
    Path(params.out_dir).mkdir(parents=True, exist_ok=True)

    results = {}
    # num_probe_params = probe.count_parameters()
    id=f'model_{params.name}_type_{params.model_type}_probe_params_{num_probe_params}'
    
    results['id'] = id
    results['model_type'] = params.model_type
    results['num_params'] = num_probe_params
    results['capacity'] = params.probe.hidden_dim
    results['mse'] = probe_test_loss #.cpu().numpy()
    results['val_rsquared_acc'] = probe_val_score#.cpu().numpy()
    results['test_rsquared_acc'] = probe_test_score#.cpu().numpy()
    results['dci_trees'] = dci_scores_trees#.cpu().numpy()

    # results['dci_mlp_val'] = dci_scores_val
    # results['dci_mlp'] = dci_scores
    results['params'] = params
    # with open(f'{params.out_dir}/results.pkl', 'wb') as fp:
    #     pickle.dump(results, fp)

    # json_results = { key : str(val) for key, val in results.items()}

    # with open(f'{params.out_dir}/results.json', 'w') as fp:
    #     json.dump(json_results, fp)
    if True:
        # sage computations might last so we save everything else before
        # evaluate the probes
        if 'MLP' in params.probe.type or 'RFF' in params.probe.type:
            probe, probe_val_loss, probe_val_score, dci_scores_val = evaluate_probe(
                    model=probe,
                    dataloader_test=dataloader_val, 
                    seed = params.seed * params.run_id + seed_bias,
                    log_interval =  params.log_interval,
                    train_name='eval_probe_val',
                    tb_writer=writer,
                    compute_dci=False,
                    group_pixels=sage_group_pixels)
                    
            probe, probe_test_loss, probe_test_score, dci_scores = evaluate_probe(
                    model=probe,
                    dataloader_test=dataloader_test, 
                    seed = params.seed * params.run_id + seed_bias,
                    log_interval =  params.log_interval,
                    train_name='eval_probe_test',
                    tb_writer=writer,
                    compute_dci=False,
                    group_pixels=sage_group_pixels)
        
        results['dci_mlp_val'] = dci_scores_val
        results['dci_mlp'] = dci_scores

        results['val_rsquared_acc'] = probe_val_score
        results['test_rsquared_acc'] = probe_test_score
        print('saving results in {params.out_dir}/results.pkl')

        with open(f'{params.out_dir}/results.pkl', 'wb') as fp:
            pickle.dump(results, fp)

        json_results = { key : str(val) for key, val in results.items()}

        with open(f'{params.out_dir}/results.json', 'w') as fp:
            json.dump(json_results, fp)

if __name__ == "__main__":
    run(parse_opts())