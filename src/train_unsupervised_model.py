import sys
sys.path.insert(0, './../')
import os
import yaml
import argparse
import numpy as np
from pathlib import Path
from PyTorchVAE.models.beta_vae import BetaVAE, SmallBetaVAE
from PyTorchVAE.models.hsicbeta_vae import HsicBetaVAE, SmallHsicBetaVAE
from PyTorchVAE.experiment import VAEXperiment
from PyTorchVAE.experiment_hsicbeta import VAEXperiment_hsicbeta
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
from loss_capacity.utils import list2tuple_, config_to_string, save_representation_dataset, get_representations_data_split
# from loss_capacity.probing import Probe
from loss_capacity.functions import HSIC

import timm

import torch
import pickle
import json
import pdb


def run(params):
    """ Entry point for liftoff. """
    
    params = list2tuple_(params)
    params.num_workers = 2
    print(config_to_string(params))

    config = params.vae
    if 'none' in  str(params.probe.max_leaf_nodes) or 'None'  in  str(params.probe.max_leaf_nodes):
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
        experiment = VAEXperiment(model, config.exp_params)
    elif config.model_params.model_type == 'big':
        model = BetaVAE(**config.model_params.__dict__)
        experiment = VAEXperiment(model, config.exp_params)
    elif config.model_params.model_type == 'smallhsicbeta':
        model = SmallHsicBetaVAE(**config.model_params.__dict__)
        experiment = VAEXperiment_hsicbeta(model, config.exp_params)
    elif config.model_params.model_type == 'bighsicbeta':
        experiment = VAEXperiment_hsicbeta(model, config.exp_params)
    
    

    # if restore:
    #     ckpt = params.out_dir + '/BetaVAE/version_0/checkpoints/last.ckpt'
    #     print(f'loading from: {ckpt}')
    #     lightning_ckpt = torch.load(ckpt)
    #     experiment.load_state_dict(lightning_ckpt['state_dict'])


    # number_of_channels = 1 if params.dataset == 'dsprites' else 3
    # print(f'number_of_channels: {number_of_channels}')
    imagenet_normalise = True if 'resnet' in params.model_type else False
    dataloader_train = load_dataset.load_dataset(
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
    
    data = DisentDatasets(
        train_batch_size=config.data_params.train_batch_size,
        val_batch_size= config.data_params.val_batch_size,
        test_batch_size= config.data_params.val_batch_size,
        train_dataset=dataloader_train.dataset,
        val_dataset=dataloader_val.dataset,
        test_dataset=dataloader_test.dataset,
        num_workers=config.data_params.num_workers,
        pin_memory=len(config.trainer_params.gpus) != 0)

    data.setup()
    savename = params.out_dir + f'/{params.dataset}_{params.model_type}_{params.name}'
    checkpoint_path = savename + '.pt'
    runner = Trainer(logger=tb_logger,
                    callbacks=[
                        LearningRateMonitor(),
                        ModelCheckpoint(save_top_k=2, 
                                        dirpath =os.path.join(tb_logger.log_dir , "checkpoints"), 
                                        monitor= "val_loss",
                                        save_last= True),
                    ],
                    strategy=DDPPlugin(find_unused_parameters=False),
                    **config.trainer_params.__dict__)

    for stage in ['val', 'test']:
        Path(f"{tb_logger.log_dir}/Samples_{stage}").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/Reconstructions_{stage}").mkdir(exist_ok=True, parents=True)
        Path(f"{tb_logger.log_dir}/InputImages_{stage}").mkdir(exist_ok=True, parents=True)
    
    if not os.path.exists(checkpoint_path):

        print(f"======= Training {config.model_params.name} =======")
        runner.fit(experiment, datamodule=data)
        
        runner.validate(experiment, datamodule=data)# TODO: maybe use trainer.test(ckpt_path='best')
        runner.test(experiment, datamodule=data)

        model.eval()

        Path(params.out_dir).mkdir(parents=True, exist_ok=True)  
        torch.save(model.state_dict(), savename +'.pt')
    else:
        ckpt = torch.load(checkpoint_path)
        model.load_state_dict(ckpt)
        model.eval()
        runner.validate(experiment, datamodule=data)
        runner.test(experiment, datamodule=data)

        
    if params.save_representation_datasets:
        device = 'cuda'
        model = model.to(device)

        # w_save_representation_dataset(device, model, dataloader_train.dataset, f'{savename}_dataset_train')
        # w_save_representation_dataset(device, model, dataloader_test.dataset, f'{savename}_dataset_test')
        # w_save_representation_dataset(device, model, dataloader_val.dataset, f'{savename}_dataset_val')
        from  loss_capacity.train_model_random_forest import train_test_random_forest
        # pdb.set_trace()
        num_probe_params, probe_train_score, probe_val_score, probe_test_score, dci_scores_trees = train_test_random_forest(
                model=model, 
                dataloader_train=dataloader_train, 
                dataloader_val=dataloader_val, 
                dataloader_test=dataloader_test, 
                method='rf',#params.probe.type,
                max_leaf_nodes=params.probe.max_leaf_nodes,
                max_depth=params.probe.max_depth,
                num_trees=params.probe.num_trees,
                seed = params.seed * params.run_id,
                lr = params.probe.rf_lr,
                epochs = params.probe.epochs, 
                log_interval =  params.log_interval, save_model = params.save_model,
                train_name='probe_trees',
                tb_writer=None, eval_model=True, savefolder=params.out_dir,
                device=device,
                data_fraction=params.probe.data_fraction,
                use_sage=False
                )

        
        with open(os.path.join(params.out_dir, f'dci_klw{config.exp_params.kld_weight}_latent_{config.model_params.latent_dim}.pickle'), 'wb') as f:
            pickle.dump(dci_scores_trees, f)
# def w_save_representation_dataset(device, model, dataset, path):
#     pdb.set_trace()
#     print(f'saving representation dataset at: {path}')
#     batch_size = 256
#     dataloader = torch.utils.data.DataLoader(dataset,
#         batch_size=batch_size,
#         shuffle=False,
#         num_workers=4)

#     inputs, targets = next(iter(dataloader))
#     feats = model.encode(inputs.to(device))
#     # outputs = model.decode(feats)


# For debug of saving models
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
#     torch.save(model.state_dict(), savename +'.pt')

if __name__ == "__main__":
    run(parse_opts())
#     all_feats = np.zeros((len(dataset), feats.shape[1])).astype(np.float32)
#     all_targets = np.zeros((len(dataset), targets.shape[1])).astype(np.float32)
#     print(f'dataloader len: {len(dataset)}')
#     data = []
#     for idx, (data, target) in enumerate(dataloader):
#         data = data.to(device)
#         output = model.encode(data)
#         output = output.detach().cpu().numpy()
#         target = target.detach().cpu().numpy()
        
#         all_feats[idx * batch_size: (idx + 1 ) * batch_size] = output
#         all_targets[idx * batch_size: (idx + 1 ) * batch_size] = target

#     # TODO: replace prev code by call to get_representations_split
#     # should we create the array at the begening??
#     # np.save(path + '_feats.npy', all_feats)
#     # np.save(path + '_targets.npy', all_targets)
# #     return all_feats, all_targets

# def get_DCI():

#     train_h, train_targets = w_save_representation_dataset(model, dataloader_train.dataset, device, len(dataloader_train.dataset))
#     val_h, val_targets = w_save_representation_dataset(model, dataloader_val.dataset, device, num_val)
#     test_h, test_targets = w_save_representation_dataset(model, dataloader_test.dataset, device, num_test)


#     factor_sizes    = dataset._factor_sizes
#     factor_discrete = dataset._factor_discrete
#     factor_discrete = dataset._factor_discrete
#     factor_names    = dataset._factor_names
#     num_factors = len(factor_sizes)

#     all_rsquared_train = np.zeros(num_factors)
#     all_rsquared_test = np.zeros(num_factors)
#     all_rsquared_val = np.zeros(num_factors)

#     eps = 0.00001
#     # time_start = time.time()
#     R = np.zeros((train_h.shape[1], num_factors))
#     R_sage = np.zeros((train_h.shape[1], num_factors))

#     # folder for saving probe models
#     Path(savefolder+'/rf_models/').mkdir(parents=True, exist_ok=True)
#     resume = True
#     # use_sage = False
#     for ind_fact in range(num_factors):
#         savefile = f"{savefolder}/rf_models/rf_datasize_{num_train}_factor_{ind_fact}.joblib"
#         savefile_sage = f"{savefolder}/rf_models/sage_rf_datasize_{num_train}_factor_{ind_fact}.joblib"

#         if factor_discrete[ind_fact]:
#             #classification
#             train_t = (train_targets[:,ind_fact] * (factor_sizes[ind_fact] - 1) + eps).astype(np.int32)
#             val_t   = (val_targets[:,ind_fact] * (factor_sizes[ind_fact] - 1) + eps).astype(np.int32)
#             test_t  = (test_targets[:,ind_fact] * (factor_sizes[ind_fact] - 1) + eps).astype(np.int32)
#             if resume and os.path.exists(savefile):
#                 print(f'Load model from: {savefile}')
#                 model = joblib.load(savefile) 
#             else:
#                 if 'xgb' in method:
#                     model = xgb.XGBClassifier(verbosity=2,
#                                 max_depth=max_depth)
#                 elif 'hgb' in method:
#                     model = HistGradientBoostingClassifier(verbose=1,
#                                 max_leaf_nodes=max_leaf_nodes,
#                                 max_depth=max_depth,
#                                 learning_rate=lr)
#                 elif 'rf' in method:
#                     model = RandomForestClassifier(verbose=1,
#                                 max_leaf_nodes=max_leaf_nodes,
#                                 max_depth=max_depth,
#                                 n_estimators=num_trees,
#                                 random_state=seed,
#                                 n_jobs=-1)
#                 model.fit(train_h, train_t)
#             # for classification this computes the mean accuracy
#             all_rsquared_train[ind_fact]  = model.score(train_h, train_t)
#             all_rsquared_val[ind_fact]  = model.score(val_h, val_t)
#             all_rsquared_test[ind_fact]  = model.score(test_h, test_t)
#             R[:,ind_fact] = np.abs(model.feature_importances_)  

#             if use_sage:
#                 # Setup and calculate
#                 imputer = sage.MarginalImputer(model.predict_proba, test_h[:sage_num_samples])
#                 estimator = sage.PermutationEstimator(imputer, 'cross entropy')
#                 sage_values = estimator(test_h, test_t, batch_size=128, thresh=sage_thresh)
#                 sage_vals = np.abs(sage_values.values)  # TODO: these are not normalised. solved
#                 sage_vals = sage_vals / sage_vals.sum()
#                 R_sage[:,ind_fact] = sage_vals

#         else:
#             # regression
#             train_t = train_targets[:, ind_fact]
#             val_t   = val_targets[:, ind_fact]
#             test_t  = test_targets[:, ind_fact]
#             if resume and os.path.exists(savefile):
#                 print(f'Load model from: {savefile}')
#                 model = joblib.load(savefile) 
#             else:
#                 if 'xgb' in method :
#                     model = xgb.XGBRegressor(verbose=1,
#                                 max_depth=max_depth)
#                 elif 'hgb' in method:
#                     model = HistGradientBoostingRegressor(verbose=1, 
#                                 max_leaf_nodes=max_leaf_nodes,
#                                 max_depth=max_depth,
#                                 learning_rate=lr)
#                 elif 'rf' in method:
#                     model = RandomForestRegressor(verbose=1,
#                                 max_leaf_nodes=max_leaf_nodes,
#                                 max_depth=max_depth,
#                                 n_estimators=num_trees,
#                                 random_state=seed,
#                                 n_jobs=-1)
                
#                 model.fit(train_h, train_t)
            
#             # TODO: we should make sure that we use the same rsquared here and elsewhere
#             # Q: here R2 is normalised by the variance on this set, while RSquaredPytorch the 
#             # variance over the whole dataset is used..
#             all_rsquared_train[ind_fact]  = model.score(train_h, train_t)
#             all_rsquared_val[ind_fact]  = model.score(val_h, val_t)
#             all_rsquared_test[ind_fact]  = model.score(test_h, test_t)
#             R[:,ind_fact] = np.abs(model.feature_importances_)

#             if use_sage:
#                 # Setup and calculate
#                 imputer = sage.MarginalImputer(model.predict, test_h[:sage_num_samples])
#                 estimator = sage.PermutationEstimator(imputer, 'mse')
#                 sage_values = estimator(test_h, test_t, batch_size=128, thresh=sage_thresh)
#                 sage_vals = np.abs(sage_values.values)  
#                 sage_vals = sage_vals / sage_vals.sum()
#                 R_sage[:,ind_fact] = sage_vals
#         # if factor_names[ind_fact] == 'height':
#         #     import imageio
#         #     imp = R[:,ind_fact]
#         #     imp = imp.reshape(3,64,64).transpose(1,2,0)
#         #     imageio.imwrite('tmp_asd.jpg', imp)

#         #     input_img = train_h[0].reshape(3,64,64).transpose(1,2,0)
#         #     ii = 1
#         #     imageio.imwrite(f'input_img_{ii}.png', train_h[ii].reshape(3,64,64).transpose(1,2,0))



#         # save
#         if not (resume and os.path.exists(savefile)):
#             joblib.dump(model, savefile)

#         # clf = load('filename.joblib') 
        
#     total_time = time.time() - time_start

#     print(f'\nTree ensemble train scores mean: {all_rsquared_train.mean()}')
#     for i, name in enumerate(factor_names):
#         print(f'{name}: {all_rsquared_train[i]}')

#     print(f'\nTree ensemble validation scores mean: {all_rsquared_val.mean()}')
#     for i, name in enumerate(factor_names):
#         print(f'{name}: {all_rsquared_val[i]}')
    
#     print(f'\nTree ensemble test scores mean: {all_rsquared_test.mean()}')
#     for i, name in enumerate(factor_names):
#         print(f'{name}: {all_rsquared_val[i]}')

#     capacity = max_depth if max_depth is not None else -10#max_leaf_nodes 

#     for i in range(10):
#         tb_writer.add_scalar(f'{train_name}-rsquared_acc/train', all_rsquared_train.mean(), capacity+i)
#         tb_writer.add_scalar(f'{train_name}-rsquared_acc/val', all_rsquared_val.mean(), capacity+i)
#         tb_writer.add_scalar(f'{train_name}-rsquared_acc/test', all_rsquared_test.mean(), capacity+i)
#         tb_writer.add_scalar(f'{train_name}-time/time_training',total_time, capacity+i)
#         for j in range(len(all_rsquared_val)):
#             tb_writer.add_scalar(f'{train_name}-scores/val_{factor_names[j]}',all_rsquared_val[j], capacity+i)
#             tb_writer.add_scalar(f'{train_name}-scores/test_{factor_names[j]}',all_rsquared_test[j], capacity+i)


#     dci_scores = {}
#     dci_scores["informativeness_train"] = all_rsquared_train.mean()
#     dci_scores["informativeness_val"] = all_rsquared_val.mean()
#     dci_scores["informativeness_test"] = all_rsquared_test.mean()
#     dci_scores["disentanglement"] = disentanglement(R)
#     dci_scores["completeness"] = completeness(R)

#     if use_sage:
#         dci_scores["sage_disentanglement"] = disentanglement(R_sage)
#         dci_scores["sage_completeness"] = completeness(R_sage)


#     for name, score in dci_scores.items():
#         print(f'DCI: {name}: {score} ')
#         for i in range(10):
#             tb_writer.add_scalar(f'{train_name}-scores/DCI_trees_ens_{name}', score, max_depth)

#     print(f'sage: sage_num_samples: {sage_num_samples}, sage_thresh: {sage_thresh}')

#     return capacity, all_rsquared_train, all_rsquared_val, all_rsquared_test, dci_scores
