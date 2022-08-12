
import numpy as np
import torch
import pdb
import os
# from sklearn.ensemble import GradientBoostingClassifier
import joblib
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from metrics.dci import disentanglement, completeness
import sage
from loss_capacity.models import ConvNet, NoisyLabels, LinearMixLabels, RawData, ResNet18
from loss_capacity.utils import RSquaredPytorch, MetricsPytorch #, MetricsPytorchL2

import xgboost as xgb
from loss_capacity.utils import get_representations_split
import time


def get_linear_least_squares_search(
        model, dataloader_train, dataloader_val, dataloader_test,
        method='tree_ens_hgb',
        max_leaf_nodes = 200,
        max_depth=20,
        num_trees=100,
        seed = 0, lr = 0.001, epochs = 100, log_interval = 50, save_model = True,
        train_name='model', 
        tb_writer=None, eval_model=False,
        savefolder='./',
        device=None,
        data_fraction=1.0):

    print(f'learn {method} probes')
    print(f'Use trees with max depth: {max_depth}. max_leaf_nodes: {max_leaf_nodes}. num_trees: {num_trees}.')
    if  isinstance(data_fraction, int):
        num_train   = data_fraction
    else:
        num_train   = int(data_fraction * len(dataloader_train.dataset))
    print(f'Train tree ensemble on: {num_train} samples')

    print('WARNING: using 10k samples for test')
    num_val     = 10000 # len(dataloader_val.dataset)
    # TODO: use the whole test dataset for eval
    num_test    = 10000 # len(dataloader_test.dataset)






    factor_sizes    = dataloader_train.dataset._factor_sizes
    factor_discrete = dataloader_train.dataset._factor_discrete
    factor_discrete = dataloader_train.dataset._factor_discrete
    factor_names    = dataloader_train.dataset._factor_names
    num_factors = len(factor_sizes)

    # all_rsquared_train = np.zeros(num_factors)
    # all_rsquared_test = np.zeros(num_factors)
    # all_rsquared_val = np.zeros(num_factors)

    # eps = 0.00001
    # time_start = time.time()
    # R = np.zeros((train_h.shape[1], num_factors))
    # R_sage = np.zeros((train_h.shape[1], num_factors))

    # folder for saving probe models
    # noise_stds = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10]
    noise_stds = [10 ** i for i in range(-5,2)]
    noise_stds = [1.0, 0.5, 0.1, 0.09, 0.08, 0.07, 0.06, 0.05, 0.03, 0.01]

    for noise_i, noise_std  in enumerate(noise_stds):

        # model = NoisyLabels(
        #         number_of_classes=num_factors, 
        #         noise_std=noise_std
        #     )

        model = LinearMixLabels(
                    number_of_classes=num_factors, 
                    noise_std=noise_std,
                    mix_type='almost_uniform_mix_labels'
        )
        model = model.to(device)


        # TODO: we should use the same training samples...
        train_h, train_targets = get_representations_split(model, dataloader_train.dataset, device, len(dataloader_train.dataset))
        val_h, val_targets = get_representations_split(model, dataloader_val.dataset, device, num_val)
        test_h, test_targets = get_representations_split(model, dataloader_test.dataset, device, num_test)
        rs = RSquaredPytorch(dataloader_test)


        R_cont_lstsq = []
        R_cont_sage = []
        all_predictions = []
        all_targers = []
        all_mse = []
        for ind_fact in range(num_factors):
            if True:
                # regression
                train_t = train_targets[:, ind_fact]
                # val_t   = val_targets[:, ind_fact]
                test_t  = test_targets[:, ind_fact]

                X = train_h
                Y =  train_t

                out = np.linalg.lstsq(X,Y)
                w = out[0]
                model = lambda x : x @ w

                pred = test_h @ w
                all_predictions.append(pred)
                mse = (pred - test_t) ** 2
                mse = mse.mean()
                all_mse.append(mse)

                R_cont_lstsq.append(np.abs(w))  


                # Setup and calculate
                # if mse.mean() < 0.000001:
                if True:
                    imputer = sage.MarginalImputer(model, test_h[:512])
                    estimator = sage.PermutationEstimator(imputer, 'mse')
                    sage_values = estimator(test_h, test_t, batch_size=128, thresh=0.05)
                    R_cont_sage.append(np.abs(sage_values.values))  
                else:
                    R_cont_sage.append(np.zeros_like(w))  


        all_mse = np.stack(all_mse,axis=0)
        all_predictions = np.stack(all_predictions, axis=1)
        rs.reset()
        rs.acum_stats(torch.Tensor(all_predictions), torch.Tensor(test_targets))
        results = rs.get_scores()
        
        R_cont_lstsq = np.stack(R_cont_lstsq,axis=0)
        R_cont_sage = np.stack(R_cont_sage,axis=0)

        d_lstsq = disentanglement(R_cont_lstsq)
        c_lstsq = completeness(R_cont_lstsq)

        d_sage = disentanglement(R_cont_sage)
        c_sage = completeness(R_cont_sage)

        print(f'noise std: {noise_std}')
        print(f'mse: {all_mse.mean()}')
        print(f'DC lstsq: {d_lstsq} , {c_lstsq}')
        print(f'DC sage: {d_sage} , {c_sage}')

        print(results)


    return 0




def get_linear_least_squares(
        model, dataloader_train, dataloader_val, dataloader_test,
        method='tree_ens_hgb',
        max_leaf_nodes = 200,
        max_depth=20,
        num_trees=100,
        seed = 0, lr = 0.001, epochs = 100, log_interval = 50, save_model = True,
        train_name='model', 
        tb_writer=None, eval_model=False,
        savefolder='./',
        device=None,
        data_fraction=1.0):

    print(f'learn {method} probes')
    print(f'Use trees with max depth: {max_depth}. max_leaf_nodes: {max_leaf_nodes}. num_trees: {num_trees}.')
    if  isinstance(data_fraction, int):
        num_train   = data_fraction
    else:
        num_train   = int(data_fraction * len(dataloader_train.dataset))
    print(f'Train tree ensemble on: {num_train} samples')

    print('WARNING: using 10k samples for test')
    num_val     = 10000 # len(dataloader_val.dataset)
    # TODO: use the whole test dataset for eval
    num_test    = 10000 # len(dataloader_test.dataset)






    factor_sizes    = dataloader_train.dataset._factor_sizes
    factor_discrete = dataloader_train.dataset._factor_discrete
    factor_discrete = dataloader_train.dataset._factor_discrete
    factor_names    = dataloader_train.dataset._factor_names
    num_factors = len(factor_sizes)


    # folder for saving probe models
    # noise_stds = [0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0, 10]
    noise_stds = [10 ** i for i in range(-5,2)]
    # noise_stds = [0.1, 1.0, 10]





    # TODO: we should use the same training samples...
    train_h, train_targets = get_representations_split(model, dataloader_train.dataset, device, len(dataloader_train.dataset))
    val_h, val_targets = get_representations_split(model, dataloader_val.dataset, device, num_val)
    test_h, test_targets = get_representations_split(model, dataloader_test.dataset, device, num_test)
    rs = RSquaredPytorch(dataloader_test)


    R_cont_lstsq = []
    R_cont_sage = []
    all_predictions = []
    all_targers = []
    all_mse = []
    for ind_fact in range(num_factors):
        if True:
            # regression
            train_t = train_targets[:, ind_fact]
            # val_t   = val_targets[:, ind_fact]
            test_t  = test_targets[:, ind_fact]

            X = train_h
            Y =  train_t
            # X = (X - X.mean(0)) 
            # X = X / X.std(0)

            # X = X - X.mean(1, keepdims=True)
            # X = X / X.std(1, keepdims=True)

            # X = X / X.sum(1, keepdims=True)
            out = np.linalg.lstsq(X,Y)
            print(f'[{ind_fact}] X.mean(0): {X.mean(0)}')
            print(f'[{ind_fact}] Y.mean(0): {Y.mean(0)}')
            
            w = out[0]
            model = lambda x : x @ w

            pred = test_h @ w
            all_predictions.append(pred)
            mse = (pred - test_t) ** 2
            mse = mse.mean()
            all_mse.append(mse)

            R_cont_lstsq.append(np.abs(w))  
            print(f'[{ind_fact}] importance: {np.abs(w)}')

            # Setup and calculate
            # if mse.mean() < 0.000001:
            if False:
                imputer = sage.MarginalImputer(model, test_h[:512])
                estimator = sage.PermutationEstimator(imputer, 'mse')
                sage_values = estimator(test_h, test_t, batch_size=128, thresh=0.05)
                R_cont_sage.append(np.abs(sage_values.values))  
            else:
                R_cont_sage.append(np.zeros_like(w))  
            print(f'[{ind_fact}] sage_importance: {R_cont_sage[-1]}')
            
            

    all_mse = np.stack(all_mse,axis=0)
    all_predictions = np.stack(all_predictions, axis=1)
    rs.reset()
    rs.acum_stats(torch.Tensor(all_predictions), torch.Tensor(test_targets))
    results = rs.get_scores()
    
    R_cont_lstsq = np.stack(R_cont_lstsq,axis=0)
    R_cont_sage = np.stack(R_cont_sage,axis=0)

    d_lstsq = disentanglement(R_cont_lstsq)
    c_lstsq = completeness(R_cont_lstsq)

    d_sage = disentanglement(R_cont_sage)
    c_sage = completeness(R_cont_sage)

    # print(f'noise std: {noise_std}')
    print(f'mse: {all_mse.mean()}')
    print(f'DC lstsq: {d_lstsq} , {c_lstsq}')
    print(f'DC sage: {d_sage} , {c_sage}')

    print(results)


    return 0

