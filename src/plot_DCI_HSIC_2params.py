
# from unittest import result
import torch
import pdb
import matplotlib.pyplot as plt
from glob import glob
import os
from loss_capacity.utils import from_pickle
import sys
import numpy as np
from liftoff import parse_opts
from matplotlib.pyplot import cm
from loss_capacity.functions import HSIC
import matplotlib as mpl
from tqdm import tqdm
import seaborn as sns
from matplotlib.colors import LogNorm

def run(params):
    mpl.rcParams['figure.dpi'] = 120
    mpl.rcParams['savefig.dpi'] = 200

    # To tune ##############################################
    # Whether sigma_x is multiplied by the latent_dim, same for y:
    s_x_latent = params.hsic.x_latent
    s_y_latent = params.hsic.y_latent
    num_seeds = params.num_seeds
    # path = './results/2022Aug11-170838_unsup_vae_dsprites'
    # x = [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]
    path = params.result_path
    score_name = params.score_name
    # x = [0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 100.0, 1000.0]
    sigma_hsic = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    sigma_hsic = [str(a) for a in sigma_hsic]
    # sigma_hsic = [1, 10, 100]

    alpha_list = [0., 100.0, 1000.0, 10000.0, 100000.0, 1000000.0]
    beta_list = [0, 0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]
    #####################################################

    if s_x_latent== True:
        title_text_x =  'xlatent'
    else:
        title_text_x = ''

    if s_y_latent== True:
        title_text_y =  '_ylatent'
    else:
        title_text_y = ''


    if score_name == 'dci':
        labels = ['informativeness_train', 'informativeness_val', 'informativeness_test',
                'disentanglement', 'completeness'] #with python 3.8 we could have used .split() but here we have to use python 3.7
        results_seeds = dict((str(el),np.zeros((len(alpha_list), len(beta_list), num_seeds))) for el in labels)
    elif score_name == 'hsic':
        labels = sigma_hsic #with python 3.8 we could have used .split() but here we have to use python 3.7
        results_seeds = dict((str(el),np.zeros((len(alpha_list), len(beta_list), num_seeds))) for el in labels)
        # if s_x_latent== True:
        #     title_text_x =  'xlatent'
        # else:
        #     title_text_x = ''

        # if s_y_latent== True:
        #     title_text_y =  '_ylatent'
        # else:
        #     title_text_y = ''
        # hsic_reg_version = params.exp_params.hsic_reg_version
        # hsic1= dict((str(el),np.zeros((len(alpha_list), len(beta_list), num_seeds))) for el in sigma_hsic) # train
        # hsic2= dict((str(el),np.zeros((len(alpha_list), len(beta_list), num_seeds))) for el in sigma_hsic) # test
    elif score_name == 'beta':
        labels = ['train_accuracy', 'eval_accuracy'] #with python 3.8 we could have used .split() but here we have to use python 3.7
        results_seeds = dict((str(el),np.zeros((len(alpha_list), len(beta_list), num_seeds))) for el in labels)
    elif score_name == 'factor':
        labels = ['train_accuracy', 'eval_accuracy', 'num_active_dims'] #with python 3.8 we could have used .split() but here we have to use python 3.7
        results_seeds = dict((str(el),np.zeros((len(alpha_list), len(beta_list), num_seeds))) for el in labels)
    
    
    counter = 0
    for folder in tqdm(sorted(glob(path+"/*/", recursive=True))):
        for seed in range(num_seeds):
            seed_folder = folder + str(seed)
            for i,alpha in enumerate(alpha_list):
                for j,beta in enumerate(beta_list):
                    for label in labels:
                        if 'v1' in folder and 'alpha_'+str(alpha) in folder and 'weight_'+str(beta)+'_' in folder:
                            # print(folder)
                            # print('alpha_'+str(alpha), 'weight_'+str(beta))
                            # print(str(alpha), str(beta))
                            for file in os.listdir(seed_folder):
                                file_name = file.split('/')[-1]
                                if score_name == 'hsic':
                                    if file_name.endswith('.pickle') and score_name in file_name and 'div_1000' in file_name:
                                        results_file = from_pickle(os.path.join(seed_folder,file))

                                        results_seeds[str(label)][i,j,seed] = results_file[str(label)]
                                        counter +=1
                                else:
                                    if file_name.endswith('.pickle') and score_name in file_name:
                                        results_file = from_pickle(os.path.join(seed_folder,file))
                                        results_seeds[label][i,j,seed] = results_file[label]
                                        counter +=1

    sns.set_theme()
    for label in labels:
        plt.figure()
        if score_name == 'hsic':
            ax = sns.heatmap(np.mean(results_seeds[label], axis = 2), xticklabels = beta_list, yticklabels=alpha_list, norm=LogNorm())
        elif score_name == 'dci':
            ax = sns.heatmap(np.mean(results_seeds[label], axis = 2), xticklabels = beta_list, yticklabels=alpha_list, vmin=0, vmax=1)
        else:
            ax = sns.heatmap(np.mean(results_seeds[label], axis = 2), xticklabels = beta_list, yticklabels=alpha_list)
        ax.set(xlabel='beta', ylabel='alpha')
        # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        if score_name == 'hsic':
            plt.title(score_name +' '+ label+ title_text_x + title_text_y)
        else:
            plt.title(score_name +' '+label)
        plt.savefig(os.path.join(path, score_name + '_'+label+'_3seeds.png'),
                                        bbox_inches='tight')

if __name__ == "__main__":
    run(parse_opts())
