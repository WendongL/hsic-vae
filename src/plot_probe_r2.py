
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


sns.set()
mpl.rcParams['figure.dpi'] = 120
mpl.rcParams['savefig.dpi'] = 200

# To tune ##############################################
# path = './results/2022Aug11-170838_unsup_vae_dsprites'
# x = [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]
path = './results/2022Oct19-160510_vae_probes_mlp'
num_seeds = 3
dataset_name = 'dsprites'
#####################################################


variant_list = ['random', 'composition', 'interpolation', 'extrapolation']
if dataset_name == 'dsprites':
    factor_names = [
                'shape', 'scale', 'orientation', 'x-position', 'y-position'
            ]
elif dataset_name == 'shapes3d':
    factor_names = [
                'floor color', 'wall color', 'object color', 'object size',
                'object type', 'azimuth'
            ]
elif dataset_name == 'mpi3d':
    factor_names = [
                'color', 'shape', 'size', 'height', 'bg color', 'x-axis',
                'y-axis'
            ]
elif dataset_name == 'cars3d':
    factor_names = [
                'elevation', 'azimuth', 'object'
            ]

x = np.arange(len(variant_list))  # the label locations
width = 0.5  # the width of the bars
alpha_list = [0,10000,100000]
for i, name in enumerate(factor_names):
    r2 = {}
    plt.figure()
    fig, ax = plt.subplots()
    r2 = {str(alpha): np.zeros(len(variant_list)) for alpha in alpha_list}
    
    for alpha in alpha_list:
        print(alpha)
        for j,variant in enumerate(variant_list):
            print(variant)
            r2_seeds = np.zeros(num_seeds)
            for seed in range(num_seeds):
                print(seed)
                for folder in sorted(glob(path+"/*/", recursive=True)):
                    if variant in folder and 'alpha_'+str(alpha)+'.0__' in folder and 'run_id_'+str(seed) in folder:
                        # print(variant, alpha, seed, folder)
                        results = from_pickle(os.path.join(folder+'0','results.pkl'))
                        # print('test_rsquared_acc', results['test_rsquared_acc'])
                        r2_seeds[seed]= results['test_rsquared_acc'][i]
                        print(r2_seeds[seed])
            r2[str(alpha)][j]=np.mean(r2_seeds)
        # pdb.set_trace()
        
        # print('r2',r2)
        
    rects1 = ax.bar(x - width/2, r2['0'], width/3, label='alpha = 0')
    rects2 = ax.bar(x - width/6, r2['10000'], width/3, label='alpha = 10000')
    rects3 = ax.bar(x + width/6, r2['100000'], width/3, label='alpha = 100000')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel(r'$R^2$')
    ax.set_title(name + r' $R^2$ over test set in different OOD setting')
    ax.set_xticks(x, variant_list)
    ax.legend(loc='lower right')
    # plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.bar_label(rects1, labels=[f'{x:,.2f}' for x in rects1.datavalues], padding=2, size = 8)
    ax.bar_label(rects2, labels=[f'{x:,.2f}' for x in rects2.datavalues], padding=2, size = 8)
    ax.bar_label(rects3, labels=[f'{x:,.2f}' for x in rects3.datavalues], padding=2, size = 8)

    fig.tight_layout()


    
    plt.savefig(os.path.join(path, name+'_r2.png'), bbox_inches='tight')
    plt.close()


