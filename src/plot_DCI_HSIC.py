
from unittest import result
import torch
import pdb
import matplotlib.pyplot as plt
from glob import glob
import os
from loss_capacity.utils import from_pickle
import sys
import numpy as np
from matplotlib.pyplot import cm

# path = './results/2022Aug11-170838_unsup_vae_dsprites'
# x = [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]
path = './results/2022Aug11-170838_unsup_vae_dsprites'
# x = [2, 5, 10, 20, 40]
x = [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]
labels = ['informativeness_train', 'informativeness_val', 'informativeness_test',
        'disentanglement', 'completeness', 'hsic']#with python 3.8 we could have used .split() but here we have to use python 3.7
informativeness_train = []
informativeness_val = []
informativeness_test = []
disentanglement = []
completeness = []

sigma_hsic = [0.01, 0.1, 1, 10, 100, 1000]
hsic = dict((str(el),[]) for el in sigma_hsic)
num_curves = 5 + len(sigma_hsic)

color_key=cm.rainbow(np.linspace(0,1,num_curves))



for folder in glob(path+"/*/*/", recursive=True):
    print(folder)
    
    for file in os.listdir(folder):
        if file.endswith('.pickle') and 'beta' in file:
            results = from_pickle(os.path.join(folder,file))
            informativeness_train.append(results['informativeness_train'])
            informativeness_val.append(results['informativeness_val'])
            informativeness_test.append(results['informativeness_test'])
            disentanglement.append(results['disentanglement'])
            completeness.append(results['completeness'])

    for file in os.listdir(folder):
        if file.endswith('.pickle') and 'hsic' in file:
            results = from_pickle(os.path.join(folder,file))
            for sigma in sigma_hsic:
                hsic[str(sigma)].append(results[str(sigma)])


plt.figure()
for i,(y,label) in enumerate(zip([informativeness_train, informativeness_val, informativeness_test,
            disentanglement, completeness], labels)):
    plt.loglog(x,y, label = label, c=color_key[i])
for j,key in enumerate(hsic.keys()):
    plt.loglog(x,hsic[key], label = 'HSIC sigma='+key, c=color_key[i+j+1])
plt.title('DCI&HSIC')
plt.xlabel('kl_weight')
# plt.xlabel('latent_dim')
plt.legend(loc='best')
plt.savefig(os.path.join(path,'DCI_HSIC2.png'),
                                bbox_inches='tight')