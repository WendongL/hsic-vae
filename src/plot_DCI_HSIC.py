
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


def adjust_lightness(color, amount=0.5):
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
def run(params):
    mpl.rcParams['figure.dpi'] = 120
    mpl.rcParams['savefig.dpi'] = 200

    # To tune ##############################################
    # Whether sigma_x is multiplied by the latent_dim, same for y:
    s_x_latent = params.hsic.x_latent
    s_y_latent = params.hsic.y_latent

    # path = './results/2022Aug11-170838_unsup_vae_dsprites'
    # x = [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0]
    path = params.result_path
    if 'bottle' in path:
        if 'dsprites' in path:
            x = [2, 5, 6, 7, 10, 20,30]
        else:
            x = [2,6,7,8,10,20,30]
    else:
        if 'dsprites' in path:
            x = [0.1, 0.5, 1.0, 10.0, 100.0, 1000.0, 5000.0]
        else:
            x = [0.001, 0.01, 0.1, 1.0, 10, 20, 100, 1000, 5000]
    # x = [0.001, 0.01, 0.1, 1.0, 10.0, 20.0, 100.0, 1000.0]
    sigma_hsic = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
    num_seeds = 5
    plot_train_hsic = params.plot_train_hsic
    #####################################################
    
    labels = ['informativeness_train', 'informativeness_val', 'informativeness_test',
            'disentanglement', 'completeness'] #with python 3.8 we could have used .split() but here we have to use python 3.7
    if s_x_latent== True:
        title_text_x =  'xlatent'
    else:
        title_text_x = ''

    if s_y_latent== True:
        title_text_y =  '_ylatent'
    else:
        title_text_y = ''

    hsic1= dict((str(el),[]) for el in sigma_hsic) # train
    hsic2= dict((str(el),[]) for el in sigma_hsic) # test
    if plot_train_hsic:
        num_curves = 5 + len(sigma_hsic)*2
    else:
        num_curves = 5 + len(sigma_hsic)

    color_key=cm.rainbow(np.linspace(0,1,num_curves))
    informativeness_train = []
    informativeness_val =[]
    informativeness_test = []
    disentanglement = []
    completeness = []

    for folder in tqdm(sorted(glob(path+"/*/", recursive=True))):
        # print(folder)
        informativeness_train_seeds = torch.zeros(num_seeds)
        informativeness_val_seeds = torch.zeros(num_seeds)
        informativeness_test_seeds = torch.zeros(num_seeds)
        disentanglement_seeds = torch.zeros(num_seeds)
        completeness_seeds = torch.zeros(num_seeds)
        hsic1_seed= dict((str(el),torch.zeros(num_seeds)) for el in sigma_hsic)
        hsic2_seed= dict((str(el),torch.zeros(num_seeds)) for el in sigma_hsic)
        for seed in range(num_seeds):
            for file in os.listdir(folder+str(seed)):
                if file.endswith('.pickle') and 'dci' in file :
                    results = from_pickle(os.path.join(folder+str(seed),file))
                    informativeness_train_seeds[seed]= results['informativeness_train']
                    informativeness_val_seeds[seed]= results['informativeness_val']
                    informativeness_test_seeds[seed]= results['informativeness_test']
                    disentanglement_seeds[seed]= results['disentanglement']
                    completeness_seeds[seed]= results['completeness']

            if s_x_latent:
                if s_y_latent:
                    for file in os.listdir(folder+str(seed)):
                        if file.endswith('_v3.pickle') and 'hsic' in file and "train" in file and 'xlatent' in file and 'ylatent' in file:
                            
                            results = from_pickle(os.path.join(folder+str(seed),file))

                            for sigma in sigma_hsic:
                                hsic1_seed[str(sigma)][seed] = results[str(sigma)]
                    for file in os.listdir(folder+str(seed)):
                        if file.endswith('_v3.pickle')  and 'hsic' in file and "test" in file and 'xlatent' in file and 'ylatent' in file:
                            results = from_pickle(os.path.join(folder+str(seed),file))
                            for sigma in sigma_hsic:
                                hsic2_seed[str(sigma)][seed] = results[str(sigma)]
                else:
                    for file in os.listdir(folder+str(seed)):
                        if file.endswith('_v3.pickle')  and 'hsic' in file and "train" in file and 'xlatent' in file and 'ylatent' not in file:
                            results = from_pickle(os.path.join(folder+str(seed),file))
                            for sigma in sigma_hsic:
                                hsic1_seed[str(sigma)][seed] = results[str(sigma)]
                    for file in os.listdir(folder+str(seed)):
                        if file.endswith('_v3.pickle')  and 'hsic' in file and "test" in file and 'xlatent' in file and 'ylatent' not in file:
                            results = from_pickle(os.path.join(folder+str(seed),file))
                            for sigma in sigma_hsic:
                                hsic2_seed[str(sigma)][seed] = results[str(sigma)]
            else:
                if s_y_latent:
                    for file in os.listdir(folder+str(seed)):
                        if file.endswith('_v3.pickle')  and 'hsic' in file and "train" in file and 'xlatent' not in file and 'ylatent' in file:
                            results = from_pickle(os.path.join(folder+str(seed),file))
                            for sigma in sigma_hsic:
                                hsic1_seed[str(sigma)][seed] = results[str(sigma)]
                    for file in os.listdir(folder+str(seed)):
                        if file.endswith('_v3.pickle')  and 'hsic' in file and "test" in file and 'xlatent' not in file and 'ylatent' in file:
                            results = from_pickle(os.path.join(folder+str(seed),file))
                            for sigma in sigma_hsic:
                                hsic2_seed[str(sigma)][seed] = results[str(sigma)]
                else:
                    for file in os.listdir(folder+str(seed)):
                        if file.endswith('_v3.pickle')  and 'hsic' in file and "train" in file and 'xlatent' not in file and 'ylatent' not in file:
                            results = from_pickle(os.path.join(folder+str(seed),file))
                            
                            for sigma in sigma_hsic:
                                hsic1_seed[str(sigma)][seed] = results[str(sigma)]
                    for file in os.listdir(folder+str(seed)):
                        if file.endswith('_v3.pickle')  and 'hsic' in file and "test" in file and 'xlatent' not in file and 'ylatent' not in file:
                            results = from_pickle(os.path.join(folder+str(seed),file))
                            
                            for sigma in sigma_hsic:
                                hsic2_seed[str(sigma)][seed] = results[str(sigma)]

        informativeness_train.append(torch.std_mean(informativeness_train_seeds))
        informativeness_val.append(torch.std_mean(informativeness_val_seeds))
        informativeness_test.append(torch.std_mean(informativeness_test_seeds))
        disentanglement.append(torch.std_mean(disentanglement_seeds))
        completeness.append(torch.std_mean(completeness_seeds))
        for sigma in sigma_hsic:
            hsic1[str(sigma)].append(torch.std_mean(hsic1_seed[str(sigma)]))
            hsic2[str(sigma)].append(torch.std_mean(hsic2_seed[str(sigma)]))
    print(hsic2)
    informativeness_train = np.array(informativeness_train)
    informativeness_val = np.array(informativeness_val)
    informativeness_test = np.array(informativeness_test)
    disentanglement = np.array(disentanglement)
    completeness = np.array(completeness)
    for sigma in sigma_hsic:
        hsic1[str(sigma)] = np.array(hsic1[str(sigma)])
        hsic2[str(sigma)] = np.array(hsic2[str(sigma)])

    if 'bottle' in path:
        plot_fig = plt.semilogy
    else:
        plot_fig = plt.loglog
    # pdb.set_trace()
    plt.figure()
    for i,(y,label) in enumerate(zip([informativeness_train, informativeness_val, informativeness_test,
                disentanglement, completeness], labels)):
        print(y.shape)
        plot_fig(x,y[:,1], label = label, c=color_key[i])
        plt.fill_between(x, y[:,1]-y[:,0], y[:,1]+y[:,0], color=adjust_lightness(color_key[i],amount=0.5), alpha=0.3)
    if plot_train_hsic:
        for j,key in enumerate(hsic1.keys()):
            plot_fig(x,hsic1[key][:,1], label = r'HSIC_train $\sigma^2$='+key, c=color_key[i+j+1])
            plt.fill_between(x, hsic1[key][:,1]-hsic1[key][:,0], hsic1[key][:,1]+hsic1[key][:,0], color=adjust_lightness(color_key[i+j+1],amount=0.5),alpha=0.3)
        for k,key in enumerate(hsic2.keys()):
            plot_fig(x,hsic2[key][:,1], label = r'HSIC_test $\sigma^2$='+key, c=color_key[i+j+k+1])
            plt.fill_between(x, hsic2[key][:,1]-hsic2[key][:,0], hsic2[key][:,1]+hsic2[key][:,0], color=adjust_lightness(color_key[i+j+k+1],amount=0.5),alpha=0.3)
    else:
        for k,key in enumerate(hsic2.keys()):
            plot_fig(x,hsic2[key][:,1], label = r'HSIC_test $\sigma^2$='+key, c=color_key[i+k+1])
            plt.fill_between(x, hsic2[key][:,1]-hsic2[key][:,0], hsic2[key][:,1]+hsic2[key][:,0], color=adjust_lightness(color_key[i+k+1],amount=0.5),alpha=0.3)
    
    if 'bottle' in path:
        plt.xlabel('latent_dim')
    else:
        plt.xlabel('kld_weight')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.title(r'DCI&HSIC' + title_text_x + title_text_y)
    if plot_train_hsic:
        plt.savefig(os.path.join(path,'DCI_HSIC_train&test_num_reparam10_div1000'+ title_text_x + title_text_y +'_v3.png'),
                                    bbox_inches='tight')
    else:
        plt.savefig(os.path.join(path,'DCI_HSIC_test_num_reparam10_div1000'+ title_text_x + title_text_y +'_v3.png'),
                                    bbox_inches='tight')

if __name__ == "__main__":
    run(parse_opts())
