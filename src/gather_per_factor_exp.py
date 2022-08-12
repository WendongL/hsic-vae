
import glob
import pickle
import os.path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdb
import sys
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
sys.path.insert(0, './../')
from metrics.explitcitness import compute_explitcitness, Explitcitness

sns.set_theme()

dataset = 'cars3d' #  'mpi3d' /  'cars3d' / 'dsprites'
plot_probe =  'rf' # 'mlp' / 'rf' / 'rff'
plot_errorbars = True


### MPI3D
# MLP
if dataset == 'mpi3d':
    if plot_probe == 'mlp':
        path_folders = '/home/anicolicioiu/data/models/results_uai/results_plot1_ind/*/*/*'
    elif plot_probe == 'rf':
        path_folders = '/home/anicolicioiu/data/models/results_uai/results_plot1_rf/*/*ens_rf*fraction_1.0*/*'
    elif plot_probe == 'rff':
        path_folders = '/home/anicolicioiu/data/models/results_rff/results_rff/*/*/*'
elif dataset == 'cars3d':
    if plot_probe == 'mlp':
        path_folders = '/home/anicolicioiu/data/models/results_cars3d/results_cars3/*/*/*'
    elif plot_probe == 'rf':
        path_folders = '/home/anicolicioiu/data/models/results_cars3d/results_cars_rf/*/*/*'
    elif plot_probe == 'rff':
        path_folders = '/home/anicolicioiu/data/models/results_cars3d/results_cars_rff/*/*/*'
elif dataset == 'dsprites':
    if plot_probe == 'mlp':
        path_folders = '/home/anicolicioiu/data/models/results_dsprites/results_dsprites/*/*/*'
    elif plot_probe == 'rf':
        path_folders = '/home/anicolicioiu/data/models/results_cars3d/results_cars_rf/*/*/*'
    elif plot_probe == 'rff':
        path_folders = '/home/anicolicioiu/data/models/results_cars3d/results_cars_rff/*/*/*'


std_type = 'conf95' # 'conf95' /  'std1'



if plot_probe == 'rf':
    ## rf
    my_plot_title = 'Random Forest Probes'
    div_capacity = 1
    use_predefined_colors = True
    plot_type = 'normal'
    bottom=-0.02
    top = 0.8
    plot_name = f'__3loss_rf_uai_excess' 
    capacity_measure = f'Max. Tree Depth'
    folders = glob.glob(path_folders)
    use_zoombox = False
elif plot_probe == 'mlp':
    ## mlp
    use_zoombox = False
    my_plot_title = 'MLP Probes'
    div_capacity = 7
    use_predefined_colors = True
    # use_predefined_colors = False
    plot_type = 'excess' #normal / excess / ratio #True
    eps_params = 1
    bottom = - 0.03
    top = 0.7
    # plot_name = f'per_factor_vae_log_loss_mlp_uai_excess_{eps_params}' 
    # plot_name = f'per_factor_log_loss_mlp_uai_excess_{eps_params}' 
    plot_name = f'individual_{plot_type}_loss_mlp_{eps_params}' 

    capacity_measure = f'log(Total #Params - Linear #Params + {eps_params})'
    # folders = glob.glob("./results/results_plot1/*/*/*")
    # folders = glob.glob("./results/results_plot1/vae_beta1/000*/*")

    folders = glob.glob(path_folders)
elif plot_probe == 'rff':
    use_zoombox = False
    my_plot_title = 'RFF Probes'
    div_capacity = 7
    use_predefined_colors = True
    # use_predefined_colors = False
    plot_type = 'normal' #normal / excess / ratio #True
    eps_params = 1
    bottom = - 0.03
    top = 0.7
    # plot_name = f'per_factor_vae_log_loss_mlp_uai_excess_{eps_params}' 
    # plot_name = f'per_factor_log_loss_mlp_uai_excess_{eps_params}' 
    plot_name = f'individual_{plot_type}_loss_RFF_normal_{eps_params}' 

    capacity_measure = f'log(Total #Params - Linear #Params + {eps_params})'
    # folders = glob.glob("./results/results_plot1/*/*/*")
    # folders = glob.glob("./results/results_plot1/vae_beta1/000*/*")

    folders = glob.glob(path_folders)


if plot_probe == 'rf':
    logscale=False
else:
    logscale=False



num_seeds = 3

# [6, 6, 2, 3, 3, 40, 40]
# [True, True, False, False, True, False, False]
# TODO: create per_dataset num_factors, per_factor values etc..
baseline_loss = {
        'mpi3d': [5.0 / 6, 5.0 / 6, 1.0, 1.0, 2.0 / 3, 1.0, 1.0],
        'cars3d': [1.0, 1.0, 182.0 / 183],
        'dsprites': [2.0 / 3, 1.0, 1.0, 1.0, 1.0]
    }
# cars3d: [4,  24, 183]
# False, False, True]

all_mse_results = {}
all_r_results = {
    'dsprites' : {}, 
    'shapes3d' : {}, 
    'mpi3d' : {}, 
    'cars3d' : {}
}
dataset_num_factors = {
    'dsprites' : 5,
    'mpi3d' : 7,
    'shapes3d': 5,
    'cars3d' : 3
}
num_factors = dataset_num_factors[dataset]

# mpi3d_factor_names = [
#                 'color', 'shape', 'size', 'height', 'bg color', 'x-axis',
#                 'y-axis'
#             ]
# mpi3d_factor_names = [
#                 'Object Color', 'Object Shape', 'Object Size', 'Camera Height', 'Background Color', 'Robotic arm x-axis',
#                 'Robotic arm y-axis'
#             ]
factor_names = {
    'mpi3d': ['Object Color', 'Object Shape', 'Object Size', 
                'Camera Height', 'Background Color', 'Robotic arm x-axis',
                'Robotic arm y-axis'
            ],
    'cars3d' : ['Elevation', 'Azimuth', 'Object' ],
    'dsprites': ['Shape', 'Scale', 'Orientation', 'x-position', 'y-position']
}
def my_label(model_type):
    mapping = {
        'resnet' : 'ImageNet',
        'ImageNet' : 'ImageNet',
        'raw' : 'Raw Data',
        'vae_beta_100' : '$\\beta$-VAE',
        'vae_beta_0.0' : 'AE',
        'vae' : 'VAE',
        'noisy' : 'Noisy Labels',
        'almost_uniform' : 'Uniform Mix'
    }
    for name, new_name in mapping.items():
        if name in model_type:
            return new_name
    return 'Probe'
def get_color(model_type):
    # colors = {
    #     'resnet' : 'g',
    #     'ImageNet' : 'g',
    #     'Raw Data' : 'r',
    #     'Beta-VAE_beta_100' : 'c',
    #     # 'vae_beta_0.0' : 'k',
    #     'VAE' : 'm',
    #     'Noisy Labels' : 'y',
    #     'Uniform Mix' : 'b'
    # }
    colors = {
        'resnet' : 'g',
        'ImageNet' : 'g',
        'raw' : 'r',
        'vae_beta_100' : 'c',
        'vae_beta_0.0' : 'k',
        'vae' : 'm',
        'noisy' : 'y',
        'almost_uniform' : 'b'
    }
    for name, color in colors.items():
        if name in model_type:
            return color
# print(folders)
if True:
    [print(f) for f in folders]
    for folder in folders:
        piclkle_name = folder + '/results.pkl'
        if os.path.exists(piclkle_name):
            with open(piclkle_name, 'rb') as f:
                results = pickle.load(f)
            id = results['id']
            mse = results['mse']
            test_rsquared_acc = results['test_rsquared_acc']
            val_rsquared_acc = results['val_rsquared_acc']
            if not isinstance(val_rsquared_acc, np.ndarray):
                print(f'{folder} does not have scores for each factor')
                val_rsquared_acc = np.array( [val_rsquared_acc] * num_factors )
                test_rsquared_acc = np.array( [test_rsquared_acc] * num_factors )

            num_probe_params = results['num_params']
            if plot_probe == 'rf':
                if num_probe_params > 60:
                    print('ignore random forest with dethp > 60')
                    continue
            # if dataset == 'cars3d':
            #     # on cars3d we should have hidden dim > 185 because that is the number of outputs
            #     if results['params'].probe.hidden_dim < 185 and results['params'].probe.hidden_dim != 0:
            #         continue
            model_type = results['model_type']
            probe_type = results['params'].probe.type
            dci_scores = None

            if plot_probe == 'rf':
                if 'dci_trees' in results:
                    dci_scores = results['dci_trees']
            else:
                if 'dci_mlp' in results:
                    dci_scores = results['dci_mlp']
            
            if dci_scores == -1:
                dci_scores = None
            # dci_scores = [
            #     results['dci']['disentanglement'],
            #     results['dci']['completeness'],
            #     results['dci']['informativeness_val']
            # ]

            dataset = results['params'].dataset
            if probe_type not in all_r_results[dataset]:
                all_r_results[dataset][probe_type] = {}
            model_txt = model_type
            # if 'params' in results:
            #     if 'nois' in model_type:
            #         model_txt = model_txt + '_noise_' + str(results['params'].probe.noise_std)
            # if 'conv_net' in model_type or 'resnet' in model_type:
            #     text = 'supervised' if results['params'].supervision else 'unsupervised'
            #     model_txt = model_txt + '_' + str(text)
        
            if 'resnet' in model_type:
                text = 'ImageNet_pretrained' if results['params'].pretrained else 'scratch'
                # model_txt = model_txt + '_' + str(text)
                model_txt = str(text)
            if 'raw' in model_type:
                if plot_probe == 'mlp':
                    if results['params'].probe.hidden_dim in [24, 30]:
                        print(f'skiping: raw small {folder}')
                        continue


            if 'vae' in model_type:
                if results['params'].vae.exp_params.kld_weight != 1.0:
                    model_txt = model_txt + '_beta_' + str(results['params'].vae.exp_params.kld_weight)
                # if results['params'].vae.exp_params.kld_weight > 100 or results['params'].vae.exp_params.kld_weight < 0.1:
                #     continue
                if results['params'].vae.exp_params.kld_weight not in [1.0, 100]:
                    continue
            if not isinstance(val_rsquared_acc, np.ndarray):
                print(f'skiping: {folder}')
                continue # skip over files that don't have per factor scores
            if -1.0 in val_rsquared_acc:
                print(f'SKIPPING folder with negative scores: {folder}')
                continue
            print(f'model_txt: {model_txt}')

            if val_rsquared_acc.mean() < -0.1 or test_rsquared_acc.mean() < -0.1 :
                print(f'SKIPPING near random runs')
                continue
            if not plot_probe == 'rf':
                num_probe_params = num_probe_params // div_capacity
                # num_probe_params = np.log(num_probe_params)
            if model_txt not in all_r_results[dataset][probe_type]:
                all_r_results[dataset][probe_type][model_txt] = [(num_probe_params, 1.0 - val_rsquared_acc, 1.0 - test_rsquared_acc, mse, dci_scores)]
            else:
                all_r_results[dataset][probe_type][model_txt].append((num_probe_params, 1.0 - val_rsquared_acc, 1.0 - test_rsquared_acc, mse, dci_scores))

            # print(folder)
            print(f'{id}  mse: {mse} R: {test_rsquared_acc} dci : {dci_scores}')

    file = 'tmp_results_mlps.pickle'

    with open(file, 'wb') as f:
        pickle.dump(all_r_results, f)
else:
    file = 'tmp_results_mlps.pickle'
    with open(file, 'rb') as f:
        all_r_results = pickle.load(f)


# per_factor_per_run

okk = False
# for model, vals in all_r_results.items():
for dataset in all_r_results.keys():
    # Exp = Explitcitness(mode='global')
    per_fact_exp = [Explitcitness(mode='baseline') for _ in range(num_factors)]
    Exp = Explitcitness(mode='baseline')


    for probe_type in all_r_results[dataset].keys():
        key = next(iter(all_r_results[dataset][probe_type]))
        num_factors = all_r_results[dataset][probe_type][key][0][1].shape[-1]

        # fig, axs = plt.subplots(num_factors+1,sharex=True, sharey=True, figsize=[6,28])
        # fig, axs = plt.subplots(2,4,sharex=True, sharey=True, figsize=[6,28])
        # fig, axs = plt.subplots(2,4,sharex=True, sharey=True, figsize=[24,4])
        fig, axs = plt.subplots(4,2,sharex=True, sharey=True, figsize=[12,8+14])


        # fig_r, axs_r = plt.subplots(1,sharex=True, sharey=True, figsize=[10,10])
        fig_r, axs_r = plt.subplots(1,sharex=True, sharey=True, figsize=[6,3.5], dpi=300)

        fig_dci, axs_dci = plt.subplots(3+1,sharex=True, sharey=True, figsize=[10,10])
        if use_zoombox:
            # define the zoom box
            # x1, x2 = 14.7, 17  # 0.97, 0.995
            x1, x2 = 15.5, 16.8  # 0.97, 0.995

            y1, y2 = -0.009, 0.025
            # axins = zoomed_inset_axes(axs_r, 2, loc='center',  bbox_to_anchor=(7, 0.4))  # zoom = 9 # 
            axins = zoomed_inset_axes(axs_r, 3.4, loc='center',  bbox_to_anchor=(300, 170))  # zoom = 9 # 

            axins.set_xlim(x1, x2)
            axins.set_ylim(y1, y2)
            axins.set_xticks([], [])
            axins.set_yticks([], [])
        okk = True

        for model in sorted(all_r_results[dataset][probe_type].keys()):
            vals_ = all_r_results[dataset][probe_type][model]

            # put the results in order of the capacity
            vals_ = sorted(vals_, key=lambda tup: tup[0])
            # dvals: dict with key = capacity, value = [val_test_score_per_factor, test_score_per_factor]
            dvals = {}
            ddci = {}
            for v in vals_:
                if v[0] in dvals:
                    # add each run with the same capacity in the same list
                    dvals[v[0]].append(np.stack([v[1],v[2]]))
                else:
                    dvals[v[0]] = [np.stack([v[1],v[2]])]
                if v[4] is not None:
                    if v[0] not in ddci:
                        ddci[v[0]] = [v[4]]
                    else:
                        ddci[v[0]].append(v[4])
            # select the maximum acording to the val set 
 
            cap_scores = []
            cap_scores_std = []
            # for each capacity
            all_runs_test_scores = [[] for _ in range(num_seeds)]
            all_runs_x = []
            for x, runs_scores in dvals.items():
                # num_runs x 2 x num_factors
                runs_scores = np.array(runs_scores)
                print(f' [{model}] cap [{x}] runs_scores shape: {runs_scores.shape}')
                val_index = 0   # compute the validation scores. 
                test_index = 1
                runs_mean_scores = runs_scores.mean(2)[:,val_index]
                best_run_index = runs_mean_scores.argmin(0)
                median_run_index = np.argsort(runs_mean_scores)[len(runs_mean_scores)//2]
                run_index = median_run_index


                # select the run with the max/median validation mean scores
                test_scores =  runs_scores[run_index,test_index]
                # all_runs_test_scores = runs_scores[:,test_index]
                # we create a number of num_seends curves. If we don't have suficient random seeds for 
                # a particular capacity, we chose a seed at random
                all_runs_x.append(x)
                for ind_run, curve_all_fact in enumerate(all_runs_test_scores):
                    if ind_run >= runs_scores.shape[0]:
                        chose_run = np.random.randint(runs_scores.shape[0])
                    else:
                        chose_run = ind_run
                    one_capacity_run = runs_scores[chose_run, test_index]
                    curve_all_fact.append(one_capacity_run)

                average_test_scores = runs_scores[:,test_index,:].mean(-1).mean()
                # TODO: maybe don't multiply here...
                if std_type == 'conf95':
                    std_test_scores = 1.96 * runs_scores[:,test_index,:].mean(-1).std(ddof=1) / np.sqrt(len(runs_scores[:,test_index,:].mean(-1)))
                    if np.isnan(std_test_scores):
                        std_test_scores = 0
                else:
                    std_test_scores = runs_scores[:,test_index,:].mean(-1).std(ddof=1)

                # print(f'std_test_scores: {std_test_scores}')
                if ddci is not None and x in ddci:
                    if run_index >= len(ddci[x]):
                    
                        # warning DCI missing. put the DCI of another run
                        print('-'*120)
                        print(' warning DCI missing. put the DCI of another run')
                        # print(f'folder: {folder}')
                        print(f'{probe_type} - {model} - {x}')
                        print('-'*120)
                        run_index = 0
                    if ddci[x][run_index] is not None:
                        # dci = ddci[x][run_index]['sage_dci']
                        dci_i =  ddci[x][run_index]['informativeness_test']
                        if plot_probe == 'rf':
                            dci_d = ddci[x][run_index]['disentanglement']
                            dci_c = ddci[x][run_index]['completeness']
                            dci_mean = (dci_i + dci_d + dci_c) / 3.0

                        else:
                            dci_d = ddci[x][run_index]['sage_disentanglement']
                            dci_c = ddci[x][run_index]['sage_completeness']
                            dci_mean = ddci[x][run_index]['sage_dci']
                    
                    all_d = []
                    all_c = []
                    for d_in in range(len(ddci[x])):
                        if plot_probe == 'rf':
                            all_d.append(ddci[x][d_in]['disentanglement'])
                            all_c.append(ddci[x][d_in]['completeness'])
                        else:
                            all_d.append(ddci[x][d_in]['sage_disentanglement'])
                            all_c.append(ddci[x][d_in]['sage_completeness'])
                    # ±1.96
                    if std_type == 'conf95':
                        std_c = 1.96 * np.array(all_c).std(ddof=1) / np.sqrt(len(all_c))
                        std_d = 1.96 * np.array(all_d).std(ddof=1) / np.sqrt(len(all_c))
                    else:
                        std_c = np.array(all_c).std(ddof=1)
                        std_d = np.array(all_d).std(ddof=1)

                    mean_c = np.array(all_c).mean()
                    mean_d = np.array(all_d).mean()

                else:
                    dci = dci_i = dci_d = dci_c = dci_mean = None
                    mean_c = mean_d = std_c = std_d = None

                cap_scores.append((x, test_scores, [dci_mean, dci_d, dci_c, dci_i]))
                cap_scores_std.append(( x, average_test_scores, std_test_scores, mean_d, std_d, mean_c, std_c))

            
            # plot mean scores
            xx = [cap for cap, sc, dci in cap_scores]
            xx = np.array(xx)

            if plot_type == 'excess':
                xx = xx - xx[0] + eps_params
            elif plot_type == 'ratio':
                xx = xx / xx[0]
            if not plot_probe == 'rf':
                xx = np.log(xx)
            # xx = np.log(xx)
            if not plot_errorbars:
                # yy = [1.0 - sc.mean()  for cap, sc, dci in cap_scores]
                yy = [sc.mean()  for cap, sc, dci in cap_scores]
                # print(f'{model} capacity: {xx}')
                # print(f'{model} scores: {yy}') 
            else:
                # mean over the factors
                # yy = [1.0 - sc.mean()  for cap, sc, sc_std in cap_scores_std]
                yy = np.array([sc for cap, sc, sc_std, d, d_std, c, c_std in cap_scores_std])
                yy_std = np.array([sc_std for cap, sc, sc_std, d, d_std, c, c_std  in cap_scores_std])

                dd = np.array([d for cap, sc, sc_std, d, d_std, c, c_std in cap_scores_std])
                dd_std = np.array([d_std for cap, sc, sc_std, d, d_std, c, c_std in cap_scores_std])

                dc = np.array([c for cap, sc, sc_std, d, d_std, c, c_std in cap_scores_std])
                dc_std = np.array([c_std for cap, sc, sc_std, d, d_std, c, c_std in cap_scores_std])

            # exp = compute_explitcitness(xx,yy, name=model)
            for ind_f, curve_all_fact in enumerate(all_runs_test_scores):
                curve = np.array(curve_all_fact)
                for f_i in range(num_factors):
                    name = model + f'_factor_{f_i}_seed_{ind_f}'
    
                    per_fact_exp[ind_f].add_curve(xx, curve[:,f_i],
                        baseline_loss[dataset][f_i],
                        name=name)
            Exp.add_curve(xx,yy,name=model)

            
            
            print(f'{model} capacity: {xx}')
            print(f'{model} scores: {1.0 - yy}')
            print(f'{model} 1.96 * std scores: {yy_std}')

            print(f'{model} disentanglement: {dd}')
            print(f'{model} disentanglement 1.96 * std: {dd_std}')
            print(f'{model} completness: {dc}')
            print(f'{model} completness 1.96 * std: {dc_std}')
            print('-'*120)
            if dd[0] is not None:
                print(f'{model} cap_0: DCI: {dd[0]:.4f}±{dd_std[0]:.4f} , {dc[0]:.4f}±{dc_std[0]:.4f} , {1.0 - yy[0]:.4f}±{yy_std[0]:.4f} ')
            else:
                print(f'{model} cap_0: I {1.0 - yy[0]:.4f}±{yy_std[0]:.4f} ')
            
            # TODO: imi e rusine cu ce e scris aici...
            if dd[-1] is not None:
                last_index = -1
            elif dd[-2] is not None:
                last_index = -2 
                print(f'WARNING using {last_index} as last probe')
            else:
                last_index = -3
                print(f'WARNING using {last_index} as last probe')

            if dd[last_index] is not None:
                print(f'{model} cap_T: DCI: {dd[last_index]:.4f}±{dd_std[last_index]:.4f} , {dc[last_index]:.4f}±{dc_std[last_index]:.4f} , {1.0 - yy[last_index]:.4f}±{yy_std[last_index]:.4f} ')
            else:
                print(f'{model} cap_T: I: {1.0 - yy[-1]:.4f}±{yy_std[-1]:.4f} ')

            # print(f'{model} E = {exp}')

            if use_predefined_colors:
                c = get_color(model)
                axs[0][0].plot(xx, yy, f'-*{c}',label=model)
                if not plot_errorbars:
                    axs_r.plot(xx, yy, f'-*{c}',label=model)
                else:
                    # axs_r.errorbar(xx, yy, yerr=yy_std, fmt=f'-*{c}',label=model)
                    axs_r.plot(xx, yy, f'-*{c}',label=my_label(model))
                    # yy_std = yy_std + 0.02
                    axs_r.fill_between(xx, yy-yy_std, yy+yy_std, alpha=0.5, color=c)
                    if use_zoombox:
                        axins.plot(xx, yy, f'-*{c}',label=my_label(model))
                        axins.fill_between(xx, yy-yy_std, yy+yy_std, alpha=0.5, color=c)
            else:
                axs[0][0].plot(xx, yy, f'-*',label=model)
                if not plot_errorbars:
                    axs_r.plot(xx, yy, f'-*',label=model)
                else:
                    axs_r.errorbar(xx, yy, yerr=yy_std, fmt=f'-*',label=model)


            if logscale:
                axs[0][0].set_xscale('log')
                
            axs[0][0].set_title('Loss Averaged across all factors')
            # axs[0][0].set_ylim(-0.03,1.01)

            if logscale:
                axs_r.set_xscale('log')
                # axs_r.set_yscale('log')
            axs_r.set_title(my_plot_title)
            # axs_r.set_ylim(bottom=bottom,top=top)
            axs_r.set_xlabel(f'Capacity = {capacity_measure}')
            axs_r.set_ylabel(f'Loss')

            if not use_zoombox:
                axs_r.legend(loc='upper right')
            # fig_r.savefig(f'figures/{dataset}_{plot_name}_{probe_type}_R2_scores.pdf', 
            #     bbox_inches="tight")
            # axs[0][0].legend(bbox_to_anchor=(1.04,0.5), loc="center left")
            for i_factor in range(num_factors):
                # xx = [cap for cap, sc in cap_scores]
                # xx = np.array(xx)
                # xx = np.log(xx)
                # yy = [1.0 - sc[i_factor] for cap, sc, _ in cap_scores]
                yy = [sc[i_factor] for cap, sc, _ in cap_scores]
                ii = (1 + i_factor) % 4
                jj = (1 + i_factor) // 4
                if use_predefined_colors:
                    axs[ii][jj].plot(xx, yy, f'-*{c}',label=my_label(model))
                else:
                    axs[ii][jj].plot(xx, yy, f'-*',label=my_label(model))
                if logscale:
                    axs[ii][jj].set_xscale('log')
                axs[ii][jj].set_title(f'{factor_names[dataset][i_factor]}')
                # axs[ii][jj].set_ylim(bottom=bottom,top=top)
                if not use_predefined_colors:
                    axs[ii][jj].legend(bbox_to_anchor=(1.04,0.5), loc="center left")


            for ax in axs.flat:
                ax.set(xlabel=f'Capacity = {capacity_measure}', ylabel='Loss')

            for ax in axs.flat:
                ax.label_outer()
            


            # fig.savefig(f'figures/{dataset}_{plot_name}_{probe_type}_scores.png', 
            #     bbox_inches="tight")#, dpi=800)

            ## ----------------- plot dci -------------
            # xx = [cap for cap, sc, dci in cap_scores]
            # xx = np.array(xx)
            dci_names = ['dci', 'disentanglement', 'completeness', 'informativeness']
            for ind in range(3+1):
                yy = [dci[ind] for cap, sc, dci in cap_scores]
                # if dci is not None:
                if yy[0] is not None:

                    if use_predefined_colors:
                        axs_dci[ind].plot(xx, yy, f'-*{c}',label=model)
                    else:
                        axs_dci[ind].plot(xx, yy, f'-*',label=model)
                    if logscale:
                        axs_dci[ind].set_xscale('log')
                    axs_dci[ind].set_title(f'{dci_names[ind]}')
                    axs_dci[ind].set_ylim(0,1.01)
                    if not use_predefined_colors:
                        axs_dci[ind].legend(bbox_to_anchor=(1.04,0.5), loc="center left")


            for ax in axs_dci.flat:
                ax.set(xlabel='capacity = depth', ylabel='R^2 scores')

            for ax in axs_dci.flat:
                ax.label_outer()
            
            handles, labels = ax.get_legend_handles_labels()
            fig_dci.legend(handles, labels, loc='lower center')


            fig_dci.savefig(f'figures/{dataset}_{plot_name}_{probe_type}_dci_scores.png', 
                bbox_inches="tight")#, dpi=800)
            # for x, accs in dvals.items():
            #     vals.append((x, np.array(accs).max(),0))  # add 0 as mse

            # print(f'model: {model} : {vals}')
            # x = [params for params,r,mse in vals]
            # y = [r for params,r,mse in vals]
        # pdb.set_trace()
        for ax in axs.flat:
            ax.label_outer()

        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center')

        if okk and use_zoombox:
            mark_inset(axs_r, axins, loc1=1, loc2=3, fc="none") #, ec="0.5")

        fig_r.savefig(f'figures/{dataset}_{plot_name}_{probe_type}_R2_scores.pdf', 
            bbox_inches="tight")
        
        fig.savefig(f'figures/{dataset}_{plot_name}_{probe_type}_scores.pdf', 
                bbox_inches="tight")#, dpi=800)


        all_E = Exp.get_explitcitness()
        [print(f'{name} E: {val}') for name, val in all_E.items()]


        # model_mean_E = [[] for _ in range(len(all_r_results[dataset][probe_type]))]
        model_mean_E = {}
        # for m_i, model in enumerate(sorted(all_r_results[dataset][probe_type].keys())):
        #     for name, E in all_E.items():
        #         if model in name:
        #             model_mean_E[m_i].append(E)


        for f_i in range(num_factors):
            all_E = per_fact_exp[f_i].get_explitcitness()
            for name, E in all_E.items():
                print(name)
                # remove suffix
                run_i = int(name.split('_')[-1])
                factor_i = int(name.split('_')[-3]) # should be the same as f_i
                name  = name[:-16]

                if name not in model_mean_E:
                    model_mean_E[name] = np.zeros((num_seeds, num_factors))
                    model_mean_E[name][run_i,factor_i] = E
                    # model_mean_E[name].append(E)
                else:
                    model_mean_E[name][run_i,factor_i] = E
        # for m_i, model in enumerate(sorted(all_r_results[dataset][probe_type].keys())):
        for name, E_scores in model_mean_E.items():
            # mean_E = np.array(E_scores)
            # mean_E = mean_E.mean()
            mean_factors_E = E_scores.mean(1)
            mean_runs_factors_E = mean_factors_E.mean()
            std_runs_factors_E = mean_factors_E.std(ddof=1)

            print(f'Model: {name} mean E: {mean_runs_factors_E} : std: {std_runs_factors_E}')
    # print(2)

        #     print(x,y)
            
        #     if 'mlp' in probe:
        #         # capacity = number of parameters more than the linear probe
        #         xx = np.array(x) # - x[0]
        #         plt.title(f'Results {dataset}')
        #         # plt.subplot(1, 2, 1)
        #         # plt.plot(xx,np.array(y),'-*',label=model)
        #         # plt.xlabel('additional number of params beside linear probe')
        #         # plt.ylabel('R squared')
        #         # plt.legend()
        #         # plt.subplot(1, 2, 2)
        #         xx = np.log(xx + 1)
        #         plt.plot(xx,np.array(y),'-*',label=model)
        #         # plt.xlabel('log(1 + number of extra params)')
        #         plt.xlabel('log(1 + number of params)')
        #     else:
        #         xx = np.array(x) # - x[0]
        #         xx = np.log(xx)
        #         plt.title(f'Results {dataset}')
        #         plt.plot(xx,np.array(y),'-*',label=model)
        #         # plt.xlabel('log(1 + number of extra params)')
        #         plt.xlabel('capacity = max depth')
        #     plt.ylabel('R squared')
        #     plt.legend()
        #         # plt.legend()
        #     plt.savefig(f'results_{probe}.png')
        # plt.show()

    # for model, vals in all_r_results.items():
    #     vals = sorted(vals, key=lambda tup: tup[0])

    #     x = [params for params,r,mse in vals]
    #     y = [mse for params,r,mse in vals]
    #     print(x,y)
    #     plt.plot(x,y,'-*',label=model)
    #     plt.ylabel('MSE')
    #     plt.legend()
    # plt.show()


