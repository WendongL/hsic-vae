
import glob
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt
import pdb

probe = 'mlp_excess_linear_scale' # / 'rf' / 'mlp'
#folders = glob.glob("./results/2022May19-165411_round8_random_forest/*/*")
# folders = glob.glob("./results/2022May19-155054_round8_random_forest/*/*")
# folders = glob.glob("./results/good_models/*/*/*")
folders = glob.glob("./results/2022May25-114330_round8_vae/*/*")
folders = glob.glob("./results/good_models/*/*/*")


all_mse_results = {}
all_r_results = {
    'dsprites' : {}, 
    'shapes3d' : {}, 
    'mpi3d' : {}, 
}

[print(f) for f in folders]
for folder in folders:
    
    piclkle_name = folder + '/results.pkl'
    if os.path.exists(piclkle_name):
        with open(piclkle_name, 'rb') as f:
            results = pickle.load(f)
        
        id = results['id']
        mse = results['mse']
        test_rsquared_acc = results['test_rsquared_acc']
        num_probe_params = results['num_params']
        model_type = results['model_type']
        dataset = results['params'].dataset
        model_txt = model_type
        # if 'params' in results:
        #     if 'nois' in model_type:
        #         pdb.set_trace()
        #         model_txt = model_txt + '_noise_' + str(results['params'].probe.noise_std)
        if 'conv_net' in model_type or 'resnet' in model_type:
            text = 'supervised' if results['params'].supervision else 'unsupervised'
            model_txt = model_txt + '_' + str(text)
    
        if 'resnet' in model_type:
            text = 'pretrained' if results['params'].pretrained else 'scratch'
            model_txt = model_txt + '_' + str(text)
        if 'vae' in model_type:
            model_txt = model_txt + '_beta_' + str(results['params'].vae.exp_params.kld_weight)
        if model_txt not in all_r_results[dataset]:
            all_r_results[dataset][model_txt] = [(num_probe_params, test_rsquared_acc, mse)]
        else:
            all_r_results[dataset][model_txt].append((num_probe_params, test_rsquared_acc, mse))

        print(f'{id}  mse: {mse} R: {test_rsquared_acc}')

# for model, vals in all_r_results.items():
for dataset in all_r_results.keys():
    for model in sorted(all_r_results[dataset].keys()):
        vals_ = all_r_results[dataset][model]

        vals_ = sorted(vals_, key=lambda tup: tup[0])

        dvals = {}
        for v in vals_:
            if v[0] in dvals:
                dvals[v[0]].append(v[1])
            else:
                dvals[v[0]] = [v[1]]
        vals = []
        for x, accs in dvals.items():
            vals.append((x,np.array(accs).max(),0))  # add 0 as mse

        print(f'model: {model} : {vals}')
        x = [params for params,r,mse in vals]
        y = [r for params,r,mse in vals]
        print(x,y)
        
        if 'mlp' in probe:
            # capacity = number of parameters more than the linear probe
            xx = np.array(x)[:-2]  - x[0] #+ 1
            plt.title(f'Results {dataset}')
            # plt.subplot(1, 2, 1)
            # plt.plot(xx,np.array(y),'-*',label=model)
            # plt.xlabel('additional number of params beside linear probe')
            # plt.ylabel('R squared')
            # plt.legend()
            # plt.subplot(1, 2, 2)
            # xx = np.log(xx + 1)
            plt.plot(xx,np.array(y)[:-2],'-*',label=model)
            # plt.xscale('log')

            # plt.xlabel('log(1 + number of extra params)')
            plt.xlabel('number of excess params (total params - linear params + 1)')
            # plt.xlabel(' Total Number of params / Linear probe params')
        else:
            xx = np.array(x) # - x[0]
            # xx = np.log(xx)
            plt.title(f'Results {dataset}')
            plt.plot(xx,np.array(y),'-*',label=model)
            plt.xscale('log')

            # plt.xlabel('log(1 + number of extra params)')
            plt.xlabel('capacity = max_depth')
        plt.ylabel('R squared')
        plt.legend()
            # plt.legend()
        plt.savefig(f'results_{probe}.png')
    plt.show()

    # for model, vals in all_r_results.items():
    #     vals = sorted(vals, key=lambda tup: tup[0])

    #     x = [params for params,r,mse in vals]
    #     y = [mse for params,r,mse in vals]
    #     print(x,y)
    #     plt.plot(x,y,'-*',label=model)
    #     plt.ylabel('MSE')
    #     plt.legend()
    # plt.show()


