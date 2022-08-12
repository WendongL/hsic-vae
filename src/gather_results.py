
import glob
import pickle
import os.path
import numpy as np
import matplotlib.pyplot as plt

folders = glob.glob("../models/dsprites/models_round2_5/*")
all_mse_results = {}
all_r_results = {}
for folder in folders:
    
    piclkle_name = folder + '/results.pkl'
    if os.path.exists(piclkle_name):
        with open(piclkle_name, 'rb') as f:
            results = pickle.load(f)
        id = results['id']
        mse = results['mse']
        rsquared = results['rsquared']
        num_probe_params = results['num_params']
        model_type = results['model_type']

        model_txt = model_type
        if 'params' in results:
            if 'nois' in model_type:
                model_txt = model_txt + '_noise_' + str(results['params'].noise_std)
        if 'conv_net' in model_type or 'resnet' in model_type:
            text = results['params'].supervision
            model_txt = model_txt + '_' + str(text)
    
        if 'resnet' in model_type:
            text = 'pretrained' if results['params'].pretrained == 'yes' else 'scratch'
            model_txt = model_txt + '_' + str(text)

        if model_txt not in all_r_results:
            all_r_results[model_txt] = [(num_probe_params, rsquared, mse)]
        else:
            all_r_results[model_txt].append((num_probe_params, rsquared, mse))

        print(f'{id}  mse: {mse} R: {rsquared}')

# for model, vals in all_r_results.items():
for model in sorted(all_r_results.keys()):
    vals = all_r_results[model]

    vals = sorted(vals, key=lambda tup: tup[0])
    print(f'model: {model} : {vals}')
    x = [params for params,r,mse in vals]
    y = [r for params,r,mse in vals]
    print(x,y)
    # capacity = number of parameters more than the linear probe
    xx = np.array(x) - x[0]
    
    plt.subplot(1, 2, 1)
    plt.plot(xx,np.array(y),'-*',label=model)
    plt.xlabel('additional number of params beside linear probe')
    plt.ylabel('R squared')
    plt.legend()
    plt.subplot(1, 2, 2)

    xx = np.log(xx + 10)
    plt.plot(xx,np.array(y),'-*',label=model)
    plt.xlabel('log(1 + number of extra params)')
    plt.ylabel('R squared')
    # plt.legend()

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


