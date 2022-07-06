# Plotting script for SyLaNN simulations with gammaIdx iteration
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import math

gamma_nSteps = 10
gamma_start, gamma_stop = 0, 1+gamma_nSteps
# as a list of integers and list of strings
gamma_list = list(range(gamma_start, gamma_stop))

# folder name and file name (without gammaIdx numbering)
date_str = '2022-07-05'
time_str = '18-16-40'
folder_name = date_str + '_SyLaNNsimulation'
if os.path == 'nt': # Windows
    folder_path = folder_name + '\\'
else: # Linux 
    folder_path = folder_name + '/'
# Note: leave out idx numbering and file format (.json) as this will change during the results loading loop
simulation_name = '_LangmuirIsotherm_gammaIdx'
time_simulation_name = time_str + simulation_name
path_name = folder_path + time_simulation_name
file_format = '.json'

# create overall dictionary
results_dict = {}

# fill results dictionary with simulation results (nested dictionary of dicionaries)
for gammaIdx in gamma_list:
    current_key = gammaIdx
    current_json_name = path_name + str(current_key) + file_format
    # read current results file and save into dictionary
    json_file_data = open(current_json_name,)
    current_file = json.load(json_file_data)
    json_file_data.close()
    # write into overall dictionary at specific key
    results_dict[current_key] = current_file

######################## Plotting functions ########################
def linePlot_pred_vs_ref(domain, eq_pred, eq_ref, plot_saved_name):
    fig = plt.figure()
    plt.plot(domain, eq_pred, 'r--', label='prediction')
    plt.plot(domain, eq_ref, 'k-', label='reference')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(loc='upper left')
    # plt.xlim plt.ylim etc
    fig.savefig(plot_saved_name)
    # plt.show()
    plt.close()

def linePlot_traintesterror(epochs, train_error, test_error, plot_saved_name):
    fig = plt.figure()
    plt.plot(epochs, train_error, 'b-', label='training error')
    plt.plot(epochs, test_error, 'r-', label='test error')
    plt.xlabel('epochs')
    plt.ylabel('sum of squared errors')
    plt.legend(loc='upper right')
    fig.savefig(plot_saved_name)
    # plt.show()
    plt.close()

def linePlot_traintesterror_logscale(epochs, train_error, test_error, plot_saved_name):
    fig = plt.figure()
    plt.semilogy(epochs, train_error, 'b-', label='training error')
    plt.semilogy(epochs, test_error, 'r-', label='test error')
    plt.xlabel('epochs')
    plt.ylabel('log(sum of squared errors)')
    plt.legend(loc='upper right')
    fig.savefig(plot_saved_name)
    # plt.show()
    plt.close()

######################## PLOTS #########################
plot_folder_name = date_str + '_plots' + simulation_name
try:
    os.mkdir(plot_folder_name)
except OSError:
    print ("Creation of the directory %s failed (already exists)" % plot_folder_name)
else:
    print ("Successfully created the directory %s " % plot_folder_name)

if os.path == 'nt': # Windows
    plots_path = plot_folder_name + '\\'
else: # Linux 
    plots_path = plot_folder_name + '/'

file_type = '.png'

# Loop over gamma values (L1 ratio of elastic net penalty) and create/save plots
for plotIdx in gamma_list:
    current_results = results_dict[plotIdx]
    x_domain = current_results['domain_test']
    x_domain_steps = 100
    x_domain = np.linspace(x_domain[0], x_domain[1], num=x_domain_steps)
    # plot ref vs prediction
    plot_name = 'gammaIdx' + str(plotIdx) + '_ref-vs-pred' + file_type
    path_saveTo = plots_path + plot_name
    ref_eq = eval(current_results['ref_fct_str'])
    ref_eq_list = [ref_eq(i) for i in x_domain]
    prediction = eval(current_results['found_eq'])
    # TODO find way to cut predicted results vars!
    # TODO why exp etc with Python's math module?
    prediction_list = [prediction(i,1,1) for i in x_domain]
    linePlot_pred_vs_ref(x_domain, prediction_list, ref_eq_list, path_saveTo)

    # plot training and testing errors
    plot_name = 'gammaIdx' + str(plotIdx) + '_errors' + file_type
    path_saveTo = plots_path + plot_name
    n_epochs = current_results['trainEpochs3']
    epochs = np.linspace(1, n_epochs, num=n_epochs+1)
    train_error_onlySSE = current_results['training_onlySSE_loss']
    # train_error = data_dict['training_loss']
    test_error = current_results['testing_loss']
    linePlot_traintesterror(epochs, train_error_onlySSE, test_error, path_saveTo)

    # plot training and testing errors (log scale)
    plot_name = 'gammaIdx' + str(plotIdx) + '_errorsLog' + file_type
    path_saveTo = plots_path + plot_name
    linePlot_traintesterror_logscale(epochs, train_error_onlySSE, test_error, path_saveTo)

print('Done plotting.')
