# TODO same as main.py
# main_BR.py for Bayesian regularization settings plus elastic net (gamma iteration)
# save simulation results per gamma iteration -> post processing evaluation script?
# load data set instead of generating it each time
# Example Freundlich isotherm equation

# packages
import torch
import numpy as np
import random
import os
from datetime import datetime

# own imports
import input_dicts
import datamanager as dm
import SyLaNN

if __name__ == "__main__":
        # set seed for reproducability
        seed_val = 42
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)

        # create folders for saving results and plots later
        curr_date = datetime.today().strftime('%Y-%m-%d')
        curr_time = datetime.today().strftime('%H-%M-%S')
        folder_name = curr_date + '_SyLaNNsimulation'
        loadData_folder = 'datasets'
        try:
                os.mkdir(folder_name)
        except OSError:
                print ("Creation of the directory %s failed (already exists)" % folder_name)
        else:
                print ("Successfully created the directory %s " % folder_name)

        if os.path == 'nt': # Windows
                save_folder_path = folder_name + '\\'
                load_folder_path = loadData_folder + '\\'
        else: # Linux 
                save_folder_path = folder_name + '/'
                load_folder_path = loadData_folder + '/'
        
        # read SyLaNN configurations and its training settings from input file (dictionaries)
        net_dict = input_dicts.readDictionaries()[1]
        trainConfig_dict = input_dicts.readDictionaries()[2]

        # generate data, if no dataset is given
        manageData_obj = dm.DataManager()
        data_file_name = "2022-06-13_data_FreundlichIsotherm.json"
        loadedDatasets_withConfigs = manageData_obj.loadDataset(load_folder_path, data_file_name)
        n_params = loadedDatasets_withConfigs['x_dim']

        # create Symbolic-Layered Neural Network (short SyLaNN)
        mySyLaNN = SyLaNN.SLNet(n_hiddenLayers=net_dict['n_hidden'], fcts=net_dict['symbolic_layer'], data_dim=n_params)

        # training of network with given data (generated or loaded and formatted previously)
        # and save to file in corresponding folder (encoded with date and time when simulation started)
        gamma_nSteps = 10
        gamma_step = 1/gamma_nSteps
        gamma_start, gamma_stop = 0, 1+gamma_step # regular values for L1 ratio
        gamma_values = [*np.arange(gamma_start, gamma_stop, gamma_step)]
        print(gamma_values)
        for gamma_idx, gamma_val in enumerate(gamma_values):
            simulationResults_dict = mySyLaNN.train(loadedDatasets_withConfigs, trainConfig_dict, gamma_val)
            # current L1 ratio (gamma_val) is saved within result dictionary
            file_name = loadedDatasets_withConfigs['saveFile_name'] + "_gammaIdx" + str(gamma_idx) + ".json"
            save_file_name = save_folder_path + curr_time + file_name
            # manageData_obj.saveSimulation(save_file_name, generateData_dict, net_dict, trainConfig_dict, simulationResults_dict)
            manageData_obj.saveSimulation(save_file_name, loadedDatasets_withConfigs, net_dict, trainConfig_dict, simulationResults_dict)
