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
        folder_name = curr_date + '_SLNNsimulation'
        try:
                os.mkdir(folder_name)
        except OSError:
                print ("Creation of the directory %s failed" % folder_name)
        else:
                print ("Successfully created the directory %s " % folder_name)

        if os.path == 'nt': # Windows
                save_folder_path = folder_name + '\\'
        else: # Linux 
                save_folder_path = folder_name + '/'
        
        # read configurations from input file (dictionaries)
        generateData_dict, net_dict, trainConfig_dict = input_dicts.readDictionaries()

        # generate data, if no dataset is given
        # TODO change to load data set and check format of dictionary which is read in
        manageData_obj = dm.DataManager()
        generatedDatasets_dict = manageData_obj.generateData(generateData_dict)

        n_params = generatedDatasets_dict['x_dim']

        # create Symbolic-Layered Neural Network (short SLNN)
        mySLnet = SyLaNN.SLNet(n_hiddenLayers=net_dict['n_hidden'], fcts=net_dict['symbolic_layer'], data_dim=n_params)

        # training of network with given data (generated or loaded and formatted previously)
        # and save to file in corresponding folder (encoded with date and time when simulation started)
        
        gamma_values = [] # TODO define gamma values as list or any other iterable construct
        for gamma_idx, gamma_val in enumerate(gamma_values):
            simulationResults_dict = mySLnet.train(generatedDatasets_dict, trainConfig_dict, gamma_val)

            # TODO needs changes if we have a dataset given and do not need the generating dict
            # current L1 ratio (gamma_val) is saved within result_dict
            file_name = generateData_dict['saveFile_name'] + "_gammaIdx" + str(gamma_idx) + ".json"
            save_file_name = save_folder_path + curr_time + file_name
            manageData_obj.saveSimulation(save_file_name, generateData_dict, net_dict, trainConfig_dict, simulationResults_dict)
