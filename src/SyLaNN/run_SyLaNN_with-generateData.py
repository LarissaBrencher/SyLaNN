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

# postprocessing runs separately by loading the save file of the simulation

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

        # generate data, if no dataset is given (later load, format etc functions for that)
        manageData_obj = dm.DataManager()
        generatedDatasets_dict = manageData_obj.generateData(generateData_dict)

        n_params = generatedDatasets_dict['x_dim']

        # create Symbolic-Layered Neural Network (short SyLaNN)
        mySyLaNN = SyLaNN.SyLaNet(net_dict, data_dim=n_params)

        # training of network with given data (generated or loaded and formatted previously)
        # and save to file in corresponding folder (encoded with date and time when simulation started)
        gamma_miniTest = 0.5
        simulationResults_dict = mySyLaNN.train(generatedDatasets_dict, trainConfig_dict, gamma_miniTest)
        file_name = generateData_dict['saveFile_name']
        save_file_name = save_folder_path + curr_time + file_name
        manageData_obj.saveSimulation(save_file_name, generateData_dict, net_dict, trainConfig_dict, simulationResults_dict)
