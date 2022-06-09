# main to generate a dataset for given settings in input_dicts.py
# in order to load an existing dataset for training, the dataset needs to be generated here first

import torch
import numpy as np
import random
import os
from datetime import datetime

# own imports
import input_dicts
import datamanager as dm

if __name__ == "__main__":
        # set seed for reproducability
        seed_val = 42
        torch.manual_seed(seed_val)
        random.seed(seed_val)
        np.random.seed(seed_val)

        curr_date = datetime.today().strftime('%Y-%m-%d')
        curr_time = datetime.today().strftime('%H-%M-%S')

        # create folder where the dataset file is to be saved
        loadData_folder = 'datasets'
        try:
                os.mkdir(loadData_folder)
        except OSError:
                print ("Creation of the directory %s failed (already exists)" % loadData_folder)
        else:
                print ("Successfully created the directory %s " % loadData_folder)

        if os.path == 'nt': # Windows
                load_folder_path = loadData_folder + '\\'
        else: # Linux
                load_folder_path = loadData_folder + '/'
        
        # read configuration dictionary for generating the dataset
        generateData_dict = input_dicts.readDictionaries()[0]

        # generate dataset
        manageData_obj = dm.DataManager()
        generatedDatasets_dict = manageData_obj.generateData(generateData_dict)

        # save generated dataset and its configurations
        file_name = "_data" + generateData_dict['saveFile_name'] + ".json"
        # TODO  + "_" + curr_time needed in addition?
        save_file_name = load_folder_path + curr_date + file_name
        manageData_obj.saveDataset(save_file_name, generateData_dict, generatedDatasets_dict)


