"""
Managing data of the Symbolic-Layered Neural Network framework

Either new input data is generated based on a given reference function or existing data can be used.
The results of the computed simulation can be saved and the predicted equation is formatted into a 
string lambda expression to simplify further processing.
"""

import inspect
import numpy as np
import torch
import json
from sympy import symbols
from sympy.utilities.lambdify import lambdastr
from scipy.stats import truncnorm
import copy
import noise

class DataManager():
    """
    A class to manage data (and to save simulation results and obtain useful formatting).
    """
    def __init__(self):
        """
        Constructs the DataManager object.
        """
        super().__init__()

    def generateData(self, generateData_dict):
        """
        Returns training and testing datasets by means of given definitions in the given dictionary.

        :param generateData\_dict: Dictionary containing the definitions for generating a dataset.
        :type generateData\_dict: dict

        :return: Returns generated dataset as dictionary.
        :rtype: dict
        """
        ref_fct = eval(generateData_dict['ref_fct_str'])
        x_dim = len(inspect.signature(ref_fct).parameters)     # Number of inputs to the function, or, dimensionality of x
        # create training data
        range_min_train = generateData_dict['domain_train'][0]
        range_max_train = generateData_dict['domain_train'][1]
        # uniform-distributed:
        # inputX_train = (range_max_train - range_min_train) * torch.rand([generateData_dict['n_train'], x_dim]) + range_min_train
        # normal-distributed:
        # mean_train = (range_max_train - range_min_train) / 2
        # std_train = np.sqrt(np.abs(range_max_train - range_min_train))
        # loc=mean_train, scale=std_train, 
        # for specific mean, std: (range_min_train - mean_train) / std_train, (range_max_train - mean_train) / std_train
        a_train, b_train = range_min_train, range_max_train
        inputX_train = torch.from_numpy(truncnorm.rvs(a_train, b_train, size=[generateData_dict['n_train'], x_dim]))
        outputY_train = torch.tensor([[ref_fct(*x_i)] for x_i in inputX_train])
        # create testing data
        range_min_test = generateData_dict['domain_test'][0]
        range_max_test = generateData_dict['domain_test'][1]
        # uniform-distributed:
        inputX_test = (range_max_test - range_min_test) * torch.rand([generateData_dict['n_test'], x_dim]) + range_min_test
        # normal-distributed:
        # mean_test = (range_max_test - range_min_test) / 2
        # std_test = 1 
        # for specific mean, std: (range_min_test - mean_test) / std_test, (range_max_test - mean_test) / std_test
        # a_test, b_test = range_min_test, range_max_test
        # inputX_test = torch.from_numpy(truncnorm.rvs(a_test, b_test, size=[generateData_dict['n_test'], x_dim]))
        outputY_test = torch.tensor([[ref_fct(*x_i)] for x_i in inputX_test])

        # if noise is chosen in dict
        if generateData_dict['checkNoise'] is True:
            tmp_saveFileName = generateData_dict['saveFile_name']
            generateData_dict['saveFile_name'] = tmp_saveFileName + '_' + generateData_dict['noise_type'] + 'Noise'
            train_size = outputY_train.size()
            test_size = outputY_test.size()
            noise_train = noise.createNoise(train_size, noise_std=generateData_dict['noise_std'], type_str=generateData_dict['noise_type'])
            noise_test = noise.createNoise(test_size, noise_std=generateData_dict['noise_std'], type_str=generateData_dict['noise_type'])
            outputY_train = torch.add(outputY_train, noise_train)
            outputY_test = torch.add(outputY_test, noise_test)
        elif generateData_dict['checkNoise'] is False:
            pass

        # write generated data sets into dictionary
        generated_dict = {
            'X_train' : inputX_train,
            'y_train' : outputY_train,
            'X_test' : inputX_test,
            'y_test' : outputY_test,
            'x_dim' : x_dim
        }

        if generateData_dict['standardize_or_centralize'] == 's':
            dataset_dict = self.standardizeData(generated_dict)
        elif generateData_dict['standardize_or_centralize'] == 'c':
            dataset_dict = self.centralizeData(generated_dict)
        else:
            dataset_dict = generated_dict

        return dataset_dict

    def saveDataset(self, save_file_name, configGenerateData_dict, dataset_dict):
        """
        Saves the previously generated dataset and its settings to a json file as a dictionary.

        :param save\_file\_name: Name of the JSON file in which the dataset is saved.
        :type save\_file\_name: str
        :param configGenerateData\_dict: Dictionary containing the definitions for generating a dataset.
        :type configGenerateData\_dict: dict
        :param dataset\_dict: Dictionary containing the dataset.
        :type dataset\_dict: dict
        """
        dataset_dict_format = self.tensor2list(dataset_dict)
        saveFile_dict = {
            **configGenerateData_dict,
            **dataset_dict_format
        }

        if saveFile_dict['standardize_or_centralize'] == 's':
            insert_s = '_s'
            sub_str = '.json'
            idx = save_file_name.index(sub_str)
            save_file_name = save_file_name[:idx] + insert_s + save_file_name[idx:]
            saveFile_dict['saveFile_name'] = saveFile_dict['saveFile_name'] + insert_s
        elif saveFile_dict['standardize_or_centralize'] == 'c':
            insert_c = '_c'
            sub_str = '.json'
            idx = save_file_name.index(sub_str)
            save_file_name = save_file_name[:idx] + insert_c + save_file_name[idx:]
            saveFile_dict['saveFile_name'] = saveFile_dict['saveFile_name'] + insert_c
        else:
            pass

        with open(save_file_name, 'w') as outfile:
            json.dump(saveFile_dict, outfile, indent = 4)

    def loadDataset(self, file_dir, data_dict):
        """
        Loads a given dataset.

        :param file\_dir: Directory in which the dataset file is located.
        :type file\_dir: str
        :param data\_dict: Name of the JSON file in which the dataset is saved.
        :type data\_dict: str
        """
        data_path = file_dir + data_dict
        dataset = open(data_path,)
        data = json.load(dataset)
        dataset_format = self.list2tensor(data) # change lists back to torch tensors
        dataset.close()
        return dataset_format

    def centralizeData(self, data_dict):
        """
        Centralizes a given dataset by its mean (per column, i.e. physical variable).

        :param data\_dict: Name of the JSON file in which the dataset is saved.
        :type data\_dict: str
        """
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        nVars = X_train.size(dim=1)
        X_train_c = []
        X_test_c = []
        # loop over the variable dimension and center each variable by its mean
        for iVar in range(nVars):
            current_train = X_train[:, iVar]
            current_test = X_test[:, iVar]
            mean_train = torch.mean(current_train)
            mean_test = torch.mean(current_test)
            central_train = torch.sub(current_train, mean_train)
            central_test = torch.sub(current_test, mean_test)
            # save centralized columns in new tensor
            X_train_c.append(central_train)
            X_test_c.append(central_test)
        X_train_c = torch.stack(X_train_c, dim=1)
        X_test_c = torch.stack(X_test_c, dim=1)
        # same computation for features saved in y_*
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        mean_train = torch.mean(y_train)
        mean_test = torch.mean(y_test)
        y_train_c = torch.sub(y_train, mean_train)
        y_test_c = torch.sub(y_test, mean_test)
        # update the dictionary with new centralized values and return it
        data_dict.update({'X_train': X_train_c, 'X_test': X_test_c, 'y_train': y_train_c, 'y_test': y_test_c})
        return data_dict

    def standardizeData(self, data_dict):
        """
        Standardizes a given dataset by its mean and its standard deviation (per column, i.e. physical variable).

        :param data\_dict: Name of the JSON file in which the dataset is saved.
        :type data\_dict: str
        """
        data_dict = self.centralizeData(data_dict)
        X_train = data_dict['X_train']
        X_test = data_dict['X_test']
        nVars = X_train.size(dim=1)
        X_train_s = []
        X_test_s = []
        # loop over the variable dimension and standardize each variable of the centered data by its standard deviation
        for iVar in range(nVars):
            current_train = X_train[:, iVar]
            current_test = X_test[:, iVar]
            std_train = torch.std(current_train)
            std_test = torch.std(current_test)
            standardize_train = torch.div(current_train, std_train)
            standardize_test = torch.div(current_test, std_test)
            # save standardized columns in new tensor
            X_train_s.append(standardize_train)
            X_test_s.append(standardize_test)
        X_train_s = torch.stack(X_train_s, dim=1)
        X_test_s = torch.stack(X_test_s, dim=1)
        # same computation for features saved in y_*
        y_train = data_dict['y_train']
        y_test = data_dict['y_test']
        std_train = torch.std(y_train)
        std_test = torch.std(y_test)
        y_train_s = torch.div(y_train, std_train)
        y_test_s = torch.div(y_test, std_test)
        # update the dictionary with new standardized values and return it
        data_dict.update({'X_train': X_train_s, 'X_test': X_test_s, 'y_train': y_train_s, 'y_test': y_test_s})
        return data_dict

    def tensor2list(self, unformattedDataset):
        """
        Formats a given dataset's tensors to lists for being able to be saved in JSON files.

        :param unfomattedDataset: Previously generated dataset which still needs formatting.
        :type unformattedDataset: dict
        """
        # training datapoints
        if torch.is_tensor(unformattedDataset['X_train']):
            # convert to list as a format savable by JSON
            unformattedDataset['X_train'] = unformattedDataset['X_train'].tolist()
        if torch.is_tensor(unformattedDataset['y_train']):
            # convert to list as a format savable by JSON
            unformattedDataset['y_train'] = unformattedDataset['y_train'].tolist()

        # testing datapoints
        if torch.is_tensor(unformattedDataset['X_test']):
            # convert to list as a format savable by JSON
            unformattedDataset['X_test'] = unformattedDataset['X_test'].tolist()
        if torch.is_tensor(unformattedDataset['y_test']):
            # convert to list as a format savable by JSON
            unformattedDataset['y_test'] = unformattedDataset['y_test'].tolist()

        return unformattedDataset  # which is now formatted after calling this method

    def list2tensor(self, unformattedDataset):
        """
        Formats lists from a loaded dataset back to tensors for further computations.

        :param unfomattedDataset: Previously loaded dataset which still needs formatting.
        :type unformattedDataset: dict
        """
        # training datapoints
        if isinstance(unformattedDataset['X_train'], list):
            # convert to tensor for evaluation after loading the dataset from JSON
            unformattedDataset['X_train'] = torch.tensor(unformattedDataset['X_train'])
        if isinstance(unformattedDataset['y_train'], list):
            # convert to tensor for evaluation after loading the dataset from JSON
            unformattedDataset['y_train'] = torch.tensor(unformattedDataset['y_train'])

        # testing datapoints
        if isinstance(unformattedDataset['X_test'], list):
            # convert to tensor for evaluation after loading the dataset from JSON
            unformattedDataset['X_test'] = torch.tensor(unformattedDataset['X_test'])
        if isinstance(unformattedDataset['y_test'], list):
            # convert to tensor for evaluation after loading the dataset from JSON
            unformattedDataset['y_test'] = torch.tensor(unformattedDataset['y_test'])

        return unformattedDataset  # which is now formatted after calling this method
            

    def sympy2str(self, sympyFct, vars_list):
        """
        Converts sympy function to lambda expression. Returns a string in a lambda function format.

        :param sympyFct: Sympy expression to be converted.
        :type sympyFct: str
        :param vars_list: List of variable names.
        :type vars_list: list\[char\]

        :return: Expression which can be interpreted as a lambda function.
        :rtype: str
        """
        var_sym = symbols(vars_list)
        lmb_fct_str = lambdastr(var_sym, sympyFct)
        return lmb_fct_str

    def saveSimulation(self, save_file_name, generateData_dict, net_dict, trainConfig_dict, simulationResults_dict):
        """
        Saves the results of the computed simulation.

        :param save\_file\_name: Name of the JSON file in which the results are saved.
        :type save\_file\_name: str
        :param generateData\_dict: Dictionary containing the definitions for generating a dataset.
        :type generateData\_dict: dict
        :param net\_dict: Dictionary containing the settings of the Symbolic-Layered Neural Network.
        :type net\_dict: dict
        :param trainConfig\_dict: Dictionary containing the configurations of the network's training.
        :type trainConfig\_dict: dict
        :param simulationResults\_dict: Dictionary containing the computed simulation's results which are to be saved.
        """
        # create copies of original dictionaries in order to avoid errors in the elastic net iteration loop
        generateData_dict_copy = copy.deepcopy(generateData_dict)
        net_dict_copy = copy.deepcopy(net_dict)
        trainConfig_dict_copy = copy.deepcopy(trainConfig_dict)
        simulationResults_dict_copy = copy.deepcopy(simulationResults_dict)
        # remove list of objects for JSON, corresponding string version in the save file indicates how the hidden layers are structured
        net_dict_copy.pop('symbolic_layer')
        net_dict_copy.pop('symbolic_layer_div')
        # change tensors (training and testing datasets) back to lists for JSON
        generateData_dict_copy = self.tensor2list(generateData_dict_copy)
        # change sympy function to string (to make it savable for json)
        found_eq_str = self.sympy2str(simulationResults_dict_copy['found_eq'], trainConfig_dict_copy['variables_str'])
        simulationResults_dict_copy.update({'found_eq' : found_eq_str})
        saveFile_dict = {
            **simulationResults_dict_copy,
            **generateData_dict_copy,
            **net_dict_copy,
            **trainConfig_dict_copy
        }

        # save dictionary to json file in corresponding simulation folder
        with open(save_file_name, 'w') as outfile:
            json.dump(saveFile_dict, outfile, indent = 4)
