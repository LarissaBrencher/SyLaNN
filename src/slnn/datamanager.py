"""
Managing data of the Symbolic-Layered Neural Network framework

Either new input data is generated based on a given reference function or existing data can be used.
The results of the computed simulation can be saved and the predicted equation is formatted into a 
string lambda expression to simplify further processing.
"""

import inspect
import torch
import json
from sympy import symbols
from sympy.utilities.lambdify import lambdastr

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

        :param generateData_dict: Dictionary containing the definitions for generating a dataset.
        :type generateData_dict: dict

        :return: Returns generated dataset as dictionary.
        :rtype: dict
        """
        ref_fct = eval(generateData_dict['ref_fct_str'])
        x_dim = len(inspect.signature(ref_fct).parameters)     # Number of inputs to the function, or, dimensionality of x
        # create training data
        range_min_train = generateData_dict['domain_train'][0]
        range_max_train = generateData_dict['domain_train'][1]
        inputX_train = (range_max_train - range_min_train) * torch.rand([generateData_dict['n_train'], x_dim]) + range_min_train
        outputY_train = torch.tensor([[ref_fct(*x_i)] for x_i in inputX_train])
        # create testing data
        range_min_test = generateData_dict['domain_test'][0]
        range_max_test = generateData_dict['domain_test'][1]
        inputX_test = (range_max_test - range_min_test) * torch.rand([generateData_dict['n_test'], x_dim]) + range_min_test
        outputY_test = torch.tensor([[ref_fct(*x_i)] for x_i in inputX_test])
        # write generated data sets into dictionary
        dataset_dict = {
            'X_train' : inputX_train,
            'y_train' : outputY_train,
            'X_test' : inputX_test,
            'y_test' : outputY_test,
            'x_dim' : x_dim
        }
        return dataset_dict

    def loadDataset(self, file):
        """
        Loads a given dataset. (Currently not implemented)
        """
        pass

    def formatDataset(self, unformattedDataset):
        """
        Formats a given dataset into the required structure. (Currently not implemented)
        """
        pass

    def sympy2str(self, sympyFct, vars_list):
        """
        Converts sympy function to lambda expression. Returns a string in a lambda function format.

        :param sympyFct: Sympy expression to be converted.
        :type sympyFct: str
        :param vars_list: List of variable names.
        :type vars_list: list[char]

        :return: Expression which can be interpreted as a lambda function.
        :rtype: str
        """
        var_sym = symbols(vars_list)
        lmb_fct_str = lambdastr(var_sym, sympyFct)
        return lmb_fct_str

    def saveSimulation(self, save_file_name, generateData_dict, net_dict, trainConfig_dict, simulationResults_dict):
        """
        Saves the results of the computed simulation.

        :param save_file_name: Name of the JSON file in which the results are saved.
        :type save_file_name: str
        :param generateData_dict: Dictionary containing the definitions for generating a dataset.
        :type generateData_dict: dict
        :param net_dict: Dictionary containing the settings of the Symbolic-Layered Neural Network.
        :type net_dict: dict
        :param trainConfig_dict: Dictionary containing the configurations of the network's training.
        :type trainConfig_dict: dict
        :param simulationResults_dict: Dictionary containing the computed simulation's results which are to be saved.
        """
        net_dict.pop('symbolic_layer') 
        # change sympy function to string (to make it savable for json)
        found_eq_str = self.sympy2str(simulationResults_dict['found_eq'], trainConfig_dict['variables_str'])
        simulationResults_dict.update({'found_eq' : found_eq_str})
        saveFile_dict = {
            **simulationResults_dict,
            **generateData_dict, 
            **net_dict, 
            **trainConfig_dict
        }

        # save dictionary to json file in corresponding simulation folder
        with open(save_file_name, 'w') as outfile:
            json.dump(saveFile_dict, outfile, indent = 4)
