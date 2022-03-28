import inspect
import torch
import json
from sympy import symbols
from sympy.utilities.lambdify import lambdastr

# Manage data (either generate or use existing dataset)
class DataManager():
    def __init__(self):
        super().__init__()

    def generateData(self, generateData_dict):
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
        pass

    def formatDataset(self, unformattedDataset):
        pass

    def sympy2str(self, sympyFct, vars_list):
        # convert sympy function to string lambda expression
        var_sym = symbols(vars_list)
        lmb_fct_str = lambdastr(var_sym, sympyFct)
        return lmb_fct_str

    def saveSimulation(self, save_file_name, generateData_dict, net_dict, trainConfig_dict, simulationResults_dict):
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
