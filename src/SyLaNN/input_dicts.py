"""
Creates dictionaries which contain the specifics for generating data,
defining the SyLaNN's architecture and its training configurations.
"""

import mathOperators as ops

def readDictionaries():
    """
    Returns dictionaries which contain the specifics for generating data,
    defining the SyLaNN's architecture and its training configurations.

    :return: generateData\_dict, net\_dict, trainConfig\_dict
    :rtype: dict
    """
    # for generating data
    # "_1Dlinear" 'lambda x : x + 8'
    # "_2Dpoly" 'lambda x,y : x**2*y + x*y**2'
    # "_Langmuir" 'lambda x : 0.00059 * (1*x) / (1 + 1*x)', # k=1 q_max = 5.9e-4
    # Note regarding Langmuir isotherm: sample only positive numbers (0 to 1 for testing)
    # and use standardization/centralization to compensate and study the effect within SyLaNN
    generateData_dict = {
        'n_train' : 2000,
        'n_test' : 1000,
        'domain_train' : [0.25, 0.75],
        'domain_test' : [0, 1],
        'ref_fct_str' : 'lambda x : 0.00059 * (1*x) / (1 + 1*x)',
        'saveFile_name' : "_Langmuir",
        'checkNoise' : True,
        'noise_type' : 'white',
        'noise_std' : 0.01,
        'standardize_or_centralize' : None
    }

    # for creating the neural network structure
    net_dict = {
        'n_hidden' : 2,
        'checkDivLayer' : False,
        'divThreshold' : 1, # everything below 1 can be expressed as a product alternatively
        'symbolic_layer' : [
                            *[ops.Constant()] * 2,
                            *[ops.Identity()] * 4,
                            *[ops.Square()] * 4,
                            *[ops.Exp()] * 2,
                            *[ops.Product()] * 2
                            ],
        'symbolic_layer_str': [
                            'Const', 'Const', 
                            'Id', 'Id', 'Id', 'Id', 
                            'Sq', 'Sq', 'Sq', 'Sq',
                            'Exp', 'Exp',
                            'Prod', 'Prod'
                            ],
        'symbolic_layer_div' : [
                            *[ops.Identity()] * 6,
                            *[ops.Quotient()] * 8 # same number of nodes like symbolic layer!
                            ],
        'symbolic_layer_str_div' : [
                            'Id', 'Id',  
                            'Id', 'Id', 'Id', 'Id', 
                            'Quot', 'Quot', 'Quot', 'Quot',
                            'Quot', 'Quot',
                            'Quot', 'Quot'
                            ]
    }

    # for configuring the settings and hyperparameters for the network's training
    # LBFGS vs Adam: 50,100, then 100:100:1000, then 1000,5000,10000,15000
    # BR's init alpha, beta according to Bayesian regularization paper
    trainConfig_dict = {
        'variables_str' : ['x'],
         # 'L12approx', 'ElasticNetapprox'
        'loop1Reg' : 'L12approx',
        'loop2Reg' : 'L12approx',
        'loop3Reg' : 'L12approx',
        'regApprox_threshold': 0.0001,
        'learning_rate' : 0.001,
        'cutWeights_threshold' : 0.01,
        'regularization_weight' : 0.001,
        'trainEpochs1' : 10,
        'trainEpochs2' : 50,
        'trainEpochs3' : 10000, # 500 for BR True, 10000 for BR False # according to Martius, Adam needs (L-1)*10000 epochs with L being the number of hidden layers
        'optimizer' : 'Adam',
        'chooseBR' : False, # choose whether Bayesian regularization (BR) is desired during training
        'error_data_factor' : 1, # choose initial value of BR's prefactor for data error
        'error_reg_factor' : 0, # choose initial value of BR's prefactor for regularization error
        'updateBRparams_every_n_epoch' : 5
    }

    return generateData_dict, net_dict, trainConfig_dict