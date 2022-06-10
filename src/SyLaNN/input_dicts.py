"""
Creates dictionaries which contain the specifics for generating data,
defining the SLNN's architecture and its training configurations.
"""

import mathOperators as ops

def readDictionaries():
    """
    Returns dictionaries which contain the specifics for generating data,
    defining the SLNN's architecture and its training configurations.

    :return: generateData\_dict, net\_dict, trainConfig\_dict
    :rtype: dict
    """
    # for generating data
    # 16-08-21: currently using LBFGS optimizer always so no need to include in json name
    # exNum = 1 # numbering for 1 to 7 (examples in research project 2020-2022)
    # 'lambda x : x + 8' # standard linear funtion to start with
    # 'lambda x : x**2 + 3*x - 7' # second degree polynomial
    # 'lambda x : x**3 + 2*x**2 - x + 1' # third degree polynomial
    # 'lambda x : torch.exp(x)' # struggle with exp detection, often some streched polynomial with even exponent
    # 'lambda x, y : x**2 + 2*x*y + y**2' # two vars polynomial, make learning rate for two vars 0.01
    # 'lambda x, y : x**2*y + x*y**2' # non-separable, needs smaller learning rate at 0.01, all others at 0.05 -> adaptive scheduler
    # 'lambda x : torch.exp(x) - x - 3'

    # LBFGS vs Adam: 2000:500:7000 (n_test accordingly)
    # for 15000 epochs 2000:500:4000
    generateData_dict = {
        'n_train' : 2000,
        'n_test' : 1000,
        'domain_train' : [0.25, 0.75],
        'domain_test' : [0, 1],
        'ref_fct_str' : 'lambda C : 0.00035 * C ** (0.874)', # K=3.5e-4, n=0.874
        'saveFile_name' : "_FreundlichIsotherm"
    }

    # for creating the neural network structure
    net_dict = {
        'n_hidden' : 2,
        'symbolic_layer' : [
                            *[ops.Constant()] * 2,
                            *[ops.Identity()] * 4,
                            *[ops.Square()] * 4,
                            *[ops.Exp()] * 2,
                            *[ops.Product()] * 2,
                            *[ops.Power()] * 2
                            ],
        'symbolic_layer_str': [
                            'Const', 'Const', 
                            'Id', 'Id', 'Id', 'Id', 
                            'Sq', 'Sq', 'Sq', 'Sq',
                            'Exp', 'Exp',
                            'Prod', 'Prod',
                            'Pow', 'Pow'
                            ]
    }

    # for configuring the settings and hyperparameters for the network's training
    # LBFGS vs Adam: 50,100, then 100:100:1000, then 1000,5000,10000,15000
    # BR's init alpha, beta according to Bayesian regularization paper
    trainConfig_dict = {
        'variables_str' : ['x', 'y', 'z'], # ['C'],
        'loop1Reg' : 'ElasticNetapprox', # None, # warm-up phase
        'loop2Reg' : 'ElasticNetapprox', # 'L12approx',
        'loop3Reg' : 'ElasticNetapprox', # 'L12approx',
        'regApprox_threshold': 0.01,
        'learning_rate' : 0.01,
        'regularization_weight' : 0.001,
        'trainEpochs1' : 5,
        'trainEpochs2' : 20,
        'trainEpochs3' : 80,
        'optimizer' : 'LBFGS',
        'chooseBR' : True, # choose whether Bayesian regularization (BR) is desired during training
        'error_data_factor' : 1, # choose initial value of BR's prefactor for data error
        'error_reg_factor' : 0 # choose initial value of BR's prefactor for regularization error
    }

    return generateData_dict, net_dict, trainConfig_dict