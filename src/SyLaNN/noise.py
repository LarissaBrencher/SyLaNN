'''
This file contains several types of noises which can be chosen to be applied during data creation.
'''

import torch

def createNoise(data_size, noise_std=0.01, type_str='white'):
    '''
    Creates noise which can be optionally chosen during data creation.
    The default setting returns a Gaussian (white) noise which is commonly used.
    Possible choices: white, more coming soon

    :param data\_size: Size of the given data tensor onto which the noise shall be added.
    :type data\_size: Tensor
    :param noise\_std: Standard deviation of the noise, default 0.01
    :type noise\_std: float
    :param type_str: Determines which type of noise should be applied, default white .
    :type type_str: str

    :return: Generated noise of chosen type
    :rtype: Tensor
    '''
    if type_str.__eq__('white'):
        # std choice according to example in
        # Learning Equations for Extrapolation and Control by Sahoo, Lampert, Martius
        # TODO make user specific choice in dict
        # TODO add user choice bool for noise
        noise = torch.normal(mean=0, std=noise_std, size=data_size)
        return noise