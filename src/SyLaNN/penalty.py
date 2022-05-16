"""
Defines the possible choices for regularization/penalty.
"""

import torch
import torch.nn as nn

class Penalty(nn.Module):
    """
    A class for the approximated penalty function which inherits from the PyTorch nn.Module.
    """
    def __init__(self, name=None):
        """
        Constructor method

        :param name: Name of the chosen penalty function
        :type name: string
        """
        # TODO add possiblities as example to documentation
        super(Penalty, self).__init__()

        self.name = name

    def forward(self, input_tensor, lmb=0.001, eps=0.001):
        """
        Returns the result of applying the penalty function to the input tensor with a
        fixed threshold (for approximations).

        :param input\_tensor: Predicted weight matrix to be evaluated via smooth penalty function
        :type input\_tensor: Tensor
        :param lmb: Regularization weight, default 0.001
        :type lmb: float
        :param eps: Threshold for the approximation, default 0.001
        :type eps: float

        :return: Penalized prediction
        :rtype: Tensor
        """
        reg_val = 0
        if self.name is None:
            return torch.tensor([reg_val])
        elif self.name.__eq__('L0'):
            for W_i in input_tensor:
                penalty = torch.count_nonzero(torch.abs(W_i))
                reg_val += penalty
            return lmb * torch.tensor([reg_val])
        elif self.name.__eq__('L1'):
            for W_i in input_tensor:
                penalty = torch.sum(torch.abs(W_i))
                reg_val += penalty
            return lmb * torch.tensor([reg_val])
        elif self.name.__eq__('L2'):
            for W_i in input_tensor:
                penalty = torch.sum(torch.square(W_i))
                reg_val += penalty
            return lmb * torch.tensor([reg_val])
        elif self.name.__eq__('L1approx'):
            return lmb * L1_approx(input_tensor, eps)
        elif self.name.__eq__('L12approx'):
            return lmb * L12_approx(input_tensor, eps)
        elif self.name.__eq__('ElasticNetapprox'):
            return lmb * elasticnet_approx(input_tensor, eps)
        else:
            raise ValueError("Wrong value chosen for regularizarion loss. \n Please re-evaluate the dictionary for training the Symbolic-Layered Neural Network.")

    def calculate_jacobian(self, weights):
        pass

    def calculate_hessian(self, weights):
        pass


def L1_approx(input_tensor, eps=0.001):
    if type(input_tensor) == list:
        return sum([L1_approx(tensor, eps) for tensor in input_tensor])
    input_squared = torch.square(input_tensor)
    approx = torch.sqrt(input_squared + eps**2) - eps
    return torch.sum(approx)

def L12_approx(input_tensor, eps=0.001):
    if type(input_tensor) == list:
        return sum([L12_approx(tensor, eps) for tensor in input_tensor])
    input_squared = torch.square(input_tensor)
    # alpha = 2
    # input_tensor * torch.tanh(alpha*input_tensor) 
    approx = torch.sqrt(torch.sqrt(input_squared + eps**2)) - eps**(1/2) # torch.sqrt(torch.sqrt(input_squared - eps**2))
    return torch.sum(approx)

def elasticnet_approx(input_tensor, eps=0.001):
    if type(input_tensor) == list:
        return sum([elasticnet_approx(tensor, eps) for tensor in input_tensor])
    L1part = L1_approx(input_tensor, eps)
    L2part = torch.sum(torch.square(input_tensor))
    return L1part + L2part # TODO add gamma