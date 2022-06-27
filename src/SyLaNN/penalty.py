"""
Defines the possible choices for regularization/penalty.
"""
# Note, the regularization weight lambda is removed for BR

import torch
import torch.nn as nn

class Penalty(nn.Module):
    """
    A class for the approximated penalty function which inherits from the PyTorch nn.Module.
    """
    def __init__(self, name=None):
        """
        Constructor method
        (Possible choices for penalty name: None, 'L0', 'L1', 'L1approx', 'L12approx', 'L2', 'ElasticNetapprox')

        :param name: Name of the chosen penalty function
        :type name: string
        """

        super(Penalty, self).__init__()

        self.name = name

    def forward(self, input_tensor, lmb=0.001, eps=0.001, gamma=0.5):
        """
        Returns the result of applying the penalty function to the input tensor with a
        fixed threshold (for approximations).

        :param input\_tensor: Predicted weight matrix to be evaluated via smooth penalty function
        :type input\_tensor: Tensor
        :param lmb: Regularization weight, default 0.001
        :type lmb: float
        :param eps: Threshold for the approximation, default 0.001
        :type eps: float
        :param gamma: L1 ratio of elastic net penalty, default 0.5
        :type gamma: float

        :return: Penalized prediction
        :rtype: Tensor
        """
        reg_val = 0
        # removed 'lmb *' at each return
        if self.name is None:
            return torch.tensor([reg_val])
        elif self.name.__eq__('L0'):
            for W_i in input_tensor:
                penalty = torch.count_nonzero(torch.abs(W_i))
                reg_val += penalty
            return torch.tensor([reg_val])
        elif self.name.__eq__('L1'):
            for W_i in input_tensor:
                penalty = torch.sum(torch.abs(W_i))
                reg_val += penalty
            return torch.tensor([reg_val])
        elif self.name.__eq__('L2'):
            for W_i in input_tensor:
                penalty = torch.sum(torch.square(W_i))
                reg_val += penalty
            return torch.tensor([reg_val])
        elif self.name.__eq__('L1approx'):
            return L1_approx(input_tensor, eps)
        elif self.name.__eq__('L12approx'):
            return L12_approx(input_tensor, eps)
        elif self.name.__eq__('ElasticNetapprox'):
            return elasticnet_approx(input_tensor, eps, gamma)
        else:
            raise ValueError("Wrong value chosen for regularizarion loss. \n Please re-evaluate the dictionary for training the Symbolic-Layered Neural Network.")

    def calculate_jacobian(self, weights, eps=0.001):
        if self.name is None:
            raise ValueError("No regularization is chosen. Consequently, the Jacobian cannot be computed.")
        elif self.name.__eq__('L0'):
            raise ValueError("L0 regularization is chosen. Due to the non-differentiability the Jacobian cannot be computed.")
        elif self.name.__eq__('L1'):
            raise ValueError("L1 regularization is chosen. Due to the non-differentiability the Jacobian cannot be computed. If L1 with BR optimization is desired, please choose the L1 approximation.")
        elif self.name.__eq__('L2'):
            pass
        elif self.name.__eq__('L1approx'):
            pass
        elif self.name.__eq__('L12approx'):
            pass
        elif self.name.__eq__('ElasticNetapprox'):
            pass
        else:
            raise ValueError("Wrong value chosen for regularizarion loss. \n Please re-evaluate the dictionary for training the Symbolic-Layered Neural Network.")

    def calculate_hessian(self, input_tensor, eps=0.001, gamma=None):
        weights = torch.nn.utils.parameters_to_vector(input_tensor)
        hessian = torch.zeros(len(weights))
        if self.name is None:
            return torch.diag(hessian) # hessian only contains zeros, so no effect when no regularization is chosen
        elif self.name.__eq__('L0'):
            raise ValueError("L0 regularization is chosen. Due to the non-differentiability the Hessian cannot be computed.")
        elif self.name.__eq__('L1'):
            raise ValueError("L1 regularization is chosen. Due to the non-differentiability the Hessian cannot be computed. If L1 with BR optimization is desired, please choose the L1 approximation.")
        elif self.name.__eq__('L2'):
            # TODO just create torch.eye of corresponding size, less computational time
            for idx, W_i in enumerate(weights):
                h_value = 2
                hessian[idx] = h_value
            return torch.diag(hessian)
        elif self.name.__eq__('L1approx'):
            for idx, W_i in enumerate(weights):
                h_value = eps**2 / ((torch.square(W_i) + eps**2)**(3/2))
                hessian[idx] = h_value
            return torch.diag(hessian)
        elif self.name.__eq__('L12approx'):
            for idx, W_i in enumerate(weights):
                h_value = (2*eps**2 - W_i**2) / (4*(torch.square(W_i) + eps**2)**(7/4))
                hessian[idx] = h_value
            return torch.diag(hessian)
        elif self.name.__eq__('ElasticNetapprox'):
            for idx, W_i in enumerate(weights):
                h_value_l1 = gamma*eps**2 / ((torch.square(W_i) + eps**2)**(3/2))
                h_value_l2 = (1-gamma)*2
                h_value = h_value_l1 + h_value_l2
                hessian[idx] = h_value
            return torch.diag(hessian)
        else:
            raise ValueError("Wrong value chosen for regularizarion loss. \n Please re-evaluate the dictionary for training the Symbolic-Layered Neural Network.")


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
    approx = torch.sqrt(torch.sqrt(input_squared + eps**2)) - eps**(1/2)
    return torch.sum(approx)

def elasticnet_approx(input_tensor, eps=0.001, l1ratio=0.5):
    if type(input_tensor) == list:
        return sum([elasticnet_approx(tensor, eps) for tensor in input_tensor])
    L1part = L1_approx(input_tensor, eps)
    L2part = torch.sum(torch.square(input_tensor))
    return l1ratio * L1part + (1-l1ratio) * L2part