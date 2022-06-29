"""
Defines the SyLaNN's operator nodes which are applied to each neuron in each custom layer.
"""

import torch
import numpy as np
import sympy as sp


class BaseFunction:
    """A class for unary mathematical operators, i.e. function which take one input."""
    def __init__(self, norm=1):
        """
        Constructor method
        
        :param norm: Normalizing factor of BaseFunction, default 1
        :type norm: int
        """
        self.norm = norm

    def sp(self, x):
        """
        No conversion needed for SymPy.
        
        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        """
        return None

    def torch(self, x):
        """
        No conversion needed for PyTorch.
        
        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        """
        return None

    def np(self, x):
        """
        Converts SymPy to NumPy
        
        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Callable NumPy function
        :rtype: lambda function
        """
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)


class Constant(BaseFunction):
    """
    A class for the mathematical operator containing constants.
    Takes BaseFunction as an argument.
    """
    def torch(self, x):
        """
        Returns a tensor filled with scalar ones in the same size as the given input.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Ones-filled tensor with input's shape
        :rtype: Tensor
        """
        return torch.ones_like(x)

    def sp(self, x):
        """
        Returns scalar 1.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: The number one as scalar value
        :rtype: int
        """
        return 1

    def np(self, x):
        """
        Returns a NumPy array filled with scalar ones in the same size as the given input.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: NumPy array

        :return: Ones-filled matrix with input's shape
        :rtype: NumPy array
        """
        return np.ones_like(x)


class Identity(BaseFunction):
    """
    A class for the mathematical operator applying the identity function.
    Takes BaseFunction as an argument.
    """
    def torch(self, x):
        """
        Returns a tensor (identity function applied to its entries).

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting tensor after the identity function has been applied to its entries.
        :rtype: Tensor
        """
        return x / self.norm

    def sp(self, x):
        """
        Returns a tensor (identity function applied to its entries).

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting tensor after the identity function has been applied to its entries.
        :rtype: Tensor
        """
        return x / self.norm

    def np(self, x):
        """
        Returns a NumPy array (identity function applied to its entries).

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting array after the identity function has been applied to its entries.
        :rtype: NumPy array
        """
        return np.array(x) / self.norm


class Square(BaseFunction):
    """
    A class for the mathematical operator applying the square function.
    Takes BaseFunction as an argument.
    """
    def torch(self, x):
        """
        Returns a tensor (square function applied to its entries).

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting tensor after the square function has been applied to its entries.
        :rtype: Tensor
        """
        return torch.square(x) / self.norm

    def sp(self, x):
        """
        Returns a tensor (square function applied to its entries).

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting tensor after the square function has been applied to its entries.
        :rtype: Tensor
        """
        return x ** 2 / self.norm

    def np(self, x):
        """
        Returns the element-wise square of the given input.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting tensor after the square function has been applied to its entries.
        :rtype: NumPy array
        """
        return np.square(x) / self.norm


class Exp(BaseFunction):
    """
    A class for the mathematical operator applying the exponential function.
    Takes BaseFunction as an argument.
    """
    def __init__(self, norm=np.e):
        """
        Constructor method, inherits from BaseFunction with adjusted norm parameter
        
        :param norm: Normalizing factor of BaseFunction, default np.e
        :type norm: int
        """
        super().__init__(norm)

    def torch(self, x):
        """
        Returns a tensor (exponential function applied to its entries).

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting tensor after the exponential function has been applied to its entries.
        :rtype: Tensor
        """
        return (torch.exp(x) - 1) / self.norm

    def sp(self, x):
        """
        Returns a tensor (exponential function applied to its entries).

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting tensor after the exponential function has been applied to its entries.
        :rtype: Tensor
        """
        return (sp.exp(x) - 1) / self.norm

    def np(self, x):
        """
        Returns the element-wise exponential of the given input.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor

        :return: Resulting tensor after the exponential function has been applied to its entries.
        :rtype: NumPy array
        """
        return (np.exp(x) - 1) / self.norm


#class Power(BaseFunction):
#    """
#    A class for the mathematical operator applying the power with a constant exponent. The first input is the base, the second one is the exponent.
#    Takes BaseFunction as an argument.
#    """
#    def __init__(self, norm=1.):
#        """
#        Constructor method, inherits from BaseFunction with adjusted norm parameter
#        
#        :param norm: Normalizing factor of BaseFunction, default 0.1
#        :type norm: int
#        """
#        super().__init__(norm=norm)
#        self.exponent = 1/2 # exponent
#
#    def torch(self, x):
#        """
#        Returns a tensor which contains the power of two inputs.
#
#        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
#        :type x: Tensor
#        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
#        :type y: Tensor
#
#        :return: Resulting tensor after the power has been applied.
#        :rtype: Tensor
#        """
#        # exponent = Constant()
#        return torch.pow(x, self.exponent) / self.norm # ** exponent.torch(x) / self.norm # torch.pow(x, self.exponent) / self.norm
#
#    def sp(self, x):
#        """
#        Returns a tensor which contains the power of two inputs.
#
#        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
#        :type x: Tensor
#        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
#        :type y: Tensor
#
#        :return: Resulting tensor after the power has been applied.
#        :rtype: Tensor
#        """
#        return x**(0.5) # x ** self.exponent / self.norm
#
#    def np(self, x):
#        """
#        Returns a NumPy array which contains the power of two inputs.
#
#        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
#        :type x: NumPy array
#        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
#        :type y: NumPy array
#
#        :return: Resulting tensor after the power has been applied.
#        :rtype: NumPy array
#        """
#        return np.sqrt(x) # np.power(x, self.exponent) / self.norm


class BaseFunction2:
    """A class for binary mathematical operators, i.e. function which take two inputs."""
    def __init__(self, norm=1.):
        """
        Constructor method
        
        :param norm: Normalizing factor of BaseFunction2, default 1
        :type norm: float
        """
        self.norm = norm

    def sp(self, x, y):
        """
        No conversion needed for SymPy.
        
        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: Tensor
        """
        return None

    def torch(self, x, y):
        """
        No conversion needed for PyTorch.
        
        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: Tensor
        """
        return None

    def np(self, x, y):
        """
        Converts SymPy to NumPy
        
        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: Tensor

        :return: Callable NumPy function
        :rtype: lambda function
        """
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)


class Product(BaseFunction2):
    """
    A class for the mathematical operator applying the product of two inputs.
    Takes BaseFunction2 as an argument.
    """
    def __init__(self, norm=0.1):
        """
        Constructor method, inherits from BaseFunction2 with adjusted norm parameter
        
        :param norm: Normalizing factor of BaseFunction2, default 0.1
        :type norm: int
        """
        super().__init__(norm=norm)

    def torch(self, x, y):
        """
        Returns a tensor which contains the product between both inputs.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: Tensor

        :return: Resulting tensor after the multiplication has been applied.
        :rtype: Tensor
        """
        return x * y / self.norm

    def sp(self, x, y):
        """
        Returns a tensor which contains the product between both inputs.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: Tensor

        :return: Resulting tensor after the multiplication has been applied.
        :rtype: Tensor
        """
        return x * y / self.norm

    def np(self, x, y):
        """
        Returns a NumPy array which contains the product between both inputs.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: NumPy array
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: NumPy array

        :return: Resulting tensor after the multiplication has been applied.
        :rtype: NumPy array
        """
        return x * y / self.norm


# TODO add some assert statements to ensure that y is never zero
class Quotient(BaseFunction2):
    """
    A class for the mathematical operator applying the quotient of two inputs.
    Takes BaseFunction2 as an argument.
    """
    def __init__(self, norm=1.):
        """
        Constructor method, inherits from BaseFunction2 with adjusted norm parameter
        
        :param norm: Normalizing factor of BaseFunction2, default 0.1
        :type norm: int
        """
        super().__init__(norm=norm)

    def torch(self, x, y):
        """
        Returns a tensor which contains the quotient between both inputs.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: Tensor

        :return: Resulting tensor after the division has been applied.
        :rtype: Tensor
        """
        return (x / y) / self.norm

    def sp(self, x, y):
        """
        Returns a tensor which contains the quotient between both inputs.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: Tensor
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: Tensor

        :return: Resulting tensor after the division has been applied.
        :rtype: Tensor
        """
        return (x / y) / self.norm

    def np(self, x, y):
        """
        Returns a NumPy array which contains the quotient between both inputs.

        :param x: Input within SyLaNN's custom layer (previous to the application of operators)
        :type x: NumPy array
        :param y: Input within SyLaNN's custom layer (previous to the application of operators)
        :type y: NumPy array

        :return: Resulting tensor after the division has been applied.
        :rtype: NumPy array
        """
        return (x / y) / self.norm


def count_inputs(fcts):
    """
    Returns the number of inputs according to the count of unary and binary functions.

    :param fcts: Activation functions
    :type fcts: list\[objects\]

    :return: Number of inputs
    :rtype: int
    """
    i = 0
    for fct in fcts:
        if isinstance(fct, BaseFunction):
            i += 1
        elif isinstance(fct, BaseFunction2):
            i += 2
    return i


def count_binaryFcts(fcts):
    """
    Returns the number of binary functions.

    :param fcts: Activation functions
    :type fcts: list\[objects\]

    :return: Number of binary functions
    :rtype: int
    """
    i = 0
    for fct in fcts:
        if isinstance(fct, BaseFunction2):
            i += 1
    return i

# implicitly gives the number of neurons per SymLayer with the int replications
# TODO document this default somewhere
default_func = [
    *[Constant()] * 2,
    *[Identity()] * 4,
    *[Square()] * 2,
    *[Exp()] * 4,
    *[Product()] * 2,
]

default_divLayer = [
    *[Quotient()] * 2
]