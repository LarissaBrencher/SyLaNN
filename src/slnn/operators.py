import torch
import numpy as np
import sympy as sp


class BaseFunction:
    def __init__(self, norm=1):
        self.norm = norm

    def sp(self, x):
        return None

    def torch(self, x):
        return None

    def np(self, x):
        z = sp.symbols('z')
        return sp.utilities.lambdify(z, self.sp(z), 'numpy')(x)


class Constant(BaseFunction):
    def torch(self, x):
        return torch.ones_like(x)

    def sp(self, x):
        return 1

    def np(self, x):
        return np.ones_like


class Identity(BaseFunction):
    def torch(self, x):
        return x / self.norm

    def sp(self, x):
        return x / self.norm

    def np(self, x):
        return np.array(x) / self.norm


class Square(BaseFunction):
    def torch(self, x):
        return torch.square(x) / self.norm

    def sp(self, x):
        return x ** 2 / self.norm

    def np(self, x):
        return np.square(x) / self.norm


class Exp(BaseFunction):
    def __init__(self, norm=np.e):
        super().__init__(norm)

    def torch(self, x):
        return (torch.exp(x) - 1) / self.norm

    def sp(self, x):
        return (sp.exp(x) - 1) / self.norm


class BaseFunction2:
    def __init__(self, norm=1.):
        self.norm = norm

    def sp(self, x, y):
        return None

    def torch(self, x, y):
        return None

    def np(self, x, y):
        a, b = sp.symbols('a b')
        return sp.utilities.lambdify([a, b], self.sp(a, b), 'numpy')(x, y)


class Product(BaseFunction2):
    def __init__(self, norm=0.1):
        super().__init__(norm=norm)

    def torch(self, x, y):
        return x * y / self.norm

    def sp(self, x, y):
        return x*y / self.norm


def count_inputs(fcts):
    i = 0
    for fct in fcts:
        if isinstance(fct, BaseFunction):
            i += 1
        elif isinstance(fct, BaseFunction2):
            i += 2
    return i


def count_binaryFcts(fcts):
    i = 0
    for fct in fcts:
        if isinstance(fct, BaseFunction2):
            i += 1
    return i

# implicitly gives the number of neurons per SR layer with the int replications
default_func = [
    *[Constant()] * 2,
    *[Identity()] * 4,
    *[Square()] * 2,
    *[Exp()] * 4,
    *[Product()] * 2,
]