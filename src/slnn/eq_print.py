"""
Contains help functions to print out the predicted expression.
"""

import sympy as sym
import operators as ops

def apply_activFcts(W, fcts, n_binary=0):
    """
    Applies unary and binary activation functions to the weight matrix.
    Returns the corresponding weight matrix with accordingly changed entries.
    
    :param W: Matrix containing the expressions (already multiplied with the weight matrix)
    :type W: sympy matrix
    :param fcts: Activation functions
    :type fcts: list[sympy functions]
    :param n_binary: Number of binary activation functions, i.e. need two inputs, default 0
    :type n_binary: int

    :return: Weight matrix to which the activation functions have been applied.
    :rtype: sympy matrix
    """
    W = sym.Matrix(W)
    if n_binary == 0:
        for i in range(W.shape[0]):
            for j in range(W.shape[1]):
                W[i, j] = fcts[j](W[i, j])
    else:
        W_new = W.copy()
        out_size = len(fcts)
        for i in range(W.shape[0]):
            in_j = 0
            out_j = 0
            while out_j < out_size - n_binary:
                W_new[i, out_j] = fcts[out_j](W[i, in_j])
                in_j += 1
                out_j += 1
            while out_j < out_size:
                W_new[i, out_j] = fcts[out_j](W[i, in_j], W[i, in_j+1])
                in_j += 2
                out_j += 1
        for i in range(n_binary):
            W_new.col_del(-1)
        W = W_new
    return W

def sparsify(M, threshold=0.01):
    """
    Sets matrix entries below (in an absolute sense) a certain threshold to zero.

    :param M: Weight matrix
    :type M: sympy matrix
    :param threshold: Fixed threshold at which the matrix entries will be set to zero, default 0.01
    :type threshold: float

    :return: Sparse matrix depending on the threshold.
    :rtype: sympy matrix
    """
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) < threshold:
                M[i, j] = 0
    return M

def slnn_print(weights, fcts, var_str, threshold=0.01, n_binary=0):
    """
    Help function in order to generate a readable version of the SLNN structure.

    :param weights: Weight matrix
    :type weights: list[numpy array]
    :param fcts: Activation functions
    :type fcts: list[objects]
    :param var_str: List of variable names.
    :type var_str: list[char]
    :param threshold: Fixed threshold at which the matrix entries will be set to zero, default 0.01
    :type threshold: float
    :param n_binary: Number of binary activation functions, i.e. need two inputs, default 0
    :type n_binary: int

    :return: Expression matrix
    :rtype: sympy matrix
    """
    vars = []
    for var in var_str:
        if isinstance(var, str):
            vars.append(sym.Symbol(var))
        else:
            vars.append(var)
    eq = sym.Matrix(vars).T
    for W in weights[:-1]:
        W = sparsify(sym.Matrix(W), threshold=threshold)
        eq = eq * W
        eq = apply_activFcts(eq, fcts, n_binary=n_binary)
    eq = eq * sparsify(sym.Matrix(weights[-1]))
    return eq

def network(weights, fcts, var_str, threshold=0.01):
    """
    Generates a readable version of the SLNN structure.

    :param weights: Weight matrix
    :type weights: list[numpy array]
    :param fcts: Activation functions
    :type fcts: list[objects]
    :param var_str: List of variable names.
    :type var_str: list[char]
    :param threshold: Fixed threshold at which the matrix entries will be set to zero, default 0.01
    :type threshold: float

    :return: Readable expression of the SLNN's predicted equation
    :rtype: str
    """
    n_binary = ops.count_binaryFcts(fcts)
    fcts = [fct.sp for fct in fcts]
    eq_slnn = slnn_print(weights, fcts, var_str, threshold=threshold, n_binary=n_binary)
    found_eq = eq_slnn[0, 0]
    return found_eq
