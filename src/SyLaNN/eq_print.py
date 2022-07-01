"""
Contains help functions to print out the predicted expression.
"""

import sympy as sym
import mathOperators as ops

def apply_activFcts(W, fcts, n_binary=0):
    """
    Applies unary and binary activation functions to the weight matrix.
    Returns the corresponding weight matrix with accordingly changed entries.
    
    :param W: Matrix containing the expressions (already multiplied with the weight matrix)
    :type W: sympy matrix
    :param fcts: Activation functions
    :type fcts: list\[sympy functions\]
    :param n\_binary: Number of binary activation functions, i.e. need two inputs, default 0
    :type n\_binary: int

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

def slnn_print(weights, fcts, var_str, threshold=0.01, n_binary=0, checkDivLayer=False, fctsDiv=[], n_binary_div=0):
    """
    Help function in order to generate a readable version of the SyLaNN structure.

    :param weights: Weight matrix
    :type weights: list\[numpy array\]
    :param fcts: Activation functions
    :type fcts: list\[objects\]
    :param var\_str: List of variable names.
    :type var\_str: list\[char\]
    :param threshold: Fixed threshold at which the matrix entries will be set to zero, default 0.01
    :type threshold: float
    :param n\_binary: Number of binary activation functions, i.e. need two inputs, default 0
    :type n\_binary: int
    :param checkDivLayer: Selection whether an additional division operator layer before the final output is desired, default False
    :type checkDivLayer: boolean
    :param fctsDiv: Activation functions of division layer (last layer before final output), default empty
    :type fctsDiv: list\[objects\]
    :param n\_binary: Number of binary activation functions (division layer), i.e. need two inputs, default 0
    :type n\_binary: int

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
    if checkDivLayer is False:
        for W in weights[:-1]:
            W = sparsify(sym.Matrix(W), threshold=threshold)
            eq = eq * W
            eq = apply_activFcts(eq, fcts, n_binary=n_binary)
        # output layer
        eq = eq * sparsify(sym.Matrix(weights[-1]))
        return eq
    if checkDivLayer is True:
        for W in weights[:-2]:
            W = sparsify(sym.Matrix(W), threshold=threshold)
            eq = eq * W
            eq = apply_activFcts(eq, fcts, n_binary=n_binary)
        # treat division layer separately
        W_div = sparsify(sym.Matrix(weights[-2]), threshold=threshold)
        eq = eq * W_div
        eq = apply_activFcts(eq, fctsDiv, n_binary_div)
        # output layer
        eq = eq * sparsify(sym.Matrix(weights[-1]))
        return eq



def network(weights, fcts, var_str, threshold=0.01, checkDivLayer=False, fctsDiv=[]):
    """
    Generates a readable version of the SyLaNN structure.

    :param weights: Weight matrix
    :type weights: list\[numpy array\]
    :param fcts: Activation functions
    :type fcts: list\[objects\]
    :param var\_str: List of variable names.
    :type var\_str: list\[char\]
    :param threshold: Fixed threshold at which the matrix entries will be set to zero, default 0.01
    :type threshold: float
    :param checkDivLayer: Selection whether an additional division operator layer before the final output is desired, default False
    :type checkDivLayer: boolean
    :param fctsDiv: Activation functions of division layer (last layer before final output), default empty
    :type fctsDiv: list\[objects\]

    :return: Readable expression of the SyLaNN's predicted equation
    :rtype: str
    """
    if checkDivLayer is False:
        n_binary = ops.count_binaryFcts(fcts)
        fcts = [fct.sp for fct in fcts]
        eq_slnn = slnn_print(weights, fcts, var_str, threshold=threshold, n_binary=n_binary, checkDivLayer=checkDivLayer)
        found_eq = eq_slnn[0, 0]
        return found_eq

    if checkDivLayer is True:
        n_binary = ops.count_binaryFcts(fcts)
        fcts = [fct.sp for fct in fcts]
        n_binary_div = ops.count_binaryFcts(fctsDiv)
        fcts_div = [fct.sp for fct in fctsDiv]
        eq_slnn = slnn_print(weights, fcts, var_str, threshold=threshold, n_binary=n_binary, checkDivLayer=checkDivLayer, fctsDiv=fcts_div, n_binary_div=n_binary_div)
        found_eq = eq_slnn[0, 0]
        return found_eq
