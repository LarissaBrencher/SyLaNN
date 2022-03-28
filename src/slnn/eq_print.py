import sympy as sym
import operators as ops

def apply_activFcts(W, fcts, n_binary=0):
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
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if abs(M[i, j]) < threshold:
                M[i, j] = 0
    return M

def slnn_print(weights, fcts, var_str, threshold=0.01, n_binary=0):
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
    n_binary = ops.count_binaryFcts(fcts)
    fcts = [fct.sp for fct in fcts]
    eq_srn = slnn_print(weights, fcts, var_str, threshold=threshold, n_binary=n_binary)
    found_eq = eq_srn[0, 0]
    return found_eq
