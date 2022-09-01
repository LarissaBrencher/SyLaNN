from sympy import *

def extract_sympyEq(lambda_str):
    split_here = lambda_str.find(':') + 2 # cut off colon and space
    expr_str = lambda_str[split_here:]
    expr = parse_expr(expr_str, evaluate=False)
    return expr

def roundDecimals(sympy_expr, roundToNthDecimal=4):
    expr_rounded = sympy_expr
    for idx in preorder_traversal(sympy_expr):
        if isinstance(idx, Float):
            expr_rounded = expr_rounded.subs(idx, round(idx, roundToNthDecimal))
    return expr_rounded

def floats2rational(sympy_expr):
    return nsimplify(sympy_expr)

def factorOut(sympy_expr):
    return sympy_expr.factor()

def autoSimplify(sympy_expr):
    return sympy_expr.simplify()

def eq_beautifPrint(eqs_dict):
    pass