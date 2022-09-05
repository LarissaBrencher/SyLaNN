from sympy import *
import json
import copy

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

def createResultsDict(checkEN, path_name, file_format, gamma_list=[]):
    # create overall dictionary
    results_dict = {}

    if checkEN is True:
        # fill results dictionary with simulation results (nested dictionary of dicionaries)
        for gammaIdx, gammaVal in enumerate(gamma_list):
            current_key = gammaIdx
            current_json_name = path_name + str(current_key) + file_format
            # read current results file and save into dictionary
            json_file_data = open(current_json_name,)
            current_file = json.load(json_file_data)
            json_file_data.close()
            # write into overall dictionary at specific key
            results_dict[current_key] = current_file
    if checkEN is False:
        current_json_name = path_name + file_format
        # read current results file and save into dictionary
        json_file_data = open(current_json_name,)
        current_file = json.load(json_file_data)
        json_file_data.close()
        # write into overall dictionary at specific key
        results_dict = current_file
    return results_dict

def createBeautifDict(checkEN, results_dict, gamma_list=[]):
    # dictionary to save beautified/simplified expressions
    # (dict in dict; gammaIdx, then key for beautified version)
    eqs_beautified = {}

    if checkEN is True:
        # loop over found equations
        for gammaIdx, gammaVal in enumerate(gamma_list):
            tmp_eqDict = {}
            current_key = gammaIdx
            currentEq_str = results_dict[gammaIdx]["found_eq"]
            currentEq_sp = extract_sympyEq(currentEq_str)

            currentEq_long = currentEq_sp.expand()
            currentEq_simplified = autoSimplify(currentEq_long)
            currentEq_factorized = factorOut(currentEq_long)
            currentEq_rational = floats2rational(currentEq_long)
            tmp_eqDict['long_expanded'] = currentEq_long
            tmp_eqDict['long_simplified'] = currentEq_simplified
            tmp_eqDict['long_factorized'] = currentEq_factorized
            tmp_eqDict['long_rational'] = currentEq_rational

            currentEq_rounded4 = roundDecimals(currentEq_long)
            currentEq_simplified4 = autoSimplify(currentEq_rounded4)
            currentEq_factorized4 = factorOut(currentEq_rounded4)
            currentEq_rational4 = floats2rational(currentEq_rounded4)
            tmp_eqDict['rounded4_expanded'] = currentEq_rounded4
            tmp_eqDict['rounded4_simplified'] = currentEq_simplified4
            tmp_eqDict['rounded4_factorized'] = currentEq_factorized4
            tmp_eqDict['rounded4_rational'] = currentEq_rational4

            currentEq_rounded0 = roundDecimals(currentEq_long, roundToNthDecimal=0)
            currentEq_simplified0 = autoSimplify(currentEq_rounded0)
            currentEq_factorized0 = factorOut(currentEq_rounded0)
            currentEq_rational0 = floats2rational(currentEq_rounded0)
            tmp_eqDict['rounded0_expanded'] = currentEq_rounded0
            tmp_eqDict['rounded0_simplified'] = currentEq_simplified0
            tmp_eqDict['rounded0_factorized'] = currentEq_factorized0
            tmp_eqDict['rounded0_rational'] = currentEq_rational0

            eqs_beautified[current_key] = tmp_eqDict

    if checkEN is False:
        # loop over found equations
        currentEq_str = results_dict["found_eq"]
        currentEq_sp = extract_sympyEq(currentEq_str)

        currentEq_long = currentEq_sp.expand()
        currentEq_simplified = autoSimplify(currentEq_long)
        currentEq_factorized = factorOut(currentEq_long)
        currentEq_rational = floats2rational(currentEq_long)
        eqs_beautified['long_expanded'] = currentEq_long
        eqs_beautified['long_simplified'] = currentEq_simplified
        eqs_beautified['long_factorized'] = currentEq_factorized
        eqs_beautified['long_rational'] = currentEq_rational

        currentEq_rounded4 = roundDecimals(currentEq_long)
        currentEq_simplified4 = autoSimplify(currentEq_rounded4)
        currentEq_factorized4 = factorOut(currentEq_rounded4)
        currentEq_rational4 = floats2rational(currentEq_rounded4)
        eqs_beautified['rounded4_expanded'] = currentEq_rounded4
        eqs_beautified['rounded4_simplified'] = currentEq_simplified4
        eqs_beautified['rounded4_factorized'] = currentEq_factorized4
        eqs_beautified['rounded4_rational'] = currentEq_rational4

        currentEq_rounded0 = roundDecimals(currentEq_long, roundToNthDecimal=0)
        currentEq_simplified0 = autoSimplify(currentEq_rounded0)
        currentEq_factorized0 = factorOut(currentEq_rounded0)
        currentEq_rational0 = floats2rational(currentEq_rounded0)
        eqs_beautified['rounded0_expanded'] = currentEq_rounded0
        eqs_beautified['rounded0_simplified'] = currentEq_simplified0
        eqs_beautified['rounded0_factorized'] = currentEq_factorized0
        eqs_beautified['rounded0_rational'] = currentEq_rational0

    return eqs_beautified

def removeDuplicates_dictValues(eqs_dict):
    result_dict = {}
    # loop over values and do not include already existing values in new dict
    for key, value in eqs_dict.items():
        if value not in result_dict.values():
            result_dict[key] = value
    # return dict with removed duplicate equations
    return result_dict

def eq_beautifPrint(checkEN, eqs_dict, gamma_list=[]):
    # TODO still ugly with long expressions...
    if checkEN is True:
        for gammaIdx, gammaVal in enumerate(gamma_list):
            print('Predicted equation: ')
            print('(trained with elastic net penalty at L1-ratio %f)' % (gammaVal))
            print(eqs_dict[gammaIdx]['long_expanded'])
            alternate_forms = removeDuplicates_dictValues(eqs_dict[gammaIdx])
            alternate_forms.pop('long_expanded') # remove initial expression
            print('Alternate (approximated) forms: ')
            for value in alternate_forms.values():
                print(value)
            # for better readability after each gamma iteration (bottom line)
            print('#--##--##--##--##--##--##--##--##--##--##--##--##--##--##--##--#')
    if checkEN is False:
        print('Predicted equation: ')
        print(eqs_dict['long_expanded'])
        alternate_forms = removeDuplicates_dictValues(eqs_dict)
        alternate_forms.pop('long_expanded') # remove initial expression
        print('Alternate (approximated) forms: ')
        for value in alternate_forms.values():
            print(value)

def sympy2str(sympyFct, vars_list):
    var_sym = symbols(vars_list)
    lmb_fct_str = utilities.lambdify.lambdastr(var_sym, sympyFct)
    return lmb_fct_str

def saveEqBeautifDict(eqs_dict, vars_list, save_file_name):
    eqs_dict_copy = copy.deepcopy(eqs_dict)
    for key, value in eqs_dict.items():
        value_str = sympy2str(value, vars_list)
        eqs_dict_copy.update({key : value_str})

    # save dictionary to json file in corresponding simulation folder
    with open(save_file_name, 'w') as outfile:
        json.dump(eqs_dict_copy, outfile, indent = 4)