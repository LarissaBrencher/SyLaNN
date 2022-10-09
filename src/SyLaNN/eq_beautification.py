from sympy import *
import json
import copy

import datamanager as dm

def replaceMathwithSympy(result_dict):
    """
    Replaces 'math.' within the found equation string, in order for it to be processable in the equation beautification.

    :param result\_dict: Dictionary containing the simulation results
    :type result\_dict: dict
    :return: Dictionary containing the updated simulation results
    :rtype: dict
    """
    dict_copy = copy.deepcopy(result_dict)
    expr = dict_copy['found_eq']
    math_str = 'math.'
    if math_str in expr.lower():
        expr_np = expr.replace(math_str, '')
        dict_copy.update({'found_eq' : expr_np})
    return dict_copy

def extract_sympyEq(lambda_str):
    """
    Extracts the predictied equation as a SymPy expression.

    :param lambda\_str: Found lambda function to parse.
    :type lambda\_str: str
    :return: Parsed equation
    :rtype: SymPy expression
    """
    split_here = lambda_str.find(':') + 2 # cut off colon and space
    expr_str = lambda_str[split_here:]
    expr = parse_expr(expr_str, evaluate=False)
    return expr

def roundDecimals(sympy_expr, roundToNthDecimal=4):
    """
    Rounds SymPy expression to a defined number of decimals.

    :param sympy\_expr: Predicted equation to be rounded
    :type sympy\_expr: SymPy expression
    :param roundToNthDecimal: Defines to how many decimals the result should be rounded, default 4
    :type roundToNthDecimal: int
    :return: Updated rounded expression
    :rtype: SymPy expression
    """
    expr_rounded = sympy_expr
    for idx in preorder_traversal(sympy_expr):
        if isinstance(idx, Float):
            expr_rounded = expr_rounded.subs(idx, round(idx, roundToNthDecimal))
    return expr_rounded

def floats2rational(sympy_expr):
    """
    Changes float numbers to rational numbers in the given SymPy expression.

    :param sympy\_expr: Predicted equation to be changed
    :type sympy\_expr: SymPy expression
    :return: Updated rational expression
    :rtype: SymPy expression
    """
    return nsimplify(sympy_expr)

def factorOut(sympy_expr):
    """
    Takes a SymPy expression and factors it into irreducible factors over the rational numbers.

    :param sympy\_expr: Predicted equation to be factored
    :type sympy\_expr: SymPy expression
    :return: Factored expression
    :rtype: SymPy expression
    """
    return sympy_expr.factor()

def autoSimplify(sympy_expr):
    """
    Takes a SymPy expression and simplifies it.

    :param sympy\_expr: Predicted equation to be simplified
    :type sympy\_expr: SymPy expression
    :return: Simplified expression
    :rtype: SymPy expression
    """
    return sympy_expr.simplify()

def createResultsDict(checkEN, path_name, file_format, gamma_list=[]):
    """
    Creates a single dictionary with all chosen simulation results.

    :param checkEN: Check if the elastic net penalty is chosen
    :type checkEN: boolean
    :param path\_name: Directory to the JSON file which contains the dictionary
    :type path\_name: str
    :param file\_format: Defines the file format in which the dictionary is saved
    :type file\_format: str
    :return: Dictionary containing the results which are to be beautified
    :rtype: dict
    """
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
    """
    Creates a dictionary with all versions of beautified equations.

    :param checkEN: Check if the elastic net penalty is chosen
    :type checkEN: boolean
    :param results\_dict: Dictionary containing the simulation results
    :type results\_dict: dict
    :param gamma\_list: Contains the values for the L1 ratio (see elastic net penalty definition)
    :type gamma\_list: list(float)
    :return: Dictionary containing the beautified equations
    :rtype: dict
    """
    # dictionary to save beautified/simplified expressions
    # (dict in dict; gammaIdx, then key for beautified version)
    eqs_beautified = {}

    if checkEN is True:
        # loop over found equations
        for gammaIdx, gammaVal in enumerate(gamma_list):
            tmp_eqDict = {}
            current_key = gammaIdx
            results_dict_sp = replaceMathwithSympy(results_dict[gammaIdx])
            currentEq_str = results_dict_sp[gammaIdx]["found_eq"]
            currentEq_sp = extract_sympyEq(currentEq_str)
            tmp_eqDict['original'] = currentEq_sp

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
        results_dict_sp = replaceMathwithSympy(results_dict)
        currentEq_str = results_dict_sp["found_eq"]
        currentEq_sp = extract_sympyEq(currentEq_str)
        eqs_beautified['original'] = currentEq_sp

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
    """
    Removes duplicates of beautified equation from a dictionary.

    :param eqs\_dict: Dictionary containing all beautified equations
    :type eqs\_dict: dict
    :return: Dictionary without duplicates
    :rtype: dict
    """
    result_dict = {}
    # loop over values and do not include already existing values in new dict
    for key, value in eqs_dict.items():
        if value not in result_dict.values():
            result_dict[key] = value
    # return dict with removed duplicate equations
    return result_dict

def printLine(nTimes=118):
    """
    Prints a line for a prettier output.

    :param nTimes: Defines how many hyphens are printed to form a line.
    :type nTimes: int
    """
    print('-'*nTimes)

def eq_beautifPrint(checkEN, eqs_dict, gamma_list=[]):
    """
    Beautifying process in order to find alternative forms of the predicted equation.

    :param checkEN: Check if the elastic net penalty is chosen
    :type checkEN: boolean
    :param eqs\_dict: Dictionary containing all beautified equations
    :type eqs\_dict: dict
    :param gamma\_list: Contains the values for the L1 ratio (see elastic net penalty definition)
    :type gamma\_list: list(float)
    """
    printLine()
    if checkEN is True:
        for gammaIdx, gammaVal in enumerate(gamma_list):
            print('Predicted equation: ')
            printLine(nTimes=19)
            print('(trained with elastic net penalty at L1-ratio %f)' % (gammaVal))
            print(eqs_dict[gammaIdx]['original'])
            alternate_forms = removeDuplicates_dictValues(eqs_dict[gammaIdx])
            alternate_forms.pop('original') # remove initial expression
            print('Alternate (approximated) forms: ')
            printLine(nTimes=31)
            for value in alternate_forms.values():
                print(value)
                printLine()
            # for better readability after each gamma iteration (bottom line)
            printLine()
    if checkEN is False:
        print('Predicted equation: ')
        printLine(nTimes=19)
        print(eqs_dict['original'])
        alternate_forms = removeDuplicates_dictValues(eqs_dict)
        alternate_forms.pop('original') # remove initial expression
        printLine()
        print('Alternate (approximated) forms: ')
        printLine(nTimes=31)
        for value in alternate_forms.values():
            print(value)
            printLine()

def saveEqBeautifDict(eqs_dict, vars_list, save_file_name):
    """
    Saves dictionary which contains the beautified equations.

    :param eqs\_dict: Dictionary containing all beautified equations
    :type eqs\_dict: dict
    :param vars\_list: Contains the variables (parameters/dimensions) within the found equations
    :type vars\_list: list(str)
    :param save\__file\_name: Defines the file name to which the dictionary gets saved.
    :type save\__file\_name: str
    """
    eqs_dict_copy = copy.deepcopy(eqs_dict)
    dm_obj = dm.DataManager()
    for key, value in eqs_dict_copy.items():
        value_str = dm_obj.sympy2str(value, vars_list)
        eqs_dict_copy.update({key : value_str})

    # save dictionary to json file in corresponding simulation folder
    with open(save_file_name, 'w') as outfile:
        json.dump(eqs_dict_copy, outfile, indent = 4)