# packages
import os
import eq_beautification as eqBeauty
from sympy import *
import numpy as np

######## User inputs: folder name, file name, EN gamma list ########
# folder name and file name (without gammaIdx numbering)
date_str = '2022-08-31'
time_str = '11-04-58'
folder_name = date_str + '_testPlotting'

# check OS and append slash or backslash accordingly
if os.path == 'nt': # Windows
    folder_path = folder_name + '\\'
else: # Linux 
    folder_path = folder_name + '/'

# Note: leave out idx numbering and file format (.json) as this will change during the results loading loop
simulation_name = '_2Dpoly_whiteNoise_gammaIdx'
time_simulation_name = time_str + simulation_name
path_name = folder_path + time_simulation_name
file_format = '.json'

# Only for elastic net simulations, else set checkEN to False
checkEN = True
gamma_nSteps = 10
gamma_start, gamma_stop = 0, 1+gamma_nSteps
# as a list of integers and list of strings
gamma_list = [*np.arange(gamma_start, gamma_stop, gamma_nSteps)]

######## Beautify predicted equation ########
# TODO needed? or just save and print within ifs
# TODO comment everything better (especially in eq_beautificaton)
results_dict = {}
eqs_beautif = {}
if checkEN is True:
    results_dict = eqBeauty.createResultsDict(checkEN, path_name, file_format, gamma_list)
    eqs_beautif = eqBeauty.createBeautifDict(checkEN, results_dict, gamma_list)
    vars_str = results_dict[0]["variables_str"]
    eqBeauty.eq_beautifPrint(checkEN, eqs_beautif, gamma_list)
    save_file_name = path_name + '_beautified' + file_format
    eqBeauty.saveEqBeautifDict(eqs_beautif, vars_str, save_file_name, gamma_list)
if checkEN is False:
    results_dict = eqBeauty.createResultsDict(checkEN, path_name, file_format)
    eqs_beautif = eqBeauty.createBeautifDict(checkEN, results_dict)
    vars_str = results_dict["variables_str"]
    eqBeauty.eq_beautifPrint(checkEN, eqs_beautif)
    save_file_name = path_name + '_beautified' + file_format
    eqBeauty.saveEqBeautifDict(eqs_beautif, vars_str, save_file_name)