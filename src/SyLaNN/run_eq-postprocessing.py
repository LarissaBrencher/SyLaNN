# packages
import os
import eq_beautification as eqBeauty
from sympy import *

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
gamma_list = list(range(gamma_start, gamma_stop))

######## Beautify predicted equation ########
results_dict = {}
eqs_beautif = {}
if checkEN is True:
    results_dict = eqBeauty.createResultsDict(checkEN, path_name, file_format, gamma_list)
    eqs_beautif = eqBeauty.createBeautifDict(checkEN, results_dict, gamma_list)
if checkEN is False:
    results_dict = eqBeauty.createResultsDict(checkEN, path_name, file_format)
    eqs_beautif = eqBeauty.createBeautifDict(checkEN, results_dict)

# TODO save several options on simplified expressions and print them in a WolframAlpha manner
# TODO include gamma specification into printing out the alternative forms