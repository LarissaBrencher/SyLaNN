# packages
import os
import json
import eq_beautification as eqBeauty
from sympy import *

############################### Read results and write to dict #######################################
# Only for elastic net simulations, else set gamma_list to one value
gamma_nSteps = 10
gamma_start, gamma_stop = 0, 1+gamma_nSteps
# as a list of integers and list of strings
gamma_list = list(range(gamma_start, gamma_stop))

# folder name and file name (without gammaIdx numbering)
date_str = '2022-08-31'
time_str = '11-04-58'
folder_name = date_str + '_testPlotting'
if os.path == 'nt': # Windows
    folder_path = folder_name + '\\'
else: # Linux 
    folder_path = folder_name + '/'
# Note: leave out idx numbering and file format (.json) as this will change during the results loading loop
simulation_name = '_2Dpoly_whiteNoise_gammaIdx'
time_simulation_name = time_str + simulation_name
path_name = folder_path + time_simulation_name
file_format = '.json'

# create overall dictionary
results_dict = {}

# fill results dictionary with simulation results (nested dictionary of dicionaries)
for gammaIdx in gamma_list:
    current_key = gammaIdx
    current_json_name = path_name + str(current_key) + file_format
    # read current results file and save into dictionary
    json_file_data = open(current_json_name,)
    current_file = json.load(json_file_data)
    json_file_data.close()
    # write into overall dictionary at specific key
    results_dict[current_key] = current_file

########################### Beautify predicted equation #########################################
# dictionary to save beautified/simplified expressions
# (dict in dict; gammaIdx, then key for beautified version)
eqs_beautified = {}

# loop over found equations
for gammaIdx in gamma_list:
    tmp_eqDict = {}
    current_key = gammaIdx
    currentEq_str = results_dict[gammaIdx]["found_eq"]
    currentEq_sp = eqBeauty.extract_sympyEq(currentEq_str)

    currentEq_long = currentEq_sp.expand()
    currentEq_simplified = eqBeauty.autoSimplify(currentEq_long)
    currentEq_factorized = eqBeauty.factorOut(currentEq_long)
    currentEq_rational = eqBeauty.floats2rational(currentEq_long)
    tmp_eqDict['long_expanded'] = currentEq_long
    tmp_eqDict['long_simplified'] = currentEq_simplified
    tmp_eqDict['long_factorized'] = currentEq_factorized
    tmp_eqDict['long_rational'] = currentEq_rational

    currentEq_rounded4 = eqBeauty.roundDecimals(currentEq_long)
    currentEq_simplified4 = eqBeauty.autoSimplify(currentEq_rounded4)
    currentEq_factorized4 = eqBeauty.factorOut(currentEq_rounded4)
    currentEq_rational4 = eqBeauty.floats2rational(currentEq_rounded4)
    tmp_eqDict['rounded4_expanded'] = currentEq_rounded4
    tmp_eqDict['rounded4_simplified'] = currentEq_simplified4
    tmp_eqDict['rounded4_factorized'] = currentEq_factorized4
    tmp_eqDict['rounded4_rational'] = currentEq_rational4

    currentEq_rounded0 = eqBeauty.roundDecimals(currentEq_long, roundToNthDecimal=0)
    currentEq_simplified0 = eqBeauty.autoSimplify(currentEq_rounded0)
    currentEq_factorized0 = eqBeauty.factorOut(currentEq_rounded0)
    currentEq_rational0 = eqBeauty.floats2rational(currentEq_rounded0)
    tmp_eqDict['rounded0_expanded'] = currentEq_rounded0
    tmp_eqDict['rounded0_simplified'] = currentEq_simplified0
    tmp_eqDict['rounded0_factorized'] = currentEq_factorized0
    tmp_eqDict['rounded0_rational'] = currentEq_rational0

    eqs_beautified[current_key] = tmp_eqDict

# TODO save several options on simplified expressions and print them in a WolframAlpha manner
# TODO just choose one gamma or leave option for user to specify more than 1 (eg as list)
# TODO include gamma specification into printing out the alternative forms