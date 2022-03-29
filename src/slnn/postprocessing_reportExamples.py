# visualization and plotting (post processing)
"""
TODO extract somewhere for examples, similar to main, maybe some tutorial setup like via jupyter notebook (but hard to maintain, not tracked)
"""
import matplotlib.pyplot as plt
from matplotlib import cm
import pylab
from pylab import subplot,subplot2grid,figure,plot,matshow
import json
import numpy as np
import torch
import matplotlib.ticker as mtick

def linePlot_pred_vs_ref(domain, eq_pred, eq_ref, save_folder_path, curr_time):
    file_type = '.png'
    fig_name = curr_time + '_pred_vs_ref' + file_type
    fig = plt.figure()

    plt.plot(domain, eq_pred, 'r--', label='prediction')
    plt.plot(domain, eq_ref, 'k-', label='reference')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(loc='upper left')
    # plt.xlim plt.ylim etc
    fig.savefig(save_folder_path + fig_name)
    plt.show()

def linePlot_pred_vs_ref_error(domain, eq_pred, eq_ref, save_folder_path, curr_time):
    file_type = '.png'
    fig_name = curr_time + '_pred_vs_ref_error' + file_type
    fig = plt.figure()

    abs_error = [np.abs(eq_ref[i] - eq_pred[i]) for i in range(len(domain))]
    plt.plot(domain, abs_error, 'r-')
    plt.xlabel('x')
    plt.ylabel('absolute error')
    # plt.legend(loc='upper left')
    # plt.xlim plt.ylim etc
    fig.savefig(save_folder_path + fig_name)
    plt.show()

def linePlot_traintesterror(epochs, train_error, test_error, save_folder_path, curr_time):
    file_type = '.png'
    fig_name = curr_time + '_train_test_error' + file_type
    fig = plt.figure()

    plt.plot(epochs, train_error, 'b-', label='training error')
    plt.plot(epochs, test_error, 'r-', label='test error')
    plt.xlabel('epochs')
    plt.ylabel('error') # y as log for error rate?
    plt.legend(loc='upper right')
    # plt.xlim plt.ylim etc
    fig.savefig(save_folder_path + fig_name)
    plt.show()

def linePlot_traintesterror_logscale(epochs, train_error, test_error, save_folder_path, curr_time):
    file_type = '.png'
    fig_name = curr_time + '_train_test_error_logscale' + file_type
    fig = plt.figure()

    plt.semilogy(epochs, train_error, 'b-', label='training error')
    plt.semilogy(epochs, test_error, 'r-', label='test error')
    plt.xlabel('epochs')
    plt.ylabel('log(error)') # y as log for error rate?
    plt.legend(loc='upper right')
    # plt.xlim plt.ylim etc
    fig.savefig(save_folder_path + fig_name)
    plt.show()

def linePlot_myResult_vs_others(domain, ref_eq, my_eq, Martius_eq, PySR_eq, save_folder_path, exNum):
    # TODO add eps?
    file_type = '.png'
    fig_name = 'ex' + str(exNum) + '-comparison' + file_type
    fig = plt.figure()

    #nplt.plot(domain, Martius_eq, '--', color=[0.5, 0., 0.7], label='EQL(Div)')
    plt.plot(domain, ref_eq, 'k-', label='reference', linewidth=2)
    plt.plot(domain, my_eq, 'r-', label='SLNN', linewidth=2)
    plt.plot(domain, PySR_eq, '--', color=[1.0, 0.8, 0.], label='PySR')
    pylab.axvspan(-1,1, facecolor='0.8', alpha=0.5)
    #for extra in [1.4,1.8]:
    #    pylab.axvline(-extra*1,c='0.8')
    #    pylab.axvline(extra*1,c='0.8')
    plt.xlabel('$x$')
    plt.ylabel('$\hat{y}(x)$') # y as log for error rate?
    plt.legend(loc='lower right')
    # plt.xlim plt.ylim etc
    fig.savefig(save_folder_path + fig_name)
    plt.show()

def plot_epochs_vs_accuracy(save_folder_path):
        epochs_x = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 5000, 10000, 15000]
        # 2000 samples (iterate epochs)
        train_error_LBFGS = [3.6337260098662227e-06, 1.4331995146221743e-07, \
            8.601781331663005e-08, 8.601781331663005e-08, 8.601781331663005e-08, \
            8.601781331663005e-08, 1.2099560819933686e-07, 1.1520510412310614e-07, \
            1.1520510412310614e-07, 1.1520510412310614e-07, 8.601781331663005e-08, \
            8.601781331663005e-08, 8.601781331663005e-08, 8.601781331663005e-08]
        train_error_Adam = [0.03107273019850254, 0.00028542010113596916, 4.3749940232373774e-05, \
            3.642198134912178e-05, 4.1180563130183145e-05, 5.466029324452393e-05, \
            3.568809188436717e-05, 1.9223261915612966e-05, 1.0436631782795303e-05, \
            7.622682005603565e-06, 5.621164746116847e-06, 3.0489934488286963e-07, \
            0.01473561953753233, 0.0002601312007755041]
        acc_LBFGS = train_error_LBFGS # [(1-train_error_LBFGS[i])*100 for i in range(len(train_error_LBFGS))]
        acc_Adam = train_error_Adam # [(1-train_error_Adam[i])*100 for i in range(len(train_error_Adam))]
        # plotting
        file_type = '.png'
        fig_name = 'ex2-epochsVSMSE-LBFGS' + file_type # 'ex2-epochsVSaccuracy-LBFGS' + file_type
        fig = plt.figure(figsize=[10.0, 6.0])
        ax = fig.add_subplot(1,1,1)
        x_axis = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        plt.plot(x_axis, acc_Adam, 'bx-', linewidth=2, label='ADAM')
        plt.plot(x_axis, acc_LBFGS, 'rx-', linewidth=2, label='LBFGS')
        plt.yscale('log')
        
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        labels = [str(x) for x in epochs_x]
        plt.xticks(x_axis)
        ax.set_xticklabels(labels)
        plt.gcf().subplots_adjust(left=0.2)
        plt.xlabel('$epochs$')
        # plt.ylabel('$accuracy$')
        plt.ylabel('$mean$ $squared$ $error$ $(MSE)$')
        plt.legend(loc='right')
        fig.savefig(save_folder_path + fig_name)
        plt.show()

        fig_name = 'ex2-epochsVSMSE-Adam' + file_type # 'ex2-epochsVSaccuracy-Adam' + file_type
        fig = plt.figure(figsize=[10.0, 6.0])
        ax = fig.add_subplot(1,1,1)
        plt.plot(x_axis, acc_Adam, 'bx-', linewidth=2, label='ADAM')
        plt.yscale('log')
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter())
        labels = [str(x) for x in epochs_x]
        plt.xticks(x_axis)
        ax.set_xticklabels(labels)
        plt.gcf().subplots_adjust(left=0.15)
        plt.xlabel('$epochs$')
        # plt.ylabel('$accuracy$')
        plt.ylabel('$mean$ $squared$ $error$ $(MSE)$')
        plt.legend(loc='right')
        fig.savefig(save_folder_path + fig_name)
        plt.show()

def plot_runtime_vs_accuracy(save_folder_path):
    # 10 epochs to 150 epochs in stepsize 10
        epochs_x = np.linspace(10, 150, 15)
        runtime = [2.985001564025879, 5.2229413986206055, 7.945849657058716, \
            10.973135709762573, 13.236000537872314, 15.52799677848816, \
            17.423001766204834, 20.304955005645752, 22.407955646514893, \
            21.877476453781128, 25.353523015975952, 28.615339517593384, \
            28.148960828781128, 30.165664672851562, 31.469665050506592]
        train_error_LBFGS = [0.0013925093226134777, 0.001103823771700263, \
            1.2794756912626326e-05, 5.2733934552406936e-08, 3.6337260098662227e-06, \
            6.614612289013166e-07, 9.414011259423205e-08, 2.476441238741245e-07, \
            1.2095398460587603e-07, 1.4331995146221743e-07, 1.1750471173854748e-07, \
            6.894762094589169e-08, 7.914741928516378e-08, 8.677631768705396e-08, \
            8.507312543315493e-08]
        acc_LBFGS = train_error_LBFGS # [(1-train_error_LBFGS[i])*100 for i in range(len(train_error_LBFGS))]
        # plotting
        file_type = '.png'
        fig_name = 'ex2-runtimeVSMSE-LBFGS' + file_type # 'ex2-runtimeVSaccuracy-LBFGS' + file_type
        fig = plt.figure()
        color = 'tab:red'
        ax1 = fig.add_subplot(1,1,1)
        # plt.xticks(epochs_x)
        plot1 = ax1.plot(epochs_x, acc_LBFGS, 'x-', color=color, linewidth=2, label='MSE')
        ax1.set_yscale('log')
        # ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.tick_params(axis='y', labelcolor=color)
        plt.gcf().subplots_adjust(left=0.2)
        ax1.set_xlabel('$epochs$')
        ax1.set_ylabel('$mean$ $squared$ $error$ $(MSE)$', color=color)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('$runtime$ $in$ $seconds$', color=color)
        plot2 = ax2.plot(epochs_x, runtime, 'x-', color=color, linewidth=2, label='runtime')
        ax2.tick_params(axis='y', labelcolor=color)
        plots = plot1+plot2
        labs = [l.get_label() for l in plots]
        ax1.legend(plots, labs, loc='right')
        # plt.legend(loc='lower right')
        fig.tight_layout()
        fig.savefig(save_folder_path + fig_name)
        plt.show()

def plot_samples_vs_runtime_accuracy(save_folder_path):
    # 50 epochs
        samples_x = np.linspace(2000, 7000, 10)
        runtime_Adam_50 = [0.792212724685669, 0.7269420623779297, \
            0.776954174041748, 1.359001636505127, 0.9740269184112549, \
            0.970564603805542, 1.4540069103240967, 1.3290050029754639, \
            1.1289973258972168, 1.0905239582061768]
        train_error_Adam_50 = [0.03107273019850254, 0.027410656213760376, \
            0.049236733466386795, 0.0025545998942106962, 0.027446197345852852, \
            0.046976733952760696, 0.0119589539244771, 0.014575955457985401, \
            0.024653539061546326, 0.014349269680678844]
        runtime_LBFGS_50 = [12.587872743606567, 12.736936569213867, \
            13.514764070510864, 14.48259711265564, 15.375001668930054, \
            15.09950304031372, 16.879002809524536, 17.647603034973145, \
            16.928297996520996, 18.81227684020996]
        train_error_LBFGS_50 = [3.6337260098662227e-06, 3.588571644286276e-07, \
            8.074730999396706e-07, 2.541364892749698e-06, 9.019602771331847e-07, \
            7.510780415032059e-05, 1.1912378795386758e-05, 9.29253928916296e-07, \
            7.24169731256552e-05, 2.03902899897912e-07]
        acc_Adam_50 = train_error_Adam_50 # [(1-train_error_Adam_50[i])*100 for i in range(len(train_error_Adam_50))]
        acc_LBFGS_50 = train_error_LBFGS_50 # [(1-train_error_LBFGS_50[i])*100 for i in range(len(train_error_LBFGS_50))]
        file_type = '.png'
        fig_name = 'ex2-50epochs-LBFGS-VS-ADAM' + file_type
        fig = plt.figure()
        color = 'tab:red'
        ax1 = fig.add_subplot(1,1,1)
        plot1 = ax1.plot(samples_x, acc_Adam_50, ':', color=color, linewidth=2, label='MSE (ADAM)')
        plot3 = ax1.plot(samples_x, acc_LBFGS_50, '--', color=color, linewidth=2, label='MSE (LBFGS)')
        # ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color)
        plt.gcf().subplots_adjust(left=0.1)
        ax1.set_xlabel('$training$ $samples$')
        ax1.set_ylabel('$mean$ $squared$ $error$ $(MSE)$', color=color)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('$runtime$ $in$ $seconds$', color=color)
        plot2 = ax2.plot(samples_x, runtime_Adam_50, ':', color=color, linewidth=2, label='runtime (ADAM)')
        plot4 = ax2.plot(samples_x, runtime_LBFGS_50, '--', color=color, linewidth=2, label='runtime (LBFGS)')
        ax2.tick_params(axis='y', labelcolor=color)
        plots = plot1+plot3+plot2+plot4
        labs = [l.get_label() for l in plots]
        ax1.legend(plots, labs, loc='right')
        fig.tight_layout()
        fig.savefig(save_folder_path + fig_name)
        plt.show()
    
    # 15.000 epochs
        samples_x = np.linspace(2000, 4000, 5)
        runtime_Adam_15k = [236.39837908744812, 262.0041091442108, \
            331.3736493587494, 299.3857216835022, 344.9626717567444]
        train_error_Adam_15k = [0.0002601312007755041, 0.00020166876493021846, \
            0.00024369012680836022, 0.0005049892351962626, 0.00019750553474295884]
        runtime_LBFGS_15k = [396.78988766670227, 377.6617875099182, \
            411.54453444480896, 416.51134490966797, 470.9359698295593]
        train_error_LBFGS_15k = [8.601781331663005e-08, 1.1932013421755983e-07, \
            1.2325794784828759e-07, 1.292514468786976e-07, 1.2183234332496795e-07]
        acc_Adam_15k = train_error_Adam_15k # [(1-train_error_Adam_15k[i])*100 for i in range(len(train_error_Adam_15k))]
        acc_LBFGS_15k = train_error_LBFGS_15k # [(1-train_error_LBFGS_15k[i])*100 for i in range(len(train_error_LBFGS_15k))]
        # plot 1: compare runtimes
        file_type = '.png'
        fig_name = 'ex2-15kepochs-LBFGS-VS-ADAM' + file_type
        fig = plt.figure()
        color = 'tab:red'
        ax1 = fig.add_subplot(1,1,1)
        plot1 = ax1.plot(samples_x, acc_Adam_15k, ':', color=color, linewidth=2, label='MSE (ADAM)')
        plot3 = ax1.plot(samples_x, acc_LBFGS_15k, '--', color=color, linewidth=2, label='MSE (LBFGS)')
        # ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
        ax1.set_yscale('log')
        ax1.tick_params(axis='y', labelcolor=color)
        plt.gcf().subplots_adjust(left=0.1)
        ax1.set_xlabel('$training$ $samples$')
        ax1.set_ylabel('$mean$ $squared$ $error$ $(MSE)$', color=color)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('$runtime$ $in$ $seconds$', color=color)
        plot2 = ax2.plot(samples_x, runtime_Adam_15k, ':', color=color, linewidth=2, label='runtime (ADAM)')
        plot4 = ax2.plot(samples_x, runtime_LBFGS_15k, '--', color=color, linewidth=2, label='runtime (LBFGS)')
        ax2.tick_params(axis='y', labelcolor=color)
        plots = plot1+plot3+plot2+plot4
        labs = [l.get_label() for l in plots]
        ax1.legend(plots, labs, loc='right')
        fig.tight_layout()
        fig.savefig(save_folder_path + fig_name)
        plt.show()

if __name__ == '__main__':
    # TODO change to current structure of results (different dicts etc) and json dict input
    save_folder_path = "2022-03-02_SRNNsimulation" + '\\' # "2021-12-22_SRNNsimulation" + '\\'
    time_file =  '10-30-19'
    plots_path = "plots" + '\\'
    exNum = 7
    file_name = save_folder_path + time_file + "_ex" + str(exNum) + ".json"
    Martius_path = "2021-12-22_EQL-jsons" + '\\'
    exNumMartius = 4
    Martius_name = Martius_path + "ex" + str(exNumMartius) + "-bestInstance-dict.json"

    with open(file_name) as json_file:
        data_dict = json.load(json_file)
    with open(Martius_name) as json_file:
        eql_dict = json.load(json_file)

    eql_eq = eql_dict['result']
    eql_runtimeAll = eql_dict['runtimeAll'] * 60 #[s]
    eql_trainLoss = eql_dict['trainLoss'] #format [epoch, loss_value]
    domain_length = len(eql_eq)

    n_epochs = data_dict['trainEpochs3']
    epochs = range(1, n_epochs+1)
    train_error_onlyMSE = data_dict['training_onlyMSE_loss']
    train_error = data_dict['training_loss']
    test_error = data_dict['testing_loss'] # old key 'testing loss'
    x_domain = data_dict['domain_test'] # domain including the extrapolation part of the testing
    x_domain = np.linspace(x_domain[0], x_domain[1], num=domain_length)
    ref_eq_str = data_dict['ref_fct_str']
    ref_eq = eval(ref_eq_str)
    ref_eq = [ref_eq(i) for i in x_domain]
    my_eq_str = data_dict['found_eq']
    my_eq = eval(my_eq_str)
    my_eq = [my_eq(i,1,1) for i in x_domain]

    pysr_result = lambda x : 1
    pysr_runtime = 0.
    if exNum == 1:
        pysr_result = lambda x : x + 8.0
        pysr_runtime = 354.1297583580017
    elif exNum == 2:
        pysr_result = lambda x : x*(x+3.0000012) - 7.0
        pysr_runtime = 509.34841203689575
    elif exNum == 3:
        pysr_result = lambda x : (np.sin(1.03580128542067*np.exp(x))*np.cos(x*np.sin(x*(x+1.577618))))**(-1)
        pysr_runtime = 606.7507166862488
    elif exNum == 4:
        pysr_result = lambda x : np.exp(x)
        pysr_runtime = 325.87172770500183
    elif exNum == 5:
        pysr_result = lambda x, y : x*(x+2*y)+y**2
        pysr_runtime = 306.74258303642273
    elif exNum == 6:
        pysr_result = lambda x, y : x*y*(x + y)
        pysr_runtime = 476.9937307834625
    elif exNum == 7:
        pysr_result = lambda x : -1.9945105*np.cos(1.3200251*x)
        pysr_runtime = 671.6263046264648

    pysr_result = [pysr_result(i) for i in x_domain]
    # precompiling julia delta
    pysr_runtime = pysr_runtime + 300

    # train test error plot
    # linePlot_traintesterror(epochs, train_error, test_error, save_folder_path, time_file)
    # linePlot_traintesterror_logscale(epochs, train_error_onlyMSE, test_error, save_folder_path, time_file)

    # compare found equation to reference
    # linePlot_pred_vs_ref(x_domain, my_eq, ref_eq, save_folder_path, time_file)
    # linePlot_pred_vs_ref_error(x_domain, my_eq, ref_eq, save_folder_path, time_file)

    # compare to other equation learning codes on same conditions
    linePlot_myResult_vs_others(x_domain, ref_eq, my_eq, eql_eq, pysr_result, plots_path, exNum)

    # plot_epochs_vs_accuracy(plots_path)
    # plot_runtime_vs_accuracy(plots_path)
    # plot_samples_vs_runtime_accuracy(plots_path)