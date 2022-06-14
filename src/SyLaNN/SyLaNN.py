"""
Defines the structure of the Symbolic-Layered Neural Network.
This includes the construction of the customized layer, the architecture of the overall network
and the definition of the approximated penalty function.
"""

import numpy as np
import torch
import torch.nn as nn
import mathOperators as ops
import time
import eq_print
from penalty import Penalty


class SymLayer(nn.Module):
    """
    Definition of the customized layer of the Symbolic-Layered Neural Network (SLNN),
    inherits from PyTorch's nn.Module.
    """
    def __init__(self, fcts=ops.default_func, in_dim=None, init_W=None):
        """
        Constructor method

        :param fcts: Activation functions per layer, default ops.default\_func
        :type fcts: list\[objects\]
        :param in_dim: Dimension of input from layer, default None
        :type in_dim: int
        :param init_W: Pre-defined weight matrix, default None
        :type init_W: Tensor
        """
        super().__init__()
        
        self.output = None
        self.n_fcts = len(fcts)
        self.fcts = [fct.torch for fct in fcts]
        self.n_binary = ops.count_binaryFcts(fcts)
        self.n_unary = self.n_fcts - self.n_binary

        self.out_dim = self.n_fcts + self.n_binary
        self.init_W = init_W
        self.W = None

        if init_W is None:
            self.W = nn.Parameter(torch.normal(mean=0.0, std=0.1, size=(in_dim, self.out_dim)))
        else:
            self.W = nn.Parameter(self.init_W.clone().detach())

    def forward(self, x):
        """
        Applies the forward propagation method (definition per layer).

        :param x: Input from previous layer
        :type x: Tensor

        :return: Input for next layer
        :rtype: Tensor
        """
        g = torch.matmul(x, self.W)
        self.output = []

        in_i = 0
        out_i = 0
        while out_i < self.n_unary:
            self.output.append(self.fcts[out_i](g[:, in_i]))
            in_i += 1
            out_i += 1
        while out_i < self.n_fcts:
            self.output.append(self.fcts[out_i](g[:, in_i], g[:, in_i+1]))
            in_i += 2
            out_i += 1

        self.output = torch.stack(self.output, dim=1)

        return self.output

    def get_weight(self):
        """
        Returns the weight matrix as NumPy array.
        """
        return self.W.cpu().detach().numpy()

    def get_weight_tensor(self):
        """
        Returns the weight matrix as PyTorch tensor.
        """
        return self.W.clone()


class SLNet(nn.Module):
    """
    Definition of the Symbolic-Layered Neural Network (SLNN) architecture,
    inherits from PyTorch's nn.Module.
    """
    def __init__(self, n_hiddenLayers=2, fcts=ops.default_func, init_W=None, data_dim=1):
        """
        Constructor method

        :param n_hiddenLayers: Number of hidden layers of the SLNN (depth), default 2
        :type n_hiddenLayers: int
        :param fcts: Activation functions per layer, default ops.default_func
        :type fcts: list\[objects\]
        :param init_W: Pre-defined weight matrix, default None
        :type init_W: Tensor
        :param data_dim: Number of variables from the given input data, default 1
        :type data_dim: int
        """
        super(SLNet, self).__init__()

        self.depth = n_hiddenLayers
        self.fcts = fcts
        layer_in_dim = [data_dim] + self.depth*[len(self.fcts)]

        if init_W is None:
            layers = [SymLayer(fcts=fcts, in_dim=layer_in_dim[i]) for i in range(self.depth)]
            self.output_weight = nn.Parameter(torch.rand((layers[-1].n_fcts, 1)))
        else:
            layers = [SymLayer(fcts=fcts, in_dim=layer_in_dim[i], init_W=init_W[i]) for i in range(self.depth)]
            self.output_weight = nn.Parameter(init_W[-1].clone().detach())
        
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input):
        """
        Applies the forward propagation through the whole SLNN.

        :param input: Initial data from the input layer
        :type input: Tensor

        :return: Result of the SLNN's forward propagation
        :rtype: Tensor
        """
        h = self.hidden_layers(input)
        return torch.matmul(h, self.output_weight)

    def get_weights(self):
        """
        Returns the weight matrices as list of NumPy arrays.
        """
        return [self.hidden_layers[i].get_weight() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]

    def get_weights_tensor(self):
        """
        Returns the weight matrices as list of PyTorch tensors.
        """
        return [self.hidden_layers[i].get_weight_tensor() for i in range(self.depth)] + \
               [self.output_weight.clone()]
    
    def loss(self):
        """
        Returns the loss value (in this case the mean-squared error is chosen).
        """
        return nn.MSELoss()

    def loss_jacobian(self, loss):
        """
        Returns the Jacobian matrix of the loss with respect to weights.

        :param loss: Error loss of the prediction compared to the reference solution (without penalty)
        :type loss: Tensor

        :return: Jacobian matrix
        :rtype: Tensor
        """
        n_samples = loss.size(dim=0)
        jacobian = []
        for sample_i in range(n_samples):
            grad_tmp = torch.autograd.grad(loss[sample_i], self.parameters(), retain_graph=True)
            grad_tmp = torch.nn.utils.parameters_to_vector(grad_tmp)
            jacobian.append(grad_tmp)
        jacobian = torch.stack(jacobian) # stack all rows to get matrix form
        return jacobian

    def loss_hessian(self, loss):
        """
        Returns the Hessian matrix of the loss with respect to weights (Gauss-Newton approximation).

        :param loss: Error loss of the prediction compared to the reference solution (without penalty)
        :type loss: Tensor

        :return: Hessian matrix approximation, i.e., multiply transpose Jacobian with Jacobian
        :rtype: Tensor
        """
        jacobian = self.loss_jacobian(loss) # size: numSamples x numWeights
        jacobian_T = torch.transpose(jacobian, 0, 1) # size: numWeights x numSamples
        # calculate Hessian matrix approximation by multiplying J^T * J
        hessian_approx = torch.matmul(jacobian_T, jacobian) # size: numWeights x numWeights
        return hessian_approx

    def trainLBFGS_iteration(self, optimizer, data, target, regObj, gamma_val, current_epoch):
        """
        Performs a single optimization step (LBFGS). Returns the training error (MSE).

        :param optimizer: Specifices the applied optimizer (from torch.optim)
        :type optimizer: object
        :param data: Training data set
        :type data: Tensor
        :param target: Corresponding training targets
        :type target: Tensor
        :param regObj: Specifies which kind of regularization method is applied
        :type regObj: Penalty object
        :param gamma\_val: Prefactor (L1 ratio) for the elastic net penalty computation, default 0.5
        :type gamma\_val: float
        :param current\_epoch: Current epoch of the training
        :type current\_epoch: int

        :return: Training losses (MSE + regularization loss and only MSE)
        :rtype: Tensor (with one element)
        """
        train_loss = 0.
        criterion = self.loss()
            
        def closure():
            """
            A closure that reevaluates the model and returns the loss.
            """
            optimizer.zero_grad()
            output = self(data)
            MSE_loss = criterion(output, target)
            reg_loss = regObj(self.get_weights_tensor(), self.lmb_reg, self.approx_eps, gamma_val)
            
            # Bayesian regularization computation
            if self.chooseBR is True:
                # update prefactors every n-th epoch
                if (current_epoch != 0) and ((current_epoch % self.updateBR_everyNepoch) == 0):
                    squared_loss = (self(data) - target).pow(2)
                    hessian_MSE = self.loss_hessian(squared_loss)
                    hessian_penalty = regObj.calculate_hessian(self.get_weights_tensor(), self.approx_eps, gamma_val)
                    H_pinv = torch.linalg.pinv(hessian_MSE+hessian_penalty)
                    tr_Hinv = torch.trace(H_pinv)
                    N_p = data.size(dim=1) # TODO add number of parameters somewhere as self.something?
                    N_eff = N_p - self.alpha_reg * tr_Hinv
                    N_D = data.size(dim=0)
                    # update prefactors
                    self.alpha_reg = N_eff / (2*MSE_loss)
                    self.beta_MSE = (N_D - N_eff) / (2*reg_loss)
                # calculate current loss and apply backpropagation
                loss = self.beta_MSE*MSE_loss + self.alpha_reg*reg_loss
                loss.backward()
                return loss

            # regular SyLaNN training when BR is off
            elif self.chooseBR is False:
                # calculate current loss and apply backpropagation
                loss = MSE_loss + reg_loss
                loss.backward()
                return loss
            
        train_loss = optimizer.step(closure)
        output_plot = self(data)
        MSE_loss_plot = criterion(output_plot, target)
        return train_loss, MSE_loss_plot

    def train(self, generatedDatasets_dict, trainConfig_dict, gamma_val=0.5):
        """
        Trains the SLNN with given datasets within set training configurations.
        Allows LBFGS and ADAM as optimizers.

        :param generatedDatasets\_dict: Contains the training and testing data
        :type generatedDatasets\_dict: dict
        :param trainConfig\_dict: Contains the specifications for the SLNN training (number of epochs, regularization weight, learning rate, ...)
        :type trainConfig\_dict: dict
        :param gamma\_val: Prefactor (L1 ratio) for the elastic net penalty computation, default 0.5
        :type gamma\_val: float

        :return: Dictionary with results of the simulation
        :rtype: dict
        """
        train_loss_list = []          # Total loss (MSE + regularization)
        train_loss_onlyMSE_list = [] # only MSE loss without regularization
        err_list = []         # MSE
        reg_list = []
        err_test_list = []
        n_epochs1 = int(trainConfig_dict['trainEpochs1']) # int((1 / 4) * epochs)
        n_epochs2 = int(trainConfig_dict['trainEpochs2']) # int((19 / 20) * epochs)
        n_epochs3 = int(trainConfig_dict['trainEpochs3']) # int(epochs+1)
        self.lmb_reg = trainConfig_dict['regularization_weight'] # regularization weight
        self.approx_eps = trainConfig_dict['regApprox_threshold'] # penalty approximation threshold

        self.chooseBR = trainConfig_dict['chooseBR']
        self.beta_MSE = trainConfig_dict['error_data_factor']
        self.alpha_reg = trainConfig_dict['error_reg_factor']
        self.updateBR_everyNepoch = trainConfig_dict['updateBRparams_every_n_epoch']
        if self.chooseBR is False: # in order to avoid wrong prefactor values when BR is off
            assert self.beta_MSE == 1, "Please set beta MSE prefactor to one, if Bayesian regularization is not used."
            assert self.alpha_reg == 1, "Please set alpha reg prefactor to one, if Bayesian regularization is not used."

        err_test_final = []
        found_eq = []

        train_loss_val = np.nan
        data = generatedDatasets_dict['X_train']
        target = generatedDatasets_dict['y_train']
        test_data = generatedDatasets_dict['X_test']
        test_target = generatedDatasets_dict['y_test']

        self.learning_rate = trainConfig_dict['learning_rate'] # 0.001 # see Martius 2016

        reg_id1 = trainConfig_dict['loop1Reg']
        regObj1 = Penalty(name=reg_id1)
        reg_id2 = trainConfig_dict['loop2Reg']
        regObj2 = Penalty(name=reg_id2)
        reg_id3 = trainConfig_dict['loop3Reg']
        regObj3 = Penalty(name=reg_id3)

        # measure computational time --- start
        t0 = time.time()

        if trainConfig_dict['optimizer'] == 'LBFGS':
            while np.isnan(train_loss_val):
                optimizer = torch.optim.LBFGS(self.parameters(), lr=self.learning_rate)

                for epo in range(n_epochs1):
                    train_loss, MSE_plot = self.trainLBFGS_iteration(optimizer, data, target, regObj1, gamma_val, epo)
                    train_loss_onlyMSE_val = MSE_plot.item()
                    train_loss_val = train_loss.item()
                    train_loss_onlyMSE_list.append(train_loss_onlyMSE_val)
                    train_loss_list.append(train_loss_val)

                    with torch.no_grad():
                        test_outputs = self(test_data)
                        test_loss = nn.functional.mse_loss(test_outputs, test_target)
                        err_test_val = test_loss.item()
                        err_test_list.append(err_test_val)

                    print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (epo, train_loss_val, err_test_val))

                    if np.isnan(train_loss_val) or train_loss_val > 1000:  # If loss goes to NaN, restart training
                        break

                for epo in range(n_epochs1, n_epochs2):
                    train_loss, MSE_plot = self.trainLBFGS_iteration(optimizer, data, target, regObj2, gamma_val, epo)
                    train_loss_onlyMSE_val = MSE_plot.item()
                    train_loss_val = train_loss.item()
                    train_loss_onlyMSE_list.append(train_loss_onlyMSE_val)
                    train_loss_list.append(train_loss_val)

                    with torch.no_grad():  # test error
                        test_outputs = self(test_data)
                        test_loss = nn.functional.mse_loss(test_outputs, test_target)
                        err_test_val = test_loss.item()
                        err_test_list.append(err_test_val)

                    print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (epo, train_loss_val, err_test_val))

                    if np.isnan(train_loss_val) or train_loss_val > 1000:  # If loss goes to NaN, restart training
                        break

                for epo in range(n_epochs2, n_epochs3+1):
                    train_loss, MSE_plot = self.trainLBFGS_iteration(optimizer, data, target, regObj3, gamma_val, epo)
                    train_loss_onlyMSE_val = MSE_plot.item()
                    train_loss_val = train_loss.item()
                    train_loss_onlyMSE_list.append(train_loss_onlyMSE_val)
                    train_loss_list.append(train_loss_val)

                    with torch.no_grad():
                        test_outputs = self(test_data)
                        test_loss = nn.functional.mse_loss(test_outputs, test_target)
                        err_test_val = test_loss.item()
                        err_test_list.append(err_test_val)

                    print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (epo, train_loss_val, err_test_val))

                    if np.isnan(train_loss_val) or train_loss_val > 1000:  # If loss goes to NaN, restart training
                        break
                        

        elif trainConfig_dict['optimizer'] == 'Adam':
            while np.isnan(train_loss_val):
                # use Adam optimizer see Martius
                optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
                regObj = Penalty(reg_id)

                # first training part
                for epo in range(n_epochs1):
                    reg_id = trainConfig_dict['loop1Reg']
                    optimizer.zero_grad() # sets gradients of all optimized tensors to zero
                    # forward pass
                    outputs = self(data)
                    criterion = self.loss()
                    MSE_loss = criterion(outputs, target)
                    reg_loss = self.reg_term(regObj, self.get_weights_tensor(), self.lmb_reg, self.approx_eps)
                    train_loss = MSE_loss + reg_loss
                    train_loss.backward()
                    optimizer.step()

                    train_loss_onlyMSE_val = MSE_loss.item()
                    train_loss_val = train_loss.item()
                    train_loss_onlyMSE_list.append(train_loss_onlyMSE_val)
                    train_loss_list.append(train_loss_val)

                    with torch.no_grad():  # test error
                        test_outputs = self(test_data)
                        test_loss = nn.functional.mse_loss(test_outputs, test_target)
                        err_test_val = test_loss.item()
                        err_test_list.append(err_test_val)

                    print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (epo, train_loss_val, err_test_val))

                    if np.isnan(train_loss_val) or train_loss_val > 1000:  # If loss goes to NaN, restart training
                        break

                # second training part
                for epo in range(n_epochs1, n_epochs2):
                    reg_id = trainConfig_dict['loop2Reg']
                    optimizer.zero_grad() # sets gradients of all optimized tensors to zero
                    outputs = self(data) # forward pass
                    criterion = self.loss()
                    MSE_loss = criterion(outputs, target)
                    reg_loss = self.reg_term(regObj, self.get_weights_tensor(), self.lmb_reg, self.approx_eps)
                    train_loss = MSE_loss + reg_loss
                    train_loss.backward()
                    optimizer.step()
                    
                    train_loss_onlyMSE_val = MSE_loss.item()
                    train_loss_val = train_loss.item()
                    train_loss_onlyMSE_list.append(train_loss_onlyMSE_val)
                    train_loss_list.append(train_loss_val)

                    with torch.no_grad():  # test error
                        test_outputs = self(test_data)
                        test_loss = nn.functional.mse_loss(test_outputs, test_target)
                        err_test_val = test_loss.item()
                        err_test_list.append(err_test_val)

                    print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (epo, train_loss_val, err_test_val))

                    if np.isnan(train_loss_val) or train_loss_val > 1000:  # If loss goes to NaN, restart training
                        break

                # third training part
                for epo in range(n_epochs2, n_epochs3+1):
                    reg_id = trainConfig_dict['loop3Reg']
                    optimizer.zero_grad() # sets gradients of all optimized tensors to zero
                    outputs = self(data) # forward pass
                    criterion = self.loss()
                    MSE_loss = criterion(outputs, target)
                    reg_loss = self.reg_term(regObj, self.get_weights_tensor(), self.lmb_reg, self.approx_eps)
                    train_loss = MSE_loss + reg_loss
                    train_loss.backward()
                    optimizer.step()
                    
                    train_loss_onlyMSE_val = MSE_loss.item()
                    train_loss_val = train_loss.item()
                    train_loss_onlyMSE_list.append(train_loss_onlyMSE_val)
                    train_loss_list.append(train_loss_val)

                    with torch.no_grad():  # test error
                        test_outputs = self(test_data)
                        test_loss = nn.functional.mse_loss(test_outputs, test_target)
                        err_test_val = test_loss.item()
                        err_test_list.append(err_test_val)

                    print("Epoch: %d\tTotal training loss: %f\tTest error: %f" % (epo, train_loss_val, err_test_val))

                    if np.isnan(train_loss_val) or train_loss_val > 1000:  # If loss goes to NaN, restart training
                        break

        else:
            raise ValueError("Chosen optimizer not implemented.")

        # measure computational time --- stop
        t1 = time.time()
        total_time = t1-t0
        print("Total time: %f" % (total_time))

        # Print the expressions
        with torch.no_grad():
            n_inputArgs = generatedDatasets_dict['x_dim']
            weights = self.get_weights()
            vars_str = trainConfig_dict['variables_str']
            expr = eq_print.network(weights, self.fcts, vars_str[:n_inputArgs])
            print(expr)


        err_test_final.append(err_test_list[-1])
        found_eq.append(expr)

        # create dictionary of results
        result_dict = {
            'found_eq' : found_eq,
            'elasticNet_L1ratio' : gamma_val,
            'simulation_time' : total_time,
            'training_loss' : train_loss_list,
            'training_onlyMSE_loss' : train_loss_onlyMSE_list,
            'testing_loss' : err_test_list
        }
        return result_dict

