# classes for SR layer, SR neural network, ... (SR architecture)
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
import operators as ops
from inspect import signature
import time
import eq_print

class L12Smooth(nn.Module):
    def __init__(self):
        super(L12Smooth, self).__init__()

    def forward(self, input_tensor, a=0.01):
        return l12_smooth(input_tensor, a)


def l12_smooth(input_tensor, a=0.01):
    if type(input_tensor) == list:
        return sum([l12_smooth(tensor) for tensor in input_tensor])

    smooth_abs = torch.where(torch.abs(input_tensor) < a,
                            torch.pow(input_tensor, 4) / (-8 * a ** 3) + torch.square(input_tensor) * 3 / 4 / a + 3 * a / 8,
                            torch.abs(input_tensor))

    return torch.sum(torch.sqrt(smooth_abs))


class SRLayer(nn.Module):
    def __init__(self, fcts=None, in_dim=None, init_W=None):
        super().__init__()

        if fcts is None:
            fcts = ops.default_func
        
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
        return self.W.cpu().detach().numpy()

    def get_weight_tensor(self):
        return self.W.clone()


class SRNet(nn.Module):
    def __init__(self, n_hiddenLayers=2, fcts=None, init_W=None, data_dim=1):
        super(SRNet, self).__init__()

        self.depth = n_hiddenLayers
        if fcts is None:
            fcts = ops.default_func
        self.fcts = fcts
        layer_in_dim = [data_dim] + self.depth*[len(self.fcts)]

        if init_W is None:
            layers = [SRLayer(fcts=fcts, in_dim=layer_in_dim[i]) for i in range(self.depth)]
            self.output_weight = nn.Parameter(torch.rand((layers[-1].n_fcts, 1)))
        else:
            layers = [SRLayer(fcts=fcts, in_dim=layer_in_dim[i], init_W=init_W[i]) for i in range(self.depth)]
            self.output_weight = nn.Parameter(init_W[-1].clone().detach())
        
        self.hidden_layers = nn.Sequential(*layers)

    def forward(self, input):
        h = self.hidden_layers(input)
        return torch.matmul(h, self.output_weight)

    def get_weights(self):
        return [self.hidden_layers[i].get_weight() for i in range(self.depth)] + \
               [self.output_weight.cpu().detach().numpy()]

    def get_weights_tensor(self):
        return [self.hidden_layers[i].get_weight_tensor() for i in range(self.depth)] + \
               [self.output_weight.clone()]

    def reg_term(self, W, lmb, regID):
        if regID is None:
            reg_val = 0
            return lmb * torch.tensor([reg_val])
        elif regID == 0:
            reg_val = 0
            for W_i in W:
                penalty = torch.count_nonzero(torch.abs(W_i))
                reg_val += penalty
            return lmb * torch.tensor([reg_val])
        elif regID == 1:
            reg_val = 0
            for W_i in W:
                penalty = torch.sum(torch.abs(W_i))
                reg_val += penalty
            return lmb * torch.tensor([reg_val])
        elif regID == 2:
            reg_val = 0
            for W_i in W:
                penalty = torch.sum(torch.square(W_i))
                reg_val += penalty
            return lmb * torch.tensor([reg_val])
        elif regID == 12:
            penalty = L12Smooth()
            reg_val = penalty(W)
            return lmb * reg_val
        else:
            raise ValueError("Wrong value chosen for regularizarion loss (possible values: None, 0, 1, 2). \n Please re-evaluate the dictionary for training the Symbolic regression network.")

    def loss(self):
        return nn.MSELoss()

    def trainLBFGS_iteration(self, optimizer, data, target, reg_id):
        train_loss = 0.
        criterion = self.loss()
            
        def closure():
            optimizer.zero_grad()
            output = self(data)
            MSE_loss = criterion(output, target)
            reg_loss = self.reg_term(self.get_weights_tensor(), self.lmb_reg, reg_id)
            loss = MSE_loss + reg_loss
            loss.backward()
            return loss
            
        train_loss = optimizer.step(closure)
        output_plot = self(data)
        MSE_loss_plot = criterion(output_plot, target)

        return train_loss, MSE_loss_plot

    def train(self, generatedDatasets_dict, trainConfig_dict):
        train_loss_list = []          # Total loss (MSE + regularization)
        train_loss_onlyMSE_list = [] # only MSE loss without regularization
        err_list = []         # MSE
        reg_list = []
        err_test_list = []
        n_epochs1 = int(trainConfig_dict['trainEpochs1']) # int((1 / 4) * epochs)
        n_epochs2 = int(trainConfig_dict['trainEpochs2']) # int((19 / 20) * epochs)
        n_epochs3 = int(trainConfig_dict['trainEpochs3']) # int(epochs+1)
        self.lmb_reg = trainConfig_dict['regularization_weight'] # regularization weight

        err_test_final = []
        found_eq = []

        train_loss_val = np.nan
        data = generatedDatasets_dict['X_train']
        target = generatedDatasets_dict['y_train']
        test_data = generatedDatasets_dict['X_test']
        test_target = generatedDatasets_dict['y_test']

        self.learning_rate = trainConfig_dict['learning_rate'] # 0.001 # see Martius 2016

        # measure computational time --- start
        t0 = time.time()

        if trainConfig_dict['optimizer'] == 'LBFGS':
            while np.isnan(train_loss_val):
                optimizer = torch.optim.LBFGS(self.parameters(), lr=self.learning_rate)

                for epo in range(n_epochs1):
                    reg_id = trainConfig_dict['loop1Reg']
                    train_loss, MSE_plot = self.trainLBFGS_iteration(optimizer, data, target, reg_id)

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
                    reg_id = trainConfig_dict['loop2Reg']
                    train_loss, MSE_plot = self.trainLBFGS_iteration(optimizer, data, target, reg_id)

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
                    reg_id = trainConfig_dict['loop3Reg']
                    train_loss, MSE_plot = self.trainLBFGS_iteration(optimizer, data, target, reg_id)

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

                # first training part
                for epo in range(n_epochs1):
                    reg_id = trainConfig_dict['loop1Reg']
                    optimizer.zero_grad() # sets gradients of all optimized tensors to zero
                    # forward pass
                    outputs = self(data)
                    criterion = self.loss()
                    MSE_loss = criterion(outputs, target)
                    reg_loss = self.reg_term(self.get_weights_tensor(), self.lmb_reg, reg_id)
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
                    reg_loss = self.reg_term(self.get_weights_tensor(), self.lmb_reg, reg_id)
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
                    reg_loss = self.reg_term(self.get_weights_tensor(), self.lmb_reg, reg_id)  # or get_weights() or self.get_weights_tensor()?
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
            'training_loss' : train_loss_list,
            'training_onlyMSE_loss' : train_loss_onlyMSE_list,
            'testing_loss' : err_test_list,
            'simulation_time' : total_time
        }
        return result_dict

