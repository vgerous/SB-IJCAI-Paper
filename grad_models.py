import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

# local libraries & helper functions
from qpth_local.qp import QPFunction, QPSolvers
from utils import requests2params
from qpth_utils import transform_qpth


class ReluFCNet(nn.Module):
    def __init__(self, input_size, pred_size, T, hid1=100, hid2=50, SGD_params=[1e-4, 0.9], nn_device=torch.device('cpu')):
        super(ReluFCNet, self).__init__()
        self.nn_device = nn_device
        # just some casually set hidden layers now.
        self.input_size = input_size
        self.pred_size = pred_size
        self.fc1 = nn.Linear(input_size, hid1)
        self.fc2 = nn.Linear(hid1, hid2)
        self.fc3 = nn.Linear(hid2, pred_size)
        self.T = T
        self.step_size, self.momentum = SGD_params
        self.to(nn_device)

    def set_params(self, **params):
        if "hid1" in params:
            self.hid1 = params["hid1"]
        if "hid2" in params:
            self.hid2 = params["hid2"] 
        if "step_size" in params:
            self.step_size = params["step_size"]
        if "momentum" in params:
            self.momentum = params["momentum"] 

        self.fc1 = nn.Linear(self.input_size, self.hid1)
        self.fc2 = nn.Linear(self.hid1, self.hid2 )
        self.fc3 = nn.Linear(self.hid2 , self.pred_size)

    def forward(self, x):
        """
        naive forward pass, output predicted capacites
        params:
            x: historical capacities
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        a_hats = self.fc3(x)
        return a_hats
    
    def forward_optnet(self, x1, req_param, y, req_core_scaledown):
        """
        optnet forward pass: naive forward pass + qpth layer
        this function should only be used only at training stage since it needs ground truth future capacites
        currently only support batch_size=1 since the request data are not size-aligned. batch SGD needs to be manually done.
        params:
            x1: history capacities
            req_param: future requests data
            y: future capacities
        returns:
            objective, sum-of-squrares of total violating capacity usage 
        """
        # predicted capcities. hopefully no negative values!
        a_hat = self.forward(x1)[0]
        if not (a_hat > 0).all():
            print("[WARNING]: negative capacity predictions. Recommend scale down the regularity factor in OptNet.")
            a_hat[a_hat < 0] = 0.01
        # retrieve request data for this training instance
        N, c, d, e, l = requests2params(req_param)
        c = c/req_core_scaledown
        epsilon, p, G, h = transform_qpth(self.T,N,c,d,e,l,sparse=False)  # qpth no sparse implementations support
        # now transform and invoke qpth
        Q = torch.Tensor(epsilon*np.eye(G.shape[1])).to(self.nn_device)
        p = torch.Tensor(p).to(self.nn_device)
        G = torch.Tensor(G).to(self.nn_device)
        h1 = torch.Tensor(h).to(self.nn_device)
        null_var = torch.Tensor()
        """
        important step: fill in the predicted capacities
        """
        h1[:self.T] = a_hat
        sol = QPFunction(verbose=False, solver=QPSolvers.CVXPY, nn_device=self.nn_device)(Q, p, G, h1, null_var, null_var)[0]
        # penalize the violation of the ground truth capacities
        h2 = torch.Tensor(h).to(self.nn_device)
        h2[:self.T] = y
        occupied = torch.matmul(G,sol)  
        ind = occupied > h2   # indices of where the volation happens
        return torch.dot(p, sol), torch.sum((occupied[ind]-h2[ind])**2)

    def fit(self, X_train, y_train, batch_size=64, total_iteration=15000):
        X_train, y_train = torch.Tensor(X_train).to(self.nn_device), torch.Tensor(y_train).to(self.nn_device)
        optimizer = optim.SGD(self.parameters(), lr=self.step_size, momentum=self.momentum)
        criterion = nn.MSELoss()
        for iteration in range(total_iteration): 
            # randomly select batch
            indices = np.random.choice(X_train.shape[0], batch_size, replace=False)
            x = X_train[indices]
            y = y_train[indices]
            optimizer.zero_grad()   # zero the gradient buffers
            output = self.forward(x)
            loss = criterion(output, y)
            if iteration % 2000 == 0:
                print("iteration", iteration, "loss:", loss.data)
            loss.backward()
            optimizer.step()    # does the update

    def fit_optnet(self, X1_train, req_params_train, y_train,
                   batch_size, total_iteration, vio_regularity, req_core_scaledown=1.0):
        X1_train, y_train = torch.Tensor(X1_train).to(self.nn_device), torch.Tensor(y_train).to(self.nn_device)
        optimizer = optim.SGD(self.parameters(), lr=self.step_size/batch_size, momentum=self.momentum)
        for iteration in range(total_iteration):
            optimizer.zero_grad()   # zero the gradient buffers
            indices = np.random.choice(X1_train.shape[0], batch_size, replace=False)
            objval_batch = torch.tensor(0.)
            vio_batch = torch.tensor(0.)
            for index in indices:
                x1 = X1_train[[index]]    # wrap it;
                y = y_train[index]
                objval, violation_sum = self.forward_optnet(x1,req_params_train[index],y,req_core_scaledown)
                loss = objval + vio_regularity*violation_sum
                loss.backward() # no zeroing gradients here since need to add the gradients over each instance in this batch
                objval_batch += objval.data
                vio_batch += violation_sum.data
            if iteration % 1 == 0:
                print("iteration", iteration, "obj_val:", objval_batch/batch_size, "violation:", vio_batch/batch_size)
            optimizer.step()    # Does the update

    def predict(self, X):
        return self.forward(torch.Tensor(X).to(self.nn_device)).cpu().detach().numpy()


class LstmNet(nn.Module):
    def __init__(self, input_feature_dim, output_size, T, hid_size=100, step_size=1e-3,
                 nn_device=torch.device('cpu')):
        super(LstmNet, self).__init__()
        self.nn_device = nn_device
        self.step_size = step_size
        self.hidden_layer_size = hid_size
        self.lstm = nn.LSTM(input_feature_dim, hid_size)
        self.linear = nn.Linear(hid_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))
        self.T = T
        self.to(nn_device)

    def set_params(self, **params):
        if "hid_size" in params:
            self.hid_size = params["hid_size"]
        if "step_size" in params:
            self.step_size = params["step_size"] 
        self.hidden_layer_size = self.hid_size

    def forward(self, x):
        """
        naive forward pass, output predicted capacites
        params:
            x: historical capacities; currently a 1-d array
        returns:
            predicted capacities, 1d array
        """
        lstm_out, self.hidden_cell = self.lstm(x.view(len(x), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(x), -1))
        return predictions[-1]

    def forward_optnet(self, x1, req_param, y, req_core_scaledown):
        """
        optnet forward pass: naive forward pass + qpth layer
        this function should only be used only at training stage since it needs ground truth future capacites
        currently only support batch_size=1 since the request data are not size-aligned. batch SGD needs to be manually done.
        params:
            x1: history capacities
            req_param: future requests data
            y: future capacities
        returns:
            objective, sum-of-squrares of total violating capacity usage
        """
        # predicted capcities. hopefully no negative values!
        a_hat = self.forward(x1)
        if not (a_hat > 0).all():
            print("[WARNING]: negative capacity predictions. Recommend scale down the regularity factor in OptNet.")
            a_hat[a_hat < 0] = 0.01
        # retrieve request data for this training instance
        N, c, d, e, l = requests2params(req_param)
        c /= req_core_scaledown
        # scale down the cores according to the core_scaledown factors
        epsilon, p, G, h = transform_qpth(self.T, N, c, d, e, l, sparse=False)  # qpth no sparse implementations support
        # now transform and invoke qpth
        Q = torch.Tensor(epsilon * np.eye(G.shape[1])).to(self.nn_device)
        p = torch.Tensor(p).to(self.nn_device)
        G = torch.Tensor(G).to(self.nn_device)
        h1 = torch.Tensor(h).to(self.nn_device)
        null_var = torch.Tensor()
        """
        important step: fill in the predicted capacities
        """
        h1[:self.T] = a_hat
        sol = \
        QPFunction(verbose=False, solver=QPSolvers.CVXPY, nn_device=self.nn_device)(Q, p, G, h1, null_var, null_var)[0]
        # penalize the violation of the ground truth capacities
        h2 = torch.Tensor(h).to(self.nn_device)
        h2[:self.T] = y
        occupied = torch.matmul(G, sol)
        ind = occupied > h2  # indices of where the volation happens
        return torch.dot(p, sol), torch.sum((occupied[ind] - h2[ind]) ** 2)

    def fit(self, X_train, y_train, total_iteration=2000):
        X_train, y_train = torch.Tensor(X_train).to(self.nn_device), torch.Tensor(y_train).to(self.nn_device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.step_size)
        criterion = nn.MSELoss()
        for iteration in range(total_iteration):
            # randomly select a sample for SGD
            index = np.random.choice(X_train.shape[0])
            seq = X_train[index]
            labels = y_train[index]
            optimizer.zero_grad()
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.nn_device),
                                torch.zeros(1, 1, self.hidden_layer_size).to(self.nn_device))
            y_pred = self.forward(seq)
            assert y_pred.shape == labels.shape
            single_loss = criterion(y_pred, labels)
            single_loss.backward()
            optimizer.step()
            if iteration % 200 == 0:
                print("iteration", iteration, "loss:", single_loss.data)

    def fit_optnet(self, X1_train, req_params_train, y_train,
                   batch_size, total_iteration, vio_regularity, req_core_scaledown=1.0):
        X1_train, y_train = torch.Tensor(X1_train).to(self.nn_device), torch.Tensor(y_train).to(self.nn_device)
        optimizer = torch.optim.Adam(self.parameters(), lr=self.step_size)
        for iteration in range(total_iteration):
            optimizer.zero_grad()  # zero the gradient buffers
            indices = np.random.choice(X1_train.shape[0], batch_size, replace=False)
            objval_batch = torch.tensor(0.)
            vio_batch = torch.tensor(0.)
            for index in indices:
                x1 = X1_train[index]
                y = y_train[index]
                self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.nn_device),
                                    torch.zeros(1, 1, self.hidden_layer_size).to(self.nn_device))
                objval, violation_sum = self.forward_optnet(x1, req_params_train[index], y, req_core_scaledown)
                loss = objval + vio_regularity * violation_sum
                loss.backward()  # no zeroing gradients here since need to add the gradients over each instance in this batch
                objval_batch += objval.data
                vio_batch += violation_sum.data
            if iteration % 1 == 0:
                print("iteration", iteration, "obj_val:", objval_batch / batch_size, "violation:",
                      vio_batch / batch_size)
            optimizer.step()  # Does the update

    def predict(self, X):
        y = []
        for x in X:
            self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to(self.nn_device),
                                torch.zeros(1, 1, self.hidden_layer_size).to(self.nn_device))
            y.append(self.forward(torch.Tensor(x).to(self.nn_device)).cpu().detach().numpy())
        return np.array(y)
