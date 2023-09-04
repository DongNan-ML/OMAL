#!/usr/bin python3
# -*- coding:utf8-*-
# @TIME     :2022/1/16 5:07 上午
# @Author   :Heather
# @File     :model.py

import torch
from torch.nn import Linear
from torch.nn import Sigmoid, ReLU
from torch.nn import Module
import numpy as np
import random
from torch.nn import Sigmoid, ReLU, BatchNorm1d, ELU

NN_setting = [50, 30, 15]   # [30, 15, 5] # For feature embeddings


class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super().__init__()

        # input to first hidden layer
        self.hidden1 = Linear(n_inputs, NN_setting[0])
        self.act1 = ELU()

        # second hidden layer
        self.hidden2 = Linear(NN_setting[0], NN_setting[1])
        self.act2 = ELU()

        # second hidden layer
        self.hidden3 = Linear(NN_setting[1], NN_setting[2])
        self.act3 = ELU()

        # self.hidden5 = Linear(NN_setting[2],  NN_setting[3])
        # self.act5 = ELU()


        # self.hidden6 = Linear(NN_setting[3],  NN_setting[4])
        # self.act6 = ELU()

        # second hidden layer and output
        self.hidden4 = Linear(NN_setting[2], 1)

    # For Red Wine: 150, 75, 35, 15, 1
    # For Boston Housing: 50, 30, 15, 1

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)

        X = self.hidden3(X)
        X = self.act3(X)

        # X = self.hidden5(X)
        # X = self.act5(X)

        # X = self.hidden6(X)
        # X = self.act6(X)
        # second hidden layer and output
        X = self.hidden4(X)

        return X




# Auto MPG

# class MLP(Module):
#     # define model elements
#     def __init__(self, n_inputs):
#         super().__init__()
#
#         # input to first hidden layer
#         self.hidden1 = Linear(n_inputs, 30)
#         self.act1 = ReLU()
#
#         # second hidden layer
#         self.hidden2 = Linear(30, 15)
#         self.act2 = ReLU()
#
#         #         # second hidden layer
#         #         self.hidden3 = Linear(30, 15)
#         #         self.act3 = ReLU()
#
#
#         # second hidden layer and output
#         self.hidden4 = Linear(15, 1)
#
#     # forward propagate input
#     def forward(self, X):
#         # input to first hidden layer
#         X = self.hidden1(X)
#         X = self.act1(X)
#         # second hidden layer
#         X = self.hidden2(X)
#         X = self.act2(X)
#
#         #         X = self.hidden3(X)
#         #         X = self.act3(X)
#         # second hidden layer and output
#         X = self.hidden4(X)
#
#         return X
# Boston:
# class MLP(Module):
#     # define model elements
#     def __init__(self, n_inputs):
#         super().__init__()
#
#         # input to first hidden layer
#         self.hidden1 = Linear(n_inputs, 50)
#         self.act1 = ReLU()
#
#         # second hidden layer
#         self.hidden2 = Linear(50, 30)
#         self.act2 = ReLU()
#
#         # second hidden layer
#         self.hidden3 = Linear(30, 15)
#         self.act3 = ReLU()
#
#         # second hidden layer and output
#         self.hidden4 = Linear(15, 1)
#
#     # forward propagate input
#     def forward(self, X):
#         # input to first hidden layer
#         X = self.hidden1(X)
#         X = self.act1(X)
#         # second hidden layer
#         X = self.hidden2(X)
#         X = self.act2(X)
#
#         X = self.hidden3(X)
#         X = self.act3(X)
#         # second hidden layer and output
#         X = self.hidden4(X)
#
#         return X


# class MLP(Module):
#     # define model elements
#     def __init__(self, n_inputs):
#         super().__init__()
#
#         # input to first hidden layer
#         self.hidden1 = Linear(n_inputs, 30)
#         self.act1 = ReLU()
#
#         # second hidden layer
#         self.hidden2 = Linear(30, 15)
#         self.act2 = ReLU()
#
#         # second hidden layer and output
#         self.hidden3 = Linear(15, 1)
#
#     # forward propagate input
#     def forward(self, X):
#         # input to first hidden layer
#         X = self.hidden1(X)
#         X = self.act1(X)
#         # second hidden layer
#         X = self.hidden2(X)
#         X = self.act2(X)
#         # second hidden layer and output
#         X = self.hidden3(X)
#
#         return X