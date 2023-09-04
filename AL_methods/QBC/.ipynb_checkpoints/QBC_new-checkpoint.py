#!/usr/bin python3
# -*- coding:utf8-*-
# @TIME     :2022/1/16 12:15 上午
# @Author   :Heather
# @File     :LAL-QBC-R.py

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from modAL.models import ActiveLearner, CommitteeRegressor
from modAL.disagreement import vote_entropy_sampling, max_std_sampling
import matplotlib.pyplot as plt
import torch
from torch.nn import MSELoss
from skorch.regressor import NeuralNetRegressor
from copy import deepcopy
from skorch.dataset import ValidSplit
from skorch.callbacks import EarlyStopping
from sklearn.cluster import DBSCAN, OPTICS, cluster_optics_dbscan, Birch, SpectralClustering
from sklearn.neighbors import NearestNeighbors
from sklearn import metrics
import matplotlib.gridspec as gridspec
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from torch.nn import Linear
from torch.nn import Sigmoid, ReLU
from torch.nn import Module
from sklearn.metrics import r2_score
import copy
from scipy.spatial import distance
from scipy.spatial.distance import euclidean
from torch import tensor
from sklearn.metrics import mean_squared_error
from scipy.spatial.distance import cdist
from sklearn.ensemble import RandomForestRegressor
from torch.nn.init import xavier_uniform_
from model import MLP
from model_state import get_model_params, get_input_for_hidden_layers
from functions import clustering, sample_the_cloest_data, data_input_space_diversity, minimum_distance_to_used_data, sample_by_input_diversity
from functions import get_variance, qbc, normalization, error_reduct_fuc, total_disagrement
import random
from skorch.callbacks import EarlyStopping, LRScheduler, Freezer, Unfreezer
from torch.optim.lr_scheduler import CyclicLR, ReduceLROnPlateau
from skorch.regressor import NeuralNetRegressor, NeuralNet
from model_state import features_concat, learning_state_features_concat

total_training_r2 = []
total_test_r2 = []
total_test_mse = []
total_test_error_r2 = []
total_test_error_mse = []
total_rf_r2 = []
total_rf_mse = []
total_rf_training_r2 = []
total_rf_training_mse = []

query_number = 30      # The number of the queries in each iteration
iteration = 20         # Total number of the experiments
NN_setting = [50, 30, 15]   # Inner model size
batch_zise = 40
total_initial = 117


# seed_nn = np.load(file="../diabetes_dataset_20test/seed_for_NN.npy")
seed_rf = np.load(file="../WhiteWine_dataset_20test/seed_for_RF.npy")
seed_initial = np.load(file="../WhiteWine_dataset_20test/seed_for_initialSamples.npy")


def main_function(query_number, iter):

    # Set Seeds here:
    # torch.manual_seed(seed_nn[iter])
    np.random.seed(seed_initial[iter])
    rf_model = RandomForestRegressor(random_state=seed_rf[iter], n_estimators=100)

    # Load the data:
    name1 = "../WhiteWine_dataset_20test/X_train_" + str(iter) + ".npy"
    name2 = "../WhiteWine_dataset_20test/X_test_" + str(iter) + ".npy"
    name3 = "../WhiteWine_dataset_20test/y_train_" + str(iter) + ".npy"
    name4 = "../WhiteWine_dataset_20test/y_test_" + str(iter) + ".npy"

    X_train = np.load(name1)
    X_test = np.load(name2)
    y_train = np.load(name3)
    y_test = np.load(name4)
    X = X_train.shape[1]
    X_index = np.arange(X_train.shape[0])

    # Number of member in committee:
    n_members = 3
    learner_list = []

    used_data = np.empty(shape=(0, X))
    used_label = np.empty(shape=(0)).reshape(-1, 1)

    X_initial = np.empty(shape=(0,X))
    y_initial = np.empty(shape=(0)).reshape(-1, 1)

    # Initialize the Committee
    # Initial samples:
    initial_size = 5
    idx = np.random.choice(range(len(X_index)), size=initial_size, replace=False)
    train_idx = X_index[idx]

    X_initial = X_train[train_idx]
    y_initial = y_train[train_idx].reshape(-1, 1)
    X_index = np.delete(X_index, idx, axis=0)
    # print(y_initial)

    # 5 another random sampling:
    sampled_index = np.empty(shape=(0))
    rest_initial_X = np.empty(shape=(0,X))
    rest_initial_y = np.empty(shape=(0)).reshape(-1, 1)

    idx = np.random.choice(range(len(X_index)), size=total_initial-5, replace=False)
    train_idx = X_index[idx]
    sampled_index = np.append(sampled_index, train_idx, axis=0).astype(np.int32)

    rest_initial_X = np.append(rest_initial_X, X_train[train_idx], axis=0).astype(np.float32)
    rest_initial_y = np.append(rest_initial_y, y_train[train_idx], axis=0).astype(np.float32).reshape(-1, 1)
    # print(rest_initial_y)

    X_index = np.delete(X_index, idx, axis=0)


    for member_idx in range(n_members):
        # Every single member:
        regressor = RandomForestRegressor(n_estimators=100)

    # initializing learner - Firstly use 5 instances to train the model <- decided by the early stopping criteria here.

        learner = ActiveLearner(
            estimator=regressor,
            X_training=X_initial,
            y_training=y_initial.ravel()
        )
        learner_list.append(learner)

    used_data = np.append(used_data, X_initial, axis=0).astype(np.float32)
    used_label = np.append(used_label, y_initial, axis=0).astype(np.float32).reshape(-1, 1)


    # Assembling the committee
    committee = CommitteeRegressor(
        learner_list=learner_list,
        query_strategy=max_std_sampling
    )

    # Committee initial R2 for Testing set:
    prediction_ = committee.predict(X_test)
    testing_performance = [r2_score(y_test, prediction_)]
    testing_mse = [mean_squared_error(y_test, prediction_)]
    # print("The initial testing R^2:", initial_score)

    # RF Prediction Res:
    rf_model.fit(used_data, used_label.ravel())
    rf_training_r2 = r2_score(used_label, rf_model.predict(used_data))
    rf_training_mse = mean_squared_error(used_label, rf_model.predict(used_data))
    rf_model_training_r2 = [rf_training_r2]
    rf_model_training_mse = [rf_training_mse]

    rf_model_prediction = rf_model.predict(X_test)
    rf_model_r2 = r2_score(y_test, rf_model_prediction)
    rf_model_mse = mean_squared_error(y_test, rf_model_prediction)
    rf_model_testing_performance = [rf_model_r2]
    rf_model_testing_mse = [rf_model_mse]
    # print(rf_model_r2)

    # Committee initial R2 for Training set:
    prediction = committee.predict(used_data)
    unqueried_score = r2_score(used_label, prediction)
    performance_history = [unqueried_score]
    # print("The initial training R^2:", unqueried_score)

    for i in range(rest_initial_X.shape[0]):  # rest_initial_X.shape[0]

        single_X = rest_initial_X[i],
        single_y = rest_initial_y[i].reshape(1, -1)
        single_X = single_X[0].reshape(1, -1)

        # Retrain the model:
        committee.teach(
            X=single_X,
            y=single_y.ravel()
            # only_new=True
        )

        used_data = np.append(used_data, single_X, axis=0).astype(np.float32)
        used_label = np.append(used_label, single_y, axis=0).astype(np.float32).reshape(-1, 1)

        # R2 Score for Training set after each query
        model_accuracy = r2_score(used_label, committee.predict(used_data))
        performance_history.append(model_accuracy)

        # R2 Score for Testing set after each query
        test_accuracy = r2_score(y_test, committee.predict(X_test))
        test_mse = mean_squared_error(y_test, committee.predict(X_test))
        testing_performance.append(test_accuracy)
        testing_mse.append(test_mse)

        # RF Regressor:
        rf_model.fit(used_data, used_label.ravel())

        rf_training_r2 = r2_score(used_label, rf_model.predict(used_data))
        rf_training_mse = mean_squared_error(used_label, rf_model.predict(used_data))
        rf_model_training_r2.append(rf_training_r2)
        rf_model_training_mse.append(rf_training_mse)


        rf_model_r2 = r2_score(y_test, rf_model.predict(X_test))
        rf_model_mse = mean_squared_error(y_test, rf_model.predict(X_test))
        rf_model_testing_performance.append(rf_model_r2)
        rf_model_testing_mse.append(rf_model_mse)

        index = sampled_index[i]


    Pool_X = X_index
    remaining_samples = len(Pool_X)

    Pool_X = [int(x) for x in Pool_X]
    remaining_data = X_train[Pool_X]
    remaining_label = y_train[Pool_X]
    # print(remaining_samples)

    prediction = committee.predict(X_test)
    unqueried_score = r2_score(y_test, prediction)

    n_queries = query_number
    np.random.seed(None)

    for idx in range(n_queries):
        np.random.seed(None)
        print('Query no. %d' % (idx+1))
        idx, query_instance = committee.query(remaining_data, n_instances=batch_zise)
        # idx = np.random.choice(range(remaining_samples), size=batch_zise, replace=False)
        Pool_X = np.array(Pool_X)

        # Query the new sample:
        X_train_index = Pool_X[idx]

        # Random:
        # idx = np.random.choice(range(remaining_samples), size=1, replace=False)
        # X_train_index = Pool_X[idx[0]]

        new_X = X_train[X_train_index].reshape(batch_zise, -1)
        new_y = y_train[X_train_index].reshape(batch_zise, -1)

        # Retrain the model
        committee.teach(
            X=new_X,
            y=new_y.ravel()
        )

        # Adding the used data to the used_data pool
        used_data = np.append(used_data, new_X, axis=0).astype(np.float32)
        used_label = np.append(used_label, new_y, axis=0).astype(np.float32).reshape(-1, 1)

        # R2 Score for Training set after each query
        model_accuracy = r2_score(used_label, committee.predict(used_data))
        # print("Committee's training score:", model_accuracy)
        performance_history.append(model_accuracy)

        # R2 Score for Testing set after each query
        test_accuracy = r2_score(y_test, committee.predict(X_test))
        test_mse = mean_squared_error(y_test, committee.predict(X_test))
        testing_performance.append(test_accuracy)
        testing_mse.append(test_mse)

        # RF Regressor:
        rf_model.fit(used_data, used_label.ravel())

        rf_training_r2 = r2_score(used_label, rf_model.predict(used_data))
        rf_training_mse = mean_squared_error(used_label, rf_model.predict(used_data))
        rf_model_training_r2.append(rf_training_r2)
        rf_model_training_mse.append(rf_training_mse)


        rf_model_r2 = r2_score(y_test, rf_model.predict(X_test))
        rf_model_mse = mean_squared_error(y_test, rf_model.predict(X_test))
        rf_model_testing_performance.append(rf_model_r2)
        rf_model_testing_mse.append(rf_model_mse)

        # remove queried instance from pool
        Pool_X = np.delete(Pool_X, idx, axis=0)
        remaining_data = X_train[Pool_X]
        remaining_samples = remaining_samples - batch_zise


    # NN
    performance_history = np.array(performance_history)
    testing_performance = np.array(testing_performance)
    testing_mse = np.array(testing_mse)

    np.save(file="./res_QBC/featureEmbeddings/used_data" + str(iter) + ".npy", arr=used_data)
    np.save(file="./res_QBC/featureEmbeddings/used_label" + str(iter) + ".npy", arr=used_label)

    np.save(file="./res_QBC/featureEmbeddings/Summary_dia/training_r2_QBC-" + str(iter) + ".npy", arr=performance_history)
    np.save(file="./res_QBC/featureEmbeddings/Summary_dia/testing_r2_QBC-" + str(iter) + ".npy", arr=testing_performance)
    np.save(file="./res_QBC/featureEmbeddings/Summary_dia/testing_mse_QBC-" + str(iter) + ".npy", arr=testing_mse)

    # RF
    rf_model_testing_performance = np.array(rf_model_testing_performance)
    rf_model_testing_mse = np.array(rf_model_testing_mse)
    rf_model_training_r2 = np.array(rf_model_training_r2)
    rf_model_training_mse = np.array(rf_model_training_mse)

    np.save(file="./res_QBC/featureEmbeddings/Summary_dia/testing_rf_r2_QBC-" + str(iter) + ".npy", arr=rf_model_testing_performance)
    np.save(file="./res_QBC/featureEmbeddings/Summary_dia/testing_rf_mse_QBC-" + str(iter) + ".npy", arr=rf_model_testing_mse)
    np.save(file="./res_QBC/featureEmbeddings/Summary_dia/training_rf_r2_QBC-" + str(iter) + ".npy", arr=rf_model_training_r2)
    np.save(file="./res_QBC/featureEmbeddings/Summary_dia/training_rf_mse_QBC-" + str(iter) + ".npy", arr=rf_model_training_mse)

    total_training_r2.append(performance_history)
    total_test_r2.append(testing_performance)
    total_test_mse.append(testing_mse)
    total_rf_r2.append(rf_model_testing_performance)
    total_rf_mse.append(rf_model_testing_mse)
    total_rf_training_r2.append(rf_model_training_r2)
    total_rf_training_mse.append(rf_model_training_mse)


for i in range(iteration):
    print("The Iteration is ", i)
    main_function(query_number, i)


averaged_training_r2 = np.array([sum(x) for x in zip(* total_training_r2)])/iteration
averaged_test_r2 = np.array([sum(x) for x in zip(* total_test_r2)])/iteration
averaged_test_mse = np.array([sum(x) for x in zip(* total_test_mse)])/iteration
averaged_rf_r2 = np.array([sum(x) for x in zip(* total_rf_r2)])/iteration
averaged_rf_mse = np.array([sum(x) for x in zip(* total_rf_mse)])/iteration
averaged_rf_training_r2 = np.array([sum(x) for x in zip(* total_rf_training_r2)])/iteration
averaged_rf_training_mse = np.array([sum(x) for x in zip(* total_rf_training_mse)])/iteration


file_name_1 = "./res_QBC/featureEmbeddings/Averaged_dia/training_r2_QBC-10"
file_name_2 = "./res_QBC/featureEmbeddings/Averaged_dia/testing_r2_QBC-10"
file_name_3 = "./res_QBC/featureEmbeddings/Averaged_dia/testing_mse_QBC-10"
file_name_6 = "./res_QBC/featureEmbeddings/Averaged_dia/testing_rf_r2_QBC-10"
file_name_7 = "./res_QBC/featureEmbeddings/Averaged_dia/testing_rf_mse_QBC-10"
file_name_8 = "./res_QBC/featureEmbeddings/Averaged_dia/training_rf_mse_QBC-10"
file_name_9 = "./res_QBC/featureEmbeddings/Averaged_dia/training_rf_r2_QBC-10"


np.save(file=file_name_1+".npy", arr=averaged_training_r2)
np.save(file=file_name_2+".npy", arr=averaged_test_r2)
np.save(file=file_name_3+".npy", arr=averaged_test_mse)
np.save(file=file_name_6+".npy", arr=averaged_rf_r2)
np.save(file=file_name_7+".npy", arr=averaged_rf_mse)
np.save(file=file_name_8+".npy", arr=averaged_rf_training_mse)
np.save(file=file_name_9+".npy", arr=averaged_rf_training_r2)







