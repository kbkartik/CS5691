import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();
import math
import random

training_data_path = 'A2Q2Data_train.csv'
validation_data_path = 'A2Q2Data_test.csv'

# Getting optimal values of w using closed-form solution for OLS
def calc_analytic_linear_reg(X_train, Y_train):
  w = np.linalg.pinv(np.dot(X_train.T, X_train)).dot(X_train.T).dot(Y_train)
  return w

def fetch_data_from_excel(file_path):
  my_data = np.genfromtxt(file_path, delimiter=',')
  return my_data

# Fetch data from excel file
training_data = fetch_data_from_excel(training_data_path)
validation_data = fetch_data_from_excel(validation_data_path)

X = training_data[:, :-1]
y = training_data[:, -1]

X_val = validation_data[:, :-1]
Y_val = validation_data[:, -1]

num_datapoints, _ = X.shape

# Adding column of bias in X
bias_ones = np.ones((num_datapoints, 1))
X = np.hstack((X, bias_ones))

# Adding column of bias in X_val
bias_ones = np.ones(((X_val.shape)[0], 1))
X_val = np.hstack((X_val, bias_ones))

_, num_features = X.shape

w_ML = calc_analytic_linear_reg(X, y)