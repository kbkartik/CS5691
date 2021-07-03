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

# Calculating gradient of linear regressor estimator
def grad_w(XTX, XTy, w_t, lambda_reg=0):
  return np.dot(XTX, w_t) - XTy + lambda_reg * w_t

# Initializing weights from uniform distribution
def init_weights(num_features):
  np.random.seed(10)
  bounds = 1 / math.sqrt(num_features)
  w = np.random.uniform(-bounds, bounds, (num_features,))
  return w

def optimize_model(X_train, Y_train, n_iterations=2000, lr=0.1, batch=num_datapoints, lambda_reg=0):
  
  # Checking if normal GD or SGD
  if batch < num_datapoints:
    n_iterations *= int(num_datapoints/batch)
    batch_idxs = np.array(range(0, batch))

  # Initializing weights
  w_initial = init_weights(num_features)

  tracking_diff_norm_w = []
  init_diff_norm_w = np.linalg.norm(w_initial - w_ML)
  tracking_diff_norm_w.append(init_diff_norm_w)
  w_t = w_initial

  if batch == num_datapoints:
    # One time calculation of $X^X$ and $X^y$.
    XTX = np.dot(X_train.T, X_train)
    XTy = np.dot(X_train.T, Y_train)

  # Optimizing weights
  for itr in range(1, n_iterations+1):

    if batch < num_datapoints:
      XTX = np.dot(X_train[batch_idxs].T, X_train[batch_idxs])
      XTy = np.dot(X_train[batch_idxs].T, Y_train[batch_idxs])

      batch_idxs = batch_idxs + batch
      if (itr%100) == 0:
        batch_idxs = np.array(range(0, batch))

    # Weights update
    w_t -= (lr/batch) * grad_w(XTX, XTy, w_t, lambda_reg=lambda_reg)

    if ((itr%100) == 0 and batch < num_datapoints) or (batch == num_datapoints):
      diff_w_norm = np.linalg.norm(w_t - w_ML)
      tracking_diff_norm_w.append(diff_w_norm)
  
  return tracking_diff_norm_w, w_t

def cross_validation_check(X_val, Y_val, trained_weights, lambda_reg=0):
  Y_pred = np.dot(X_val, trained_weights)
  val_mse = np.mean(0.5 * (Y_val - Y_pred)**2 + lambda_reg * 0.5 * np.dot(trained_weights.T, trained_weights))
  return val_mse

def plot_diff_norm_w(tracking_diff_norm_w):
  sns.set();
  plt.plot(range(0, len(tracking_diff_norm_w)), tracking_diff_norm_w)
  plt.title('$||w_t - w_{ML}||_2$ vs number of iterations')
  plt.xlabel('Number of iterations')
  plt.ylabel('$||w_t - w_{ML}||_2$')

# Normal GD (Part Q2b)
tracking_diff_norm_w, w_t = optimize_model(X, y, n_iterations=500, lr=0.05)
plot_diff_norm_w(tracking_diff_norm_w)

# SGD with batch_size=100 (Part Q2c)
tracking_diff_norm_w, w_t = optimize_model(X, y, n_iterations=50, lr=0.01, batch=100)
plot_diff_norm_w(tracking_diff_norm_w)

# Ridge regression (Part Q2d)
reg_lambdas = [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2]
validation_mse = []
trained_weights = []

for reg in reg_lambdas:
  _, w_t = optimize_model(X, y, n_iterations=50, lr=0.001, batch=100, lambda_reg=reg)
  trained_weights.append(w_t)
  validation_mse.append(cross_validation_check(X_val, Y_val, w_t, lambda_reg=reg))

sns.set();
plt.plot(reg_lambdas, validation_mse)
plt.title('Validation set MSE vs Lambdas_ridge')
plt.xlabel('Lambdas_ridge')
plt.ylabel('Validation set MSE');

# Getting the lambda with minimum test error for ridge regression
min_lambda = reg_lambdas[np.argmin(validation_mse)]

w_ML_mse = cross_validation_check(X_val, Y_val, w_ML, lambda_reg=0)
w_r_mse = cross_validation_check(X_val, Y_val, trained_weights[np.argmin(validation_mse)], lambda_reg=min_lambda)
print('Test error for w_ML:', np.round(w_ML_mse, 3))
print('Test error for w_r:', np.round(w_r_mse, 3))