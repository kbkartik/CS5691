import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();
import math
import random

data_path = 'A2Q1Data.csv'

def fetch_data_from_excel(file_path):
  my_data = np.genfromtxt(file_path, delimiter=',')
  return my_data

# Fetch data from excel file
X = fetch_data_from_excel(data_path)
X = X.reshape(-1, 1)

print(X.shape)

# Define number of clusters
K_clusters = 4

# Initializing params for EM algorithm
def init_params(K_clusters):
  u_ks = np.random.uniform(0, 3, K_clusters).reshape(1, -1) 
  u_ks = u_ks.reshape(1, -1)

  sig_ks = np.random.uniform(0.05, 1, K_clusters).reshape(1, -1)
  sig_ks = sig_ks.reshape(1, -1)

  pi_ks = np.random.dirichlet(np.ones(K_clusters)*100, size=1)
  pi_ks = pi_ks.reshape(1, -1)
  return u_ks, sig_ks, pi_ks

# Calculate P(xi, zi=k; theta)
def calc_data_probabilities(u_ks, sigma_ks, pi_ks, X):
  
  log_prob = -0.5 * ((u_ks ** 2) - 2 * np.dot(X, u_ks) + X)
  log_prob /= sigma_ks
  prob = log_prob + np.log(pi_ks) + np.log(np.reciprocal(np.sqrt(sigma_ks)))

  return prob

# Calculating lambdas during Expectation step
# Initially log probabilities are calculated to avoid numerical errors.
# Then all values are exponentiated to get back values as per standard equations.
def expectation(u_ks, sigma_ks, pi_ks, X):
  data_probs = calc_data_probabilities(u_ks, sigma_ks, pi_ks, X)

  exp_data_probs = np.exp(data_probs)
  lambdas_normalizing_factor = np.log(np.sum(exp_data_probs, axis=1)).reshape(-1, 1)
  lambdas_iks = data_probs - lambdas_normalizing_factor
  lambdas_iks = np.exp(lambdas_iks)
  
  return exp_data_probs, lambdas_iks

# Updating parameters using lambdas calculated in Expectation step
def maximization(lambdas_iks, X):
  
  n_samples = (X.shape)[0]
  
  nk = lambdas_iks.sum(axis=0)
  nk = nk.reshape(1, -1)

  # Updating pi's
  pi_ks = nk/n_samples

  # Updating means
  u_ks = np.dot(X.T, lambdas_iks)
  u_ks /= nk
  
  # Updating variances
  avg_X2 = np.dot(X.T, lambdas_iks) 
  avg_means2 = np.sum((u_ks ** 2) * lambdas_iks, axis=0).reshape(1, -1)
  avg_X_means = u_ks * np.dot(X.T, lambdas_iks)
  sigma_ks = avg_X2 - 2 * avg_X_means + avg_means2
  sigma_ks /= nk

  sigma_ks = sigma_ks.reshape(1, -1)

  return u_ks, sigma_ks, pi_ks

# This method avoids numerical errors such NaN and divide by 0.
def check_nans(u_ks, sigma_ks, pi_ks):  

  if np.any(np.isnan(u_ks)):
    u_ks[np.where(np.isnan(u_ks))] = float(np.random.uniform(0.05, 1, 1))
  if np.any(u_ks < 1e-3):
    u_ks[np.where(u_ks < 1e-3)] = float(np.random.uniform(0.05, 1, 1))

  if np.any(np.isnan(sigma_ks)):
    sigma_ks[np.where(np.isnan(sigma_ks))] = float(np.random.uniform(0.05, 1, 1))
  if np.any(sigma_ks < 1e-3):
    sigma_ks[np.where(sigma_ks < 1e-3)] = float(np.random.uniform(0.05, 1, 1))

  if np.any(np.isnan(pi_ks)):
    pi_ks[np.where(np.isnan(pi_ks))] = float(np.random.uniform(0.05, 0.5, 1))
  if np.any(pi_ks < 1e-3):
    pi_ks[np.where(pi_ks < 1e-3)] = float(np.random.uniform(0.05, 0.5, 1))

  return u_ks, sigma_ks, pi_ks

# EM algorithm
def training_MM(n_iterations, X):
  
  for itr in range(0, 100):

    loglikelihood_iterations = []

    # Initialize parameters
    u_ks, sigma_ks, pi_ks = init_params(K_clusters)

    for t in range(0, n_iterations):
      
      data_probs, lambdas_iks = expectation(u_ks, sigma_ks, pi_ks, X)
      u_ks, sigma_ks, pi_ks = maximization(lambdas_iks, X)
      u_ks, sigma_ks, pi_ks = check_nans(u_ks, sigma_ks, pi_ks)

      # Calculated log-likelihood
      log_data_p = np.log(np.sum(data_probs, axis=1))
      lambdas_k = np.sum(lambdas_iks, axis=1)
      loglikelihood = np.sum(lambdas_k * (log_data_p-np.log(lambdas_k)))
      loglikelihood_iterations.append(loglikelihood)

    loglikelihood_iterations = np.array(loglikelihood_iterations).reshape(1, -1)

    u_ks = u_ks.reshape(1, -1)
    sigma_ks = sigma_ks.reshape(1, -1)
    pi_ks = pi_ks.reshape(1, -1)

    if itr == 0:
      all_loglikelihood = loglikelihood_iterations
      all_u_ks = u_ks
      all_sigma_ks = sigma_ks
      all_pi_ks = pi_ks
    else:
      all_loglikelihood = np.vstack((all_loglikelihood, loglikelihood_iterations))
      all_u_ks = np.vstack((all_u_ks, u_ks))
      all_sigma_ks = np.vstack((all_sigma_ks, sigma_ks))
      all_pi_ks = np.vstack((all_pi_ks, pi_ks))

  # Averaging log-likehood, means, variances, and pi's for 100 random initializations
  all_loglikelihood = np.mean(all_loglikelihood, axis=0)
  all_u_ks = np.mean(all_u_ks, axis=0)
  all_sigma_ks = np.mean(all_sigma_ks, axis=0)
  all_pi_ks = np.mean(all_pi_ks, axis=0)

  return all_loglikelihood, all_u_ks, all_sigma_ks, all_pi_ks

def plot_likelihood(loglikelihood_iterations):
  sns.set();
  plt.plot(range(1, len(loglikelihood_iterations)+1), loglikelihood_iterations)
  plt.title('LLFn vs iterations (GMM)')
  plt.xlabel('Number of iterations')
  plt.ylabel('LLFn')

n_iterations = 5
loglikelihood_iterations, u_ks, sigma_ks, pi_ks = training_MM(n_iterations, X)

plot_likelihood(loglikelihood_iterations)
u_ks, sigma_ks, pi_ks