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
  p_ks = np.random.uniform(0.1, 1, K_clusters).reshape(1, -1)
  p_ks = np.around(p_ks, 2)

  pi_ks = np.random.dirichlet(np.ones(K_clusters)*100, size=1)
  return p_ks, pi_ks

# Calculate P(xi, zi=k; theta)
def calc_data_probabilities(p_ks, pi_ks, X):
  #p_ks = np.clip(p_ks, 1e-10, 1 - 1e-10, out=p_ks)

  p_ks_data = p_ks ** X
  one_minus_p_ks = (1 - p_ks) ** (1 - X)
  
  prob = p_ks_data * one_minus_p_ks * pi_ks
  return prob

# Calculating lambdas during Expectation step
def expectation(p_ks, pi_ks, X):

  data_probs = calc_data_probabilities(p_ks, pi_ks, X)
  lambdas_normalizing_factor = np.sum(data_probs, axis=1).reshape(-1, 1)
  lambdas_iks = data_probs/lambdas_normalizing_factor
  
  return data_probs, lambdas_iks

# Updating parameters using lambdas calculated in Expectation step
def maximization(lambdas_iks, X):
  
  n_samples = (X.shape)[0]

  nk = np.sum(lambdas_iks, axis=0)
  nk = nk.reshape(1, -1)

  # Update pi's
  pi_ks = nk/n_samples
  
  # Update cluster probabilities
  p_ks = np.dot(X.T, lambdas_iks)
  p_ks = p_ks/nk
  p_ks = p_ks.reshape(1, -1)

  return p_ks, pi_ks

# EM algorithm
def training_MM(n_iterations, X):
  
  for itr in range(0, 100):

    loglikelihood_iterations = []
    p_ks, pi_ks = init_params(K_clusters)

    for t in range(0, n_iterations):
      data_probs, lambdas_iks = expectation(p_ks, pi_ks, X)
      p_ks, pi_ks = maximization(lambdas_iks, X)
      
      # Calculate log-likelihood
      log_data_p = np.log(np.sum(data_probs, axis=1))
      lambdas_k = np.sum(lambdas_iks, axis=1)
      loglikelihood = np.sum(lambdas_k * (log_data_p-np.log(lambdas_k)))
      loglikelihood_iterations.append(loglikelihood)

    loglikelihood_iterations = np.array(loglikelihood_iterations).reshape(1, -1)
    p_ks = p_ks.reshape(1, -1)
    pi_ks = pi_ks.reshape(1, -1)
    if itr == 0:
      all_loglikelihood = loglikelihood_iterations
      all_p_ks = p_ks
      all_pi_ks = pi_ks
    else:
      all_loglikelihood = np.vstack((all_loglikelihood, loglikelihood_iterations))
      all_p_ks = np.vstack((all_p_ks, p_ks))
      all_pi_ks = np.vstack((all_pi_ks, pi_ks))

  # Averaging log-likehood, means, variances, and pi's for 100 random initializations
  all_loglikelihood = np.round(np.mean(all_loglikelihood, axis=0), 2)
  all_p_ks = np.mean(all_p_ks, axis=0)
  all_pi_ks = np.mean(all_pi_ks, axis=0)

  return all_loglikelihood, all_p_ks, all_pi_ks

def plot_likelihood(loglikelihood_iterations):
  sns.set();
  plt.plot(range(1, len(loglikelihood_iterations)+1), loglikelihood_iterations)
  plt.title('LLFn vs iterations')
  plt.xlabel('Number of iterations')
  plt.ylabel('LLFn')

n_iterations = 3
loglikelihood_iterations, p_ks, pi_ks = training_MM(n_iterations, X)

plot_likelihood(loglikelihood_iterations)
p_ks, pi_ks