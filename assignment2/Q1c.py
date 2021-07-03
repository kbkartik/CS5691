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

def init_rand_clusters(K, X):
  z_vec = np.random.randint(0, 4, size=len(X))
  return z_vec

def calc_cluster_centers(z_vec, X):
  cluster_means = []
  for l in range(4):
    idx_l = np.array(np.where(z_vec == l)).reshape(-1, 1)
    if len(idx_l) == 0:
      cluster_means.append(0)
    else:
      cluster_means.append(np.mean(X[idx_l]))

  return cluster_means

def calc_best_clusters(X, cluster_means, z_vec, cluster_loss=True):
  closest_cluster_matrix = (X - cluster_means) ** 2
  new_cluster_labels = np.argmin(closest_cluster_matrix, axis=1)

  if cluster_loss == True:
    loss = np.mean(closest_cluster_matrix[:, z_vec])
    return new_cluster_labels, loss
  
  return new_cluster_labels

def train_kmeans(X, num_clusters, num_iterations):

  K = num_clusters

  z_vec = init_rand_clusters(K, X)
  objective_loss_iterations = []

  for itr in range(num_iterations):
    cluster_means = calc_cluster_centers(z_vec, X)
    z_new_vec, loss = calc_best_clusters(X, cluster_means, z_vec, cluster_loss=True)

    # Check if previous assignment is same as new assignment
    if np.any(z_new_vec != z_vec):
      z_vec = z_new_vec
      objective_loss_iterations.append(loss)
    else:
      objective_loss_iterations.append(0)
      break
      
  objective_loss_iterations = np.array(objective_loss_iterations).reshape(-1, 1)
  cluster_means = np.array(cluster_means).reshape(1, -1)

  return objective_loss_iterations, cluster_means

def plot_loss(objective_loss_iterations):
  sns.set();
  plt.plot(range(1, len(objective_loss_iterations)+1), objective_loss_iterations)
  plt.title('K-means obj loss vs iterations')
  plt.xlabel('Number of iterations')
  plt.ylabel('obj loss')

num_clusters = 4
num_iterations = 10

objective_loss_iterations, cluster_means = train_kmeans(X, num_clusters, num_iterations)

plot_loss(objective_loss_iterations)
cluster_means