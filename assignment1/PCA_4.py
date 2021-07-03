from PIL import Image
import os, sys
import re
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();

# Give path to load images from dataset
load_dataset_path = "my_dataset/"
dirs_load_dataset = sorted(os.listdir(load_dataset_path), key=lambda f: int(re.sub('\D', '', f)))

def calculate_eigendecomposition(X_train):

    # Centering our data (Step 1)
    X_mean = np.mean(X_train, axis=1)
    X_mean = X_mean.reshape(-1, 1)
    X_mean = X_mean.astype('int')
    X_train -= X_mean

    num_examples = (X_train.shape)[1]
    constant = 1/num_examples

    # Calculating covariance matrix (Step 2)
    cov_matrix = constant * np.dot(X_train, X_train.T)
    cov_matrix = np.array(cov_matrix, dtype=float)

    # Step 3: Calculating eigen values and eigen vectors
    eigvals, eigvecs = np.linalg.eigh(cov_matrix)
    return cov_matrix, eigvals, eigvecs

def filter_eigenvectors_and_project(X_train, eigvals, eigvecs, num_components):
  # Step 4: Select Top K eigenvectors
  idx = eigvals.argsort()[::-1]
  eigvals = eigvals[idx][:num_components]
  eigvecs = np.atleast_1d(eigvecs[:, idx])[:, :num_components]
  eigvecs = eigvecs.T

  X_projected = np.dot(eigvecs, X_train)
  return X_projected, eigvals, eigvecs

# Calculate distance between projected test images and projected training images
def calculate_distance(X_projected_train, X_projected_test):
  X_cat1 = X_projected_train[:, :20]
  X_cat2 = X_projected_train[:, 20:]

  X_test_dist = np.zeros((10,2))
  for i in range(10):
    X_ti = X_projected_test[:, i].reshape(-1, 1)
    dist_cat1 = np.mean(np.sqrt(np.sum((X_ti - X_cat1)**2, axis=0)))
    dist_cat2 = np.mean(np.sqrt(np.sum((X_ti - X_cat2)**2, axis=0)))
    X_test_dist[i] = dist_cat1, dist_cat2
    
  return X_test_dist

# Load images
def load_images(load_dataset_path):
  i = 0
  for item in dirs_load_dataset:
    if os.path.isfile(load_dataset_path+item):
      # load the image
      image = Image.open(load_dataset_path+item).convert('L')
      # convert image to numpy array
      data = np.array(image)
      # Binarize image
      data = data > 128
      # Creating dataset D
      if i == 0:
        D = data.reshape((-1, 1))
        i += 1
      else:
        D = np.hstack((D, data.reshape((-1, 1))))
    D = D.astype('int')
  return D

# Plot clustering projections of test images w.r.t projection of training images
# using distance metric.
def plot_cluster(Y_label_plot):
  sns.set();
  fig = plt.figure(figsize=(10,10))
  category = Y_label_plot[:, 2]
  g = sns.scatterplot(x=Y_label_plot[:, 0], y=Y_label_plot[:, 1], hue=category, palette='flare')
  g.legend(loc='upper right', frameon=True, ncol=2, title="category", fancybox=True, borderpad=1, framealpha=1, shadow=True)
  g.set(xlim=(0,11),ylim=(0,120))
  plt.xticks(Y_label_plot[:, 0])
  plt.yticks(Y_label_plot[:, 1])
  plt.title("Clustering of test datapoint projections based on L2 norm distance with training projections")
  plt.xlabel("Test datapoint")
  plt.ylabel("Top K% eigenvectors")
  plt.show();

# Plot the elbow plot
def variance_knee_plot(variances):
  plt.plot(range(1, len(variances)+1), variances)
  plt.xlabel('number of components')
  plt.ylabel('variance explained by each PC')
  plt.show();

# Fetch all images from folder
D = load_images(load_dataset_path)

# Split into training and test set
X_train = np.hstack((D[:, :20], D[:, 25:45]))
X_test = np.hstack((D[:, 20:25], D[:, 45:]))

# Perform Eigendecomposition
C, eigvals, eigvecs = calculate_eigendecomposition(X_train)

# Perform standard PCA on input data X
num_components = [0.1, 0.2, 0.3, 0.5, 0.75, 1]
for i in range(len(num_components)):
    
    max_components = int(num_components[i]*6400)

    X_projected_train, filtered_eigvals, filtered_eigvecs = filter_eigenvectors_and_project(X_train, eigvals, eigvecs, max_components)
    X_projected_test = np.dot(filtered_eigvecs, X_test)
    X_test_dist = calculate_distance(X_projected_train, X_projected_test)
    
    # Cluster projection of test labels based on projections of training labels.
    cluster_label = (X_test_dist[:, 0] > X_test_dist[:, 1]).astype('int')
    cluster_label[cluster_label==1]=2
    cluster_label[cluster_label==0]=1
    cluster_label_plot = np.zeros((10, 2))
    cluster_label_plot[:, 0] = np.arange(1, 11, 1)
    cluster_label_plot[:, 1] = int(num_components[i]*100)
    cluster_label_plot = np.hstack((cluster_label_plot, cluster_label.reshape((-1, 1))))
    cluster_label_plot = cluster_label_plot.astype('int')
    if i == 0:
        Y_label_plot = cluster_label_plot
    else:
        Y_label_plot = np.vstack((Y_label_plot, cluster_label_plot))

plot_cluster(Y_label_plot)
# To get the variance knee plot at top K component,
# choose a specific value in the num_components list
variance_knee_plot(filtered_eigvals)