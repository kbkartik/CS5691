# Provide path to Dataset3.csv
path = 'C:/Users/karme/Documents/MS_IITM/Courses/CS5691/PRML_assignment1_2020/PRML_assignment1_2020/Dataset3.csv'
# Provide path to save plots
plot_save_path = 'C:/Users/karme/Desktop/'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();

# Kernelization happens here
def generate_kernels(X_input, ker_name='poly', degree=2, sigma=0.1):
  if ker_name == 'poly':
    K = np.dot(X_input.T, X_input)
    K += 1
    K **= degree
  elif ker_name == 'Gaussian':
    X_sum = np.sum(X_input ** 2, axis=0)
    K = X_sum[:, None] + X_sum[None, :] - 2 * np.dot(X_input.T, X_input)

    K *= -0.5/(sigma*sigma)
    K = np.exp(K)
  return K

def centering_kernel(K):
  num_samples = K.shape[0]
  num_samples_const = 1/num_samples

  K_col_avg = np.sum(K, axis=0) / num_samples
  K_row_avg = (np.sum(K, axis=1) / num_samples)[:, np.newaxis]
  K_total_avg = np.sum(K_col_avg)/ num_samples
  Gram_K = K + K_total_avg - K_col_avg - K_row_avg
  return Gram_K

def kernel_PCA(X_input, num_components, kernel_name='poly', degree=2, sigma=0.1):
  # Generating and centering kernel matrix K
  K = generate_kernels(X_input, ker_name=kernel_name, degree=degree, sigma=sigma)
  K = centering_kernel(K)

  # Calculating eigen values and eigen vectors of kernel matrix K
  eigvals, eigvecs = np.linalg.eigh(K)

  #Sorting and filtering eigenvectors based on eigenvalues
  idx = eigvals.argsort()[::-1]
  eigvals = eigvals[idx][:num_components]
  eigvecs = np.atleast_1d(eigvecs[:, idx])[:, :num_components]

  # Normalizing eigenvalues and eigen vectors
  eigvals = 1/np.sqrt(1000*eigvals)
  eigvals = eigvals.reshape((len(eigvals), 1))
  normalized_eigvals = np.diagflat(eigvals)
  normalized_eigenvecs = np.dot(eigvecs, normalized_eigvals)
  
  # Projecting datapoints into feature space
  X_projected = np.dot(K, normalized_eigenvecs)
  return X_projected, normalized_eigenvecs, normalized_eigvals

def fetch_data_from_excel(file_path):
  my_data = np.genfromtxt(file_path, delimiter=',')
  my_data = np.transpose(my_data)
  return my_data

def plot_graphs(X_projected, j, k, sigma, poly_degree):
  sns.set();
  
  if sigma > 0 and poly_degree == 0:
    ax2[j, k].scatter(X_projected[:, 0], X_projected[:, 1], c='red', s=15)
    ax2[j, k].set_title(r"Projection by KPCA in feature space for sigma=%0.1f" %sigma)
    ax2[j, k].set_xlabel("1st PC")
    ax2[j, k].set_ylabel("2nd PC")
  elif sigma == 0 and poly_degree > 0:
    ax1[j].scatter(X_projected[:, 0], X_projected[:, 1])
    ax1[j].set_title(r"Projection by KPCA in feature space for d=%i" %poly_degree)
    ax1[j].set_xlabel("1st PC")
    ax1[j].set_ylabel("2nd PC")

# Fetch data from excel file
X = fetch_data_from_excel(path)

# Computing and plotting Polynomial kernel
max_components = 2
fig1, ax1 = plt.subplots(1, 2, figsize=(20, 5))
degrees = [2, 3]
j = 0
k = 0
for i in range(len(degrees)):
  X_projected, principal_components, variances = kernel_PCA(X, max_components, kernel_name='poly', degree=degrees[i], sigma=0.1)
  plot_graphs(X_projected, j, k, 0, degrees[i])
  j += 1

sns.set();
plt.savefig(plot_save_path +'Q3a_polynomial_kernel_plots_CS20S020_Assignment1_CS5691.png')

# Computing and plotting Gaussian kernel
max_components = 2
fig2, ax2 = plt.subplots(5, 2, figsize=(25, 30))
sigmas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
j = -1
k = 0
for i in range(len(sigmas)):
  X_projected, principal_components, variances = kernel_PCA(X, max_components, kernel_name='Gaussian', degree=3, sigma=sigmas[i])
  j += 1
  plot_graphs(X_projected, j, k, sigmas[i], 0)
  if i == 4:
    j = -1
    k = 1

sns.set();
plt.savefig(plot_save_path +'Q3b_gaussian_kernel_plots_CS20S020_Assignment1_CS5691.png')