# Provide path to Dataset3.csv
path = 'C:/Users/karme/Documents/MS_IITM/Courses/CS5691/PRML_assignment1_2020/PRML_assignment1_2020/Dataset3.csv'

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set();

def linear_pca(X_input, num_components):

  # Centering our data (Step 1)
  X_mean = np.mean(X_input, axis=1)
  X_mean = X_mean.reshape(-1, 1)
  X_input -= X_mean

  num_examples = (X_input.shape)[1]
  constant = 1/num_examples

  # Calculating covariance matrix (Step 2)
  cov_matrix = constant * np.dot(X_input, X_input.T)
  cov_matrix = np.array(cov_matrix, dtype=float)

  # Step 3: Calculating eigen values and eigen vectors
  eigvals, eigvecs = np.linalg.eigh(cov_matrix)

  # Step 4: Sorting and filtering eigenvectors based on eigenvalues
  idx = eigvals.argsort()[::-1]
  eigvals = eigvals[idx][:num_components]
  eigvecs = np.atleast_1d(eigvecs[:, idx])[:, :num_components]

  return eigvecs, eigvals

def fetch_data_from_excel(file_path):
  my_data = np.genfromtxt(file_path, delimiter=',')
  my_data = my_data.T
  return my_data

# Fetch data from excel file
X = fetch_data_from_excel(path)

# Perform standard PCA on input data X
max_components = 2
principal_components, variances = linear_pca(X, max_components)

print("PCA components: ", principal_components)
print("PCA variance: ", variances)