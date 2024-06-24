import numpy as np

# Load the similarity matrix from the .npy file
similarity_matrix = np.load('similarity_matrix.npy')

# Print the shape of the matrix
print("Shape of the similarity matrix:", similarity_matrix.shape)

# Print the first few rows and columns of the matrix for inspection
print(similarity_matrix[:5, :5])
