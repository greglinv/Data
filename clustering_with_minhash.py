# clustering_with_minhash.py
import time
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Minhash matrix
minhash_matrix = np.load('minhash_matrix.npy', allow_pickle=True)

start_time = time.time()
# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(minhash_matrix)
end_time = time.time()

# Visualize Clustering
plt.scatter(minhash_matrix[:, 0], minhash_matrix[:, 1], c=labels)
plt.xlabel('Minhash Dim 1')
plt.ylabel('Minhash Dim 2')
plt.title('Clustering with Minhash')
plt.savefig('clustering_with_minhash.png')
plt.show()

print(f"Time taken for Minhash clustering: {end_time - start_time:.2f} seconds")
