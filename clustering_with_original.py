# clustering_with_original.py
import time
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer

# Fetch dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups_data.data

# Convert text documents to a matrix of token counts
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(documents).toarray()

start_time = time.time()
# Perform KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X)
end_time = time.time()

# Visualize Clustering
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel('Original Dim 1')
plt.ylabel('Original Dim 2')
plt.title('Clustering with Original Data')
plt.savefig('clustering_with_original.png')
plt.show()

print(f"Time taken for original data clustering: {end_time - start_time:.2f} seconds")
