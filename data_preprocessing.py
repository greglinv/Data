from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os

# Fetch the dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups_data.data

# Convert the collection of text documents to a matrix of token counts
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(documents)

# Compute the cosine similarity matrix
similarity_matrix = cosine_similarity(X)

# Print the shape of the similarity matrix
print("Similarity matrix shape:", similarity_matrix.shape)

# Save the similarity matrix to a file
np.save('similarity_matrix.npy', similarity_matrix)

# Confirming the file location
print("File saved at:", os.path.join(os.getcwd(), 'similarity_matrix.npy'))

