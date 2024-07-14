# data_preprocessing_with_fingerprint.py
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from fingerprint import create_fingerprint_dict  # Ensure this import is correct

# Fetch dataset
newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
documents = newsgroups_data.data

# Convert text documents to a matrix of token counts
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
X = vectorizer.fit_transform(documents)

# Compute Fingerprint Dictionary
fingerprint_dict = create_fingerprint_dict(documents)

# Save Fingerprint Dictionary
import json
with open('fingerprint_dict.json', 'w') as fp:
    json.dump(fingerprint_dict, fp)
print("Fingerprint Dictionary created.")
