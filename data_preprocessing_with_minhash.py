# data_preprocessing_with_minhash.py
import time
from sklearn.datasets import fetch_20newsgroups
import numpy as np
from minhash import compute_minhash_matrix


def fetch_and_preprocess_data():
    # Fetch dataset
    newsgroups_data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
    documents = newsgroups_data.data
    return documents


def main():
    start_time = time.time()

    documents = fetch_and_preprocess_data()
    # Compute Minhash matrix
    minhash_matrix = compute_minhash_matrix(documents, num_hashes=100)

    # Save Minhash matrix
    np.save('minhash_matrix.npy', minhash_matrix)
    print("Minhash matrix shape:", minhash_matrix.shape)

    end_time = time.time()
    print(f"Time taken for Minhash computation: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()
