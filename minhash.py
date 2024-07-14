# minhash.py
import hashlib
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

def simple_hash(value, seed):
    result = hashlib.sha256((str(value) + str(seed)).encode()).hexdigest()
    return int(result, 16) % (2**32)  # Ensure the hash value is within the range of 32-bit integers

def generate_minhash_signature(data_point, num_hashes=10):
    minhash_signature = [float('inf')] * num_hashes
    for i in range(num_hashes):
        hash_val = simple_hash(data_point, i)
        if hash_val < minhash_signature[i]:
            minhash_signature[i] = hash_val
    return minhash_signature

def compute_signature(data_point, num_hashes):
    return generate_minhash_signature(data_point, num_hashes)

def compute_minhash_matrix(data, num_hashes=10):
    minhash_matrix = []
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(compute_signature, data, [num_hashes]*len(data)), total=len(data), desc="Computing Minhash Signatures"))
        minhash_matrix = np.array(results)
    # Normalize Minhash matrix
    minhash_matrix = (minhash_matrix - minhash_matrix.min(axis=0)) / (minhash_matrix.ptp(axis=0))
    return minhash_matrix
