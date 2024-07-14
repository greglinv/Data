# fingerprint.py
import hashlib

def compute_fingerprint(data):
    return hashlib.sha256(data.encode()).hexdigest()

def create_fingerprint_dict(data):
    fingerprint_dict = {}
    for i, data_point in enumerate(data):
        fingerprint = compute_fingerprint(data_point)
        fingerprint_dict[fingerprint] = i
    return fingerprint_dict
