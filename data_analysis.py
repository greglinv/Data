import os
import pandas as pd
from hashlib import sha256

def load_csv_files(directory_path):
    """
    Loads all CSV files in the specified directory into a list of pandas DataFrames.

    :param directory_path: Path to the directory containing CSV files.
    :return: List of DataFrames.
    """
    dataframes = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            df = pd.read_csv(file_path, header=None, names=['hex_data'])
            dataframes.append(df)
    return dataframes

def inspect_dataframes(dataframes):
    """
    Inspects the loaded DataFrames by printing their info and head.

    :param dataframes: List of DataFrames.
    """
    for i, df in enumerate(dataframes):
        print(f"DataFrame {i}:")
        print(df.info())
        print(df.head())
        print("\n")

def compute_hash(hex_data):
    """
    Computes SHA-256 hash of the given hexadecimal data.

    :param hex_data: Hexadecimal data to hash.
    :return: SHA-256 hash.
    """
    binary_data = bytes.fromhex(hex_data)  # Convert hex to binary
    return sha256(binary_data).hexdigest()

def add_hashes(dataframes):
    """
    Adds a hash column to each DataFrame.

    :param dataframes: List of DataFrames.
    :return: List of DataFrames with hash column added.
    """
    for df in dataframes:
        df['hash'] = df['hex_data'].apply(compute_hash)
    return dataframes

if __name__ == "__main__":
    # Load the CSV files
    directory_path = r'C:\Users\gll4kc\Downloads\fslhomes-user000-2015-04-10'
    fsl_data = load_csv_files(directory_path)

    # Inspect the DataFrames
    inspect_dataframes(fsl_data)

    # Add hash columns
    fsl_data = add_hashes(fsl_data)

    # Example: Find duplicates in the first DataFrame
    duplicates = fsl_data[0][fsl_data[0].duplicated('hash')]
    print("Duplicates:")
    print(duplicates)
