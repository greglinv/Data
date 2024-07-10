import pandas as pd
import matplotlib.pyplot as plt
import os
from data_analysis import load_csv_files, add_hashes  # Assuming these functions are in data_analysis.py

def plot_hash_distribution(dataframes):
    """
    Plots the distribution of SHA-256 hash lengths for each DataFrame.

    :param dataframes: List of DataFrames.
    """
    for i, df in enumerate(dataframes):
        df['hash_length'] = df['hash'].apply(len)
        plt.hist(df['hash_length'], bins=20)
        plt.xlabel('Hash Length')
        plt.ylabel('Frequency')
        plt.title(f'Distribution of Hash Lengths (DataFrame {i})')
        plt.show()

if __name__ == "__main__":
    # Load the CSV files
    directory_path = r'D:\FSl Data\fslhomes-user000-2015-04-10'
    fsl_data = load_csv_files(directory_path)

    # Add hash columns
    fsl_data = add_hashes(fsl_data)

    # Plot the distribution of hash lengths
    plot_hash_distribution(fsl_data)
