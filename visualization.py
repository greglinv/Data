import pandas as pd
import matplotlib.pyplot as plt
import os
from data_analysis import load_csv_files  # Assuming this function is in data_analysis.py

def plot_hex_length_distribution(dataframes):
    """
    Plots the distribution of lengths of hexadecimal strings for each DataFrame.

    :param dataframes: List of DataFrames.
    """
    for i, df in enumerate(dataframes):
        try:
            df['hex_length'] = df['hex_data'].apply(len)
            plt.hist(df['hex_length'], bins=20)
            plt.xlabel('Hexadecimal String Length')
            plt.ylabel('Frequency')
            plt.title(f'Distribution of Hexadecimal String Lengths (DataFrame {i})')
            plt.show()
        except Exception as e:
            print(f"Error for DataFrame {i}: {e}")

if __name__ == "__main__":
    # Load the CSV files
    directory_path = r'D:\FSl Data\fslhomes-user000-2015-04-10'
    fsl_data = load_csv_files(directory_path)

    # Plot the distribution of hexadecimal string lengths
    plot_hex_length_distribution(fsl_data)
