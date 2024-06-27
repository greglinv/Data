import os
import binascii
import csv

def convert_anon_to_csv_binary(anon_file_path, csv_file_path):
    """
    Converts a binary .anon file to a .csv file with hexadecimal representation of binary data.

    :param anon_file_path: Path to the .anon file.
    :param csv_file_path: Path to save the .csv file.
    """
    try:
        with open(anon_file_path, 'rb') as anon_file:  # Read as binary
            with open(csv_file_path, 'w', newline='') as csv_file:
                writer = csv.writer(csv_file)
                for line in anon_file:
                    # Convert the binary line to a hex string
                    hex_string = binascii.hexlify(line).decode('ascii')
                    writer.writerow([hex_string])
        print(f"Successfully converted {anon_file_path} to {csv_file_path}")
    except Exception as e:
        print(f"Failed to convert {anon_file_path} due to error: {e}")

def process_directory_binary(directory_path):
    """
    Processes all .anon files in the directory and converts them to .csv files using binary read.

    :param directory_path: Path to the directory containing .anon files.
    """
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".anon"):
            anon_file_path = os.path.join(directory_path, file_name)
            csv_file_path = os.path.join(directory_path, file_name.replace('.anon', '.csv'))
            convert_anon_to_csv_binary(anon_file_path, csv_file_path)

# Set the directory path to where your .anon files are located
directory_path = r'D:\FSl Data\fslhomes-user000-2015-04-10'  # Update this path as needed

# Process the directory to convert all .anon files to .csv using binary handling
process_directory_binary(directory_path)
