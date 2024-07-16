import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K


# Load the VAE model
def load_vae(input_dim, latent_dim):
    # Encoder (redefine but not used here)
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)
    z = layers.Lambda(lambda args: args[0] + K.exp(0.5 * args[1]) * K.random_normal(K.shape(args[0])),
                      output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='sigmoid')(x)
    decoder = models.Model(latent_inputs, outputs, name='decoder')

    return encoder, decoder


# Function to load the compressed data
def load_compressed_data(file_path, latent_dim):
    with open(file_path, 'rb') as f:
        compressed_data = np.frombuffer(f.read(), dtype=np.float32)
    return compressed_data.reshape(-1, latent_dim)


# Function to decompress the data
def decompress_data(compressed_data, decoder, original_shape):
    decompressed_data = decoder.predict(compressed_data)
    return decompressed_data.reshape(original_shape)


# Function to save decompressed data to CSV
def save_to_csv(decompressed_data, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i, data in enumerate(decompressed_data):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(output_dir, f'decompressed_{i}.csv'), index=False, header=False)


# Main function
if __name__ == "__main__":
    compressed_file_path = 'compressed_data.bin'  # Path to the compressed binary file
    output_dir = 'decompressed_output'  # Directory to save decompressed CSV files
    input_dim = 8314  # Update this to match the input dimension used during compression
    latent_dim = 2  # Update this to match the latent dimension used during compression
    original_shape = (-1, input_dim)  # The shape of the original data

    directory_path = r'D:\FSl Data\fslhomes-user000-2015-04-10'  # Update this path as needed

    # Load the VAE model
    encoder, decoder = load_vae(input_dim, latent_dim)

    # Load the compressed data
    compressed_data = load_compressed_data(compressed_file_path, latent_dim)

    # Decompress the data
    decompressed_data = decompress_data(compressed_data, decoder, original_shape)

    # Save the decompressed data to CSV files
    save_to_csv(decompressed_data, output_dir)

    print(f'Decompressed data saved to {output_dir}')
