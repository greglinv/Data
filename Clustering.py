import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from data_analysis import load_csv_files, add_hashes  # Assuming these functions are in data_analysis.py
import time

def hex_to_binary(hex_str):
    return bytes.fromhex(hex_str)

def preprocess_data(dataframes, sample_size=None):
    combined_df = pd.concat(dataframes)
    combined_df.reset_index(drop=True, inplace=True)
    combined_df['binary_data'] = combined_df['hex_data'].apply(hex_to_binary)
    if sample_size:
        combined_df = combined_df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    return combined_df

def pad_data(data, max_length):
    return np.pad(data, (0, max_length - len(data)), 'constant')

def compute_vae_loss(x, x_decoded_mean, z_mean, z_log_var, input_dim):
    reconstruction_loss = tf.keras.losses.binary_crossentropy(x, x_decoded_mean)
    reconstruction_loss *= input_dim
    kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
    kl_loss = tf.keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    return tf.keras.backend.mean(reconstruction_loss + kl_loss)

def train_vae_on_batches(combined_df, batch_size=100, epochs=2):
    # Determine the maximum length of binary data
    max_length = max(combined_df['binary_data'].apply(len))
    input_dim = max_length
    latent_dim = 2  # Dimensionality of the latent space

    # Define VAE model
    encoder_inputs = tf.keras.Input(shape=(input_dim,), dtype=tf.float32)
    x = layers.Dense(64, activation='relu')(encoder_inputs)
    x = layers.Dense(32, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    decoder_inputs = tf.keras.Input(shape=(latent_dim,))
    x = layers.Dense(32, activation='relu')(decoder_inputs)
    x = layers.Dense(64, activation='relu')(x)
    decoder_outputs = layers.Dense(input_dim, activation='sigmoid')(x)

    encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
    decoder = tf.keras.Model(decoder_inputs, decoder_outputs, name='decoder')
    vae_outputs = decoder(encoder(encoder_inputs)[2])
    vae = tf.keras.Model(encoder_inputs, vae_outputs, name='vae')

    # Compile the model with a dummy loss to initialize
    vae.compile(optimizer='adam', loss='mse')

    # Custom training loop
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train_step(x):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = encoder(x)
            x_decoded_mean = decoder(z)
            loss = compute_vae_loss(x, x_decoded_mean, z_mean, z_log_var, input_dim)
        gradients = tape.gradient(loss, vae.trainable_variables)
        optimizer.apply_gradients(zip(gradients, vae.trainable_variables))
        return loss

    for epoch in range(epochs):
        print(f'Start of epoch {epoch}')
        start_time = time.time()
        for start in range(0, len(combined_df), batch_size):
            end = min(start + batch_size, len(combined_df))
            batch_data = np.stack(combined_df['binary_data'].iloc[start:end].apply(lambda x: pad_data(np.frombuffer(x, dtype=np.uint8), max_length)).values)
            batch_data = batch_data.astype('float32') / 255.0  # Normalize the data
            loss = train_step(batch_data)
            print(f'Processed batch {start // batch_size + 1}/{len(combined_df) // batch_size + 1}, Batch Loss: {loss.numpy()}')
        end_time = time.time()
        print(f'Epoch {epoch} Loss: {loss.numpy()}, Time taken: {end_time - start_time:.2f} seconds')

    return encoder

def cluster_and_plot(encoder, combined_df, plot_filename='vae_clustering.png'):
    max_length = max(combined_df['binary_data'].apply(len))
    data = np.stack(combined_df['binary_data'].apply(lambda x: pad_data(np.frombuffer(x, dtype=np.uint8), max_length)).values)
    data = data.astype('float32') / 255.0  # Normalize the data
    z_mean, _, _ = encoder.predict(data)
    kmeans = KMeans(n_clusters=3)
    clusters = kmeans.fit_predict(z_mean)
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=clusters)
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.title('VAE Clustering')
    plt.savefig(plot_filename)
    plt.close()

if __name__ == "__main__":
    directory_path = r'D:\FSl Data\fslhomes-user000-2015-04-10'
    fsl_data = load_csv_files(directory_path)
    combined_df = preprocess_data(fsl_data, sample_size=1000)  # Use a smaller subset of data for testing
    encoder = train_vae_on_batches(combined_df)
    cluster_and_plot(encoder, combined_df)
