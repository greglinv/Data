import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras import layers, models, backend as K
from tensorflow.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt


# Load CSV files function
def load_csv_files(directory_path):
    dataframes = []
    for file_name in os.listdir(directory_path):
        if file_name.endswith(".csv"):
            file_path = os.path.join(directory_path, file_name)
            df = pd.read_csv(file_path, header=None, names=['hex_data'])
            dataframes.append(df)
    return dataframes


# Data preprocessing functions
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


# Custom layer for VAE loss
class VAELossLayer(layers.Layer):
    def __init__(self, input_dim, **kwargs):
        super(VAELossLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    def call(self, inputs):
        x, z_mean, z_log_var, outputs = inputs
        reconstruction_loss = binary_crossentropy(x, outputs) * self.input_dim
        kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
        kl_loss = K.sum(kl_loss, axis=-1)
        kl_loss *= -0.5
        total_loss = K.mean(reconstruction_loss + kl_loss)

        self.add_loss(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        return outputs

    def compute_output_shape(self, input_shape):
        return input_shape[0]

    @property
    def metrics(self):
        return [self.reconstruction_loss_tracker, self.kl_loss_tracker]


# Define the VAE model
def create_vae(input_dim, latent_dim):
    # Encoder
    inputs = layers.Input(shape=(input_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    z_mean = layers.Dense(latent_dim, name='z_mean')(x)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

    def sampling(args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + K.exp(0.5 * z_log_var) * epsilon

    z = layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

    encoder = models.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    # Decoder
    latent_inputs = layers.Input(shape=(latent_dim,), name='z_sampling')
    x = layers.Dense(64, activation='relu')(latent_inputs)
    x = layers.Dense(128, activation='relu')(x)
    outputs = layers.Dense(input_dim, activation='sigmoid')(x)

    decoder = models.Model(latent_inputs, outputs, name='decoder')

    # VAE with custom loss layer
    vae_outputs = decoder(encoder(inputs)[2])
    vae_outputs = VAELossLayer(input_dim)([inputs, encoder(inputs)[0], encoder(inputs)[1], vae_outputs])
    vae = models.Model(inputs, vae_outputs, name='vae')

    vae.compile(optimizer='adam')
    return vae, encoder, decoder


# Train VAE and track performance metrics
def train_vae(dataframes, latent_dim=2, batch_size=50, epochs=10):
    combined_df = preprocess_data(dataframes, sample_size=500)  # Adjust sample size as needed
    max_length = max(combined_df['binary_data'].apply(len))
    input_dim = max_length
    data = np.stack(
        combined_df['binary_data'].apply(lambda x: pad_data(np.frombuffer(x, dtype=np.uint8), max_length)).values)
    data = data.astype('float32') / 255.0  # Normalize the data

    vae, encoder, decoder = create_vae(input_dim, latent_dim)
    history = vae.fit(data, data, epochs=epochs, batch_size=batch_size, verbose=1)

    return vae, encoder, decoder, history


# Visualize VAE results
def visualize_latent_space(encoder, dataframes, plot_filename='vae_latent_space.png'):
    combined_df = preprocess_data(dataframes, sample_size=500)  # Adjust sample size as needed
    max_length = max(combined_df['binary_data'].apply(len))
    data = np.stack(
        combined_df['binary_data'].apply(lambda x: pad_data(np.frombuffer(x, dtype=np.uint8), max_length)).values)
    data = data.astype('float32') / 255.0  # Normalize the data
    z_mean, _, _ = encoder.predict(data)
    plt.scatter(z_mean[:, 0], z_mean[:, 1])
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.title('VAE Latent Space')
    plt.savefig(plot_filename)
    plt.close()


# Plot training loss
def plot_loss(history, plot_filename='vae_loss.png'):
    plt.plot(history.history['loss'], label='Total Loss')
    plt.plot(history.history['reconstruction_loss'], label='Reconstruction Loss')
    plt.plot(history.history['kl_loss'], label='KL Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(plot_filename)
    plt.close()


if __name__ == "__main__":
    directory_path = r'D:\FSl Data\fslhomes-user000-2015-04-10'  # Update this path as needed
    fsl_data = load_csv_files(directory_path)
    vae, encoder, decoder, history = train_vae(fsl_data, latent_dim=2, batch_size=50, epochs=10)
    visualize_latent_space(encoder, fsl_data)
    plot_loss(history)
