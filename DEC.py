import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

# Data loading function
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

# Define the autoencoder model
def create_autoencoder(input_dim, latent_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoder = layers.Dense(128, activation='relu')(input_layer)
    encoder = layers.Dense(64, activation='relu')(encoder)
    latent = layers.Dense(latent_dim, name='latent')(encoder)

    decoder = layers.Dense(64, activation='relu')(latent)
    decoder = layers.Dense(128, activation='relu')(decoder)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(decoder)

    autoencoder = models.Model(inputs=input_layer, outputs=output_layer)
    encoder_model = models.Model(inputs=input_layer, outputs=latent)

    return autoencoder, encoder_model

# Clustering layer definition
class ClusteringLayer(layers.Layer):
    def __init__(self, n_clusters, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters

    def build(self, input_shape):
        self.clusters = self.add_weight(shape=(self.n_clusters, input_shape[1]),
                                        initializer='glorot_uniform', name='clusters')

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (tf.reduce_sum(tf.square(tf.expand_dims(inputs, axis=1) - self.clusters), axis=2) / 1.0))
        q = tf.transpose(tf.transpose(q) / tf.reduce_sum(q, axis=1))
        return q

# Target distribution function
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T

# Training DEC with reduced complexity
def train_dec(dataframes, n_clusters=3, latent_dim=2, batch_size=50, maxiter=100, update_interval=10, tol=1e-3):
    # Preprocess data
    combined_df = preprocess_data(dataframes, sample_size=500)  # Use a smaller subset for testing
    max_length = max(combined_df['binary_data'].apply(len))
    input_dim = max_length
    data = np.stack(combined_df['binary_data'].apply(lambda x: pad_data(np.frombuffer(x, dtype=np.uint8), max_length)).values)
    data = data.astype('float32') / 255.0  # Normalize the data

    # Initialize autoencoder
    autoencoder, encoder = create_autoencoder(input_dim, latent_dim)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(data, data, batch_size=batch_size, epochs=10, verbose=1)  # Reduce epochs

    # Initialize cluster centers using K-Means
    kmeans = KMeans(n_clusters=n_clusters)
    y_pred = kmeans.fit_predict(encoder.predict(data))
    y_pred_last = np.copy(y_pred)

    # Create DEC model
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
    dec = models.Model(inputs=encoder.input, outputs=[clustering_layer, autoencoder.output])
    dec.compile(optimizer=optimizers.Adam(0.0001, 0.99), loss=['kld', 'mse'])

    # Set initial cluster weights
    dec.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

    index = 0
    with tqdm(total=int(maxiter)) as pbar:
        for ite in range(int(maxiter)):
            if ite % update_interval == 0:
                q, _ = dec.predict(data, verbose=0)
                p = target_distribution(q)
                y_pred = q.argmax(1)
                acc = np.round(accuracy_score(y_pred, y_pred_last), 5)
                pbar.set_description(f"Iter {ite}, Acc {acc}")
                if ite > 0 and acc < tol:
                    print(f'Converged at iteration {ite}, accuracy: {acc}')
                    break
                y_pred_last = np.copy(y_pred)

            idx = np.random.randint(0, data.shape[0], size=batch_size)
            loss = dec.train_on_batch(x=data[idx], y=[p[idx], data[idx]])
            index = (index + 1) % data.shape[0]
            pbar.update(1)
            pbar.set_postfix(loss=np.round(loss, 5))  # Fixing the error here

    return dec, encoder

# Visualize clustering result
def visualize_clustering(encoder, dataframes, plot_filename='dec_clustering.png'):
    combined_df = preprocess_data(dataframes, sample_size=500)  # Use a smaller subset for testing
    max_length = max(combined_df['binary_data'].apply(len))
    data = np.stack(combined_df['binary_data'].apply(lambda x: pad_data(np.frombuffer(x, dtype=np.uint8), max_length)).values)
    data = data.astype('float32') / 255.0  # Normalize the data
    latent = encoder.predict(data)

    kmeans = KMeans(n_clusters=3)
    y_pred = kmeans.fit_predict(latent)
    plt.scatter(latent[:, 0], latent[:, 1], c=y_pred)
    plt.xlabel('Latent Dim 1')
    plt.ylabel('Latent Dim 2')
    plt.title('DEC Clustering')
    plt.savefig(plot_filename)
    plt.close()

if __name__ == "__main__":
    directory_path = r'D:\FSl Data\fslhomes-user000-2015-04-10'
    fsl_data = load_csv_files(directory_path)
    dec, encoder = train_dec(fsl_data, n_clusters=3, latent_dim=2, batch_size=50, maxiter=100, update_interval=10, tol=1e-3)
    visualize_clustering(encoder, fsl_data)
