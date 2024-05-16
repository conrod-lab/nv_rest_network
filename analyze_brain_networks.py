import numpy as np
import pandas as pd
from nilearn.datasets import fetch_coords_seitzman_2018

# Function to generate a random correlation matrix
def generate_random_correlation_matrix(num_rois):
    data = np.random.rand(num_rois, num_rois)
    data = (data + data.T) / 2  # Make the matrix symmetric
    np.fill_diagonal(data, 1)  # Set diagonal to 1 to simulate correlation matrix
    return data

# Generate correlation matrices for two time points
num_rois = 300
correlation_matrix_t1 = generate_random_correlation_matrix(num_rois)
correlation_matrix_t2 = generate_random_correlation_matrix(num_rois)

# Simulate a case where two networks were originally highly connected but are not at the end
# For simplicity, we'll assume the first 50 ROIs belong to Network 1 and the next 50 belong to Network 2
network1_indices = list(range(50))
network2_indices = list(range(50, 100))

# Increase the connectivity between Network 1 and Network 2 at timepoint 1
for i in network1_indices:
    for j in network2_indices:
        correlation_matrix_t1[i, j] = correlation_matrix_t1[j, i] = 0.8  # Strong connection

# Decrease the connectivity between Network 1 and Network 2 at timepoint 2
for i in network1_indices:
    for j in network2_indices:
        correlation_matrix_t2[i, j] = correlation_matrix_t2[j, i] = 0.2  # Weak connection

# Fetch the Seitzman atlas using nilearn
seitzman_atlas = fetch_coords_seitzman_2018()

# Get the network labels and assign them to the first 300 ROIs
network_labels = seitzman_atlas['networks'][:num_rois]

# Create a dataframe for network labels
df_network = pd.Series(network_labels, name='network')

# Function to calculate within-network means
def calculate_within_network_means(correlation_matrix, df_network):
    df = pd.DataFrame(correlation_matrix)
    df['network'] = df_network
    network_indices = {network_value: df.index[df['network'] == network_value].tolist()
                       for network_value in df['network'].unique()}
    within_network_means = {}
    for network_value, indices in network_indices.items():
        if len(indices) > 1:
            sub_matrix = correlation_matrix[np.ix_(indices, indices)]
            within_network_means[network_value] = sub_matrix[np.triu_indices_from(sub_matrix, k=1)].mean()
    return within_network_means

# Function to calculate between-network means
def calculate_between_network_means(correlation_matrix, df_network):
    df = pd.DataFrame(correlation_matrix)
    df['network'] = df_network
    network_indices = {network_value: df.index[df['network'] == network_value].tolist()
                       for network_value in df['network'].unique()}
    between_network_means = {}
    network_values = list(network_indices.keys())
    for i, network_value1 in enumerate(network_values):
        for network_value2 in network_values[i+1:]:
            indices1 = network_indices[network_value1]
            indices2 = network_indices[network_value2]
            sub_matrix = correlation_matrix[np.ix_(indices1, indices2)]
            between_network_means[(network_value1, network_value2)] = sub_matrix.mean()
    return between_network_means

# Calculate within-network and between-network means for both time points
within_network_means_t1 = calculate_within_network_means(correlation_matrix_t1, df_network)
within_network_means_t2 = calculate_within_network_means(correlation_matrix_t2, df_network)
between_network_means_t1 = calculate_between_network_means(correlation_matrix_t1, df_network)
between_network_means_t2 = calculate_between_network_means(correlation_matrix_t2, df_network)

# Output results
print("Within-Network Means at Timepoint 1:")
print(within_network_means_t1)
print("\nWithin-Network Means at Timepoint 2:")
print(within_network_means_t2)

print("\nBetween-Network Means at Timepoint 1:")
print(between_network_means_t1)
print("\nBetween-Network Means at Timepoint 2:")
print(between_network_means_t2)

# Calculate changes in between-network means
between_network_changes = {}
for networks, mean_t1 in between_network_means_t1.items():
    mean_t2 = between_network_means_t2[networks]
    between_network_changes[networks] = mean_t2 - mean_t1

print("\nChanges in Between-Network Means:")
print(between_network_changes)
