import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster

# Load the data
df = pd.read_csv('C:/Users/sushr/Downloads/ltogslc53vnx5llf.csv', low_memory=False)

# Convert 'RET' column to numeric, coercing non-numeric values to NaN
df['RET'] = pd.to_numeric(df['RET'], errors='coerce')

# Drop rows with NaN values in the 'RET' column
df = df.dropna(subset=['RET'])

# Filter the data based on NAICS codes starting with 21, 22, 23, or 33
df = df[df['NAICS'].astype(str).str.startswith(('21', '22', '23', '33'))]

# Get the subset of data where NFG has returns
nfg_data = df[df['TICKER'] == 'NFG']

# Filter the main DataFrame to include only dates where NFG has data
df = df[df['date'].isin(nfg_data['date'])]

# Pivot the DataFrame to get returns for each ticker on each date
pivoted_df = df.pivot_table(index='date', columns='TICKER', values='RET', aggfunc='mean')

# Drop any tickers that have missing values after this operation
pivoted_df.dropna(axis=1, how='any', inplace=True)

# Calculate the correlation matrix for the pivoted data
correlation_matrix = pivoted_df.corr()

# We are interested in the correlation of all stocks with NFG
nfg_correlation = correlation_matrix['NFG']

# Convert to a distance matrix
distance_matrix = 1 - correlation_matrix.abs()

# Extract the lower triangular part of the distance matrix (excluding diagonal)
distance_vector = distance_matrix.values[np.tril_indices(distance_matrix.shape[0], k=-1)]

# Compute the mean distance from the distance vector
mean_distance = distance_vector.mean()

# Set max_d as a multiple of the mean distance
max_d = 1.5 * mean_distance

# Perform hierarchical clustering directly using the distance vector
Z = linkage(distance_vector, 'ward')

# Perform hierarchical clustering
clusters = fcluster(Z, max_d, criterion='distance')

# Map clusters back to tickers
ticker_clusters = pd.Series(clusters, index=nfg_correlation.index)

# Find all stocks in the same cluster as NFG
nfg_cluster = ticker_clusters['NFG']
clustered_tickers = ticker_clusters[ticker_clusters == nfg_cluster]

# Print the tickers of stocks in the same cluster as NFG
print("Tickers in the same cluster as NFG:", clustered_tickers.index.tolist())
