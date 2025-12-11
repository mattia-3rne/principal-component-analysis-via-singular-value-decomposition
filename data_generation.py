# Libraries Import
import pandas as pd
import numpy as np

# Data Configuration
n_samples = 100
n_features = 10
n_dominant = 3

# Generating the variables
latent_vars = np.random.normal(0, 10, size=(n_samples, n_dominant))

# Creating a mixing matrix
mixing_matrix = np.random.randn(n_dominant, n_features)

# Computing the raw features
X = np.dot(latent_vars, mixing_matrix)

# Adding Gaussian noise
noise = np.random.normal(0, 3.0, size=(n_samples, n_features))

data = X + noise

# Creating the dataframe
cols = [f"feature_{i+1}" for i in range(n_features)]
df = pd.DataFrame(data, columns=cols)

output_filename = 'data.csv'
df.to_csv(output_filename, index=False)

print(f"Success: {output_filename} was generated.")
print(f"Dimensions: {len(df)} samples with {len(df.columns)} features.")