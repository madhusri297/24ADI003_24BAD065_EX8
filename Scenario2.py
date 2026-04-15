print("Madhusri S-24BAD065")
# 1. Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# 2. Load dataset (Iris dataset)
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)

print("\nOriginal Data:")
print(df.head())

# 3. Preprocessing (check missing values)
print("\nMissing Values:\n", df.isnull().sum())

# 4. Standardize features
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# 5. Apply PCA
pca = PCA()
pca_data = pca.fit_transform(scaled_data)

# 6. Explained variance ratio
explained_variance = pca.explained_variance_ratio_

print("\nExplained Variance Ratio:")
print(explained_variance)

# 7. Reduce dimensions (2D)
pca_2 = PCA(n_components=2)
reduced_data = pca_2.fit_transform(scaled_data)

# Convert to DataFrame
pca_df = pd.DataFrame(reduced_data, columns=['PC1', 'PC2'])

print("\nReduced Data (2D):")
print(pca_df.head())

# 8. Visualization

# Scree Plot
plt.figure()
plt.plot(range(1, len(explained_variance)+1), explained_variance, marker='o')
plt.title("Scree Plot")
plt.xlabel("Principal Components")
plt.ylabel("Variance")
plt.show()

# Cumulative Variance
plt.figure()
plt.plot(np.cumsum(explained_variance), marker='o')
plt.title("Cumulative Variance")
plt.xlabel("Number of Components")
plt.ylabel("Cumulative Variance")
plt.show()

# 2D Scatter Plot
plt.figure()
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.title("PCA 2D Scatter Plot")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.show()