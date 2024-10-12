import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Create a synthetic dataset
rng = np.random.RandomState(42)

# Generate normal data points
normal_data = rng.randn(100, 2)

# Generate some outliers
outliers = rng.uniform(low=-6, high=6, size=(10, 2))

# Combine normal data and outliers
data = np.concatenate([normal_data, outliers])

# Plot the dataset
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Normal Data')
plt.scatter(outliers[:, 0], outliers[:, 1], color='red', label='Outliers')
plt.title('Dataset with Normal Points and Outliers')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# Fit the Isolation Forest model
iso_forest = IsolationForest(contamination=0.1)  # Assume 10% of the data is outliers
iso_forest.fit(data)

# Predict anomalies
predictions = iso_forest.predict(data)

# Mark the anomalies in the dataset
anomalies = data[predictions == -1]

# Plot the results
plt.scatter(data[:, 0], data[:, 1], color='blue', label='Normal Data')
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='red', label='Detected Anomalies')
plt.title('Anomaly Detection using Isolation Forest')
plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
