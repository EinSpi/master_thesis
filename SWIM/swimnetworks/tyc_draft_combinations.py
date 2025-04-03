import numpy as np

# Sample input: shape (M, 2, 11)
M, K = 5, 3
A = np.random.randint(0, 100, size=(M, 2, K))  # Example data

# Vectorized computation of all combinations
# Step 1: Reshape to (M, 11, 1) and (M, 1, 11)
A1 = A[:, 0, :, np.newaxis]  # shape (M, 11, 1)
A2 = A[:, 1, np.newaxis, :]  # shape (M, 1, 11)

# Step 2: Broadcast to shape (M, 11, 11, 2)
pairs = np.stack([A1.repeat(K, axis=2), A2.repeat(K, axis=1)], axis=-1)  # shape (M, 11, 11, 2)

# Step 3: Reshape to (M, 121, 2)
pairs_reshaped = pairs.reshape((M, K * K, 2))

print(pairs_reshaped.shape)
print(A)
print(pairs_reshaped)
