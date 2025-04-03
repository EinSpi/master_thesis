import numpy as np
from scipy.spatial import KDTree


def compute_noise_level_batch(X: np.ndarray, Y: np.ndarray, query_indices: np.ndarray, M: int) -> np.ndarray:
    """
    Compute the noise level for multiple sets of points in parallel.

    Parameters:
    - X: (N, d) ndarray, dataset input points.
    - Y: (N, 1) ndarray, ground truth function values.
    - query_indices: (L, d) ndarray, indices of query points within X.
    - M: int, number of nearest neighbors to consider.

    Returns:
    - noise_levels: (L, d) ndarray, noise level for each query point.
    """
    L, d = query_indices.shape  # Extract dimensions

    # Extract actual query points from X using the provided indices
    query_points = X[query_indices]  # Shape: (L, d, feature_dim)

    # Flatten query points for batch querying (L*d, feature_dim)
    query_points_flat = query_points.reshape(-1, X.shape[1])

    # Build KDTree for efficient nearest neighbor search
    tree = KDTree(X)

    # Query M nearest neighbors for all query points at once
    distances, indices = tree.query(query_points_flat, k=M + 1)  # Shape: (L*d, M+1)

    # Exclude first column (self-neighbor)
    neighbor_indices = indices[:, 1:]  # Shape: (L*d, M)

    # Get function values for neighbors (L*d, M, 1)
    y_neighbors = Y[neighbor_indices]

    # Get function values for query points (L*d, 1)
    y_query = Y[indices[:, 0]]

    # **Fix broadcasting issue: Ensure y_neighbors is (L*d, M)**
    noise_values = np.sum(np.abs(y_neighbors.squeeze(-1) - y_query), axis=1)/M  # Shape: (L*d,)

    # Reshape back to (L, d)
    noise_levels = noise_values.reshape(L, d)

    return noise_levels


# Example Usage
N, feature_dim, L, d = 1000, 2, 10, 5  # 1000 data points, 2D space, 10 sets, 5 samples per set
M = 15  # Number of nearest neighbors

X = np.random.rand(N, feature_dim)  # Random dataset
Y = np.random.rand(N, 1)  # Random function values
query_indices = np.random.randint(0, N, size=(L, d))
query_indices = np.array([[2,4,6],[7,8,9]])# Random L sets of d indices from X

noise_matrix = compute_noise_level_batch(X, Y, query_indices, M)
print("Noise levels (L, d):\n", noise_matrix)
