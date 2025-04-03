import numpy as np
from scipy.optimize import minimize
from joblib import Parallel, delayed
from scipy.spatial import KDTree


class AdaptiveWavyRat:
    def __init__(self, order:int, sample_uniformly:bool, min_a_para:float, max_a_para:float, num_neighbors:int):
        self.order = order
        self.sample_uniformly = sample_uniformly
        self.a_paras: list = [0.01]*self.order
        self.min_a_para = min_a_para
        self.max_a_para = max_a_para
        self.num_neighbors = num_neighbors


    def sample_points_with_probability(self,x,y,rng,layer_width:int, repetition_scaler:int):
        # define repitition numbers just as original
        n_repetitions = max(1, int(np.ceil(layer_width / x.shape[0]))) * repetition_scaler

        # randomly generate point sets, each contains set_size points
        candidate_sets = []
        for i in range(x.shape[0] * n_repetitions):
            candidate_sets.append(rng.choice(x.shape[0], size=self.order, replace=False))
        candidate_sets = np.array(candidate_sets)

        #assign each randomly chosen point set a probability, using the strategy defined in probability_evaluator
        probability = self.probability_evaluator(candidate_sets, x, y)

        # select point sets according to the associated probabilities
        selected_set_indices = rng.choice(len(candidate_sets), size=layer_width, replace=True, p=probability)
        query_indices = candidate_sets[selected_set_indices]
        #use the result to compute noisy level, then use noisy level to compute a_paras
        noisy_level=self.compute_noisy_level(x,y,query_indices)
        self.compute_a_para(noisy_level)
        #return the results
        return query_indices


    def probability_evaluator(self,candidate_sets: np.ndarray,x,y):
        sum_y = np.sum(y[candidate_sets, ...], axis=1)
        sum_y = sum_y - np.min(sum_y)
        sum_y = sum_y.squeeze()
        if self.sample_uniformly:
            probab = np.ones_like(sum_y) / len(sum_y)
        else:
            probab = sum_y / np.sum(sum_y)

        return probab

    def parameter_calculator(self,x,y,rng,layer_width:int, repetition_scaler:int):
        #generate peaks by order
        peaks=np.array(range(self.order))
        #select x point sets, idx
        selected_point_sets = self.sample_points_with_probability(x, y, rng, layer_width, repetition_scaler)
        #use the idx to slice the x point sets out
        x_array = x[selected_point_sets] #(M,2,2)
        #all one vector
        ones=np.ones(self.order)

        #use optimizer to compute the constrained problem
        results = Parallel(n_jobs=-1)(delayed(lambda s:minimize(lambda w:np.sum((s@w[:self.order]+w[self.order]*ones-peaks)**2),
                                                               np.append(np.ones(self.order),0),
                                                               constraints={'type':'eq', 'fun': lambda w: np.linalg.norm(w[:self.order])-1}).x)(x_array[i])
                                     for i in range(layer_width))
        weights = np.array([result[:self.order] for result in results]).T
        biases  = np.array([result[self.order]  for result in results]).reshape(1,layer_width)
        idx_from, idx_to = np.array([0]), np.array([0])

        return weights, biases, idx_from, idx_to

    def infer(self,x:np.ndarray):
        # 将 x 扩展为 (B, M, 1)，a_para 扩展为 (1, M, D)
        x_expanded = x[:, :, np.newaxis]  # 形状: (B, M, 1)
        a_para_expanded = self.a_paras[np.newaxis, :, :]  # 形状: (1, M, D)
        square_term =(x_expanded-np.arange(self.order))**2
        result = np.sum(np.divide(10-a_para_expanded*square_term,10+a_para_expanded*square_term)+1, axis=-1)
        return result


    def compute_noisy_level(self,x:np.ndarray,y: np.ndarray, query_indices: np.ndarray):
        #given the selected x sets indices, compute how noisy at this set
        L, d = query_indices.shape  # Extract dimensions

        # Extract actual query points from X using the provided indices
        query_points = x[query_indices]  # Shape: (L, d, feature_dim)

        # Flatten query points for batch querying (L*d, feature_dim)
        query_points_flat = query_points.reshape(-1, x.shape[1])

        # Build KDTree for efficient nearest neighbor search
        tree = KDTree(x)

        # Query M nearest neighbors for all query points at once
        distances, indices = tree.query(query_points_flat, k=self.num_neighbors + 1)  # Shape: (L*d, M+1)

        # Exclude first column (self-neighbor)
        neighbor_indices = indices[:, 1:]  # Shape: (L*d, M)

        # Get function values for neighbors (L*d, M, 1)
        y_neighbors = y[neighbor_indices]

        # Get function values for query points (L*d, 1)
        y_query = y[indices[:, 0]]

        # **Fix broadcasting issue: Ensure y_neighbors is (L*d, M)**
        noise_values = np.sum(np.abs(y_neighbors.squeeze(-1) - y_query), axis=1) / self.num_neighbors  # Shape: (L*d,)

        # Reshape back to (L, d)
        noise_levels = noise_values.reshape(L, d)

        return noise_levels

    def compute_a_para(self,noise_levels:np.ndarray):
        max_noise=np.max(noise_levels)
        min_noise=np.min(noise_levels)
        if max_noise == min_noise:
            self.a_paras= np.full_like(noise_levels, self.min_a_para)
        else:
            self.a_paras = self.min_a_para+(noise_levels-min_noise)/(max_noise-min_noise)*(self.max_a_para-self.min_a_para)
