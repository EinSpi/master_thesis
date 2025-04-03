from __future__ import annotations, division

from dataclasses import dataclass
from typing import Callable, Union
import numpy as np
from scipy.spatial import KDTree

from memory_profiler import profile
from .base import Base

@dataclass
class Dense(Base):
    parameter_sampler: Union[Callable, str] = "relu"
    sample_uniformly: bool = False
    prune_duplicates: bool = False
    random_seed: int = 1
    dist_min: np.float64 = 1e-10
    repetition_scaler: int = 1

    idx_from: np.ndarray = None
    idx_to: np.ndarray = None

    def __post_init__(self):
        super().__post_init__()
        self.n_pruned_neurons = 0

        if not isinstance(self.parameter_sampler, Callable):
            if self.parameter_sampler == "relu":
                self.parameter_sampler = self.sample_parameters_act
            elif self.parameter_sampler == "relu_2nd_grd":
                self.parameter_sampler = self.sample_parameters_relu_2nd_grd
                self.probability_evaluator=self.middle_point_slope_difference_probability_evaluator
            elif self.parameter_sampler == "tanh":
                self.parameter_sampler = self.sample_parameters_act
            elif self.parameter_sampler == "random":
                self.parameter_sampler = self.sample_parameters_randomly
            elif self.parameter_sampler == "rat":
                self.parameter_sampler = self.sample_parameters_act
            elif self.parameter_sampler == "sigmoid":
                self.parameter_sampler = self.sample_parameters_act
            elif self.parameter_sampler == "relu_like_rat_news1s2":
                self.parameter_sampler = self.sample_parameters_rat_news1s2
            elif self.parameter_sampler == "gaussian":
                self.parameter_sampler = self.sample_parameters_gaussian
            elif self.parameter_sampler == "peaky_rat":
                self.parameter_sampler = self.sample_parameters_peaky_rat
                self.probability_evaluator=self.y_sum_probability_evaluator
            elif self.parameter_sampler == "wavy_rat":
                self.parameter_sampler = self.sample_parameters_wavy_rat
                self.probability_evaluator=self.y_sum_probability_evaluator
            elif self.parameter_sampler == "adpt_rat" or self.parameter_sampler == "adpt_relu" or self.parameter_sampler == "adpt_sigmoid" or self.parameter_sampler == "adpt_tanh":
                self.parameter_sampler = self.sample_parameters_act
            else:
                raise ValueError(f"Unknown parameter sampler {self.parameter_sampler}.")

    def fit(self, x, y=None,error_map=None,candidate_sets:np.ndarray=None):
        if self.layer_width is None:
            raise ValueError("layer_width must be set.")
        
        x, y = self.clean_inputs(x, y)

        rng = np.random.default_rng(self.random_seed)
        self.weights, self.biases, self.idx_from, self.idx_to = self.parameter_sampler(x, y, rng=rng,candidate_sets=candidate_sets,error_map=error_map)


        self.n_parameters = np.prod(self.weights.shape) + np.prod(self.biases.shape)
        return self

    def fit_transform(self, x, y=None,error_map=None,candidate_sets:np.ndarray=None):
        self.fit(x, y,error_map=error_map,candidate_sets=candidate_sets)
        return self.transform(x, y)

    def sample_parameters_tanh(self, x, y, rng,candidate_sets,error_map):
        scale = 0.5 * (np.log(1 + 1/2) - np.log(1 - 1/2))

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng,candidate_sets,error_map)
        weights = (2 * scale * directions / dists).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1) - scale

        return weights, biases, idx_from, idx_to

    def sample_parameters_relu_1st_grd(self, x, y, rng,candidate_sets):
        scale = 1.0

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)
        weights = (scale / dists.reshape(-1, 1) * directions).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1)

        return weights, biases, idx_from, idx_to

        
    def sample_parameters_rat(self, x, y, rng,candidate_sets):
        #this is the w and b sampler function for rational func. simply copy of relu
        x1x2=[]
        with open("GD_Results/Rational/layers1/width"+str(self.layer_width)+"/x1x2.txt", "r") as f:
            for line in f:
                x1x2.append(float(line.strip()))

        scale =x1x2[1]-x1x2[0]

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)
        weights = (scale / dists.reshape(-1, 1) * directions).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1)+x1x2[0]

        return weights, biases, idx_from, idx_to

    def sample_parameters_peaky_rat(self, x, y, rng,candidate_sets):
        #sample parameters for multi-peak rational functions
        #default by 3 peaks
        selected_point_sets=self.sample_point_sets_with_probability(x,y,rng,3)

        #use the x values as the peak points to parametrize the multi-peak rational functions
        weights=x[selected_point_sets]
        weights=weights.reshape(selected_point_sets.shape[0],-1)

        #store all parameters in weights, no use for biases, etc. These are just place holders
        biases, idx_from, idx_to=np.array([0]),np.array([0]),np.array([0])

        return weights, biases, idx_from, idx_to

    def sample_parameters_wavy_rat(self, x, y, rng,candidate_sets):
        selected_point_sets = self.sample_point_sets_with_probability(x, y, rng, 3)
        M=selected_point_sets.shape[0]
        #find w and b for affine transformation so that wx+b for the 3 xs align with the peak value
        #wx1+b=a1,wx2+b=a2,wx3+b=a3,
        selected_X = x[selected_point_sets]
        # Construct F matrices (M, 3, 3),that contains selected X points
        ones_column = np.ones((M, 3, 1))  # Shape (M, 3, 1) for 1s column
        F = np.concatenate([selected_X, ones_column], axis=-1)  # Shape (M, 3, 3)
        # predefined peaky values at 1,0,-1, let them align with the 3 sampled x after the affine transformation
        peaky_values = np.array([-1, 0, 1]).reshape(3, 1)
        #intialize weight and bias matrix
        w_and_b=np.zeros((M,3))
        #for each F, check if it's invertible, if not retreat to lstsq.
        for i in range(M):
            if np.linalg.cond(F[i]) < 1e12:
                # F invertible, solve as normal
                w_and_b[i] = np.linalg.solve(F[i], peaky_values).flatten()
            else:
                # Use least squares as fallback for singular matrices
                w_and_b[i], _, _, _ = np.linalg.lstsq(F[i], peaky_values, rcond=None)

        #splite the weights and biases to store them
        weights, biases = w_and_b[:, :2], w_and_b[:, 2]
        #ensure multiplication compatibility
        weights=weights.T
        biases=biases.reshape(1,M)
        idx_from, idx_to = np.array([0]), np.array([0])

        return weights, biases, idx_from, idx_to

    #2nd gradient oriented sample for relu

    def sample_parameters_relu_2nd_grd(self, x, y, rng,candidate_sets):
        selected_point_sets = self.sample_point_sets_with_probability(x, y, rng, 2)
        #place the relu where x1 is 1,x2 is -1
        x1=x[selected_point_sets[:,0]] #(M,2)
        x2=x[selected_point_sets[:,1]] #(M,2)
        diff=x2-x1#M.2
        l2_norms = np.linalg.norm(diff, axis=1, keepdims=True)
        s1 ,s2= -1,1 #this makes x2 -1 and x1 1
        weights=s1*(diff/l2_norms)#(M,2)
        biases=-1*(np.sum(weights*x1, axis=1, keepdims=True)+s2)#(M,1)

        #ensure multiplication compatibility
        weights=weights.T#(2,M)
        biases=biases.T#(1,M)
        idx_from=selected_point_sets[:, 0]
        idx_to=selected_point_sets[:, 1]

        return weights, biases, idx_from, idx_to


    def sample_parameters_rat_after_GD(self, x, y, rng,candidate_sets):
        # this is the w and b sampler function for rational func. simply copy of relu
        scale = 1.0

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)
        weights = (scale / dists.reshape(-1, 1) * directions).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1)

        return weights, biases, idx_from, idx_to

    def sample_parameters_rat_news1s2 (self, x, y, rng,candidate_sets):
        #this is the w and b sampler function for rational func. This is the intuitive s1 s2 design for relu like rat.
        scale = 0.875

        directions, dists, idx_from, idx_to = self.sample_parameters(x, y, rng)
        weights = (scale / dists.reshape(-1, 1) * directions).T
        biases = -np.sum(x[idx_from, :] * weights.T, axis=-1).reshape(1, -1)-0.375

        return weights, biases, idx_from, idx_to
    
    def sample_parameters_randomly(self, x, y, rng,candidate_sets):
        weights = rng.normal(loc=0, scale=1, size=(self.layer_width, x.shape[1])).T
        biases = rng.uniform(low=-np.pi, high=np.pi, size=(self.layer_width, 1)).T
        idx0 = None
        idx1 = None
        return weights, biases, idx0, idx1

    def sample_point_sets_with_probability(self,x,y,rng,set_size):
        #define repitition numbers just as original
        n_repetitions = max(1, int(np.ceil(self.layer_width / x.shape[0]))) * self.repetition_scaler

        #randomly generate point sets, each contains set_size points
        candidate_sets=[]
        for i in range(x.shape[0]*n_repetitions):
            candidate_sets.append(rng.choice(x.shape[0],size=set_size,replace=False))
        candidate_sets = np.array(candidate_sets)

        probability=self.probability_evaluator(candidate_sets,x,y)

        #select point sets according to the associated probabilities
        selected_set_indices=rng.choice(len(candidate_sets),size=self.layer_width,replace=True,p=probability)

        return candidate_sets[selected_set_indices]



    def sample_parameters(self, x, y, rng,candidate_sets,error_map):
        """
        Sample directions from points to other points in the given dataset (x, y).
        """

        # n_repetitions repeats the sampling procedure to find better directions.
        # If we require more samples than data points, the repetitions will cause more pairs to be drawn.
        n_repetitions = max(1, int(np.ceil(self.layer_width / x.shape[0]))) * self.repetition_scaler

        # This guarantees that:
        # (a) we draw from all the N(N-1)/2 - N possible pairs (minus the exact idx_from=idx_to case)
        # (b) no indices appear twice at the same position (never idx0[k]==idx1[k] for all k)
        candidates_idx_from = rng.integers(low=0, high=x.shape[0], size=x.shape[0]*n_repetitions)
        delta = rng.integers(low=1, high=x.shape[0]-1, size=candidates_idx_from.shape[0])
        candidates_idx_to = (candidates_idx_from + delta) % x.shape[0]
        
        directions = x[candidates_idx_to, ...] - x[candidates_idx_from, ...]
        dists = np.linalg.norm(directions, axis=1, keepdims=True)
        dists = np.clip(dists, a_min=self.dist_min, a_max=None)
        directions = directions / dists

        dy = y[candidates_idx_to, :] - y[candidates_idx_from, :]
        if self.is_classifier:
            dy[np.abs(dy) > 0] = 1

        # We always sample with replacement to avoid forcing to sample low densities
        probabilities = self.weight_probabilities(dy, dists)
        selected_idx = rng.choice(dists.shape[0],
                                  size=self.layer_width,
                                  replace=True,
                                  p=probabilities)
        
        if self.prune_duplicates:
            selected_idx = np.unique(selected_idx)
            self.n_pruned_neurons = self.layer_width - len(selected_idx)
            self.layer_width = len(selected_idx)

        directions = directions[selected_idx]
        dists = dists[selected_idx]
        idx_from = candidates_idx_from[selected_idx]
        idx_to = candidates_idx_to[selected_idx]
        
        return directions, dists, idx_from, idx_to
    @staticmethod
    def gaussian(x: np.ndarray, miu: np.ndarray, sigma: np.ndarray):

        return (1 / sigma ) * np.exp(
            (-1 / 2) * (np.square(x[0] - miu[0]) + np.square(x[1] - miu[1])) / sigma )



    def sample_parameters_gaussian(self, x, y, rng,candidate_sets):
        remaining_indices=np.arange(x.shape[0])
        y_prime=y-np.min(y)
        p=y_prime/np.sum(y_prime)
        p=p.squeeze()
        n_rep=100
        mius=[]
        sigmas=[]
        while n_rep>0 and remaining_indices.shape[0]>0:
            #construct probability distribution
            mask = np.ones_like(p, dtype=bool)
            mask[remaining_indices]=False
            p[mask]=0
            p=p/np.sum(p)
            tree=KDTree(x[remaining_indices])
            #choose a seed x according to the probability
            chosen_x_idx=np.random.choice(np.arange(x.shape[0]),size=1,p=p)
            chosen_x_idx=int(chosen_x_idx)
            miu=x[chosen_x_idx]
            mius.append(miu)
            neighbors=set()
            neighbors.add(chosen_x_idx)
            local_sigmas_sum=0.01
            y_value_squared_sum=y_prime[chosen_x_idx]**2
            iteration_count=0
            #search for valid neighbors using KD tree, collect them in neighbors
            while True:
                iteration_count +=1
                distance, nb_idx=tree.query(x[chosen_x_idx],k=iteration_count+1)
                nb_idx=int(nb_idx[-1])
                distance=distance[-1]
                y_ratio=y_prime[chosen_x_idx] / y_prime[nb_idx]
                if y_ratio>1:
                    sigma_local=distance**2/2*np.log(y_ratio)
                else:
                    sigma_local=0.01
                sigma_before_this=local_sigmas_sum/iteration_count
                sigma=(local_sigmas_sum+sigma_local)/(iteration_count+1)
                inner_product=0
                gaussian_squared_sum=0
                for i in neighbors:
                    gs=Dense.gaussian(miu=miu,sigma=sigma,x=x[i])
                    inner_product=inner_product+gs*y_prime[i]
                    gaussian_squared_sum=gaussian_squared_sum+gs**2
                gs_nb_idx=Dense.gaussian(miu=miu,sigma=sigma,x=x[nb_idx])
                inner_product=inner_product+gs_nb_idx*y_prime[nb_idx]
                y_magnitude=np.sqrt(y_value_squared_sum+y_prime[nb_idx]**2)
                gaussian_magnitude=np.sqrt(gaussian_squared_sum+gs_nb_idx**2)
                cos=inner_product/(y_magnitude*gaussian_magnitude)
                if cos<0.95:
                    sigmas.append(sigma_before_this)
                    break
                else:
                    neighbors.add(nb_idx)
            remaining_indices=np.setdiff1d(remaining_indices, np.array(list(neighbors)))
            n_rep-=len(neighbors)

        return np.array(mius),np.array(sigmas), 0, 0



    def sample_parameters_act(self,x,y,rng=None,candidate_sets=None,error_map=None):
        return self.act.parameter_calculator(x,y,self.layer_width,self.repetition_scaler,rng=rng,candidate_sets=candidate_sets,error_map=error_map)




    def weight_probabilities(self, dy, dists):
        """Compute probability that a certain weight should be chosen as part of the network.
        This method computes all probabilities at once, without removing the new weights one by one.

        Args:
            dy: function difference
            dists: distance between the base points
            rng: random number generator

        Returns:
            probabilities: probabilities for the weights.
        """
        # compute the maximum over all changes in all y directions to sample good gradients for all outputs
        gradients = (np.max(np.abs(dy), axis=1, keepdims=True) / dists).ravel()

        if self.sample_uniformly or np.sum(gradients) < self.dist_min:
            # When all gradients are small, avoind dividing by a small number
            # and default to uniform distribution.
            probabilities = np.ones_like(gradients) / len(gradients)
        else:
            probabilities = gradients / np.sum(gradients)

        return probabilities


    def y_sum_probability_evaluator(self,candidate_sets: np.ndarray,x,y):
        """
        Compute the probability that the given candidate set should be chosen as part of the network.
        The greater is the set's summation y value, the higher is the probability.
        """
        sum_Y=np.sum(y[candidate_sets,...],axis=1)
        sum_Y=sum_Y-np.min(sum_Y)
        sum_Y=sum_Y.squeeze()
        if self.sample_uniformly:
            probab = np.ones_like(sum_Y) / len(sum_Y)
        else:
            probab=sum_Y/np.sum(sum_Y)

        return probab

    def middle_point_slope_difference_probability_evaluator(self,candidate_sets: np.ndarray,x,y):
        """
        given 2 x points x1 and x2, compute the middle point x3=(x1+x2)/2
        calculate the slope difference between x1x3 and x3x2
        """
        #compute the coordinate of the middle points
        x_middle_points=(x[candidate_sets[:, 0]] + x[candidate_sets[:, 1]]) / 2 #(M,2)
        tree = KDTree(x)
        #use KD Tree to find the nearest neighbor of the middle points, retrieve the corresponding y value of that middle point
        _, y_indices=tree.query(x_middle_points)
        y_middle_points=y[y_indices] #(M,1)
        #retrieve the corresponding y values at x1 and x2
        y1=y[candidate_sets[:, 0]] #(M,1)
        y2=y[candidate_sets[:, 1]] #(M,1)
        dy1=np.max(y1-y_middle_points,axis=1,keepdims=True) #(M,1)
        dy2=np.max(y_middle_points-y2,axis=1,keepdims=True) #(M,1)
        #compute the distance of x1 and x2
        direction = x[candidate_sets[:, 0]] - x[candidate_sets[:, 1]]
        dist = np.linalg.norm(direction, axis=1, keepdims=True)
        dist = np.clip(dist, a_min=self.dist_min, a_max=None)
        #compute the slopes and slope difference_rate, i.e. 2nd gradient
        slope1 =(dy1/(dist/2)) #(M,1)
        slope2 =(dy2/(dist/2)) #(M,1)
        slope_difference=(np.abs(slope1-slope2)/dist).ravel() #(M,1)/(M,1)=(M,1), ravel (M,)

        if self.sample_uniformly:
            probab = np.ones_like(slope_difference) / len(slope_difference)
        else:
            probab=slope_difference/np.sum(slope_difference)

        return probab





