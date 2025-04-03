from sklearn.pipeline import Pipeline
from swimnetworks import Dense, Linear
import numpy as np
import os
from typing import Union, Tuple
from scipy.spatial import KDTree

class SwimModel:
    def __init__(self,w:int=8,l:int=1,feed_back:int=0,feed_back_factor:int=2,act:str="adpt_tanh") -> None:
        self.linear_weights = None
        self.linear_biases = None
        self.width_per_subnetwork = w // feed_back_factor
        self.width_last_subnetwork = w % feed_back_factor
        self.random_seed=1
        self.feed_back = feed_back

        if feed_back:
            model=[]
            layers = [self.width_per_subnetwork for i in range(l)]
            if not self.width_per_subnetwork == 0:
                for i in range(8):
                    steps = []
                    for k_layer in range(len(layers)):
                        steps.append((f"fcn{k_layer + 1}",
                                      Dense(layer_width=self.width_per_subnetwork, layer_idx=k_layer, layer_num=l,
                                            activation=act, parameter_sampler=act, sample_uniformly=False,
                                            random_seed=42)))
                    steps.append(("lin", Linear(regularization_scale=1e-10)))
                    model.append(Pipeline(steps))

            if not self.width_last_subnetwork == 0:
                steps = []
                for k_layer in range(len(layers)):
                    steps.append((f"fcn{k_layer + 1}",
                                  Dense(layer_width=self.width_last_subnetwork, layer_idx=k_layer, layer_num=l,
                                        activation=act, parameter_sampler=act, sample_uniformly=False,
                                        random_seed=42)))
                steps.append(("lin", Linear(regularization_scale=1e-10)))
                model.append(Pipeline(steps))
            self.model = model
        else:
            layers = [w for i in range(l)]
            steps = []
            for k_layer in range(len(layers)):
                steps.append((f"fcn{k_layer + 1}", Dense(layer_width=w, layer_idx=k_layer, layer_num=l, activation=act,
                                                         parameter_sampler=act, sample_uniformly=False,
                                                         random_seed=42)))
            steps.append(("lin", Linear(regularization_scale=1e-10)))
            self.model = [Pipeline(steps)]


    def fit(self,x:np.ndarray,y:np.ndarray):
        for i in range(len(self.model)):
            if i == 0:
                self.model[i].fit(x, y, fcn1__error_map=None,fcn1__candidate_sets=None)
                if i==len(self.model) - 1:
                    self.construct_linear_w_and_b(None,self.model[i].named_steps['lin'].weights,self.model[i].named_steps['lin'].biases)
            else:
                print("fit subnetwork",i)
                #train the i th sub_network
                #random generate samples for the respective model
                candidate_sets = self.random_sample_generator(x,width=self.width_per_subnetwork,repitition_factor=1) #(B,2)
                if len(y.shape) < 2:
                    y = y.reshape(-1, 1)
                candidate_x_centers,ground_truth=self.centers_and_ground_truth_evaluator(x,y,candidate_sets)#(B,2),(B,out)
                hidden_layer = np.hstack([self.model[j].named_steps['fcn1'].transform(candidate_x_centers) for j in range(i)])#(B,M=m1+m2+m3...)
                # prepare inputs as convention fit in linear
                if len(hidden_layer.shape) > 2:
                    hidden_layer = hidden_layer.reshape(hidden_layer.shape[0], -1)

                #append the output of Denser layer with 1 to solve lstsq problem
                 # (M)
                hidden_layer = np.column_stack([hidden_layer, np.ones((hidden_layer.shape[0], 1))]) #(B,M+1)
                w_and_b_linear = np.linalg.lstsq(hidden_layer, ground_truth, rcond=1e-8)[0] #(M+1,out)
                #generate the intermediate predict result to compute error map, to guide the next sampling
                intermediate_result = hidden_layer @ w_and_b_linear #(B,M+1)@(M+1,out)=(B,output)
                error_map = np.abs(intermediate_result - ground_truth) #(B,output)
                error_map = np.sum(error_map, axis=1) #(B,)
                self.model[i].fit(x, y, fcn1__error_map=error_map,fcn1__candidate_sets=candidate_sets)
                #detect if is last subnetwork:
                if i==len(self.model) - 1:
                    self.construct_linear_w_and_b(w_and_b_linear,self.model[i].named_steps['lin'].weights,self.model[i].named_steps['lin'].biases)
        return self

    def transform(self,x:np.ndarray) -> np.ndarray:
        hidden_layer = np.hstack([md.named_steps['fcn1'].transform(x) for md in self.model])
        return hidden_layer @ self.linear_weights + self.linear_biases

    def construct_linear_w_and_b(self,w_and_b_1:Union[np.ndarray,None],w:np.ndarray,b:np.ndarray):
        #w_and_b_1 base linear w_and_b sofar (M+1,out_dim)
        #w the last subnetwork's linear layer's weights (m,out_dim)
        #b the last subnetwork's linear layer's biases (1,out_dim)
        if w_and_b_1 is None:
            self.linear_weights=w
            self.linear_biases=b
        else:
            weights_1=w_and_b_1[:-1, :] #(M,out_dim)
            biases_1=w_and_b_1[-1:, :]  #(1,out_dm)
            biases=biases_1+b #(1,out_dm)
            self.linear_weights = np.vstack((weights_1, w)) #(M+m+1,out_dim)
            self.linear_biases = biases

    def random_sample_generator(self,x:np.ndarray,width:int,repitition_factor:int) -> np.ndarray:
        rng = np.random.default_rng(self.random_seed)
        n_repetitions = max(1, int(np.ceil(width / x.shape[0]))) * repitition_factor
        candidate_sets = []
        for i in range(x.shape[0] * n_repetitions):
            candidate_sets.append(rng.choice(x.shape[0], size=2, replace=False))
        candidate_sets = np.array(candidate_sets)

        return candidate_sets

    def centers_and_ground_truth_evaluator(self,x:np.ndarray,y:np.ndarray,candidate_sets:np.ndarray) -> Tuple[np.ndarray,np.ndarray]:
        candidate_x_pairs = x[candidate_sets]  # (B,2,2)
        candidate_x_centers = (candidate_x_pairs[:, 0, ...] + candidate_x_pairs[:, 1, ...]) / 2  # (B,2)
        tree=KDTree(x)
        _, indices=tree.query(candidate_x_centers) #(B,)
        ground_truth=y[indices] #(B,output)
        return candidate_x_centers, ground_truth #(B,2),(B,output)



