import numpy as np
from scipy.spatial import KDTree




class AdptAct:
    def __init__(self, sample_uniformly: bool, min_a_para: float, max_a_para: float, num_neighbors: int):
        self.sample_uniformly = sample_uniformly
        self.a_paras = min_a_para
        self.min_a_para = min_a_para
        self.max_a_para = max_a_para
        self.num_neighbors = num_neighbors

    def parameter_calculator(self, x, y, layer_width: int, repetition_scaler: int, candidate_sets=None, rng=None,
                             error_map=None):
        # select x point sets, idx
        selected_point_sets = self.sample_points_with_probability(x, y, layer_width=layer_width,
                                                                  repetition_scaler=repetition_scaler, rng=rng,
                                                                  candidate_sets=candidate_sets,
                                                                  error_map=error_map)  # (M,2)
        """
               s2=np.log(3)/2
               s1=2*s2
               diff=x_array[:, 1, :] - x_array[:, 0, :]#(M,2)
               squared_length=np.sum(diff**2, axis=1, keepdims=True) #(M,1)
               weights=s2*(x_array[:, 1, :] - x_array[:, 0, :])/squared_length #(M,2)/(M,1)= (M,2)
               biases = weights * x_array[:, 1, :] #(M,2)
               biases =np.sum(biases, axis=1, keepdims=True) #(M,1)
               idx_from, idx_to = np.array([0]), np.array([0])
               """
        return AdptAct.adpt_standard_w_b_calculator(x, selected_point_sets)



    def sample_points_with_probability(self, x, y, layer_width: int, repetition_scaler: int,rng=None,candidate_sets=None,error_map=None):

        if candidate_sets is None:
            # define repitition numbers just as original
            n_repetitions = max(1, int(np.ceil(layer_width / x.shape[0]))) * repetition_scaler
            candidate_sets = []
            for i in range(x.shape[0] * n_repetitions):
                candidate_sets.append(rng.choice(x.shape[0], size=2, replace=False))
            candidate_sets= np.array(candidate_sets)
            probability=self.probability_evaluator(candidate_sets,x,y,error_map=error_map)
        else:
            probability = self.probability_evaluator(candidate_sets, x, y,error_map=error_map)


        # select point sets according to the associated probabilities

        selected_set_indices = rng.choice(len(candidate_sets), size=layer_width, replace=True, p=probability)
        query_indices = candidate_sets[selected_set_indices]

        # use the result to compute noisy level, then use noisy level to compute a_paras

        noisy_level = self.compute_noisy_level(x, y, query_indices)#(M,)
        #noisy_level=np.exp(noisy_level-np.mean(noisy_level))

        #---------------sampling based noise computation
        #noisy_level=self.sample_noisy_level(x,y,query_indices)

        self.compute_a_para(noisy_level)


        """
        selected_x_points=x[query_indices,...] #(M,2,2)
        dx=selected_x_points[:,0,...]-selected_x_points[:,1,...]#(M,2)
        distances=np.linalg.norm(dx, axis=-1)#(M,)
        #distances=distances/np.max(distances)

        #self.a_paras=np.log((1/np.sqrt(distances))+(np.sqrt((1/distances)-1+1e-6)))
        self.a_paras=(noisy_level)*2/(distances+1e-10)
        """
        # return the results
        return query_indices

    def probability_evaluator(self, candidate_sets: np.ndarray, x, y, error_map=None):
        """
        given 2 x points x1 and x2, compute the middle point x3=(x1+x2)/2
        calculate the slope difference between x1x3 and x3x2
        """
        # compute probability candidate_sets (B,2)
        probability_indicator = self.__class__.probability_indicator(candidate_sets, x, y).squeeze()  # (B,)
        probability_indicator -= np.min(probability_indicator)
        if error_map is not None:
            # error_map (B,),use this to modify the resulting probability
            punishment = np.exp(error_map - error_map.mean())
            probability_indicator *= punishment

        probab = probability_indicator / np.sum(probability_indicator)

        return probab
    def compute_noisy_level(self,x,y,candidate_sets:np.ndarray):
        #candidate_sets (M,2)
        selected_x=x[candidate_sets,...] #x[candidate_sets,...] (M,2,2)

        tree = KDTree(x)
        _, indices = tree.query(selected_x, k=self.num_neighbors+1) #indices (M,2,11) assume num_neighbors=10 here
        center_indices = indices[..., 0, np.newaxis]  # center_indices (M,2,1)
        indices = AdptAct.explore_all_combinations(indices) #(M,121,2)

        center_indices=center_indices.swapaxes(1,2) #center_indices (M,1,2) 1 here is the K

        #direction_filter=AdptAct.direction_filter(x,center_indices,indices) #(M,121,1)
        #middle_point_filter=AdptAct.middle_point_filter(x,center_indices,indices)#(M,121,1)
        #print('dir',direction_filter)
        #print('mid',middle_point_filter)

        probab_indicator_neighbor = self.__class__.probability_indicator(indices, x, y) #(M,121,1)
        probab_indicator_center =  self.__class__.probability_indicator(center_indices, x, y) #(M,1,1)

        #print(probab_indicator_center-probab_indicator_neighbor)


        differences_to_neighbor=np.abs(probab_indicator_center - probab_indicator_neighbor) #(M,121,1)
        #filtered_differences=(differences_to_neighbor*direction_filter*middle_point_filter).squeeze(-1)#(M,121,1)*(M,121,1)*(M,121,1).squeeze=(M,121,1).squeeze=(M,121)
        filtered_differences=differences_to_neighbor.squeeze(-1)#(M,121,1).squeeze (M,121)
        #noisy_level=np.sum(filtered_differences,axis=1)/np.count_nonzero(filtered_differences,axis=1)#(M,)/(M,)=(M,)
        noisy_level=np.mean(filtered_differences,axis=1)#(M,)

        return noisy_level

    def sample_noisy_level(self,x,y,candidate_sets):
        selected_x_pairs=x[candidate_sets,...] #x[candidate_sets,...] (M,2,2)
        pair_distance=np.linalg.norm( selected_x_pairs[:,0,...]-selected_x_pairs[:,1,...], axis=1) #(M,)
        selected_x_middle_points= (selected_x_pairs[:,0,...]+selected_x_pairs[:,1,...])/2#(M,2)
        num_should_sample=25
        sampled_distance=np.random.uniform(0,1,size=(len(pair_distance),num_should_sample))*pair_distance[:,np.newaxis] #(M,25)*(M,1)=(M,25)
        direction=(selected_x_pairs[:,0,...]-selected_x_pairs[:,1,...])/pair_distance[:,np.newaxis] #(M,2)
        delta=sampled_distance[:,:,np.newaxis]*direction[:,np.newaxis,:]#(M,25,1)*(M,1,2)=(M,25,2)
        sampled_x0=selected_x_middle_points[:,np.newaxis,:]+delta #(M,1,2)+(M,25,2)=(M,25,2)
        sampled_x1=selected_x_middle_points[:,np.newaxis,:]-delta #(M,1,2)+(M,25,2)=(M,25,2)

        tree=KDTree(x)
        _,nearest_sampled_x0_indices=tree.query(sampled_x0) #(M,25)
        _,nearest_sampled_x1_indices=tree.query(sampled_x1) #(M,25)
        sampled_indices=np.stack([nearest_sampled_x0_indices,nearest_sampled_x1_indices],axis=-1)#(M,25,2)
        probab_indicator=self.__class__.probability_indicator(sampled_indices, x, y).squeeze(-1) #(M,25)
        var=np.var(probab_indicator,axis=-1)#(M,)
        var=var/pair_distance#(M,)/(M,)
        return var











    def infer(self, x: np.ndarray):
        # x (B,M), a_paras (M,) return (B,M)
        theta = self.a_paras[np.newaxis, :]  # (1,M)
        return self.__class__.act_infer(theta , x)

    def compute_a_para(self,noisy_level:np.ndarray):
        #noisy_level (M,)
        #assign the neuron with maximum noisy level the maximum a_para, minimum neuron with minimum a_para
        max_noise = np.max(noisy_level)
        min_noise = np.min(noisy_level)
        if max_noise == min_noise:
            print("all noisy levels are equal")
            self.a_paras = np.full_like(noisy_level, self.min_a_para)
        else:
            self.a_paras = self.min_a_para + (noisy_level - min_noise) / (max_noise - min_noise) * (self.max_a_para - self.min_a_para)

        print("average adaptive a_para:", np.mean(self.a_paras))

    @staticmethod
    def probability_indicator(indices:np.ndarray, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Probability indicator not implemented")
    @staticmethod
    def act_infer(theta:np.ndarray, x: np.ndarray):
        raise NotImplementedError("Actual inference not implemented")

    @staticmethod
    def adpt_standard_w_b_calculator(x: np.ndarray, selected_point_sets: np.ndarray):
        x_array = x[selected_point_sets]  # (M,2,2)

        # compute w
        weights = x_array[:, 0, :] - x_array[:, 1, :]  # (M,2) - (M,2)
        norms = np.linalg.norm(weights, axis=1, keepdims=True)  # (M,1)
        weights /= norms  # (M,2)/(M,1)=(M,2)
        #print("average adaptive w norm:",np.mean(np.linalg.norm(weights, axis=1)))

        # compute b
        middle_points = (x_array[:, 0, :] + x_array[:, 1, :]) / 2  # (M,2)
        biases = -np.sum(middle_points * weights, axis=1, keepdims=True)  # (M,1)
        idx_from, idx_to = np.array([0]), np.array([0])

        return weights.T, biases.T, idx_from, idx_to

    @staticmethod
    def direction_filter(x,center_indices,indices):
        #indices(M,121,2)
        #center_indices(M,1,2)
        center_x_pairs=x[center_indices,...]#(M,1,2,2)
        pertubed_x_pairs=x[indices,...]#(M,121,2,2)
        center_directions=center_x_pairs[:,:,0,:]-center_x_pairs[:,:,1,:] #(M,1,2)
        pertubed_directions=pertubed_x_pairs[:,:,0,:]-pertubed_x_pairs[:,:,1,:] #(M,121,2)
        inner_product=np.sum(center_directions*pertubed_directions, axis=-1,keepdims=True) #(M,121,1)
        magnitude=np.linalg.norm(center_directions, axis=-1,keepdims=True)*np.linalg.norm(pertubed_directions,axis=-1,keepdims=True) #(M,1,1)*(M,121,1)=(M,121,1)
        abs_cos=np.abs(inner_product/(magnitude+1e-6)) #(M,121,1)
        #abs_cos[abs_cos<0.9] = 0
        return abs_cos#(M,121,1) min 0 max 1, much is better

    @staticmethod
    def middle_point_filter(x,center_indices,indices):
        # indices(M,121,2)
        # center_indices(M,1,2)
        center_x_pairs = x[center_indices, ...]  # (M,1,2,2)
        middle_point_center_x_pairs=(center_x_pairs[:,:,0,:]+center_x_pairs[:,:,1,:])/2 #(M,1,2)
        pertubed_x_pairs=x[indices,...]#(M,121,2,2)
        middle_point_pertubed_x_pairs=(pertubed_x_pairs[:,:,0,:]+pertubed_x_pairs[:,:,1,:])/2 #(M,121,2)
        middle_point_error=np.linalg.norm(middle_point_pertubed_x_pairs-middle_point_center_x_pairs, axis=-1,keepdims=True)#(M,121,1), min 0 max inf
        return np.exp(-middle_point_error) #(M,121,1) min 0 max 1, much is better



    @staticmethod
    def explore_all_combinations(indices:np.ndarray):
        # indices (M, 2, 11)

        K=indices.shape[2]
        M=indices.shape[0]
        # Vectorized computation of all combinations
        # Step 1: Reshape to (M, 11, 1) and (M, 1, 11)
        A1 = indices[:, 0, :, np.newaxis]  # shape (M, 11, 1)
        A2 = indices[:, 1, np.newaxis, :]  # shape (M, 1, 11)

        # Step 2: Broadcast to shape (M, 11, 11, 2)
        pairs = np.stack([A1.repeat(K, axis=2), A2.repeat(K, axis=1)], axis=-1)  # shape (M, 11, 11, 2)

        # Step 3: Reshape to (M, 121, 2)
        pairs_reshaped = pairs.reshape((M, K * K, 2))

        return pairs_reshaped #(M,121,2)













