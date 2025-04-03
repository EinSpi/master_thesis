import  numpy as np

class Act:
    def __init__(self, sample_uniformly: bool):
        self.sample_uniformly = sample_uniformly


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
        return self.w_b_calculator(x, selected_point_sets)


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

    def infer(self, x: np.ndarray):
        # x (B,M), a_paras (M,) return (B,M)
        return self.__class__.act_infer(x)

    def w_b_calculator(self, x: np.ndarray, selected_point_sets: np.ndarray):
        raise NotImplementedError("W-B calculator not implemented")

    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        raise NotImplementedError("Probability indicator not implemented")

    @staticmethod
    def act_infer(x: np.ndarray):
        raise NotImplementedError("Actual inference not implemented")

    @staticmethod
    def w_b_calculator_s1_s2(x: np.ndarray, selected_point_sets: np.ndarray,s1,s2):
        x_array = x[selected_point_sets]
        diff = x_array[:, 1, :] - x_array[:, 0, :]  # (M,2)
        squared_length = np.sum(diff ** 2, axis=1, keepdims=True)  # (M,1)
        weights = s1 * (x_array[:, 1, :] - x_array[:, 0, :]) / squared_length  # (M,2)/(M,1)= (M,2)
        #check a para
        print(np.mean(np.linalg.norm(weights,axis=1)))

        biases = weights * x_array[:, 1, :]  # (M,2)
        biases = np.sum(biases, axis=1, keepdims=True)  # (M,1)
        biases = -biases - s2
        idx_from, idx_to = np.array([0]), np.array([0])

        return weights.T, biases.T, idx_from, idx_to
