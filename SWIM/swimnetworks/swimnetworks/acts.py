from .act import Act
import numpy as np
from .probability_indicator import ProbabilityIndicator

class Relu(Act):
    def __init__(self, sample_uniformly: bool):
        super().__init__(sample_uniformly=sample_uniformly)
        self.s1=2
        self.s2=1

    def w_b_calculator(self, x: np.ndarray, selected_point_sets: np.ndarray):
        return Act.w_b_calculator_s1_s2(x, selected_point_sets,self.s1, self.s2)

    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return ProbabilityIndicator.large_2nd_grd_probability_indicator(indices, x, y)

    @staticmethod
    def act_infer(x: np.ndarray):
        return np.maximum(0,x)





class Tanh(Act):
    def __init__(self, sample_uniformly: bool):
        super().__init__(sample_uniformly=sample_uniformly)
        self.s2 = 2*np.log(3)
        self.s1 = 4*np.log(3)

    def w_b_calculator(self, x: np.ndarray, selected_point_sets: np.ndarray):
        return Act.w_b_calculator_s1_s2(x, selected_point_sets, self.s1, self.s2)

    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return ProbabilityIndicator.large_1st_grd_probability_indicator(indices, x, y)

    @staticmethod
    def act_infer(x: np.ndarray):
        return np.tanh(x)






class Sigmoid(Act):
    def __init__(self, sample_uniformly: bool):
        super().__init__(sample_uniformly=sample_uniformly)
        self.s2 = -0.5
        self.s1 = -1

    def w_b_calculator(self, x: np.ndarray, selected_point_sets: np.ndarray):
        return Act.w_b_calculator_s1_s2(x, selected_point_sets, self.s1, self.s2)

    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return ProbabilityIndicator.large_1st_grd_probability_indicator(indices, x, y)

    @staticmethod
    def act_infer(x: np.ndarray):
        return 1/(1+np.exp(-x))





class Rat(Act):
    def __init__(self, sample_uniformly: bool):
        super().__init__(sample_uniformly=sample_uniformly)
        self.s2 = 3
        self.s1 = 6

    def w_b_calculator(self, x: np.ndarray, selected_point_sets: np.ndarray):
        return Act.w_b_calculator_s1_s2(x, selected_point_sets, self.s1, self.s2)

    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return ProbabilityIndicator.large_value_probability_indicator(indices, x, y)

    @staticmethod
    def act_infer(x: np.ndarray):
        return np.divide(10-x**2,10+x**2)+1
