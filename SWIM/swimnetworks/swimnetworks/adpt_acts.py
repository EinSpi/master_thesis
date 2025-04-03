from .adpt_act import AdptAct
import numpy as np
from scipy.special import erf
from .probability_indicator import ProbabilityIndicator




class AdptRelu(AdptAct):
    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return ProbabilityIndicator.large_2nd_grd_probability_indicator(indices, x, y)

    @staticmethod
    def act_infer(theta:np.ndarray, x: np.ndarray):
        # x (B,M), a_paras (M,) return (B,M)
        sigma = 1 / theta  # (1,M)
        term1 = 0.5 * x * (1 + erf(x / (sigma * np.sqrt(2))))
        term2 = (sigma / np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / (2 * sigma ** 2))
        return term1 + term2





class AdptTanh(AdptAct):
    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return ProbabilityIndicator.large_1st_grd_probability_indicator(indices, x, y)

    @staticmethod
    def act_infer(theta:np.ndarray, x: np.ndarray):
        return np.tanh(theta * x)





class AdptRat(AdptAct):
    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return ProbabilityIndicator.large_value_probability_indicator(indices, x, y)

    @staticmethod
    def act_infer(theta:np.ndarray, x: np.ndarray):
        #theta (1,M) x(B,M)
        return np.divide(10-theta*(x**2),10+theta*(x**2))+1#(B,M)



class AdptSigmoid(AdptAct):
    @staticmethod
    def probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return ProbabilityIndicator.large_1st_grd_probability_indicator(indices, x, y)
    @staticmethod
    def act_infer(theta:np.ndarray, x: np.ndarray):
        return 1/(1+np.exp(-theta * x))
