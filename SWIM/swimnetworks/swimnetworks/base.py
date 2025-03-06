from __future__ import annotations, division

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Tuple, Union
import numpy as np
from sklearn.base import BaseEstimator


@dataclass
class Base(BaseEstimator, ABC):
    is_classifier: bool = False
    layer_width: int = None
    layer_idx: int = None
    layer_num: int = None

    activation: Union[Callable[[np.ndarray], np.ndarray], str] = "none"
    weights: np.ndarray = None
    biases: np.ndarray = None
    n_parameters: int = 0

    input_shape: Tuple[int, ...] = None
    output_shape: Tuple[int, ...] = None

    @staticmethod
    def identity_activation(x):
        return x

    @staticmethod
    def relu_activation(x):
        res=np.maximum(x, 0)
        #print(res.shape)
        return res


    @staticmethod
    def tanh_activation(x):
        return np.tanh(x)

    @staticmethod
    def reluLike_rat_activation(x):
        return np.divide(np.polyval([1.1915, 1.5957, 0.5, 0.0218],x),np.polyval([2.383, 0.0, 1.0],x))


    def gaussian_act(self,x: np.ndarray):

        x_exp = x[:, np.newaxis, :]
        w_exp = self.weights[np.newaxis, :, :]
        b_exp = self.biases[np.newaxis, :]
        sq_distance=np.sum((x_exp-w_exp)**2, axis=-1)
        coeff=1/b_exp
        exp_term=np.exp(-sq_distance/(2*b_exp))
        result=coeff*exp_term
        print(result.shape)
        return result

    def peaky_rat(self,x: np.ndarray):

        """
        batch computing peaky-rational function parametrized by peaks
        self.weights  (L,M) representing L peaky-rational functions, each parametrized by M parameters.
        The M parameters represents the coordinate of M/2 peak points
        For example, act_func1 has it's peaks at (3,2) (2,8) (1,3)
        act_func2 has it's peaks at (3,4) (0,8) (1,0)
        then the self.weights should be
        ((3,2,2,8,1,3)
         (3,4,0,8,1,0))
        """

        x_exp = x[:, np.newaxis, :]
        w_exp = self.weights[np.newaxis, :, :]

        x1, x2 = x_exp[..., 0], x_exp[..., 1]

        peak_idx=0
        multiply_term = 1.0
        while peak_idx<=self.weights.shape[1]-2:
            multiply_term = multiply_term * ((x1-w_exp[...,peak_idx])**2+(x2-w_exp[...,peak_idx+1])**2)
            peak_idx+=2

        return (10-0.01*multiply_term)/(10+0.01*multiply_term) + 1

    @staticmethod
    def wavy_rat(x: np.ndarray):
        multiply_term = 0.01*((x+1)**2)*(x**2)*((x-1)**2)
        return np.divide(10-multiply_term,10+multiply_term)+1







    def rat_activation_generator(self):
        with open("GD_Results/Rational/layers%d/width%d/rat_coeffsP.txt" % (self.layer_num, self.layer_width), "r") as f1:
            coeffsP=[float(line.strip()) for line in f1]
        with open("GD_Results/Rational/layers%d/width%d/rat_coeffsQ.txt" % (self.layer_num, self.layer_width), "r") as f2:
            coeffsQ=[float(line.strip()) for line in f2]

        return lambda x : np.divide(np.polyval(coeffsP[self.layer_idx:self.layer_idx+4].copy(),x),np.polyval(coeffsQ[self.layer_idx:self.layer_idx+3].copy(),x))

    
    

    def __post_init__(self):
        self._classes = None

        if not isinstance(self.activation, Callable):
            if self.activation == "none" or self.activation is None:
                self.activation = Base.identity_activation
            elif self.activation == "relu_1st_grd" or self.activation == "relu_2nd_grd":
                self.activation = Base.relu_activation
            elif self.activation == "tanh":
                self.activation = Base.tanh_activation
            elif self.activation == "rat":
                self.activation = self.rat_activation_generator()
            elif self.activation == "relu_like_rat":
                self.activation = Base.reluLike_rat_activation
            elif self.activation == "relu_like_rat_news1s2":
                self.activation = Base.reluLike_rat_activation
            elif self.activation == "gaussian":
                self.activation = self.gaussian_act
            elif self.activation == "peaky_rat":
                self.activation = self.peaky_rat
            elif self.activation == "wavy_rat":
                self.activation = Base.wavy_rat
            else:
                raise ValueError(f"Unknown activation {self.activation}.")

    @abstractmethod
    def fit(self, x, y=None):
        pass

    def transform(self, x, y=None):
        if self.layer_width is None:
            raise ValueError("The fit method did not set the number of outputs, i.e. layer_width.")
        
        x = self.prepare_x(x)
        if self.activation == self.gaussian_act or self.activation == self.peaky_rat:
            result = self.activation(x)
        else:
            print(self.weights.shape)
            result = self.activation(x @ self.weights + self.biases)
        return result

    def fit_transform(self, x, y=None):
        self.fit(x, y)
        return self.transform(x, y)

    def predict(self, x):
        return self.transform(x)
    
    def prepare_x(self, x):
        if len(x.shape) > 2:
            x = x.reshape(x.shape[0], -1)
        return x
    
    def prepare_y(self, y):
        """Prepares labels for the sampling.

        For the classification problem, applies one-hot encoding for the labels.
        For the regression problem, adds a dimension to the labels if neccesary.
        """
        if len(y.shape) < 2:
            y = y.reshape(-1, 1)

        if not self.is_classifier:
            return y, y 
        
        self._classes = np.unique(y)  
        n_classes = len(self._classes)
        y_encoded_index = np.argmax(y == self._classes, axis=1)
        y_encoded_onehot = np.eye(n_classes)[y_encoded_index]
        return y_encoded_onehot, y_encoded_index.reshape(-1, 1)  


    def prepare_y_inverse(self, y):
        """Inverse to prepare_y(self, y).

        For the classification problem, restores labels from the one-hot predictions.
        For the regression problem, has no effect on the labels.
        """
        if not self.is_classifier:
            return y 
        
        probability_max = np.argmax(y, axis=1)
        predictions = self._classes[probability_max].reshape(-1, 1)
        return predictions
    
    def clean_inputs(self, x, y):
        x = self.prepare_x(x)
        y, _ = self.prepare_y(y)
        return x, y
