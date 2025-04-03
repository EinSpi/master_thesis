
import numpy as np
from scipy.spatial import KDTree


class ProbabilityIndicator:
    @staticmethod
    def large_value_probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        # compute the coordinate of the middle points
        # indices (M,K,2) or (M,2)
        # K is one when computing centred sets (probability)
        # K is more than one when computing pertubed sets (compare smoothness)
        # here we assume y be (N,1)
        avg_y = (y[indices[..., 0], ...] + y[indices[..., 1], ...]) / 2  #(M,K,1)
        distance=np.linalg.norm(x[indices[..., 0], ...] - x[indices[..., 1], ...],axis=-1,keepdims=True)#(M,K,1)
        return avg_y/(distance+1e-10)#(M,K,1)

    @staticmethod
    def large_absolute_probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        return np.abs(ProbabilityIndicator.large_value_probability_indicator(indices, x, y))

    @staticmethod
    def large_1st_grd_probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        # compute the coordinate of the middle points
        # indices (M,K,2) or (M,1,2)
        # K is one when computing centred sets (probability)
        # K is more than one when computing pertubed sets (compare smoothness)
        # here we assume y be (N,1)
        # compute the distance of x1 and x2
        direction = x[indices[..., 0], ...] - x[indices[..., 1], ...]  # (M,K,2) or (M,1,2)
        dist = np.linalg.norm(direction, axis=-1, keepdims=True)  # (M,K,1) or (M,1,1)
        dist = np.clip(dist, a_min=1e-10, a_max=None)  # (M,K,1) or  (M,1,1)
        dy = y[indices[..., 0], ...] - y[indices[..., 1], ...]  # (M,K,1) or (M,1,1)
        slope = dy / dist # (M,K,1) or (M,1,1)

        return np.abs(slope)  # (M,K,1) or (M,1,1)



    @staticmethod
    def large_2nd_grd_probability_indicator(indices: np.ndarray, x: np.ndarray, y: np.ndarray):
        x1=x[indices[..., 0], ...]
        x2=x[indices[..., 1], ...]
        x_middle_points = (x1+x2) / 2# (M,K,2) or (M,2)
        tree = KDTree(x)
        # use KD Tree to find the nearest neighbor of the middle points, retrieve the corresponding y value of that middle point
        _, y_indices = tree.query(x_middle_points)  # y_indices(M,K) or (M,) ,nearest indices, not (M,K,1) or (M,1)
        y_middle_points = y[y_indices, ...]  # (M,K,1) or (M,1)
        # retrieve the corresponding y values at x1 and x2
        y1 = y[indices[..., 0], ...]  # (M,K,1) or (M,1)
        y2 = y[indices[..., 1], ...]  # (M,K,1) or (M,1)
        dy1, dy2 = y1 - y_middle_points, y_middle_points - y2  # (M,K,1) or (M,1)
        # compute the distance of x1 and x2
        direction = x1-x2  # (M,K,2) or (M,2)
        dist = np.linalg.norm(direction, axis=-1, keepdims=True)  # (M,K,1) or (M,1)
        dist = np.clip(dist, a_min=1e-10, a_max=None)  # (M,K,1) or  (M,1)
        # compute the slopes and slope difference_rate, i.e. 2nd gradient
        slope1 = (dy1 / (dist / 2))  # (M,K,1) or (M,1)
        slope2 = (dy2 / (dist / 2))  # (M,K,1) or (M,1)
        slope_difference = (np.abs(slope1 - slope2) / dist)  # (M,K,1) or (M,1)
        return slope_difference# (M,K,1) or (M,1)

