import numpy as np
from scipy.spatial import KDTree

def probability_indicator(candidate_sets: np.ndarray, x, y):
    print(candidate_sets[:,:,0])
    print(y[candidate_sets, ...])
    print(y[candidate_sets[:,:,0], ...])
    return np.sum(y[candidate_sets, ...], axis=2)

def probability_evaluator(candidate_sets: np.ndarray, x, y):
    probab_indicator = probability_indicator(candidate_sets, x, y)
    probab_indicator = (probab_indicator - np.min(probab_indicator)).squeeze()
    return probab_indicator / np.sum(probab_indicator)


def noisy_level(x:np.ndarray, query_indices: np.ndarray,y:np.ndarray):
    selected_x = x[query_indices, ...]
    tree=KDTree(x)
    _, indices = tree.query(selected_x,k=4+1)
    center_indices=indices[..., 0,np.newaxis]
    indices = indices.swapaxes(1, 2)
    print(indices)
    center_indices = center_indices.swapaxes(1, 2)
    print(center_indices)
    probab_indicator_neighbor = probability_indicator(indices, x, y)
    probab_indicator_center = probability_indicator(center_indices, x, y)
    print(probab_indicator_neighbor)
    print('-------------------')
    print(probab_indicator_center)
    print('-------------------')
    noisy_level = np.sum(np.abs(probab_indicator_center-probab_indicator_neighbor),axis=1).squeeze()


    return noisy_level

x=np.random.random((1000,2))
y=np.random.random((1000,))
y_prime=y.reshape((1000,1))
q=np.array([[300,200,100],[18,190,34]])
print(noisy_level(x,q,y))
print(noisy_level(x,q,y_prime))

