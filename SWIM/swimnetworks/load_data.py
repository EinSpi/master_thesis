import scipy.io
import numpy as np
import os

data_idn=scipy.io.loadmat("../../rational_neural_network/RationalNets/src/Approximation/Data/KdV_sine.mat")

t_idn = data_idn['t'].flatten()[:,None]
x_idn = data_idn['x'].flatten()[:,None]
Exact_idn = np.real(data_idn['usol'])

T_idn, X_idn = np.meshgrid(t_idn,x_idn)

keep = 1
index = int(keep*t_idn.shape[0])
T_idn = T_idn[:,0:index]
X_idn = X_idn[:,0:index]
Exact_idn = Exact_idn[:,0:index]

t_idn_star = T_idn.flatten()[:,None]
x_idn_star = X_idn.flatten()[:,None]
X_idn_star = np.hstack((t_idn_star, x_idn_star))
u_idn_star = Exact_idn.flatten()[:,None]
# Create results folder
if not os.path.exists("Results/Rational"):
	os.makedirs("Results/Rational" )

# Save exact as CSV file
np.savetxt("Results/Rational/domain.csv" , X_idn_star, delimiter=',')
np.savetxt("Results/Rational/exact_sol.csv" , Exact_idn, delimiter=',')
 
### Training Data ###

# For identification and validation: 10^4 each
N_train = 10**4
N_val = 10**4
idx = np.random.choice(t_idn_star.shape[0], N_train+N_val, replace=False)    
idx_train = idx[0:N_train]
idx_val = idx[N_train:]
t_train = t_idn_star[idx_train,:]
x_train = x_idn_star[idx_train,:]
u_train = u_idn_star[idx_train,:]
t_val = t_idn_star[idx_val,:]
x_val = x_idn_star[idx_val,:]
u_val = u_idn_star[idx_val,:]

noise = 0.00
u_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])

print("t_size")
print(t_train.shape)
print("x_size")
print(x_train.shape)
print("u_size")
print(u_train.shape)
