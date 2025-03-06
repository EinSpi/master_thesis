"""
Created on Sat Aug  8 11:13:24 2020

@author: Nicolas Boulle
"""

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 14})


## Validation

fig, ax = plt.subplots()
val = np.genfromtxt('Results/ReLU/val_relu.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'b', label="ReLU")

val = np.genfromtxt('Results/Sine/val_sine.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'g', label="Sinusoid")

val = np.genfromtxt('Results/Rational_3_2/val_rat.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'r', label="Rational")

val = np.genfromtxt('Results/Polynomial/val_poly.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'm', label="Polynomial")

val = np.genfromtxt('Results/tanh/val_tanh.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'c', label="tanh")

val = np.genfromtxt('Results/adaptive_tanh/val_adaptive_tanh.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'k', label="adpt_tanh")

plt.xlim(1,10**4)
plt.ylim(10**-8,10)
plt.yticks([10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,1,10])

plt.legend(loc="lower left")
ax.set_aspect(0.35)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

plt.savefig('Results/validation_loss.pdf')  

plt.close('all')
plt.cla()

## Rational comparison

fig, ax = plt.subplots()
val = np.genfromtxt('Results/Rational_2_2/val_rat.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'b', label="(2, 2)")

val = np.genfromtxt('Results/Rational_3_2/val_rat.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'r', label="(3, 2)")

val = np.genfromtxt('Results/Rational_4_3/val_rat.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'g', label="(4, 3)")

val = np.genfromtxt('Results/Rational_5_4/val_rat.csv', delimiter=',')
plt.loglog(val[:,0],val[:,1], 'm', label="(5, 4)")

plt.xlim(1,10**4)
plt.ylim(10**-7,10)
plt.yticks([10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,1,10])

plt.legend(loc="upper right")
ax.set_aspect(0.35)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

plt.savefig('Results/validation_rational.pdf')  
plt.close('all')

## Adaptive Tanh comparison

fig, ax = plt.subplots()

for init_a,color  in zip([1.0,2.0,3.0,4.0],['b','r','g','m']):
    val = np.genfromtxt("Results/adaptive_tanh_a_init%f/val_adaptive_tanh_a_init%f.csv" % (init_a,init_a), delimiter=',')
    plt.loglog(val[:, 0], val[:, 1], color, label='init'+str(init_a))

plt.xlim(1,10**4)
plt.ylim(10**-7,10)
plt.yticks([10**-8,10**-7,10**-6,10**-5,10**-4,10**-3,10**-2,10**-1,1,10])

plt.legend(loc="upper right")
ax.set_aspect(0.35)
ax.xaxis.set_ticks_position('both')
ax.yaxis.set_ticks_position('both')

plt.savefig('Results/validation_adaptive_tanh.pdf')
plt.close('all')

