import numpy as np

import matplotlib.pyplot as plt

def tanh(a,x):
    #x(N,)
    return (1/a)*np.tanh(a*x)#(N,)
def gaussian(a,x):
    return np.exp(-(x**2/a))
def cosine(a,x,h):
    return np.exp(x**2/h)*(np.cos(a*x)+1)

def probability_evaluator(a, distance,h,act,eta):
    indicator= probability_indicator(a, distance,h,act,eta)#(N,)
    return indicator/np.sum(indicator)#->(N,)

def probability_indicator(a,distance,h,act,eta):
    #distance(N,)
    if act == 'tanh':
        return (tanh(a,distance/2)-tanh(a,-distance/2))/distance #(N,)-(N,)/(N,) ->(N,)
    elif act == 'gaussian':
        return (gaussian(a,distance/2)+gaussian(a,-distance/2))/(2*distance)
    else:
        return (act(a,distance/2,h)+act(a,-distance/2,h))/(2*(distance**eta))


def sum_of_m_distances(m,a,h,act,d,eta):
    distance = np.linspace(0.001,d,10000*d/10)
    probability = probability_evaluator(a, distance,h,act,eta)
    sample_distances=np.random.choice(distance,size=m,replace=True,p=probability) #(m,)
    return np.sum(1/sample_distances) #(1,)


def mean_of_sum_of_m_distances(m,a,h,act,d,eta):
    n=200
    avg=0
    for i in range(n):
        avg+=sum_of_m_distances(m,a,h,act,d,eta)
    avg/=n
    return avg
length=10
M_list=[i*10+10 for i in range(length)]
try_a_list=[0.01+0.05*i for i in range(160)]
A=np.array(try_a_list)
h_list=[10]
act=cosine

D_list=[10]
eta_list=[2]
for eta in eta_list:
    for d in D_list:
        for h in h_list:
            result_list=[np.array([mean_of_sum_of_m_distances(m,a,h,act,d,eta) for m in M_list]) for a in try_a_list]
            slope_array=np.array([np.mean(res) for res in result_list])
            plt.plot(A,slope_array,label=r'$\eta=$'+str(eta))

"""
x=np.linspace(-5,5,10000)
for a in try_a_list:
    plt.plot(x,cosine(a,x),label=r'$a_i*=$'+str(a))
"""




plt.yscale('linear')

plt.xlabel(r'$a_i*$')
plt.ylabel(r'$A_i$')
#plt.xlabel('x')
#plt.ylabel(r'$f_i(x)$')
plt.legend()
plt.legend()
plt.grid(True)
plt.savefig('A_i a_istar cosines_7.png')
plt.show()




