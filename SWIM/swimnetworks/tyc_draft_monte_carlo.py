import numpy as np

x_candidates=np.linspace(0,1,1000)
probability=np.divide((5*x_candidates+3),(x_candidates+1/2))
probability=probability/probability.sum()
samples=np.random.choice(x_candidates, size=5000, p=probability,replace=True)

means=[]
for i in range(10000):
    samples=np.random.choice(x_candidates, size=50, p=probability,replace=True)
    means+=[np.mean(samples)]
print(np.mean(np.array(means)))