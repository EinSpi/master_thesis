import numpy as np
import matplotlib.pyplot as plt
a,C,b=1,5,1
x=np.linspace(0,5,1000)
y=np.linspace(0,5,1000)

X,Y=np.meshgrid(x,y)

#numerator=C-(X**2-a**2)**2-Y**4
#denominator=(0.01*X**2+0.01*Y**2+b)**3
numerator=10-((X-1)**2+(Y-1)**2)*((X)**2+(Y-3)**2)*((X-3)**2+(Y-1)**2)
denominator=10+((X-1)**2+(Y-1)**2)*((X)**2+(Y-3)**2)*((X-3)**2+(Y-1)**2)

Z=numerator/denominator

fig = plt.figure()
ax=fig.add_subplot(111, projection='3d')
ax.plot_surface(X,Y,Z,cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('f(x,y)')
plt.title('Rat with 2 Maxima')
plt.show()