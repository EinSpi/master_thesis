import numpy as np
import matplotlib.pyplot as plt

    #return np.divide(np.polyval([1.1915, 1.5957, 0.5, 0.0218],x),np.polyval([2.383, 0.0, 1.0],x))
    #return np.divide(np.polyval([0.913795,1.873937,-2.155352,-0.915598], x), np.polyval([2.544869, 1.240194,0.396071], x))
#generate variable and function values

x=np.linspace(-10,10,4000)

for width in [8,12,25,50,100,200,400]:
    P=[]
    Q=[]
    with open("GD_Results/Rational/layers1/width"+str(width)+"/rat_coeffsP.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            P.append(float(line))
    with open("GD_Results/Rational/layers1/width"+str(width)+"/rat_coeffsQ.txt","r") as f:
        lines = f.readlines()
        for line in lines:
            Q.append(float(line))

    y=np.divide(np.polyval(P,x),np.polyval(Q,x))
    dydx=np.gradient(y,x)
    dydxdx=np.gradient(dydx,x)
    """
    #compute gradients
    dy_dx = np.gradient(y, x)
    # Find indices of max and min gradient values
    max_idx = np.argmax(dy_dx)
    min_idx = np.argmin(dy_dx)

    # Get corresponding x values
    x_max_grad = x[max_idx]
    x_min_grad = x[min_idx]

    # Save as simple floats in a text file
    with open("GD_Results/Rational/layers1/width"+str(width)+"/x1x2.txt", "w") as f:
        f.write(f"{x_min_grad}\n{x_max_grad}")
    """


    # Plot the function
    plt.plot(x, dydx, label='the 1st deriv')

    # Labels and title
    plt.xlabel('x')
    plt.ylabel('1st deriv')
    plt.title('Rat 1st deriv after GD, layer1 width'+str(width))
    plt.legend()
    plt.grid()
    plt.savefig('rat_func_plots/rat_after_GD_1st_deriv_layer1_width'+str(width)+'.png')
    plt.close()


"""
y=np.divide(np.polyval([1.1915, 1.5957, 0.5, 0.0218],x),np.polyval([2.383, 0.0, 1.0],x))
dydx=np.gradient(y,x)
dydxdx=np.gradient(dydx,x)

plt.plot(x, dydxdx, label='the 2nd deriv')
plt.xlabel('x')
plt.ylabel('dydxdx')
plt.title('2nd deriv')
plt.legend()
plt.grid()
plt.show()
"""
