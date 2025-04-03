import numpy as np
import scipy.io


def generate_and_save_mat(filename, N_t=100, N_x=100,t_range=(0, 1), x_range=(-1, 1)):
    """
    Generates synthetic PDE-like data and saves it to a MATLAB `.mat` file.

    Parameters:
    - filename (str): Name of the file to save (e.g., "custom_data.mat").
    - N_t (int): Number of time points.
    - N_x (int): Number of space points.
    """

    # Step 1: Define the domain
    t = np.linspace(t_range[0], t_range[1], N_t).reshape(-1, 1)  # Time values (N_t, 1)
    x = np.linspace(x_range[0], x_range[1], N_x).reshape(-1, 1)  # Space values (N_x, 1)

    # Step 2: Create a time-space meshgrid
    T, X = np.meshgrid(t, x)

    # Step 3: Define the function (Example: Decaying sine wave solution)
    #usol = np.maximum(2+X+1.5*T-30,-X-0.75*T+15)  # Shape (N_x, N_t)
    #usol = np.exp(-(((X)**2)/7)-(((T-20)**2)/10))
    #usol = (np.exp(0.1*X+0.2*(T-10)))+5*(np.sin(X**2))/(T**2+1)
    #usol = (1/5)*(X**2)
    """
    scale_factor=0.6
    mean_1_x,mean_1_t=20*scale_factor, 5*scale_factor+20
    mean_2_x, mean_2_t = -20 * scale_factor, 20 * scale_factor + 20
    mean_3_x, mean_3_t = -3 * scale_factor, -20 * scale_factor + 20
    sigma = 1.2
    #usol = 3*np.exp(-(((X-mean_1_x) ** 2) / 0.2) - (((T - mean_1_t) ** 2) / 0.2))+np.exp(-(((X-mean_2_x) ** 2) / 0.2) - (((T - mean_2_t) ** 2) / 0.2))+2*np.exp(-(((X-mean_3_x) ** 2) / 0.2) - (((T - mean_3_t) ** 2) / 0.2))
    usol = 3*np.exp(-(((X-mean_1_x) ** 2) / sigma) - (((T - mean_1_t) ** 2) / sigma))+2*np.exp(-(((X-mean_2_x) ** 2) / (10*sigma)) - (((T - mean_2_t) ** 2) / (10*sigma)))+2*np.exp(-(((X-mean_3_x) ** 2) / sigma) - (((T - mean_3_t) ** 2) / sigma))
    """

    """
    #Locus=np.sin(0.1*X)+np.cos(0.05*T)-0.5*np.sin(0.02*X*T)
    Locus=0.8*T+0.6*X-16
    compression_factor=0.5 #loose 5 times
    sigma=Locus*compression_factor
    usol=np.where(sigma<=0,np.sin(6*sigma),5+0.5*sigma*np.cos(6*sigma))
    """
    Locus = 0.8 * T + 0.6 * X - 16
    compression_factor = 0.5
    sigma = Locus*compression_factor
    usol=np.exp(sigma**2/200)*np.cos(6*sigma)

    # Step 4: Save the data to a `.mat` file
    data_dict = {
        "t": t,  # (N_t, 1)
        "x": x,  # (N_x, 1)
        "usol": usol  # (N_x, N_t)
    }

    scipy.io.savemat(filename, data_dict)

    print(f"Data saved to {filename}")


# Example: Save a custom `.mat` file
generate_and_save_mat("../../rational_neural_network/RationalNets/src/Approximation/Data/butterfly.mat",
                      N_t=300,N_x=300,t_range=(0,40), x_range=(-20, 20))