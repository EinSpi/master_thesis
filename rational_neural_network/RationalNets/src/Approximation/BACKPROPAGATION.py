import argparse
import os

import numpy as np
from DeepHPM import DeepHPM
import scipy.io





def prepare_data(name:str='KdV_sine'):
    data_idn = scipy.io.loadmat('Data/'+name+'.mat')

    t_idn = data_idn['t'].flatten()[:, None]
    x_idn = data_idn['x'].flatten()[:, None]
    Exact_idn = np.real(data_idn['usol'])

    T_idn, X_idn = np.meshgrid(t_idn, x_idn)

    keep = 1
    index = int(keep * t_idn.shape[0])
    T_idn = T_idn[:, 0:index]
    X_idn = X_idn[:, 0:index]
    Exact_idn = Exact_idn[:, 0:index]

    t_idn_star = T_idn.flatten()[:, None]
    x_idn_star = X_idn.flatten()[:, None]
    X_idn_star = np.hstack((t_idn_star, x_idn_star))
    u_idn_star = Exact_idn.flatten()[:, None]

    # Save exact as CSV file
    #np.savetxt("GD_Results/Rational/domain.csv", X_idn_star, delimiter=',')
    #np.savetxt("GD_Results/Rational/exact_sol.csv", Exact_idn, delimiter=',')

    ### Training Data ###

    # For identification and validation: 10^4 each
    N_train = 10 ** 4
    N_val = 10 ** 4
    idx = np.random.choice(t_idn_star.shape[0], N_train + N_val, replace=False)
    idx_train = idx[0:N_train]
    idx_val = idx[N_train:]
    t_train = t_idn_star[idx_train, :]
    x_train = x_idn_star[idx_train, :]
    u_train = u_idn_star[idx_train, :]
    t_val = t_idn_star[idx_val, :]
    x_val = x_idn_star[idx_val, :]
    u_val = u_idn_star[idx_val, :]

    noise = 0.00
    u_train = u_train + noise * np.std(u_train) * np.random.randn(u_train.shape[0], u_train.shape[1])
    return t_train, x_train, u_train, t_val, x_val, u_val, t_idn_star, x_idn_star, u_idn_star,X_idn_star, T_idn,X_idn,Exact_idn,keep

def prepare_working_directory(obj:str, amt:str, shp:str, act:str, exp_name:str):
    work_dir = 'Results/'+exp_name+'/obj_'+obj+'/amt_'+amt+'/shape_'+shp+'/act_'+act
    if not os.path.exists(work_dir):
        os.makedirs(work_dir)
    return work_dir













if __name__ == "__main__":

    # Get the type of the rational
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default='KdV_sine')
    parser.add_argument("--para_amount", type=str, default='A')
    parser.add_argument("--shape", type=str, default='s')
    parser.add_argument("--activation", type=str, default='relu')
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--experiment_name", type=str, default='exp')
    args, _ = parser.parse_known_args()
    obj=args.objective
    amt=args.para_amount
    shp=args.shape
    act=args.activation
    max_epochs=args.max_epochs
    exp_name=args.experiment_name

    #prepare working directory
    work_dir = prepare_working_directory(obj=obj, amt=amt, shp=shp, act=act, exp_name=exp_name)

    #prepare data
    t_train, x_train, u_train, t_val, x_val, u_val, t_idn_star, x_idn_star, u_idn_star, X_idn_star, T_idn, X_idn, Exact_idn,keep=prepare_data(name=obj)

    #domain bounds
    lb_idn = np.array([0.0, -20.0])
    ub_idn = np.array([40.0, 20.0])

    #create model
    nn=DeepHPM(t_train=t_train, x_train=x_train, u_train=u_train, t_val=t_val, x_val=x_val, u_val=u_val,lb_idn=lb_idn,ub_idn=ub_idn,act=act,max_epochs=max_epochs,shape=shp,amt=amt, work_dir=work_dir)

    #train
    nn.idn_u_train()

    #save loss curve as csv file
    nn.save_loss()

    #record mse loss and draw predicted vs ground truth dynamic and error map in the working directory.
    nn.record_mse_and_plot_results(t_idn_star=t_idn_star, x_idn_star=x_idn_star,u_idn_star=u_idn_star,X_idn_star=X_idn_star,T_idn=T_idn, X_idn=X_idn, Exact_idn=Exact_idn,keep=keep)

    # write a_paras after gd
    nn.save_a_paras_after_gd()


