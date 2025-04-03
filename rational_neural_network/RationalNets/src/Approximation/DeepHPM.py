import tensorflow as tf
import numpy as np
import random
from scipy.interpolate import griddata
from plotting import newfig, savefig
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set seeds
random.seed(10)
np.random.seed(10)
tf.random.set_random_seed(10)


###############################################################################
############################## Helper Functions ###############################
###############################################################################

def initialize_NN(num_hidden_layers:int,width:int,act:str):
    weights = []
    biases = []
    input_hidden_output=[2]+[width]*num_hidden_layers+[1] #[2,W,W,W,1]
    for i in range(len(input_hidden_output) - 1): #[0,1,2,3]
        W = xavier_init(size=[input_hidden_output[i], input_hidden_output[i+1]])
        b = tf.Variable(tf.zeros([1, input_hidden_output[i + 1]]), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
    #if rational function. initialize variable poly coefficients
    if act=='rat':
        ratweightsP = []
        ratweightsQ = []
        for l in range(0, num_hidden_layers):
            RP = [tf.Variable([coeffs], dtype=tf.float32) for coeffs in [1.1915, 1.5957, 0.5, 0.0218]]
            ratweightsP = ratweightsP + RP
            RQ = [tf.Variable([coeffs], dtype=tf.float32) for coeffs in [2.383, 0.0, 1.0]]
            ratweightsQ = ratweightsQ + RQ
        return weights,biases,ratweightsP,ratweightsQ
    return weights, biases, None,None


def xavier_init(size):
    in_dim = size[0]
    out_dim = size[1]
    xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
    return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)


def initialize_a_paras(num_hidden_layers, initial_a_para):
    a_paras = []
    for l in range(num_hidden_layers):
        a = [tf.Variable([initial_a_para], dtype=tf.float32)]
        a_paras = a_paras + a
    return a_paras

def act_infer_mapper(act:str):
    if act.endswith('relu'):
        return tf.nn.relu
    elif act.endswith('sigmoid'):
        return tf.nn.sigmoid
    elif act.endswith('tanh'):
        return tf.nn.tanh
    elif act.endswith('rat'):
        return rat_infer
    else:
        raise ValueError('Unrecognized activation function')

def layers_and_width(amt:str,shp:str):
    if amt == 'A':
        if shp == 's':return 1, 800
        elif shp == 'm':return 5, 27
        elif shp == 'd':return 10, 18
        else:raise ValueError('Unrecognized shape')
    elif amt == 'B':
        if shp == 's':return 1, 1600
        elif shp == 'm':return 5, 39
        elif shp == 'd':return 10, 26
        else:raise ValueError('Unrecognized shape')
    elif amt == 'C':
        if shp == 's':return 1, 3200
        elif shp == 'm':return 5, 56
        elif shp == 'd':return 10, 37
        else:raise ValueError('Unrecognized shape')
    else:
        raise ValueError('Unrecognized amount of parameters')

def rat_infer(X, weights, biases, a_paras, ratweightsP, ratweightsQ):
    num_layers = len(weights) + 1
    H = X
    degP = int(len(ratweightsP) / (num_layers - 2) - 1)
    degQ = int(len(ratweightsQ) / (num_layers - 2) - 1)
    for l in range(0, num_layers - 2):
        W = weights[l]
        b = biases[l]
        H = tf.add(tf.matmul(H, W), b)
        H = tf.math.divide(tf.math.polyval(ratweightsP[(degP + 1) * l:(degP + 1) * l + (degP + 1)], a_paras*H),
                           tf.math.polyval(ratweightsQ[(degQ + 1) * l:(degQ + 1) * l + (degQ + 1)], a_paras*H))
    W = weights[-1]
    b = biases[-1]
    Y = tf.add(tf.matmul(H, W), b)
    return Y





class DeepHPM:
    def __init__(self, t_train, x_train, u_train, t_val, x_val, u_val , lb_idn, ub_idn,act:str,max_epochs:int,shape:str,amt:str,work_dir:str):
        # Domain Boundary
        self.lb_idn = lb_idn
        self.ub_idn = ub_idn
        self.act=act
        self.act_infer = act_infer_mapper(act)

        #working directory
        self.work_dir = work_dir

        # Init for Identification
        self.idn_init(t_train, x_train, u_train, t_val, x_val, u_val,act,max_epochs,shape,amt)

        # tf session
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        init = tf.global_variables_initializer()
        self.sess.run(init)

    ###########################################################################
    ############################# Identifier ##################################
    ###########################################################################

    def idn_init(self, t_train, x_train, u_train, t_val, x_val, u_val,act:str,max_epochs:int,shape:str,amt:str):
        # Training Data for Identification
        self.t = t_train
        self.x = x_train
        self.u = u_train

        # Validation data
        self.t_val = t_val
        self.x_val = x_val
        self.u_val = u_val

        # Initialize NNs for Identification
        num_hidden_layers , width = layers_and_width(amt,shape)
        self.u_weights, self.u_biases, self.ratweightsP,self.ratweightsQ= initialize_NN(num_hidden_layers,width,act)
        # detect whether it is adaptive, if yes, include a_paras into optimizer. else not
        self.a_paras = initialize_a_paras(num_hidden_layers, 1.0)


        # tf placeholders for Identification
        self.t_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        # tf placeholders for Validation
        self.t_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_val_tf = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])

        # tf graphs for Identification
        self.idn_u_pred = self.idn_net_u(self.t_tf, self.x_tf)
        self.idn_u_pred_val = self.idn_net_u(self.t_val_tf, self.x_val_tf)

        # Training and validation losses for Identification
        self.LossArray = []
        self.ValArray = []
        self.idn_u_loss = tf.reduce_mean(tf.square(self.idn_u_pred - self.u_tf))
        self.idn_u_val = tf.reduce_mean(tf.square(self.idn_u_pred_val - self.u_val_tf))

        # Optimizer for Identification
        if act.startswith('adpt'):
            var_list=self.u_weights + self.u_biases+self.a_paras
        else:
            if act=='rat':
                var_list=self.u_weights + self.u_biases+self.ratweightsP+self.ratweightsQ
            else:
                var_list=self.u_weights+self.u_biases

        self.idn_u_optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.idn_u_loss,
                                                                      var_list=var_list,
                                                                      method='L-BFGS-B',
                                                                      options={'maxiter': max_epochs,
                                                                               'iprint': 10,
                                                                               'maxcor': 50,
                                                                               'maxls': 50,
                                                                               'ftol': 1e-20,
                                                                               'gtol': 1.0 * np.finfo(float).eps})

    def idn_net_u(self, t, x):
        X = tf.concat([t, x], 1)
        H = 2.0 * (X - self.lb_idn) / (self.ub_idn - self.lb_idn) - 1.0
        u = self.neural_net(H, self.u_weights, self.u_biases, self.a_paras, self.ratweightsP, self.ratweightsQ)
        return u

    def callback(self, loss, validation):
        # Save training loss
        self.LossArray = self.LossArray + [loss]

        # Save validation loss
        self.ValArray = self.ValArray + [validation]

    def save_loss(self):
        its = [i for i in range(len(self.LossArray))]

        # Training loss
        L = np.vstack((its, self.LossArray)).transpose()
        np.savetxt(self.work_dir+"/loss_curve.csv", L, delimiter=',')

        # Validation loss
        L = np.vstack((its, self.ValArray)).transpose()
        np.savetxt(self.work_dir+"/val_loss_curve.csv", L, delimiter=',')

    def idn_u_train(self):
        tf_dict = {self.t_tf: self.t, self.x_tf: self.x, self.u_tf: self.u,
                   self.t_val_tf: self.t_val, self.x_val_tf: self.x_val, self.u_val_tf: self.u_val}
        self.idn_u_optimizer.minimize(self.sess,
                                      feed_dict=tf_dict,
                                      fetches=[self.idn_u_loss, self.idn_u_val],
                                      loss_callback=self.callback)

    def idn_predict(self, t_star, x_star):
        tf_dict = {self.t_tf: t_star, self.x_tf: x_star}
        u_star = self.sess.run(self.idn_u_pred, tf_dict)
        return u_star

    def neural_net(self,X, weights, biases, a_paras, ratweightsP, ratweightsQ):

        if self.act == 'rat':
            return self.act_infer(X, weights, biases, a_paras, ratweightsP, ratweightsQ)

        num_layers = len(weights) + 1
        H = X
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            a = a_paras[l]
            H = tf.add(tf.matmul(H, W), b)
            H = self.act_infer(a * H)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def record_mse_and_plot_results(self,t_idn_star,x_idn_star,u_idn_star,X_idn_star,T_idn,X_idn,Exact_idn,keep):
        ###############save mse########################
        u_pred_identifier = self.idn_predict(t_idn_star, x_idn_star)
        error_u_identifier = np.mean((u_idn_star - u_pred_identifier) ** 2)
        print('Mean Squared Error: %e' % (error_u_identifier))
        with open(self.work_dir+"/errors.txt", "w") as f:
            f.write("MSE: %e" % error_u_identifier)

        ################plot dynamics##################
        ###############################################

        U_pred = griddata(X_idn_star, u_pred_identifier.flatten(), (T_idn, X_idn), method='cubic')
        fig, ax = newfig(1.0, 0.6)
        ax.axis('off')

        ######## Exact solution #######################
        gs = gridspec.GridSpec(1, 3)
        gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
        ax = plt.subplot(gs[:, 0])
        h = ax.imshow(Exact_idn, interpolation='nearest', cmap='jet',
                      extent=[self.lb_idn[0], self.ub_idn[0] * keep, self.lb_idn[1], self.ub_idn[1]],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(h, cax=cax)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Exact Dynamics', fontsize=10)

        ######## Approximate ###########
        ax = plt.subplot(gs[:, 1])
        h = ax.imshow(U_pred, interpolation='nearest', cmap='jet',
                      extent=[self.lb_idn[0], self.ub_idn[0] * keep, self.lb_idn[1], self.ub_idn[1]],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(h, cax=cax)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Predicted', fontsize=10)

        ######## Error ###########
        ax = plt.subplot(gs[:, 2])
        h = ax.imshow(abs(Exact_idn - U_pred), interpolation='nearest', cmap='jet',
                      extent=[self.lb_idn[0], self.ub_idn[0] * keep, self.lb_idn[1], self.ub_idn[1]],
                      origin='lower', aspect='auto')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        fig.colorbar(h, cax=cax)
        ax.set_xlabel('$t$')
        ax.set_ylabel('$x$')
        ax.set_title('Error', fontsize=10)

        savefig(self.work_dir+"/dynamics")




    def save_a_paras_after_gd(self):
        optimized_a_paras = [self.sess.run(p) for p in self.a_paras]
        for i in range(len(optimized_a_paras)):
            if i == 0:
                with open(self.work_dir+"/a_paras_after_gd.txt", "w") as f:
                    f.write(f"{optimized_a_paras[i][0]:.6f}\n")
            else:
                with open(self.work_dir+"/a_paras_after_gd.txt", "a") as f:
                    f.write(f"{optimized_a_paras[i][0]:.6f}\n")
