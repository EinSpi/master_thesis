from sklearn.pipeline import Pipeline
from swimnetworks import Dense, Linear
import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import random
import argparse
from scipy.interpolate import griddata
from plotting import newfig, savefig
from swim_model import SwimModel


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--layer", type=int, default = 4)
	parser.add_argument("--width", type=int, default = 50)
	parser.add_argument("--repeat_times", type=int, default=0)
	parser.add_argument("--act", type=str, default="relu")
	parser.add_argument("--objective", type=str, default="KdV_sine")
	parser.add_argument("--experiment_name", type=str, default="experiment")
	parser.add_argument("--feed_back", type=int, default=0)
	args, _ = parser.parse_known_args()

	#prepare swim model with rat function
	l=args.layer
	w=args.width
	times=args.repeat_times
	act=args.act
	objective=args.objective
	experiment_name=args.experiment_name
	feed_back=args.feed_back
	sample_uniformly=False
	swim_model=SwimModel(w=w,l=l,feed_back=feed_back,feed_back_factor=2,act=act)

	"""
	model=[]
	if not feed_back:
		layers=[w for i in range(l)]
		steps=[]
		for k_layer in range(len(layers)):
			steps.append((f"fcn{k_layer+1}", Dense(layer_width=w, layer_idx= k_layer, layer_num=l ,activation=act,parameter_sampler=act,sample_uniformly=sample_uniformly,random_seed=42)))
		steps.append(("lin", Linear(regularization_scale=1e-10)))
		model = Pipeline(steps)
	else:
		width_per_subnetwork=w//8
		width_last_subnetwork=w%8
		layers = [width_per_subnetwork for i in range(l)]
		layers_last_subnetwork=[width_last_subnetwork for i in range(l)]
		if not width_per_subnetwork==0:
			for i in range(8):
				steps=[]
				for k_layer in range(len(layers)):
					steps.append((f"fcn{k_layer + 1}",Dense(layer_width=width_per_subnetwork, layer_idx=k_layer, layer_num=l, activation=act, parameter_sampler=act,sample_uniformly=sample_uniformly, random_seed=42)))
				steps.append(("lin", Linear(regularization_scale=1e-10)))
				model.append(Pipeline(steps))

		if not width_last_subnetwork==0:
			steps = []
			for k_layer in range(len(layers)):
				steps.append((f"fcn{k_layer + 1}", Dense(layer_width=width_last_subnetwork, layer_idx=k_layer, layer_num=l, activation=act,parameter_sampler=act, sample_uniformly=sample_uniformly,random_seed=42)))
			steps.append(("lin", Linear(regularization_scale=1e-10)))
			model.append(Pipeline(steps))
	"""







	def l2_error_relative(f_approx, f_true):
		return np.linalg.norm(f_approx-f_true, ord=2) / np.linalg.norm(f_true, ord=2)

	def mse(f_approx,f_true):
		return np.mean((f_approx-f_true)**2)



	# Doman bounds
	lb_idn = np.array([0.0, -20.0])
	ub_idn = np.array([40.0, 20.0])
	#prepare data from mat file
	data_idn=scipy.io.loadmat("../../rational_neural_network/RationalNets/src/Approximation/Data/"+objective+".mat")

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
	if not os.path.exists("Results/"+experiment_name+"/obj_"+objective+"/act_"+act+"/fdbk_%d/layers%d/width%d/" % (feed_back,l,w)):
		os.makedirs("Results/"+experiment_name+"/obj_"+objective+"/act_"+act+"/fdbk_%d/layers%d/width%d/" % (feed_back,l,w))

	# Save exact as CSV file
	np.savetxt("Results/"+experiment_name+"/"+"obj_"+objective+"/domain.csv" , X_idn_star, delimiter=',')
	np.savetxt("Results/"+experiment_name+"/"+"obj_"+objective+"/exact_sol.csv" , Exact_idn, delimiter=',')

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
	U_val = u_idn_star[idx_val,:]

	noise = 0.00
	U_train = u_train + noise*np.std(u_train)*np.random.randn(u_train.shape[0], u_train.shape[1])
	X_train=np.hstack([t_train,x_train])



	#train
	"""
	if not feed_back:
		model.fit(X_train, U_train)
	else:
		for i in range(len(model)):
			if i==0:
				model[i].fit(X_train, U_train,error_map=None)
			else:
				hidden_layer=np.hstack([model[j].named_steps['fcn1'].transform(X_train) for j in range(i)])
				#prepare inputs as convention fit in linear
				if len(hidden_layer.shape) > 2:
					hidden_layer = hidden_layer.reshape(hidden_layer.shape[0], -1)
				if len(U_train.shape) < 2:
					U_train = U_train.reshape(-1, 1)
				hidden_layer = np.column_stack([hidden_layer, np.ones((hidden_layer.shape[0], 1))])
				w_and_b_linear = np.linalg.lstsq(hidden_layer, U_train, rcond=1e-8)[0]
				result=hidden_layer@w_and_b_linear
				error_map=np.abs(result-U_train)
				model[i].fit(X_train, U_train,error_map=error_map)


			probability_advisor=model[i].named_steps['fcn1'].transform(X_train)
	"""
	swim_model.fit(X_train, U_train)

	#infer,and evaluate
	"""
	callable_model= lambda x: model.transform(x)
	"""
	callable_model=lambda  x:swim_model.transform(x)
	u_pred_identifier=callable_model(X_idn_star)
	#compute mse and rel l2 loss
	mean_squared_error=mse(u_pred_identifier,u_idn_star)
	rel_l2_error=l2_error_relative(u_pred_identifier,u_idn_star)
	#print('Mean Squared Error: %e' % mean_squared_error)
	#print('Relative l2 Error: %e' % rel_l2_error)
	with open("Results/"+experiment_name+"/"+"obj_"+objective+"/"+"act_"+act+"/fdbk_%d/layers%d/width%d/errors.txt" % (feed_back,l,w), "a") as f:
		f.write("MSE: %e, rel.L2: %e\n" % (mean_squared_error, rel_l2_error))
	#print("wrote SWIM err to file")



	U_pred = griddata(X_idn_star, u_pred_identifier.flatten(), (T_idn, X_idn), method='cubic')

	# Save identifier as CSV file
	#np.savetxt("Results/"+experiment_name+"/"+act+"/layers%d/width%d/idn_rat_repeat%d.csv" % (len(layers),layers[0],times) , U_pred, delimiter=',')

	######################################################################
	############################# Plotting ###############################
	######################################################################

	fig, ax = newfig(1.0, 0.6)
	ax.axis('off')

	######## Exact solution #######################
	########      Predicted p(t,x,y)     ###########
	gs = gridspec.GridSpec(1, 3)
	gs.update(top=0.8, bottom=0.2, left=0.1, right=0.9, wspace=0.5)
	ax = plt.subplot(gs[:, 0])
	h = ax.imshow(Exact_idn, interpolation='nearest', cmap='jet',
				  extent=[lb_idn[0], ub_idn[0]*keep, lb_idn[1], ub_idn[1]],
				  origin='lower', aspect='auto')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)

	fig.colorbar(h, cax=cax)
	ax.set_xlabel('$t$')
	ax.set_ylabel('$x$')
	ax.set_title('Exact Dynamics', fontsize = 10)

	######## Approximate ###########
	ax = plt.subplot(gs[:, 1])
	h = ax.imshow(U_pred, interpolation='nearest', cmap='jet',
				  extent=[lb_idn[0], ub_idn[0]*keep, lb_idn[1], ub_idn[1]],
				  origin='lower', aspect='auto')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)

	fig.colorbar(h, cax=cax)
	ax.set_xlabel('$t$')
	ax.set_ylabel('$x$')
	ax.set_title('Predicted' , fontsize = 10)

	######## Approximation Error ###########
	ax = plt.subplot(gs[:, 2])
	h = ax.imshow(np.abs(Exact_idn-U_pred), interpolation='nearest', cmap='jet',
				  extent=[lb_idn[0], ub_idn[0] * keep, lb_idn[1], ub_idn[1]],
				  origin='lower', aspect='auto')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes("right", size="5%", pad=0.05)

	fig.colorbar(h, cax=cax)
	ax.set_xlabel('$t$')
	ax.set_ylabel('$x$')
	ax.set_title('Error', fontsize=10)

	savefig("Results/"+experiment_name+"/"+"obj_"+objective+"/"+"act_"+act+"/fdbk_%d/layers%d/width%d/swim_repeat%d" % (feed_back,l,w,times))


