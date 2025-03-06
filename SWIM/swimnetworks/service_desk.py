import subprocess

import os
cwd = os.path.dirname(os.path.realpath(__file__))

#configure the experiment setting. Fill the following menu.
repeat_times = 5 #how many times you wanna repeat
layers=[1] #depth of SWIM network
width_per_layer=[200,400,800,1600] #layer width of SWIM network
acts=['relu_2nd_grd','relu_1st_grd'] # what activation function you wanna use?
objectives=['gaussian','laptop'] # what dynamics you wanna approximate?

for obj in objectives:
    for act in acts:
        for layer in layers:
            for width in width_per_layer:
                for j in range(repeat_times):
                    subprocess.call("python3 SWIM.py --layer {} --width {} --repeat_times {} --act {} --objective {}".format(layer, width, j, act, obj), shell=True, cwd=cwd)

                subprocess.call("python3 compute_average_loss.py --layer {} --width {} --act {} --objective {}".format(layer, width,act, obj), shell=True, cwd=cwd)

    subprocess.call("python3 error_curve_plotter.py --objective {}".format(obj), shell=True, cwd=cwd)