import subprocess
import tqdm

import os
cwd = os.path.dirname(os.path.realpath(__file__))

#configure the experiment setting. Fill the following menu.
experiment_name ="exp_0325_adpt_1vs0"
repeat_times = 5 #how many times you wanna repeat
layers=[1] #depth of SWIM network
width_per_layer=[8,12,25,50,100,200,400,800,1600,3200] #layer width of SWIM network 8,12,25,50,100,200,400,800,1600
activations=['wavy_rat'] # what activation function you wanna use?
objectives=['discontinuous_complicated'] # what dynamics you wanna approximate?
feed_backs=[0]#feed back

total_iterations = len(objectives) * len(activations) *len(feed_backs)* len(layers) * len(width_per_layer) * repeat_times
with tqdm.tqdm(total=total_iterations, desc="Progress", unit="task", colour="green") as pbar:
    for obj in objectives:
        for act  in activations:
            for fdbk in feed_backs:
                for layer in layers:
                    for width in width_per_layer:
                        for j in range(repeat_times):
                            status_msg = f"Processing: obj={obj}, act={act},fdbk={fdbk} width={width}, repeat={j}"
                            pbar.set_postfix_str(status_msg)  # Updates the tqdm status line
                            # Run SWIM.py

                            subprocess.call("python3 SWIM.py --layer {} --width {} --repeat_times {} --act {} --objective {} --experiment_name {} --feed_back {}".format(layer, width, j, act, obj, experiment_name,fdbk), shell=True, cwd=cwd)
                            pbar.update(1)
                        subprocess.call("python3 compute_average_loss.py --layer {} --width {} --act {} --objective {} --experiment_name {} --feed_back {}".format(layer, width,act, obj, experiment_name,fdbk), shell=True, cwd=cwd)

                subprocess.call("python3 error_curve_plotter.py --objective {} --activation {} --experiment_name {} --feed_back {}".format(obj, act, experiment_name,fdbk), shell=True, cwd=cwd)
