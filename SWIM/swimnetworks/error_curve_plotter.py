import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict


def extract_errors_from_file(file_path):
    """Extract MSE value from avg_error.txt."""
    with open(file_path, "r") as f:
        lines = f.readlines()
        mse_value = float(lines[0].split(":")[1].strip())# Extract MSE
        rel_l2 = float(lines[1].split(":")[1].strip())# Extract rel.l2
    return mse_value,rel_l2


def parse_results(base_dir,who_first:int):
    """
    Recursively traverse the Results directory, extract objective, activation, width, and MSE values.

    Returns:
    - data: Dict of {objective: {activation: (widths, mse_values)}}
    """
    data = defaultdict(lambda: defaultdict(lambda :defaultdict( list)) ) # {objective: {activation: [(width, mse)]}}

    for root, _, files in os.walk(base_dir):
        if "avg_error.txt" in files:
            # Extract objective, activation, width from the directory structure
            path_parts = root.split(os.sep)
            objective = path_parts[2]  # objective_*
            activation = path_parts[3]
            feedback = path_parts[4]# activation_*
            width = int(re.search(r'width(\d+)', root).group(1))  # Extract width as int

            # Get MSE value
            mse_value,rel_l2 = extract_errors_from_file(os.path.join(root, "avg_error.txt"))

            # Store in dictionary
            if who_first==0:
                data[objective][feedback][activation].append((width, mse_value, rel_l2))
            elif who_first==1:
                data[activation][feedback][objective].append((width, mse_value, rel_l2))
            else:
                data[feedback][activation][objective].append((width, mse_value, rel_l2))

    return data


def plot_results(data, base_dir="Results"):
    """Generate and save plots for each objective dynamic in respective folders."""
    for label1, label2s in data.items():
        num_label2s = len(label2s)
        fig, axes = plt.subplots(num_label2s, 2, figsize=(12, 6*num_label2s))
        fig.suptitle(f'Error Curves {label1}', fontsize=16)
        if num_label2s == 1:
            axes =np.array([axes])

        for row_idx, (label2,label3s) in enumerate(label2s.items()):

            mse_ax  = axes[row_idx,0]
            rel_ax = axes[row_idx,1]
            mse_ax.set_title(f'MSE {label2}')
            rel_ax.set_title(f'REL {label2}')

            mse_ax.set_xscale("log")
            mse_ax.set_yscale("log")
            rel_ax.set_xscale("log")
            rel_ax.set_yscale("Linear")

            mse_ax.set_xlabel("Network Width")
            rel_ax.set_xlabel("Network Width")
            mse_ax.set_ylabel("MSE")
            rel_ax.set_ylabel("REL")

            mse_ax.grid(True,which="both",linestyle="--",linewidth=0.5)
            rel_ax.grid(True,which="both",linestyle="--",linewidth=0.5)

            for label3, values in label3s.items():
                # Sort values by width (ensures correct plotting order)
                sorted_values=sorted(values,key=lambda x:x[0])
                widths, mse_values,rel_l2 = zip(*sorted_values)

                mse_ax.plot(widths, mse_values, marker="o",label=label3)
                rel_ax.plot(widths, rel_l2, marker="o",label=label3)
            mse_ax.legend()
            rel_ax.legend()
        if not os.path.exists(base_dir):
            os.makedirs(base_dir)
        save_path = os.path.join(base_dir, f"{label1}.png")
        plt.savefig(save_path,dpi=300)
        plt.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default="KdV_sine")
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--experiment_name", type=str, default="experiment")
    parser.add_argument("--feed_back", type=int, default=0)

    args=parser.parse_args()
    obj=args.objective
    act=args.activation
    experiment_name=args.experiment_name
    feed_back=args.feed_back

    data_directory="Results/"+experiment_name
    save_directory="Results/"+experiment_name+"/error_curves_obj"
    data_obj_first= parse_results(data_directory,who_first=0)
    plot_results(data_obj_first, save_directory)

    save_directory = "Results/"+experiment_name+"/error_curves_act"
    data_act_first = parse_results(data_directory, who_first=1)
    plot_results(data_act_first, save_directory)

    save_directory = "Results/" + experiment_name + "/error_curves_fdbk"
    data_fdbk_first = parse_results(data_directory, who_first=2)
    plot_results(data_fdbk_first, save_directory)



