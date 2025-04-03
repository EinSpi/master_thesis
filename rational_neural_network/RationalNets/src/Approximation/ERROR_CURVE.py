import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
from itertools import cycle
from pathlib import Path

def prepare_working_directory(obj:str, amt:str, shp:str, exp_name:str):
    work_dir = 'Results/'+exp_name+'/obj_'+obj+'/amt_'+amt+'/shape_'+shp
    if not os.path.exists(work_dir):
        raise ValueError('the error curve data is not complete')
    else:
        return work_dir

if __name__ == "__main__":

    # Get the type of the rational
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default='KdV_sine')
    parser.add_argument("--para_amount", type=str, default='A')
    parser.add_argument("--shape", type=str, default='s')
    parser.add_argument("--max_epochs", type=int, default=10000)
    parser.add_argument("--experiment_name", type=str, default='exp')
    args, _ = parser.parse_known_args()
    obj=args.objective
    amt=args.para_amount
    shp=args.shape
    max_epochs=args.max_epochs
    exp_name=args.experiment_name

    work_dir = prepare_working_directory(obj=obj, amt=amt, shp=shp, exp_name=exp_name)

    plt.rcParams.update({'font.size': 14})

    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 6))


    # Get all subdirectories containing err.csv files
    csv_files = []
    for root, dirs, files in os.walk(work_dir):
        for file in files:
            if file.endswith('val_loss_curve.csv'):
                csv_files.append((root, file))

    # --- Color & Line Style Strategy ---
    # 1. Combine multiple high-contrast colormaps (60+ colors)
    color_palette = []
    for cmap in [plt.cm.tab20, plt.cm.tab20b, plt.cm.Set3, plt.cm.Pastel1]:
        color_palette.extend(cmap(np.linspace(0, 1, cmap.N)))

    # 2. Cyclic line styles (4 options)
    line_styles = cycle(['-', '--', '-.', ':'])

    # Plot each curve
    for idx, (folder_path, csv_file) in enumerate(csv_files):
        # Get folder name for label
        folder_name = Path(folder_path).name

        # Load data
        try:
            data = np.genfromtxt(os.path.join(folder_path, csv_file), delimiter=',')
            if data.size == 0:
                continue
            # Assign color and line style
            color = color_palette[idx % len(color_palette)]
            linestyle = next(line_styles)

            # Plot with unique style
            plt.loglog(data[:, 0], data[:, 1],
                       color=color, linestyle=linestyle,
                       label=folder_name)
        except Exception as e:
            print(f"Error processing {folder_path}/{csv_file}: {e}")

    # Configure plot
    plt.yticks([10 ** -8, 10 ** -7, 10 ** -6, 10 ** -5, 10 ** -4, 10 ** -3, 10 ** -2, 10 ** -1, 1, 10])

    # --- Aesthetics ---
    plt.grid(True, which="both", alpha=0.3)
    ax.set_aspect(0.35)
    ax.xaxis.set_ticks_position('both')
    ax.yaxis.set_ticks_position('both')

    # Legend outside to avoid overlap
    plt.legend(
        bbox_to_anchor=(1.05, 1),
        loc='upper left',
        fontsize=10,
        framealpha=0.9  # Semi-transparent background
    )

    plt.tight_layout()
    plt.savefig(work_dir+'/val_err_curves.pdf', bbox_inches='tight')
    plt.close()





