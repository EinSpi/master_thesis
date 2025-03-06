import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import re
from collections import defaultdict


def extract_mse_from_file(file_path):
    """Extract MSE value from avg_error.txt."""
    with open(file_path, "r") as f:
        lines = f.readlines()
        mse_value = float(lines[0].split(":")[1].strip())  # Extract MSE
    return mse_value


def parse_results(base_dir):
    """
    Recursively traverse the Results directory, extract objective, activation, width, and MSE values.

    Returns:
    - data: Dict of {objective: {activation: (widths, mse_values)}}
    """
    data = defaultdict(lambda: defaultdict(list))  # {objective: {activation: [(width, mse)]}}

    for root, _, files in os.walk(base_dir):
        if "avg_error.txt" in files:
            # Extract objective, activation, width from the directory structure
            path_parts = root.split(os.sep)
            objective = path_parts[1]  # objective_*
            activation = path_parts[2]  # activation_*
            width = int(re.search(r'width(\d+)', root).group(1))  # Extract width as int

            # Get MSE value
            mse_value = extract_mse_from_file(os.path.join(root, "avg_error.txt"))

            # Store in dictionary
            data[objective][activation].append((width, mse_value))

    return data


def plot_results(data, base_dir="Results"):
    """Generate and save plots for each objective dynamic in respective folders."""
    for objective, activations in data.items():
        plt.figure(figsize=(8, 6))

        for activation, values in activations.items():
            # Sort values by width (ensures correct plotting order)
            values.sort(key=lambda x: x[0])
            widths, mse_values = zip(*values)

            plt.plot(widths, mse_values, marker="o", label=activation)

        # Plot settings
        plt.xscale("log")  # Linear x-axis (widths)
        plt.yscale("log")  # Log-scale for MSE values
        plt.xlabel("Network Width")
        plt.ylabel("Average MSE")
        plt.title(objective)  # Title is the objective dynamic
        plt.legend()
        plt.grid(True)

        # Save plot inside the respective objective folder
        save_path = os.path.join(base_dir, "error_curves", f"{objective}.png")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  # Ensure directory exists
        plt.savefig(save_path, dpi=300)
        plt.close()

    print("Plots saved inside respective objective folders.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--objective", type=str, default="KdV_sine")

    obj=parser.parse_args().objective
    base_directory="Results/"+"obj_"+obj
    data = parse_results(base_directory)
    plot_results(data, base_directory)



