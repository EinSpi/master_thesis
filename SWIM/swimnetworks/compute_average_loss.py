import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default = 4)
    parser.add_argument("--width", type=int, default = 50)
    parser.add_argument("--act", type=str, default="relu")
    parser.add_argument("--objective", type=str, default="KdV_sine")
    args, _ = parser.parse_known_args()
    l = args.layer
    w = args.width
    act = args.act
    objective = args.objective
    layers = [w for i in range(l)]

    with open("Results/"+"obj_"+objective+"/"+"act_"+act+"/layers%d/width%d/errors.txt" % (len(layers),layers[0]), "r") as file:
        mse_values = []
        rel_l2_values = []

        for line in file:
            parts = line.split(",")
            mse = float(parts[0].split(":")[1].strip())
            rel_l2 = float(parts[1].split(":")[1].strip())

            mse_values.append(mse)
            rel_l2_values.append(rel_l2)

    # Compute the averages
    avg_mse = sum(mse_values) / len(mse_values) if len(mse_values) > 0 else 0.0
    avg_rel_l2 = sum(rel_l2_values) / len(rel_l2_values) if len(rel_l2_values) > 0 else 0.0

    # Write the result to a new file
    with open("Results/"+"obj_"+objective+"/"+"act_"+act+"/layers%d/width%d/avg_error.txt" % (len(layers),layers[0]), "w") as output_file:
        output_file.write(f"Average MSE: {avg_mse:.6f}\n")
        output_file.write(f"Average rel.L2: {avg_rel_l2:.6f}\n")