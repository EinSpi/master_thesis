import subprocess
import os
import tqdm

cwd = os.path.dirname(os.path.realpath(__file__))

# Run the different experiments
experiment_name = 'experiment0401'
para_amounts=['A','B','C']
shapes=['s','m','d']
acts=['rat']
objs=['KdV_sine','discontinuous_complicated','discontinuous_trivial']
total_iterations=len(para_amounts)*len(shapes)*len(acts)*len(objs)
epochs=10000


with tqdm.tqdm(total=total_iterations, desc="Progress", unit="task", colour="green") as pbar:
    for obj in objs:
        for amt in para_amounts:
            for shape in shapes:
                for act in acts:
                    status_msg = f"Processing: obj={obj}, amt={amt},shape={shape} act={act}"
                    pbar.set_postfix_str(status_msg)  # Updates the tqdm status line
                    subprocess.call('python3 BACKPROPAGATION.py --objective {} --para_amount {} --shape {} --activation {} --max_epochs {} --experiment_name {}'.format(obj,amt,shape,act,epochs,experiment_name), shell=True,cwd=cwd)
                    pbar.update(1)
                subprocess.call('python3 ERROR_CURVE.py --objective {} --para_amount {} --shape {}  --max_epochs {} --experiment_name {}'.format(obj,amt,shape,epochs,experiment_name), shell=True,cwd=cwd)


