import yaml
import os

params = yaml.safe_load(open("params.yaml"))["train"]
os.makedirs("models", exist_ok=True)

if not params['do_training']:
    print("Training flag is set to False, skipping training")
    exit()


print("Taking params from train_params.json")

print("Training")