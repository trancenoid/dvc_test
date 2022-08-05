import yaml
import os
import json
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json

params = yaml.safe_load(open("params.yaml"))["train"]
os.makedirs("models", exist_ok=True)

if not params['do_training']:
    print("Training flag is set to False, skipping training")
    exit()

print("Taking params from train_params.json")
with open("pipeline/train_params.json", 'rb') as f:
    train_config = json.load(f)

data = pd.read_csv("data/all_data.csv")

print("Training")
model = Prophet(changepoint_prior_scale = train_config['changepoint_prior_scale'],
                    changepoint_range = train_config['changepoint_range'],
                    seasonality_prior_scale = train_config['seasonality_prior_scale'],
                    seasonality_mode = 'multiplicative',
                    growth = 'linear')
model.fit(data)

print("saving model")

with open('models/serialized_model.json', 'w') as fout:
    fout.write(model_to_json(model))  # Save model





