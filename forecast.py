from prophet.serialize import model_from_json
from prophet import Prophet
import yaml
import pandas as pd

params = yaml.safe_load(open("params.yaml"))["train"]

with open('models/serialized_model.json', 'r') as fin:
    model = model_from_json(fin.read())  # Load model

