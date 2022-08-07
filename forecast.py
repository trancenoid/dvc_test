from prophet.serialize import model_from_json
from prophet import Prophet
import yaml
import pandas as pd

params = yaml.safe_load(open("params.yaml"))["forecast"]

with open('../models/serialized_model.json', 'r') as fin:
    model = model_from_json(fin.read())  # Load model

test = pd.DataFrame({"ds" : [ pd.to_datetime(params['start_date']) + x*pd.DateOffset(days = 7) for x in range(params['num_weeks'])]})

forecast = model.predict(test)
forecast.to_csv("forecast.csv")