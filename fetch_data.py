from pipeline.core import *
from pipeline.settings import Settings
import sys
import yaml
import json 

params = yaml.safe_load(open("params.yaml"))["fetch_data"]

with open("pipeline/training_data_config.json", 'rb') as f:
    train_data_config = json.load(f)

        
with open("pipeline/connection.json", 'rb') as f:
    conn_config = json.load(f)


with open("pipeline/data_filters.json", 'rb') as f:
    filters_config = json.load(f)


# connect to DB 
connection = connect_to_db(conn_config['connection_string'][params['server_type']])

# fetch data with supplied filters
data = fetch_data(
    connection, 
    filters_config['product_ids'],
    filters_config['channel_ids']
    )

# preprocess data and compute target (weight) columns
data = process_df(data)

# create moving window splits for evaluation
create_sliding_window_splits(data,train_data_config['moving_window_size'])