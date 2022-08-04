from pipeline.core import *
from pipeline.settings import Settings
import sys
import yaml

params = yaml.safe_load(open("params.yaml"))["fetch_data"]

# load config files
settings = Settings(
    conn_json_path="pipeline/connection.json",
    train_data_json_path="pipeline/training_data_config.json", 
    filter_json_path="pipeline/data_filters.json", 
    train_json_path="pipeline/train_params.json", 
    test_json_path="pipeline/test_params.json")

# connect to DB 
connection = connect_to_db(settings.conn_config['connection_string'][params['server_type']])

# fetch data with supplied filters
data = fetch_data(
    connection, 
    settings.filters_config['product_ids'],
    settings.filters_config['channel_ids']
    )

# preprocess data and compute target (weight) columns
data = process_df(data)

# create moving window splits for evaluation
create_sliding_window_splits(data,settings.train_data_config['moving_window_size'])