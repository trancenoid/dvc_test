from core import *
from settings import Settings


# load config files
settings = Settings(
    conn_json_path="connection.json",
    train_data_json_path="training_data_config.json", 
    filter_json_path="data_filters.json", 
    train_json_path="train_params.json", 
    test_json_path="test_params.json")

# connect to DB 
connection = connect_to_db(settings.conn_config['connection_string']['local'])

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

# if tuning is requested then initiate tuning pipe
if settings.train_config['tune']['enabled']:
    tune_flag = True
    tuner = Tuner(settings.train_config['tune'])
    tuner.tune()


# get params to train on ; Can be best of grid search or supplied in config
if tune_flag:
    pass
else:
    pass    


# create model with best/supplied params

model = Prophet(changepoint_prior_scale = min_comb['changepoint_prior_scale'][0],
                    changepoint_range = min_comb['changepoint_range'][0],
                    seasonality_prior_scale = min_comb['seasonality_prior_scale'][0],
                    seasonality_mode = 'multiplicative',
                    growth = 'linear')

model.fit(train)

# Test on supplied test data 

model.pred()
