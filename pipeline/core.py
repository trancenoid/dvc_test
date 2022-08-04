import os
import pandas as pd
import typing
import itertools
import math
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics import mean_squared_error

#prophet model
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

import numpy as np
# for SQL connection
import pyodbc
from  datetime import datetime

# Connect to SQL Server using pyodbc given a connection string

def connect_to_db(conn_str):

    conn_str = f"driver={pyodbc.drivers()[-1]};" + conn_str
    return pyodbc.connect(conn_str)


def fetch_data(connection, product_ids = ["254"], channel_ids = ["6"] ):

    statement = f"""SELECT orders.customer_channel_id , child.order_id, child.amount, child.width, child.height, child.page_count, child.shipping_date, papers.grammage, papers.name 
    FROM [dbo].[items] parent 
    INNER JOIN dbo.items child 
    ON child.parent_item_id = parent.id 
    INNER JOIN dbo.papers 
    ON child.paper_id = papers.id 
    INNER JOIN dbo.orders 
    ON child.order_id = orders.id 
    WHERE parent.product_id in (""" + ','.join(product_ids) + """) and orders.customer_channel_id in (""" + ','.join(channel_ids) + ") "

    # Get Data from server, Selects all orders made for list of product_ids by all customers by list channel_id
    data = pd.read_sql(statement, connection)

    assert data.shape[0] != 0  # not empty 

    assert data.isna().sum(axis = 1).sum() == 0  # no missing values

    return data



def process_df(df):
    df = df.copy(deep = True)
    df = df.sort_values(by=['shipping_date'])
    df['weight'] = (df['amount'] * df['width'] * df['height'] * df['grammage'] * df['page_count'])/1000000000
    df['shipping_date'] = pd.to_datetime(df['shipping_date'])
    df.index = pd.to_datetime(df['shipping_date'])
    df = df.groupby(pd.Grouper(freq = "W-SUN")).sum()
    df["ds"], df["y"] = df.index, df["weight"]
    current_date = datetime.today().strftime('%Y-%m-%d')
    df = df[df.ds < current_date]
    df = df[['ds', 'y']].groupby(['ds']).sum().reset_index()
    return df

def create_sliding_window_splits(df, moving_window = 5):    
    os.makedirs("data/train", exist_ok=True)
    os.makedirs("data/eval", exist_ok=True)

    start = len(df[(df.ds.dt.year <= 2021)])
    
    for i in range(start, len(df[:-(moving_window-1) ])):
        train, val = df[0:i], df[i:i+ moving_window]
        
        train.to_csv('data/train/{}.csv'.format(i))
        val.to_csv('data/eval/{}.csv'.format(i))
        
    test = df[-(moving_window-1):]    
    test.to_csv('data/test.csv')
    

class Tuner():

    def __init__(self, tune_params):

        self.param_grid = {  
                    'changepoint_prior_scale': Tuner.process_field(tune_params['changepoint_prior_scale']),
                    'changepoint_range': Tuner.process_field(tune_params['changepoint_range']),
                    'seasonality_prior_scale': Tuner.process_field(tune_params['seasonality_prior_scale']),
                    'seasonality_mode': Tuner.process_field(tune_params['seasonality_mode']),
                    'growth': Tuner.process_field(tune_params['growth'])
                    }

        self.params_df = self.create_param_combinations()

    @staticmethod
    def process_field(val):
        if isinstance(val,dict):
            return list(np.arange(val['start'], val['stop'], val['step']))
        elif isinstance(val, (list,np.array,tuple) ):
            return val


    @staticmethod
    def create_param_combinations(**param_dict):
        param_iter = itertools.product(*param_dict.values())
        params =[]
        for param in param_iter:
            params.append(param) 
        params_df = pd.DataFrame(params, columns=list(param_dict.keys()))
        return params_df
    

    def tune(self):

        param_combinations = self.create_param_combinations(**self.param_grid)
        os.makedirs("results", exist_ok=True)
        with ThreadPoolExecutor(max_workers = 5) as executor:
            index = range(0, len(param_combinations))
            changepoint_prior_scale_list = list(param_combinations["changepoint_prior_scale"])
            changepoint_range_list = list(param_combinations["changepoint_range"])
            seasonality_prior_scale_list = list(param_combinations["seasonality_prior_scale"])
            seasonality_mode_list = list(param_combinations["seasonality_mode"])
            growth_list = list(param_combinations["growth"])
            results = executor.map(self.train_model, index, changepoint_prior_scale_list, 
            changepoint_range_list, seasonality_prior_scale_list, 
            seasonality_mode_list, growth_list)
        
        
        
    def train_model(self, index, changepoint_prior_scale, changepoint_range, seasonality_prior_scale, seasonality_mode, growth):
        
        files = os.listdir("data/train")

        result_error = []
        for file in files:
            train = pd.read_csv("data/train/"+file)
            train = train.sort_values(by=['ds'])
            val = pd.read_csv("data/eval/"+file)
            val = val.sort_values(by=['ds'])
        
            model = Prophet(changepoint_prior_scale=changepoint_prior_scale,
                    changepoint_range=changepoint_range,
                    seasonality_prior_scale=seasonality_prior_scale,
                    seasonality_mode=seasonality_mode,
                    growth=growth)
            
            model.fit(train)
            forecast = model.predict(val)
            error = math.sqrt(mean_squared_error(val.y, forecast.yhat))
            result_error.append(float(error))
            
        aggregated_results = pd.DataFrame()

        aggregated_results["file"] = files
        aggregated_results["error"] = result_error
        aggregated_results["changepoint_prior_scale"] = changepoint_prior_scale
        aggregated_results["changepoint_range"] = changepoint_range
        aggregated_results["seasonality_prior_scale"] = seasonality_prior_scale
        aggregated_results["seasonality_mode"] = seasonality_mode
        aggregated_results["growth"] = growth    
        aggregated_results.to_csv("results/{}.csv".format(index))


