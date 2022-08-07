import pandas as pd
import pyodbc
import os

def get_connection():
    drivers = pyodbc.drivers()
    driver = [x for x in drivers if "ODBC" in x]
    driver = driver[0]
    conn_str = (
        f"DRIVER={driver};server=moveproject.database.windows.net;database=smart_bi;UID=shivansh@lytiq.de;Authentication=ActiveDirectoryInteractive;"
    )
    connection = pyodbc.connect(conn_str)
    return connection

connection = get_connection()

forecast = pd.read_csv("forecast/forecast.csv")
forecast = forecast[['ds','yhat']]
forecast['yhat'] = forecast['yhat'].astype(int)
forecast['week'] = forecast['ds'].dt.isocalendar().week
forecast['year'] = forecast['ds'].dt.year


cursor = connection.cursor()

sql_ins = '''insert into forecasts_model (customer_channel_id, product_id, type, model_tag, year, week, amount) values ''' + \
    ','.join( [ '(' + ','.join( ['6' , '254' , "'model_forecast'" , "'ch6pid254'", str(x[1]["year"]) , 
    str(x[1]["week"]), str(x[1]["yhat"]) ]) + ')' for x in forecast.iterrows() ] ) + ";"


cursor.execute(sql_ins)

cursor.commit()

connection.close()