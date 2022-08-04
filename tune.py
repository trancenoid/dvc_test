from pipeline.core import *
import json 
import yaml

params = yaml.safe_load(open("params.yaml"))["tune"]

if not params['do_tuning']:
    print("Tuning flag is set to False, skipping")
    exit()

with open('pipeline/tune_params.json', 'rb') as f:
    tune_config = json.load(f)
    
tuner = Tuner(tune_config)
tuner.tune()

min_err = []
min_err_fbp = []
results = []
error_fbp = []
#results = os.listdir("results")

results = sorted(os.listdir("results"), key = lambda x : int(x[:-4]))

for result in results:
    result = pd.read_csv("results/"+ result)
    #result = result.sort_values(by=['file'])
    error = result['error'].mean()
    error_fbp.append(float(error))


len(results)

def getMinErr(inputlist):
    min_value = min(inputlist)
    min_index=inputlist.index(min_value)
    return min_value, min_index

min_value, min_index = getMinErr(error_fbp)

min_str = str("results/"+ str(min_index) + ".csv")
min_comb = pd.read_csv(min_str)

best_params = {
    'changepoint_prior_scale' : min_comb['changepoint_prior_scale'][0],
    'changepoint_range' : min_comb['changepoint_range'][0],
    'seasonality_prior_scale' : min_comb['seasonality_prior_scale'][0],
    'seasonality_mode' : 'multiplicative',
    'growth' : 'linear'
}

with open("./pipeline/train_params.json" ,'w') as f:
    json.dump(best_params,f)


                    