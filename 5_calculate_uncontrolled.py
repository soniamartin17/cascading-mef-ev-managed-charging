import pickle
import pandas as pd
import numpy as np
import datetime
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from charging import ChargingData
from charging import ChargingAutomation

#read EV and generator data
gds = {}
df = pd.read_csv('Data/processed_EV_data.csv', index_col=0) #insert CSV of processed data
folder_root = None #insert file location here
save_str = folder_root+'/all_uncontrolled_demand'
gds[2019] = pickle.load(open('Data/generator_data_short_WECC_2019.obj', 'rb'))
gds[2020] = pickle.load(open('Data/generator_data_short_WECC_2020.obj', 'rb'))

access_series_name = 'access_series' # could be plugged_series
all_control_names = {'varying':'varying_allaccess', 'annual_withinrange':'monthlyaverage_allaccess'}

#set simulation range and get uncontrolled demand data
month = 1
min_date = datetime.date(2020, month, 1)
max_date = datetime.date(2020, month, 31)
period_string = str(min_date) + '_to_' + str(max_date)

#set simulation weeks
data = ChargingData(df.copy(deep=True))
data.define_weeks(min_date=min_date, max_date=max_date)
save_str = folder_root+'/'+'all_uncontrolled_demand'

#calculate uncontrolled demand for all the cars only once
print('-----'*5)
print('Uncontrolled Demand Computation')        

model_uncontrolled = ChargingAutomation(min_date, max_date, data=data)

for week in range(data.num_weeks):
    tic = time.time()
    print('Week starting on : ', data.mondays[week])
    model_uncontrolled.calculate_uncontrolled_only_oneweek(week, verbose=True)
    toc = time.time()
    print('Elapsed time: ', toc-tic)

cols = model_uncontrolled.uncontrolled_charging_demand.columns[1:]
model_uncontrolled.uncontrolled_charging_demand.rename(columns = {i:str(int(i)) for i in cols}, inplace = True)

#save file
model_uncontrolled.uncontrolled_charging_demand.to_csv(save_str+'_individualdrivers_'+period_string+'.csv')
