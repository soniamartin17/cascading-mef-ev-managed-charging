import pickle
import pandas as pd
import numpy as np
import datetime
import time
import warnings
import sys
warnings.simplefilter(action='ignore', category=FutureWarning)

from charging import ChargingData
from charging import ChargingAutomation

#read EV and generator data
gds = {}
file_name = sys.argv[8]  # The first argument is the data file name without GPS
df = pd.read_csv(file_name + '_PROCESSED_withAccess_withSpeeds.csv', index_col=0) #insert CSV of processed data
folder_root = 'Data' 
save_str = folder_root+'/all_uncontrolled_demand'
gds[2019] = pickle.load(open('Data/generator_data_short_WECC_2019.obj', 'rb'))
gds[2020] = pickle.load(open('Data/generator_data_short_WECC_2020.obj', 'rb'))

access_series_name = 'access_series' # could be plugged_series
all_control_names = {'varying':'varying_allaccess', 'annual_withinrange':'monthlyaverage_allaccess'}

#set simulation range and get uncontrolled demand data
month1 = int(sys.argv[1])
month2 = int(sys.argv[2])
year1 = int(sys.argv[3])
year2 = int(sys.argv[4])
day1 = int(sys.argv[5])
day2 = int(sys.argv[6])

min_date = datetime.date(year1, month1, day1)
max_date = datetime.date(year2, month2, day2)
period_string = str(min_date) + '_to_' + str(max_date)

#set simulation weeks
chg_timer = sys.argv[7]

data = ChargingData(df.copy(deep=True), chg_timer)
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
if chg_timer:
    model_uncontrolled.uncontrolled_charging_demand.to_csv(save_str+'_individualdrivers_'+period_string+'_charging_timer.csv')
    print
else:
    model_uncontrolled.uncontrolled_charging_demand.to_csv(save_str+'_individualdrivers_'+period_string+'.csv')
    print('Uncontrolled demand data saved to: ', save_str+'_individualdrivers_'+period_string+'.csv')
