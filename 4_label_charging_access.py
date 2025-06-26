import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys

file_name = sys.argv[1]  # The first argument is the data file name 
df = pd.read_csv(file_name + '_PROCESSED_withAccess_withoutSpeeds.csv')
df['datetime'] = pd.to_datetime(df['datetime'])

# Charging speed at each location

n_drivers = int(df['VINID'].max())
df.loc[df.index, 'Average_Charging_Rate_kW'] = np.nan
df.loc[df.index, 'Max_Charging_Rate_kW'] = np.nan
battcap = 83.6 # kWh #insert battery capacity here
for vinid in np.arange(1, n_drivers+1):
    print('VINID: ', vinid)
    inds = df.loc[df['VINID']==vinid].index

    subset = df.loc[inds].copy(deep=True)
    mileages = subset[subset['SessionStart']]['Mileage'].values
    for mile in mileages:
#         first_point = subset.loc[(subset['Mileage']==mile)].index.values[0]
        first_point = subset.loc[(subset['Mileage']==mile)&(subset['SOC']==subset.loc[subset['Mileage']==mile]['SOC'].min())].index.values[-1]
        max_soc_point = subset.loc[(subset['Mileage']==mile)&(subset['SOC']==subset[(subset['Mileage']==mile)]['SOC'].max())].index.values[0]
        time_seconds = (subset.loc[max_soc_point, 'datetime'] - subset.loc[first_point, 'datetime']).total_seconds()
        soccharging = subset.loc[(subset['Mileage']==mile)]['SOC'].max() - subset.loc[(subset['Mileage']==mile)]['SOC'].min()
        if soccharging > 2:
            if time_seconds > 0:
                average_rate = soccharging / 100 * battcap / (np.maximum(time_seconds, 60) / (60*60))
                subset.loc[subset.loc[(subset['Mileage']==mile)].index, 'Average_Charging_Rate_kW'] = average_rate
            else:
                subset.loc[subset.loc[(subset['Mileage']==mile)].index, 'Average_Charging_Rate_kW'] = 0

    for places in subset['ClusterLabel_50m'].unique():
        if places == places:
            inds1 = subset.loc[subset['ClusterLabel_50m']==places].index
            mileages = subset.loc[inds1, 'Mileage'].unique()
            
            avg_rate_values=pd.Series(subset.loc[subset['Mileage'].isin(mileages)]['Average_Charging_Rate_kW'].sort_values().dropna().unique())
            for outlier in range(1): # loop
                if len(avg_rate_values)>1 and (avg_rate_values.iloc[-1]-avg_rate_values.iloc[-2])>10:
                    avg_rate_values = avg_rate_values.drop(avg_rate_values.tail(1).index)
            avg_rate_values = avg_rate_values.clip(0, 50)
            subset.loc[subset.loc[subset['Mileage'].isin(mileages)].index, 'Max_Charging_Rate_kW'] = avg_rate_values.max()
            
        
    df.loc[inds, ['Average_Charging_Rate_kW', 'Max_Charging_Rate_kW']] = subset.loc[:, ['Average_Charging_Rate_kW', 'Max_Charging_Rate_kW']].values
    
df.to_csv(file_name + '_PROCESSED_withAccess_withSpeeds.csv')
print('Saved charging access data with speeds to '+ file_name + '_PROCESSED_withAccess_withSpeeds.csv')

