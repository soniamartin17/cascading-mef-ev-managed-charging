import pandas as pd
import numpy as np
import sys
from sklearn.cluster import AgglomerativeClustering


# Get file names
print('Beginning preprocessing of data and clustering.')
data_file_name = sys.argv[1]  # The first argument is the data file name without GPS
data_file_name_withLatLon = sys.argv[2]  # The second argument is the data file name with latitude and longitude

# Load data and process columns
df = pd.read_csv(data_file_name+'.csv', index_col=0)
df = df.rename(columns={'soc_pct':'SOC', 'odometer_km':'Mileage', 'vehicle_home':'Home'})
for i, vin in enumerate(df['vin_id'].unique()):
    df.loc[df.loc[df['vin_id']==vin].index, 'VINID'] = i+1 # Make integer version of vehicle ID number
number_of_drivers = i+1

# Convert timestamp to local time
tmp = df.copy(deep=True)
tmp['vehicle_timestamp_utc'] = pd.to_datetime(df['vehicle_timestamp_utc'], utc=True)
tmp.index = tmp['vehicle_timestamp_utc']
tmp = tmp.tz_convert("US/Pacific")
tmp = tmp.tz_localize(None)
df['Timestamp Local'] = tmp.index.copy(deep=True)
df['datetime'] = pd.to_datetime(df['Timestamp Local'])

# Compare with previous time stamps
for col in ['SOC', 'Home', 'Mileage']:
    for driver in np.arange(1, number_of_drivers+1):
        inds = df[df['VINID']==driver].index
        df.loc[inds[np.arange(1, len(inds))], col+'-1'] = np.copy(df.loc[inds[np.arange(0, len(inds)-1)], col].values)
        df.loc[inds[0], col+'-1'] = df.loc[inds[0], col]

# Identify when drivers were stopped and/or charging
df2 = None

df.loc[df.index, 'Status1'] = 'Driving'
df.loc[df.index, 'Status1-1'] = 'Driving'
df.loc[df.index, 'Status2'] = 'Unplugged'
df.loc[df.index, 'SessionStart'] = False
df.loc[df.index, 'ParkingStart'] = False

for vinid in np.arange(1, number_of_drivers+1):
    if np.mod(vinid, 5) == 0:
        print('On driver ', vinid)

    subset = df[df['VINID']==vinid].copy(deep=True)
    # look for plateaus in mileage
    stopped_mileages = subset[subset['Mileage-1']==subset['Mileage']]['Mileage'].unique() # stopped mileages
    for mileage in stopped_mileages:
        inds = subset[subset['Mileage']==mileage].index
        subset.loc[inds, 'Status1'] = 'Parked'
        socs = subset.loc[inds, 'SOC']
        if (socs.max() - socs.min()) > 2: # Assume very small changes in SOC were just recalibrations/errors in the estimate
            subset.loc[inds, 'Status2'] = 'Charging'

    # Fill in status1 at the previous time step
    subset.loc[subset.index[np.arange(1, len(subset))], 'Status1-1'] = np.copy(subset.loc[subset.index[np.arange(0, len(subset)-1)], 'Status1'].values)
    subset.loc[subset.index[0], 'Status1-1'] = subset.loc[subset.index[0], 'Status1']
    
    # Identify starts of charging or parking
    inds1 = subset.loc[(subset['Status1']=='Parked')&(subset['Status1-1']=='Driving')].index
    inds2 = subset.loc[(subset['Status1']=='Parked')&(subset['Mileage-1']!=subset['Mileage'])].index
    subset.loc[np.unique(np.sort(np.concatenate((inds1, inds2)))), 'ParkingStart'] = True
    subset.loc[subset.loc[(subset['ParkingStart'])&(subset['Status2']=='Charging')].index, 'SessionStart'] = True
    

    # Concatenate with previous drivers
    if df2 is not None:
        df2 = pd.concat((df2, subset.copy(deep=True)), axis=0, sort=True)
    else:
        df2 = subset.copy(deep=True)


# Save
df2.to_csv(data_file_name+'_PROCESSED.csv', index=None)

# indices of parking sessions to look at the locations: 
indices = pd.DataFrame({'Index':df2[df2['ParkingStart']==True].index})


# Cluster data with locations
# Use Agglomerative Clustering with complete linkage: all points within a cluster are within 0.0005 (about 50 meters) of eachother. Does not require presetting the number of clusters.

df_LatLon = pd.read_csv(data_file_name_withLatLon+'.csv', index_col=0) # the file with the gps data
df_LatLon_subset = df_LatLon.loc[indices['Index'].values, :].copy(deep=True)

all_vins = df_LatLon_subset['vin_id'].unique() # list of vin_ids
df_LatLon_subset.loc[df_LatLon_subset.index, 'ClusterLabel'] = np.nan # initialize new column

for i, vinid in enumerate(all_vins): # for each driver
    print('Driver Number '+str(i+1))
    inds = df_LatLon_subset.loc[(df_LatLon_subset['vin_id']==vinid)].index # driver's indices
    X = df_LatLon_subset.loc[inds, ['latitude', 'longitude']].values # data for clustering
    clustering = AgglomerativeClustering(n_clusters=None, distance_threshold=0.0005, metric='euclidean', linkage='complete').fit(X) # fit clustering
    df_LatLon_subset.loc[inds, 'ClusterLabel_50m'] = clustering.labels_ # record labels
    
# Save labeling
df_labels = df_LatLon_subset.loc[:, ['vin_id', 'odometer_km', 'vehicle_timestamp_utc', 'ClusterLabel_50m']] # changed from 'ClusterLabel' to 'ClusterLabel_50m'


# Add to data file without locations and save

df2.loc[df_labels.index.values, 'ClusterLabel_50m'] = df_labels['ClusterLabel_50m'].values # changed here from df_labels.index to df_labels['index'].values
df2['datetime'] = pd.to_datetime(df2['datetime'])
df2.loc[df2.index, 'EndTime'] = np.nan
df2.loc[df2.index, 'Access_50m'] = False
for vinid in np.arange(1, number_of_drivers+1):
    # Look at all the stops made by this driver
    inds = df2.loc[(df2['VINID']==vinid)&(df2['ClusterLabel_50m']==df2['ClusterLabel_50m'])].index
    for i in inds: # Take the starting index of each stop
        inds1 = df2.loc[(df['VINID']==df2.loc[i, 'VINID'])&(df2['Mileage']==df2.loc[i, 'Mileage'])].index
        df2.loc[i, 'EndTime'] = df2.loc[inds1[-1], 'datetime'] # Note the end time of that session

    # Look at each location labeled for this driver
    locs = df2.loc[inds, 'ClusterLabel_50m'].unique()
    for loc in locs:
        inds2 = df2.loc[(df2['VINID']==vinid)&(df2['ClusterLabel_50m']==loc)].index
        if df2.loc[inds2, 'SessionStart'].sum() > 0: # Did they ever charge there?
            df2.loc[inds2, 'Access_50m'] = True  # Then they had access to charging there

df2.to_csv(data_file_name+'_PROCESSED_withAccess_withoutSpeeds.csv', index=None)
print('Completed preprocessing and clustering of data. Saved to '+data_file_name+'_PROCESSED_withAccess_withoutSpeeds.csv')