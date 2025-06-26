import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import cvxpy as cvx
import copy


#create two separate lists for latitude and longitude
locations_list_lat = [37.8651, 36.4864, 36.8879, 40.4977, 41.2132, 36.4906, 33.8734, 36.4636, 34.2461]
locations_list_long = [-119.5383, -118.5658, -118.5551, -121.4200, -124.0046, -121.1825, -115.9010, -116.8656, -119.2644]

#create a DataFrame with the specified columns
synthetic_df = pd.DataFrame(columns=['index', 'vin_id', 'vehicle_timestamp_utc', 'vehicle_home', 'soc_pct', 'odometer_km', 'latitude', 'longitude'])

# Generate synthetic data

start_time= datetime.datetime(month=1, day=1, year=2020, hour=0, minute=0, second=0)
end_time = datetime.datetime(month=2, day=10, year=2020, hour=23, minute=59, second=59)

#Generate n "home charging" vehicles
n = 6
index=-1
for i in range(n):
    index += 1
    # Generate a random VIN
    vin_id = str(np.random.randint(100000, 999999))
    
    #Generate the first timestamp in the range
    vehicle_timestamp_utc = start_time 

    # Randomly select a home location from the list
    home_loc_idx = np.random.randint(0, len(locations_list_lat))
    work_loc_idx = home_loc_idx + 1 if home_loc_idx + 1 < len(locations_list_lat) else 0
    
    
    # Generate a random state of charge percentage
    soc_pct = np.random.uniform(20, 50)
    
    # Generate a random odometer reading
    odometer_km = np.random.uniform(0, 100000)
    
    # Randomly select latitude and longitude from the lists
    
    latitude = locations_list_lat[home_loc_idx]
    longitude = locations_list_long[home_loc_idx]
    vehicle_home = 1

    
    # Append the generated data to the DataFrame
    synthetic_df = synthetic_df.append({
        'index': index,
        'vin_id': vin_id,
        'vehicle_timestamp_utc': vehicle_timestamp_utc,
        'vehicle_home': vehicle_home,
        'soc_pct': soc_pct,
        'odometer_km': odometer_km,
        'latitude': latitude,
        'longitude': longitude
    }, ignore_index=True)

    # Generate additional events for each vehicle
    for day in range(41):
        #1) append a row for "plug out" event
        index += 1
        # Increment the timestamp by one day
        vehicle_timestamp_utc = start_time + datetime.timedelta(days=day)

        #Plug out time
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(4, 8), minutes=np.random.randint(0, 60))
        
        # Generate a random end state of charge percentage
        soc_pct += np.clip(np.random.normal(20, 10), 0, 100- soc_pct)

        # Append the plug out time to the DataFrame
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

        #2) append a row for drive start event
        index += 1
        # Increment the timestamp by a random number of hours
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(1, 2), minutes=np.random.randint(0, 60))
        # Append the drive start to the DataFrame
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

        #3) append a row for drive end event
        index += 1
        # Increment the timestamp by a random number of hours
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(1, 2), minutes=np.random.randint(0, 60))
        # Generate a random end state of charge percentage
        delta_soc = np.random.uniform(0, 20)
        soc_pct -= delta_soc
        soc_pct = np.clip(soc_pct, 0, 100)  # Ensure SOC does not go below 0 or above 100
        odometer_km += 4 * delta_soc/100 * 83.6 #4 km/kwh rate

        latitude = locations_list_lat[work_loc_idx]
        longitude = locations_list_long[work_loc_idx]

        if latitude == locations_list_lat[home_loc_idx]:
            vehicle_home = 1
        else:
            vehicle_home = 0

        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

        #4) append a row for drive start event
        index += 1
        # Increment the timestamp by a random number of hours
        vehicle_timestamp_utc = vehicle_timestamp_utc + datetime.timedelta(hours=np.random.randint(4, 7), minutes=np.random.randint(0, 60))
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

        #5) append a row for drive end event ending at home
        index += 1
        # Increment the timestamp by a random number of hours
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(1, 2), minutes=np.random.randint(0, 60))
        # Generate a random end state of charge percentage
        delta_soc = np.random.uniform(10, 40)
        soc_pct -= delta_soc
        soc_pct = np.clip(soc_pct, 0, 100)  # Ensure SOC does not go below 0 or above 100
        odometer_km += 4 * delta_soc/100 * 83.6
        latitude = locations_list_lat[home_loc_idx]
        longitude = locations_list_long[home_loc_idx]
        vehicle_home = 1
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

        #6) append a row for "plug in" event
        index += 1
        # Increment the timestamp by a random number of hours   
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(1, 2), minutes=np.random.randint(0, 60))
        
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)
        
        


#Generate m "work charging" vehicles
m=3
for i in range(m):
    str(np.random.randint(100000, 999999))

    # Generate a random VIN
    vin_id = str(np.random.randint(100000, 999999))
    
    #Generate the first timestamp in the range
    vehicle_timestamp_utc = start_time 

    # Randomly select a home location from the list
    home_loc_idx = np.random.randint(0, len(locations_list_lat))
    work_loc_idx = home_loc_idx + 1 if home_loc_idx + 1 < len(locations_list_lat) else 0
    
    # Generate a random state of charge percentage
    soc_pct = np.random.uniform(20, 50)
    
    # Generate a random odometer reading
    odometer_km = np.random.uniform(0, 100000)
    
    # Randomly select latitude and longitude from the lists
    
    latitude = locations_list_lat[home_loc_idx]
    longitude = locations_list_long[home_loc_idx]
    vehicle_home = 1

    index += 1
    # Append the generated data to the DataFrame
    synthetic_df = synthetic_df.append({
        'index': index,
        'vin_id': vin_id,
        'vehicle_timestamp_utc': vehicle_timestamp_utc,
        'vehicle_home': vehicle_home,
        'soc_pct': soc_pct,
        'odometer_km': odometer_km,
        'latitude': latitude,
        'longitude': longitude
    }, ignore_index=True)

    for day in range(41):
        #1) append row for drive start event
        index += 1
        # Increment the timestamp by one day    
        vehicle_timestamp_utc = start_time + datetime.timedelta(days=day)
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(4, 8), minutes=np.random.randint(0, 60))
        # Append the drive start to the DataFrame
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

        #2) append a row for drive end event
        index += 1
        # Increment the timestamp by a random number of hours
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(1, 2), minutes=np.random.randint(0, 60))
        # Generate a random end state of charge percentage
        delta_soc = np.random.uniform(10, 40)
        soc_pct -= delta_soc
        soc_pct = np.clip(soc_pct, 0, 100)
        odometer_km += 4 * delta_soc/100 * 83.6
        latitude = locations_list_lat[work_loc_idx]
        longitude = locations_list_long[work_loc_idx]
        if latitude == locations_list_lat[home_loc_idx]:
            vehicle_home = 1
        else:
            vehicle_home = 0
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

        #3) append a row for charge end event
        index += 1
        # Increment the timestamp by a random number of hours
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(4, 7), minutes=np.random.randint(0, 60))
        # Generate a random end state of charge percentage
        soc_pct += np.clip(np.random.normal(20, 10), 0, 100- soc_pct)
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

        #4) append a row for drive end event to home
        index += 1
        # Increment the timestamp by a random number of hours
        vehicle_timestamp_utc += datetime.timedelta(hours=np.random.randint(1, 2), minutes=np.random.randint(0, 60))
        # Generate a random end state of charge percentage
        delta_soc = np.random.uniform(0, 20)
        soc_pct -= delta_soc
        soc_pct = np.clip(soc_pct, 0, 100)
        odometer_km += 4 * delta_soc/100 * 83.6
        latitude = locations_list_lat[home_loc_idx]
        longitude = locations_list_long[home_loc_idx]
        vehicle_home = 1
        synthetic_df = synthetic_df.append({
            'index': index,
            'vin_id': vin_id,
            'vehicle_timestamp_utc': vehicle_timestamp_utc,
            'vehicle_home': vehicle_home,
            'soc_pct': soc_pct,
            'odometer_km': odometer_km,
            'latitude': latitude,
            'longitude': longitude
        }, ignore_index=True)

#Save the synthetic data to a CSV file
synthetic_df.to_csv('Data/synthetic_data_with_gps.csv', index=False) 
#Save the synthetic data to a CSV without the latitude and longitude columns
synthetic_df_no_gps = synthetic_df.drop(columns=['latitude', 'longitude'])
synthetic_df_no_gps.to_csv('Data/synthetic_data_without_gps.csv', index=False)