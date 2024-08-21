import pickle
import pandas as pd
import numpy as np
import datetime
import os
import time
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from simple_dispatch import GridModel
from charging import ChargingData
from charging import ChargingAutomation

#read EV and generator data
gds = {}
df = pd.read_csv('Data/processed_EV_data.csv', index_col=0) #insert CSV of processed data
folder_root = None #insert file location here
save_str = folder_root+'/all_uncontrolled_demand'
gds[2019] = pickle.load(open('Data/generator_data_short_WECC_2019.obj', 'rb'))
gds[2020] = pickle.load(open('Data/generator_data_short_WECC_2020.obj', 'rb'))

all_control_names = {'varying':'varying_allaccess', 'annual_withinrange':'monthlyaverage_allaccess'}

#set charging rates and charging efficiency loss
l1_rate = 1.3
l2_rate = 6.6
l3_rate = 50.
l4_rate = 150.
eta=.1

#set simulation range and get uncontrolled demand data
month = 1
min_date = datetime.date(2020, month, 1)
max_date = datetime.date(2020, month, 31)
period_string = str(min_date) + '_to_' + str(max_date)
 
all_uncontrolled_demand = pd.read_csv(save_str+'_individualdrivers_'+period_string+'.csv', index_col=0)
all_uncontrolled_demand.datetime = pd.to_datetime(all_uncontrolled_demand.datetime)

results_date = '20230802' #set simulation date for saving files
folder = folder_root + '/AEF/'
for run_number in range(15): #iterate to make multiple runs
    nev_set = [1000, 100000, 500000, 1000000, 1500000, 2000000] #run for different numbers of added EVs
    for n_evs_added in nev_set:
        print('-----'*5)
        print('Number of EVs added: ', n_evs_added)

        #setup folders
        folder_numevs = str(n_evs_added)+'EVs_added'
        if not os.path.isdir(folder+'/'+folder_numevs):
            os.mkdir(folder+'/'+folder_numevs)
        if not os.path.isdir(folder+'/'+folder_numevs+'/Uncontrolled'):
            os.mkdir(folder+'/'+folder_numevs+'/Uncontrolled')

        print('-----'*5)
        print('Starting and Sampling')
        tic = time.time()
        #randomly sample drivers from set of possible drivers (ones that have data for the given simulation week)
        data = ChargingData(df.copy(deep=True))
        data.define_weeks(min_date=min_date, max_date=max_date)
        indices_df = pd.DataFrame(np.zeros((n_evs_added, len(data.mondays))), columns=data.mondays)
        for i, monday in enumerate(data.mondays):
            possible_drivers = data.df.loc[(data.df.datetime.dt.date>=monday)&(data.df.datetime.dt.date<=data.sundays[i])&(data.df['Access_50m'])]['VINID'].unique()
            driver_inds = np.random.choice(possible_drivers, n_evs_added, replace=True)
            indices_df[monday] = driver_inds.astype(int)

        #save driver indices
        indices_df.to_csv(folder_root+'/'+'EV_indices'+'/'+folder_numevs+'/'+folder_numevs + '_indices'+'_'+period_string+'_run'+str(run_number)+'_'+results_date+'.csv')
        toc = time.time()
        print('Elapsed time: ', toc-tic)

        print('-----'*5)
        print('Dispatch 1: Uncontrolled')        
        model = ChargingAutomation(min_date, max_date, data=data)
        total_uncontrolled_demand = pd.DataFrame(columns=['datetime', 'total_demand', 'l1_demand', 'l2_demand', 'l3_demand'])
        
        #calculate aggregate uncontrolled demand profile for all vehicles, split into different charging levels
        for week in range(data.num_weeks):
            all_uncontrolled_demand_l1 = all_uncontrolled_demand.copy(deep=True)
            all_uncontrolled_demand_l1.iloc[:,np.arange(1, 749)] = all_uncontrolled_demand_l1.iloc[:,np.arange(1, 749)].where(all_uncontrolled_demand_l1.iloc[:,np.arange(1, 749)]<l1_rate/(1+eta), 0)

            all_uncontrolled_demand_l2 = all_uncontrolled_demand.copy(deep=True)
            all_uncontrolled_demand_l2.iloc[:,np.arange(1, 749)] = all_uncontrolled_demand_l2.iloc[:,np.arange(1, 749)].where((all_uncontrolled_demand_l2.iloc[:,np.arange(1, 749)]<=(l2_rate/(1+eta)+.01)), 0)

            all_uncontrolled_demand_l3 = all_uncontrolled_demand.copy(deep=True)
            all_uncontrolled_demand_l3.iloc[:,np.arange(1, 749)]=all_uncontrolled_demand_l3.iloc[:,np.arange(1, 749)].where(all_uncontrolled_demand_l3.iloc[:,np.arange(1, 749)]>(l2_rate/(1+eta)+.01), 0)

            #obtain time intervals and driver frequency
            #inds represents time
            #dot_values represent number of each driver selected
            inds = all_uncontrolled_demand.loc[(all_uncontrolled_demand.datetime.dt.date >= data.mondays[week])&(all_uncontrolled_demand.datetime.dt.date <= data.sundays[week])].index
            dot_values = indices_df[data.mondays[week]].value_counts().sort_index().reindex(np.arange(1, 749), fill_value=0)

            #multiply drivers by frequency
            all_uncontrolled_demand.loc[inds, 'total_demand'] = all_uncontrolled_demand.loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)
            all_uncontrolled_demand.loc[inds, 'l1_demand'] = all_uncontrolled_demand_l1.loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)
            all_uncontrolled_demand.loc[inds, 'l2_demand'] = all_uncontrolled_demand_l2.loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)
            all_uncontrolled_demand.loc[inds, 'l3_demand'] = all_uncontrolled_demand_l3.loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)


            total_uncontrolled_demand = pd.concat((total_uncontrolled_demand, all_uncontrolled_demand.loc[inds, ['datetime', 'total_demand', 'l1_demand', 'l2_demand', 'l3_demand']]), ignore_index=True)


        total_uncontrolled_demand.datetime = pd.to_datetime(total_uncontrolled_demand.datetime)

        #multiply by efficiency to feed grid-side power into dispatch model
        save_str = folder+'/'+folder_numevs+'/Uncontrolled/'+'demand_run'+str(run_number)
        total_uncontrolled_demand.total_demand = total_uncontrolled_demand.total_demand * (1+eta)
        total_uncontrolled_demand.l1_demand = total_uncontrolled_demand.l1_demand  * (1+eta)
        total_uncontrolled_demand.l2_demand  = total_uncontrolled_demand.l2_demand  * (1+eta)
        total_uncontrolled_demand.l3_demand  = total_uncontrolled_demand.l3_demand  * (1+eta)

        #save uncontrolled demand
        total_uncontrolled_demand.to_csv(save_str+'_'+period_string+'_'+results_date+'.csv')

        tic = time.time()
        dpdfs = {}    
        year_set = list(set([total_uncontrolled_demand.datetime.dt.year.min(), total_uncontrolled_demand.datetime.dt.year.max()]))
        for year in year_set:
            save_str = folder+'/'+folder_numevs+'/Uncontrolled/'+'results_'+str(year)+'_run'+str(run_number)+'_'+period_string

            #convert added demand to hourly in MW
            added_demand = total_uncontrolled_demand.loc[total_uncontrolled_demand.loc[total_uncontrolled_demand.datetime.dt.year==year].index, ['datetime', 'total_demand']].copy(deep=True).rename(columns={'total_demand':'demand'}).reset_index(drop=True)
            added_demand.datetime = pd.to_datetime(added_demand.datetime)
            added_demand = added_demand.resample('H', on='datetime').sum().reset_index()
            added_demand.demand = (1/60)*added_demand.demand # as part of the hourly conversion
            added_demand_mw = added_demand.copy(deep=True)
            added_demand_mw.demand = (1/1000)*added_demand_mw.demand 

            #instantiate grid model object and add demand
            grid1 = GridModel(year=str(year), gd=gds[year])
            grid1.add_demand(added_demand_mw)
            minweek = np.maximum(total_uncontrolled_demand.loc[total_uncontrolled_demand.datetime.dt.year==year].datetime.dt.week.min()-2, 0)
            maxweek = np.minimum(total_uncontrolled_demand.loc[total_uncontrolled_demand.datetime.dt.year==year].datetime.dt.week.max()+2, 51)
            print('Weeks: ', minweek, maxweek)

            #run grid dispatch
            if len(added_demand_mw) > 0:
                grid1.run_dispatch(save_str=save_str, result_date = results_date, time_array=np.arange(minweek, maxweek+1)+1)
            else:
                grid1.run_dispatch(save_str=save_str, result_date = results_date, time_array=[1])
            dpdfs[year] = grid1.dp.df.copy(deep=True)
        toc = time.time()
        print('Elapsed time: ', toc-tic)
        if len(year_set) == 1:
            dpdf_total = dpdfs[year].copy(deep=True)
        else:
            dpdf_total = pd.concat((dpdfs[2019], dpdfs[2020]), axis=0, ignore_index=True)

        #now move on to controlling EVs
        print('-----'*5) 
        print('Controlling EVs')

        for aef_type in ['varying']:
            for access_series_name in ['access_series', 'plugged_series']:
                print('-----'*5)
                print('Control type: ', aef_type, access_series_name)
                control_name = all_control_names[aef_type]
                control_folder_name = control_name+'_'+access_series_name
                if not os.path.isdir(folder+'/'+folder_numevs+'/Controlled_'+control_folder_name):
                    os.mkdir(folder+'/'+folder_numevs+'/Controlled_'+control_folder_name)
                if not os.path.isdir(folder+'/'+folder_numevs+'/Controlled_'+control_folder_name + '/demand_details'):
                    os.mkdir(folder+'/'+folder_numevs+'/Controlled_'+control_folder_name+ '/demand_details')
                mindate=total_uncontrolled_demand.datetime.dt.date.min()
                maxdate=total_uncontrolled_demand.datetime.dt.date.max()

                #run control for each week
                for week in range(data.num_weeks):
                    print('Week starting on : ', data.mondays[week])
                    tic = time.time()
                    detail_save_str = folder+'/'+folder_numevs+'/Controlled_'+control_folder_name+ '/demand_details/'+str(run_number)+'_'+period_string+'_'+results_date
                    model.run_control_oneweek(week, name=control_folder_name, access_series_name=access_series_name, 
                                              objective_type='aef', reg=1, dpdf=dpdf_total, 
                                              driver_subset=indices_df[data.mondays[week]].unique(), verbose=False, mindate=mindate, maxdate=maxdate,
                                              force_nouncontrolled=True, aef_type = aef_type)
                    toc = time.time()
                    print('Elapsed time: ', toc-tic)

                cols = model.controlled_charging_demand[control_folder_name].columns[1:]
                model.controlled_charging_demand[control_folder_name].rename(columns = {i:str(int(i)) for i in cols}, inplace = True)
                total_controlled_demand = pd.DataFrame(columns=['datetime', 'total_demand', 'l1_demand', 'l2_demand', 'l3_demand'])
                model.controlled_charging_demand[control_folder_name].datetime = pd.to_datetime(model.controlled_charging_demand[control_folder_name].datetime)

                #calculate aggregate controlled demand profile for all vehicles, split into different charging levels
                for week in range(data.num_weeks):
                    all_controlled_demand_l1 = model.controlled_charging_demand[control_folder_name].copy(deep=True)
                    all_controlled_demand_l1.iloc[:,np.arange(1, 749)] = all_controlled_demand_l1.iloc[:,np.arange(1, 749)].where(all_controlled_demand_l1.iloc[:,np.arange(1, 749)]<l1_rate/(1+eta), 0)

                    all_controlled_demand_l2 = model.controlled_charging_demand[control_folder_name].copy(deep=True)
                    all_controlled_demand_l2.iloc[:,np.arange(1, 749)] = all_controlled_demand_l2.iloc[:,np.arange(1, 749)].where((all_controlled_demand_l2.iloc[:,np.arange(1, 749)]<=(l2_rate/(1+eta)+.01)), 0)

                    all_controlled_demand_l3 = model.controlled_charging_demand[control_folder_name].copy(deep=True)
                    all_controlled_demand_l3.iloc[:,np.arange(1, 749)]=all_controlled_demand_l3.iloc[:,np.arange(1, 749)].where(all_controlled_demand_l3.iloc[:,np.arange(1, 749)]>(l2_rate/(1+eta)+.01), 0)

                    #obtain time intervals and driver frequency
                    inds = model.controlled_charging_demand[control_folder_name].loc[(model.controlled_charging_demand[control_folder_name].datetime.dt.date >= data.mondays[week])&(model.controlled_charging_demand[control_folder_name].datetime.dt.date <= data.sundays[week])].index
                    dot_values = indices_df[data.mondays[week]].value_counts().sort_index().reindex(np.arange(1, 749), fill_value=0)
                    
                    #multiply drivers by frequency
                    model.controlled_charging_demand[control_folder_name].loc[inds, 'total_demand'] = model.controlled_charging_demand[control_folder_name].loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)
                    model.controlled_charging_demand[control_folder_name].loc[inds, 'l1_demand'] = all_controlled_demand_l1.loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)
                    model.controlled_charging_demand[control_folder_name].loc[inds, 'l2_demand'] = all_controlled_demand_l2.loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)
                    model.controlled_charging_demand[control_folder_name].loc[inds, 'l3_demand'] = all_controlled_demand_l3.loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)

                    total_controlled_demand = pd.concat((total_controlled_demand, model.controlled_charging_demand[control_folder_name].loc[inds, ['datetime', 'total_demand', 'l1_demand', 'l2_demand', 'l3_demand']]), ignore_index=True)

                total_controlled_demand.datetime = pd.to_datetime(total_controlled_demand.datetime)
                save_str = folder+'/'+folder_numevs+'/Controlled_'+control_folder_name+'/'+'demand_run'+str(run_number)
                model.controlled_charging_demand[control_folder_name].to_csv(save_str+'_individualdrivers_'+period_string+'_'+results_date+'.csv')
                
                #multiply by efficiency to feed grid-side power into dispatch model
                total_controlled_demand.total_demand = total_controlled_demand.total_demand * (1+eta)
                total_controlled_demand.l1_demand = total_controlled_demand.l1_demand * (1+eta)
                total_controlled_demand.l2_demand = total_controlled_demand.l2_demand * (1+eta)
                total_controlled_demand.l3_demand = total_controlled_demand.l3_demand * (1+eta)

                #save controlled demand
                total_controlled_demand.to_csv(save_str+'_'+period_string+'_'+results_date+'.csv')

                print('-----'*5)
                print('Dispatch 2: Controlled')
                tic = time.time()
                dpdfs_controlled = {}    
                for year in year_set:
                    save_str = folder+'/'+folder_numevs+'/Controlled_'+control_folder_name+'/'+'results_'+str(year)+'_run'+str(run_number)+'_'+period_string

                    #convert added demand to hourly in MW
                    added_demand = total_controlled_demand.loc[total_controlled_demand.loc[total_controlled_demand.datetime.dt.year==year].index, ['datetime', 'total_demand']].copy(deep=True).rename(columns={'total_demand':'demand'}).reset_index(drop=True)
                    added_demand.datetime = pd.to_datetime(added_demand.datetime)
                    added_demand = added_demand.resample('H', on='datetime').sum().reset_index()
                    added_demand.demand = (1/4)*added_demand.demand # as part of the hourly conversion - controlled profile is on 15min basis
                    added_demand_mw = added_demand.copy(deep=True)
                    added_demand_mw.demand = (1/1000)*added_demand_mw.demand 

                    #instantiate grid model object and add demand
                    grid1 = GridModel(year=str(year), gd=gds[year])
                    grid1.add_demand(added_demand_mw)
                    minweek = np.maximum(total_controlled_demand.loc[total_controlled_demand.datetime.dt.year==year].datetime.dt.week.min()-2, 0)
                    maxweek = np.minimum(total_controlled_demand.loc[total_controlled_demand.datetime.dt.year==year].datetime.dt.week.max()+2, 51)
                    print('Week numbers: ', minweek,'-', maxweek)

                    #run grid dispatch
                    if len(added_demand_mw) > 0:
                        grid1.run_dispatch(save_str=save_str, result_date = results_date, time_array=np.arange(minweek, maxweek+1)+1)
                    else:
                        grid1.run_dispatch(save_str=save_str, result_date = results_date, time_array=[1])
                    dpdfs_controlled[year] = grid1.dp.df.copy(deep=True)

                toc = time.time()
                print('Elapsed time: ', toc-tic)