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

future_year = 2030

#set simulation range and get uncontrolled demand data
month = 1
min_date = datetime.date(2020, month, 1)
max_date = datetime.date(2020, month, 31)
period_string = str(min_date) + '_to_' + str(max_date)
 
all_uncontrolled_demand = pd.read_csv(save_str+'_individualdrivers_'+period_string+'.csv', index_col=0)
all_uncontrolled_demand.datetime = pd.to_datetime(all_uncontrolled_demand.datetime)


results_date = '20230912' #set simulation date for saving files
folder = folder_root + '/MEF_2030_Cascade/'
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

        #set MEF type and access level
        mef_type = 'varying'
        access_series_name ='access_series' #or 'plugged_series'
        control_name = all_control_names[mef_type]
        control_folder_name = control_name+'_'+access_series_name
        if not os.path.isdir(folder+'/'+folder_numevs+'/Controlled_'+control_folder_name):
            os.mkdir(folder+'/'+folder_numevs+'/Controlled_'+control_folder_name)
        if not os.path.isdir(folder+'/'+folder_numevs+'/Controlled_'+control_folder_name + '/demand_details'):
            os.mkdir(folder+'/'+folder_numevs+'/Controlled_'+control_folder_name+ '/demand_details')
        

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

        #Run and save uncontrolled dispatch
        tic = time.time()
        year=2020
        save_str = folder+'/'+folder_numevs+'/Uncontrolled/'+'results_'+str(year)+'_run'+str(run_number)+'_'+period_string

        #convert added demand to hourly in MW
        added_demand = total_uncontrolled_demand.loc[total_uncontrolled_demand.loc[total_uncontrolled_demand.datetime.dt.year==year].index, ['datetime', 'total_demand']].copy(deep=True).rename(columns={'total_demand':'demand'}).reset_index(drop=True)
        added_demand.datetime = pd.to_datetime(added_demand.datetime)
        added_demand = added_demand.resample('H', on='datetime').sum().reset_index()
        added_demand.demand = (1/60)*added_demand.demand # as part of the hourly conversion
        added_demand_mw = added_demand.copy(deep=True)
        added_demand_mw.demand = (1/1000)*added_demand_mw.demand 

        #instantiate grid model object and add demand
        grid_u = GridModel(year=future_year, reference_year=year, added_evs=False)
        grid_u.add_demand(added_demand_mw)
        minweek = np.maximum(total_uncontrolled_demand.loc[total_uncontrolled_demand.datetime.dt.year==year].datetime.dt.week.min()-2, 0)
        maxweek = np.minimum(total_uncontrolled_demand.loc[total_uncontrolled_demand.datetime.dt.year==year].datetime.dt.week.max()+2, 51)
        print('Weeks: ', minweek, maxweek)

        #run grid dispatch
        if len(added_demand_mw) > 0:
            grid_u.run_dispatch(save_str=save_str, result_date = results_date, time_array=np.arange(minweek, maxweek+1)+1)
        else:
            grid_u.run_dispatch(save_str=save_str, result_date = results_date, time_array=[1])
        toc = time.time()
        print('Elapsed time: ', toc-tic)

        print('-----'*5)
        print('Dispatch Loops - Uncontrolled')
        #grid dispatch for subsets of uncontrolled

        #set size of each group of EVs
        MEF_breakdown = int(n_evs_added/20)
        num_mef_signals = np.ceil(n_evs_added/MEF_breakdown)
        uncontrolled_fraction = np.arange(1,num_mef_signals+1) * MEF_breakdown / n_evs_added
        uncontrolled_fraction[-1] =1.0
        
        #set up dataframes 
        dpdfs = {}    
        year_set = list(set([total_uncontrolled_demand.datetime.dt.year.min(), total_uncontrolled_demand.datetime.dt.year.max()]))
        year=2020
        total_controlled_demand_running = pd.DataFrame(columns=['datetime', 'total_demand'])
        column_names = []
        for idx in range(data.num_weeks):
            column_names.append(str(idx))
        dot_values_frac = pd.DataFrame(columns=column_names)

        #loop through each group of EVs
        for signal_num, fraction in enumerate(uncontrolled_fraction):
            fraction_uncontrolled_demand = pd.DataFrame(columns=['datetime', 'frac_demand'])
            print('Signal Number '+ str(signal_num+1) + ' out of '+str(num_mef_signals)+' total signals')
            print('Dispatching with Controlled Fraction')
            #for each group, we dispatch with a new "added demand". This includes controlled demand from all vehicles in previous groups, as well as the uncontrolled demand from the current group.
            for week in range(data.num_weeks):
                if signal_num==0:
                    dot_values_frac[str(week)] = indices_df[data.mondays[week]][:int(fraction*n_evs_added)].value_counts().sort_index().reindex(np.arange(1, 749), fill_value=0)
                else:
                    dot_values_frac[str(week)] = indices_df[data.mondays[week]][int(uncontrolled_fraction[signal_num-1]*n_evs_added):int(fraction*n_evs_added)].value_counts().sort_index().reindex(np.arange(1, 749), fill_value=0)
                    
                inds = all_uncontrolled_demand.loc[(all_uncontrolled_demand.datetime.dt.date >= data.mondays[week])&(all_uncontrolled_demand.datetime.dt.date <= data.sundays[week])].index
                all_uncontrolled_demand.loc[inds, 'frac_demand'] = all_uncontrolled_demand.loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values_frac[str(week)].values).sum(axis=1)
                fraction_uncontrolled_demand = pd.concat((fraction_uncontrolled_demand, all_uncontrolled_demand.loc[inds, ['datetime', 'frac_demand']]), ignore_index=True)

            #convert to hourly
            fraction_uncontrolled_demand.frac_demand = fraction_uncontrolled_demand.frac_demand * (1+eta)
            added_demand = fraction_uncontrolled_demand.loc[fraction_uncontrolled_demand.loc[fraction_uncontrolled_demand.datetime.dt.year==year].index, ['datetime', 'frac_demand']].copy(deep=True).rename(columns={'frac_demand':'demand'}).reset_index(drop=True)
            added_demand.datetime = pd.to_datetime(added_demand.datetime)
            added_demand = added_demand.resample('H', on='datetime').sum().reset_index()
            added_demand.demand = (1/60)*added_demand.demand # as part of the hourly conversion
            
            #add controlled demand from previous groups
            if len(total_controlled_demand_running.total_demand) != 0:
                temp_ctrl_run = total_controlled_demand_running.copy(deep=True)
                temp_ctrl_run = temp_ctrl_run.resample('H', on='datetime').sum().reset_index()
                temp_ctrl_run.total_demand = (1/4)*temp_ctrl_run.total_demand # as part of the hourly conversion, controlled demand 15 min increments
                added_demand.demand += temp_ctrl_run.total_demand #add controlled portion (from prior loop)

            #convert to MW
            tic = time.time()
            added_demand_mw = added_demand.copy(deep=True)
            added_demand_mw.demand = (1/1000)*added_demand_mw.demand 

            #instantiate grid model object and add demand
            grid1 = GridModel(year=future_year, reference_year=year, added_evs=False)
            grid1.add_demand(added_demand_mw)
            minweek = np.maximum(total_uncontrolled_demand.loc[total_uncontrolled_demand.datetime.dt.year==year].datetime.dt.week.min()-2, 0)
            maxweek = np.minimum(total_uncontrolled_demand.loc[total_uncontrolled_demand.datetime.dt.year==year].datetime.dt.week.max()+2, 51)
            print('Weeks: ', minweek, maxweek)

            #run grid dispatch
            if len(added_demand_mw) > 0:
                grid1.run_dispatch(save_str=None, result_date = results_date, time_array=np.arange(minweek, maxweek+1)+1)
            else:
                grid1.run_dispatch(save_str=None, result_date = results_date, time_array=[1])
            dpdfs[year] = grid1.dp.df.copy(deep=True)
            toc = time.time()
            print('Elapsed time: ', toc-tic)

            if len(year_set) == 1:
                dpdf_total = dpdfs[year].copy(deep=True)
            else:
                dpdf_total = pd.concat((dpdfs[2019], dpdfs[2020]), axis=0, ignore_index=True)

            print('-----'*5)
            print('Control type: ', mef_type, access_series_name)
            mindate=total_uncontrolled_demand.datetime.dt.date.min()
            maxdate=total_uncontrolled_demand.datetime.dt.date.max()
            model = ChargingAutomation(min_date, max_date, data=data)
            
            #run controlled charging for each week
            for week in range(data.num_weeks):
                print('Week starting on : ', data.mondays[week])
                tic = time.time()
                #get driver indices for current group
                driver_set = indices_df[data.mondays[week]][int(signal_num*uncontrolled_fraction[0]*n_evs_added):int((signal_num+1)*uncontrolled_fraction[0]*n_evs_added)].unique()
                model.run_control_oneweek(week, name=control_folder_name, access_series_name=access_series_name, 
                                            objective_type='mef', reg=1, dpdf=dpdf_total, mef_type=mef_type,
                                            driver_subset=driver_set, verbose=False, mindate=mindate, maxdate=maxdate,
                                            force_nouncontrolled=True)#, mef_unique = mef_signal_df)
                toc = time.time()
                print('Elapsed time: ', toc-tic)

            cols = model.controlled_charging_demand[control_folder_name].columns[1:]
            model.controlled_charging_demand[control_folder_name].rename(columns = {i:str(int(i)) for i in cols}, inplace = True)
            total_controlled_demand_temp = pd.DataFrame(columns=['datetime', 'total_demand'])
            model.controlled_charging_demand[control_folder_name].datetime = pd.to_datetime(model.controlled_charging_demand[control_folder_name].datetime)

            #calculate aggregate controlled demand profile for all vehicles
            for week in range(data.num_weeks):
                #obtain time intervals and driver frequency
                inds = model.controlled_charging_demand[control_folder_name].loc[(model.controlled_charging_demand[control_folder_name].datetime.dt.date >= data.mondays[week])&(model.controlled_charging_demand[control_folder_name].datetime.dt.date <= data.sundays[week])].index
                dot_values = indices_df[data.mondays[week]][int(signal_num*uncontrolled_fraction[0]*n_evs_added):int((signal_num+1)*uncontrolled_fraction[0]*n_evs_added)].value_counts().sort_index().reindex(np.arange(1, 749), fill_value=0)
                
                #multiply drivers by frequency
                model.controlled_charging_demand[control_folder_name].loc[inds, 'total_demand'] = model.controlled_charging_demand[control_folder_name].loc[inds, np.arange(1, 749).astype(str)].multiply(dot_values.values).sum(axis=1)
                total_controlled_demand_temp = pd.concat((total_controlled_demand_temp, model.controlled_charging_demand[control_folder_name].loc[inds, ['datetime', 'total_demand']]), ignore_index=True)

            total_controlled_demand_temp.datetime = pd.to_datetime(total_controlled_demand_temp.datetime)
            
            #multiply by efficiency to feed grid-side power into dispatch model
            total_controlled_demand_temp.total_demand = total_controlled_demand_temp.total_demand * (1+eta)

            #add to running total
            if signal_num==0:
                total_controlled_demand_running.total_demand = total_controlled_demand_temp.total_demand
                total_controlled_demand_running.datetime = total_controlled_demand_temp.datetime
            else:
                total_controlled_demand_running.total_demand += total_controlled_demand_temp.total_demand

        save_str = folder+'/'+folder_numevs+'/Controlled_'+control_folder_name+'/'+'demand_run'+str(run_number)
        total_controlled_demand = total_controlled_demand_running
        total_controlled_demand.datetime = total_controlled_demand_temp.datetime
        total_controlled_demand.to_csv(save_str+'_'+period_string+'_'+results_date+'.csv')

        print('-----'*5)
        print('Dispatch 2 - Controlled demand dispatch with all vehicles controlled')
        tic = time.time()
        dpdfs_controlled = {}    
        for year in year_set:
            save_str = folder+'/'+folder_numevs+'/Controlled_'+control_folder_name+'/'+'results_'+str(year)+'_run'+str(run_number)+'_'+period_string

            #
            added_demand = total_controlled_demand.loc[total_controlled_demand.loc[total_controlled_demand.datetime.dt.year==year].index, ['datetime', 'total_demand']].copy(deep=True).rename(columns={'total_demand':'demand'}).reset_index(drop=True)
            added_demand.datetime = pd.to_datetime(added_demand.datetime)
            added_demand = added_demand.resample('H', on='datetime').sum().reset_index()
            added_demand.demand = (1/4)*added_demand.demand # as part of the hourly conversion - controlled profile is on 15min basis
            added_demand_mw = added_demand.copy(deep=True)
            added_demand_mw.demand = (1/1000)*added_demand_mw.demand 

            #instantiate grid model object and add demand
            grid1 = GridModel(year=future_year, reference_year=year, added_evs=False)
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