""" Grid modeling file.
There are three classes: GridModel, FutureDemand, generatorData, generatorDataShort, bidStack, and dispatch

The first two classes were developed for this project by Siobhan Powell and Sonia Martin.

The last four classes are borrowed from here: https://github.com/SiobhanPowell/speech-grid-impact. 
They are an extension of the model originally published by Thomas Deetjen and adjusted or added to by Siobhan Powell: https://github.com/tdeetjen/simple_dispatch
The following classes were written by Thomas Deetjen and adjusted or added to by Siobhan Powell.
"""

import numpy as np
import pandas as pd
import pickle
import scipy
import os.path
import pandas
import matplotlib
import copy
import os


class GridModel(object):
    """ Uses the simple dispatch model from https://github.com/tdeetjen/simple_dispatch to calculate emissions in the WECC grid."""
    
    def __init__(self, year=2020, gd=None, reference_year=2020, calculate_aef=True, co2_price = None, added_evs = True):
         
        #open correct generator data (gd) object, depending if year is in the future or not
        if gd is None:
            if year > 2020:
                self.gd = pickle.load(open('Data/generator_data_short_WECC_'+str(reference_year)+'.obj', 'rb'))
            else:
                self.gd = pickle.load(open('Data/generator_data_short_WECC_'+str(year)+'.obj', 'rb'))
        else:
            self.gd = gd
        self.year = year
        self.evs = added_evs
        if int(reference_year) == 2020:
            self.num_hours_year = 366 * 24
        elif int(reference_year) == 2019:
            self.num_hours_year = 365 * 24
        self.demand = self.gd.demand_data.copy(deep=True)[:self.num_hours_year] # only get demand data for one year if there is more in the dataframe
        

        #set CO2 price if applicable
        if co2_price is None: 
            self.co2_price = 0
        else:
            self.co2_price = co2_price
         
        #if looking at future grid:
        if int(year) > 2020:
            self.reference_year = reference_year
            self.add_generators(self.year)
            self.drop_generators(self.year)
            self.future_demand = FutureDemand(self.gd, self.year, self.reference_year)
            self.future_demand.set_up()
            self.original_demand = self.gd.demand_data.copy(deep=True)[:self.num_hours_year] # only get demand data for one year if there is more in the dataframe
            self.demand = self.future_demand.demand.copy(deep=True)

        self.bs = bidStack(self.gd, co2_dol_per_kg=self.co2_price, time=1, dropNucHydroGeo=True, include_min_output=False, mdt_weight=0.5, include_easiur=False)
            
        #if calculating AEFs:
        if int(year) > 2020: #this step is already done in the FutureDemand function
            pass
        elif calculate_aef:
            self.all_generation = pd.read_csv('Data/region_eia_generation_data_'+str(year)+'.csv', index_col=0)
            self.demand['total_incl_noncombustion'] = self.demand['demand'].values + self.all_generation['WECC_notcombustion'].values

    def change_demand(self, new_demand):
        '''Change electricity demand to a new set of values'''
        inds = self.demand.loc[self.demand.datetime.isin(new_demand.datetime)].index
        self.demand.loc[inds, 'demand'] = new_demand.demand.values
    
    def add_demand(self, added_demand):
        '''Add electricity demand'''
        inds = self.demand.loc[self.demand.datetime.isin(added_demand.datetime)].index
        self.demand.loc[inds, 'demand'] += added_demand.demand.values
        
    def check_overgeneration(self, save_str=None, change_demand=True):
        '''Check for negative demand. Clip and save overgeneration amount'''
        if self.demand['demand'].min() < 0:
            if save_str is not None:
                self.demand.loc[self.demand['demand'] < 0].to_csv(save_str+'_overgeneration.csv', index=None)
        if change_demand:
            self.demand['demand'] = self.demand['demand'].clip(0, 1e10)

    def run_dispatch(self, save_str=None, result_date='20220522', time_array=np.arange(52)+1, include_storage=False):
        '''Run the dispatch. max_penlevel indicates whether storage will be needed or whether the model will break
        without it, but the try except clause will ensure the simulation is run if that is incorrect.''' 
        self.check_overgeneration(save_str=save_str)
        self.dp = dispatch(self.bs, self.demand, time_array=time_array, return_generator_limits=False, include_storage=include_storage)
        self.dp.calcDispatchAll()
        if save_str is not None:
            self.dp.df.to_csv(save_str+'_dpdf_'+result_date+'.csv', index=False)
            
    def add_generators(self, future_year):
        '''Duplicate generators to simulate new additions in the future WECC grid'''
        self.additions_df = pd.read_csv('GridInputData/generator_additions.csv', index_col=0)
        gd_short_final = copy.deepcopy(self.gd)
        added_units = self.additions_df[self.additions_df['Year']<future_year]['orispl_unit'].values
        for i, val in enumerate(added_units):
            idx = len(gd_short_final.df)
            loc1 = gd_short_final.df[gd_short_final.df['orispl_unit']==val].index
            gd_short_final.df = pd.concat((gd_short_final.df, gd_short_final.df.loc[loc1]), ignore_index=True)
            gd_short_final.df.loc[idx, 'orispl_unit'] = 'added_'+str(i)
            
        self.gd = copy.deepcopy(gd_short_final)
        
    def drop_generators(self, future_year):
        '''Drop generators to match announced retirements in the WECC grid'''
        self.retirements_df = pd.read_csv('GridInputData/generator_retirements.csv', index_col=0)
        gd_short_final = copy.deepcopy(self.gd)
        dropped_units = self.retirements_df[self.retirements_df['retirement_year']<future_year]['orispl_unit'].values
        gd_short_final.df = gd_short_final.df[~gd_short_final.df['orispl_unit'].isin(dropped_units)].copy(deep=True).reset_index(drop=True)
        self.gd = copy.deepcopy(gd_short_final)

        
class FutureDemand(object):
    '''Calculate net demand in a future year.'''

    def __init__(self, gd_short, year=2030, renewables_year=2020):
        if int(renewables_year) == 2020:
            self.num_hours_year = 366 * 24
        elif int(renewables_year) == 2019:
            self.num_hours_year = 365 * 24
        self.baseline_demand = gd_short.demand_data.copy(deep=True)[:self.num_hours_year]# only get demand data for one year if there is more in the dataframe
        self.year = int(year)
        self.all_generation = pd.read_csv('Data/region_eia_generation_data_'+str(renewables_year)+'.csv', index_col=0)
        self.renewables_year = renewables_year
        self.baseline_demand['datetime'] = pd.to_datetime(self.baseline_demand['datetime'])
        self.baseline_demand['total_incl_noncombustion'] = self.baseline_demand['demand'].values + self.all_generation['WECC_notcombustion'].values
        self.not_combustion = pd.DataFrame({'dt': self.all_generation['dt'], 'generation': self.all_generation['WECC_notcombustion']})
        self.demand = self.baseline_demand.copy(deep=True)[:self.num_hours_year] # only get demand data for one year if there is more in the dataframe

        #find electrification scaling factor by interpolating. Currently functional for self.renewables_year = 2020
        self.electrification_scaling = {self.renewables_year: 1, 2032: 1.15}
        for year in np.arange(self.renewables_year, 2035):
            self.electrification_scaling[int(year)] = ((year - self.renewables_year) / (2032 - self.renewables_year)) * (self.electrification_scaling[2032] - 1) + 1

        #find ev scaling factor (if desired)
        self.ev_add_scaling = {2020: 0, 2035: 0.3}
        for year in np.arange(self.renewables_year, 2035):
            self.ev_add_scaling[int(year)] = ((year - self.renewables_year) / (2035 - self.renewables_year)) * self.ev_add_scaling[2035]
    
        #find various renewables scaling factors
        self.solar_multiplier = {self.renewables_year: 1, 2035: 3.64}
        self.wind_multiplier = {self.renewables_year: 1, 2035: 1.75}
        self.hydro_multiplier = {self.renewables_year: 1, 2035: 1.04}
        for year in np.arange(self.renewables_year, 2035):
            self.solar_multiplier[int(year)] = ((year - self.renewables_year) / (2035 - self.renewables_year)) * (self.solar_multiplier[2035] - 1) + 1
            self.wind_multiplier[int(year)] = ((year - self.renewables_year) / (2035 - self.renewables_year)) * (self.wind_multiplier[2035] - 1) + 1
            self.hydro_multiplier[int(year)] = ((year - self.renewables_year) / (2035 - self.renewables_year)) * (self.hydro_multiplier[2035] - 1) + 1

        #set EV load profile if adding EVs to baseline demand
        self.ev_load_weekend = None
        self.ev_load_weekend_add = None
        
    
    def set_up(self, evs=False, storage=True):
        '''Update demand, renewables, and dispatch storage'''
        self.electrification()
        self.solar()
        self.wind()
        self.hydro()
        if evs:
            self.evs()
        if storage: #if storage before dispatch
            self.storage()
        self.update_total()

    def electrification(self):
        '''Scale demand then subtract renewable generation'''
        self.demand['demand'] = self.electrification_scaling[self.year] * self.baseline_demand['total_incl_noncombustion'] - self.all_generation['WECC_notcombustion'].values

    def solar(self):
        '''Subtract the increase in solar from net demand'''
        solar_base = self.all_generation['WECC_SUN'].values
        self.not_combustion['generation'] += (self.solar_multiplier[self.year]-1) * solar_base
        self.demand['demand'] -= (self.solar_multiplier[self.year]-1) * solar_base

    def wind(self):
        '''Subtract the increase in wind from net demand'''
        wind_base = self.all_generation['WECC_WND'].values
        self.not_combustion['generation'] += (self.wind_multiplier[self.year]-1) * wind_base
        self.demand['demand'] -= (self.wind_multiplier[self.year]-1) * wind_base

    def hydro(self):
        '''Subtract the increase in solar from net demand'''
        hydro_base = self.all_generation['WECC_WAT'].values
        self.not_combustion['generation'] += (self.hydro_multiplier[self.year]-1) * hydro_base
        self.demand['demand'] -= (self.hydro_multiplier[self.year]-1) * hydro_base

    def storage(self):
        '''Calculate charge and discharge times for grid scale storage and add to demand'''
        #Calculate total storage capacity depending on year and max charge rate for 4 hour storage
        storage_cap = (24700 / (2035 - self.renewables_year)) * (self.year - self.renewables_year) + 200
        storage_max_rate = storage_cap / 4

        chg_idxs = []
        dischg_idxs = []
        for idx in range(len(self.demand.datetime)):
            #find indicies of charging time
            if self.demand.datetime[idx].hour>=11 and self.demand.datetime[idx].hour<15:
                chg_idxs.append(idx)
            #find indicies of discharging time
            if idx%24 ==0:
                dischg_idxs.append(np.argsort(self.demand.demand[idx+15:idx+35]).iloc[-1] +15 + idx)
                dischg_idxs.append(np.argsort(self.demand.demand[idx+15:idx+35]).iloc[-2] +15 + idx)
                dischg_idxs.append(np.argsort(self.demand.demand[idx+15:idx+35]).iloc[-3] +15 + idx)
                dischg_idxs.append(np.argsort(self.demand.demand[idx+15:idx+35]).iloc[-4] +15 + idx)
        
        #subtract charging power from generation, add it to demand
        self.not_combustion.loc[chg_idxs, 'generation'] -= storage_max_rate
        self.demand.loc[chg_idxs, 'demand'] += storage_max_rate 

        #add discharging power to generation, subtract it from demand
        self.not_combustion.loc[dischg_idxs, 'generation'] += storage_max_rate 
        self.demand.loc[dischg_idxs, 'demand'] -= storage_max_rate 

    def evs(self):
        '''Add baseline/uncontrolled EV demand if desired'''
        pen_level = self.ev_add_scaling[int(self.year)]

        self.evs_update_status = True
        self.ev_pen_level = pen_level

        basecase = pd.read_csv('Data/BusinessAsUsual_100p_WECC_20220313.csv', index_col=0)
        basecase_we = pd.read_csv('Data/BusinessAsUsual_100p_weekend_WECC_20220313.csv', index_col=0)                       
        basecase['Total'] = basecase.sum(axis=1)
        basecase_we['Total'] = basecase_we.sum(axis=1)
        
        # apply pen level and convert to MW
        self.ev_load_add = pen_level * (1/1000) * basecase['Total'].values # this is in 5 min intervals
        self.ev_load_weekend_add = pen_level * (1/1000) * basecase_we['Total'].values

        for i in range(365):
            if pd.to_datetime(self.demand.loc[24*i, 'datetime']).weekday() in [0, 1, 2, 3, 4]:
                self.demand.loc[24*i+np.arange(0, 24), 'demand'] += self.ev_load_add[np.arange(0, 1440, 60)]
            else:
                self.demand.loc[24*i+np.arange(0, 24), 'demand'] += self.ev_load_weekend_add[np.arange(0, 1440, 60)]

    def update_total(self):
        '''Update demand after adding electrification, renewables, EVs (if desired), and storage'''
        #total demand = net demand + renewables (not combustion) demand
        #total demand is the same, but now there is less net demand and more "not combustion" demand
        self.demand['total_incl_noncombustion'] = self.demand['demand'] + self.not_combustion['generation']


"""
The following code was originally included in https://github.com/tdeetjen/simple_dispatch and minimally modified by Siobhan Powell

# simple_dispatch
# Thomas Deetjen
# v_21
# last edited: 2019-07-02
# class "generatorData" turns CEMS, eGrid, FERC, and EIA data into a cleaned up dataframe for feeding into a "bidStack" object
# class "bidStack" creates a merit order curve from the generator fleet information created by the "generatorData" class
# class "dispatch" uses the "bidStack" object to choose which power plants should be operating during each time period to meet a demand time-series input
# ---
# v21:
# set up to simulate the 2017 historical dispatch for the NERC regional level
# v22:
# added some better functionality to allow co2, so2, and nox prices to be included in the dispatch. They were only being included in non-default calls of the bidStack.calcGenCost() function in v21. 
# fixed an error where bidStack was basing 'demand', 'f', 's', and 'a' off of 'mw' instead of ('mw' + str(self.time))
# fixed an error in bidStack.returnTotalFuelCost where [ind+1] was looking for a row in df that wasn't there. Changed this to [ind]. I'm not really sure how the model was solving before this without throwing errors about the index value not being there?
# updated the calculation of total production cost to reflect the FullMeritOrder
# fixed the calculation of full_"fuel"_mix_marg which was going above 1 and summing to more than 1 or less than 1 often
# added 'og' fuel to the 'gas' fuel_type in the generatorData class
# updated dispatch object to use returnFullTotalValue(demand, 'gen_cost_tot') for its 'gen_cost_tot' calculation
# v23:
# fixed an issue where putting in negative CO2 price could lead to a negative gen_cost which messed up the merit order (added a scipy.maximum(0.01, script here for calculating gen_cost) to the calcGenCost function).
# added an initialization variable to the bidStack object. It prevents the bidStack object from continually adding on new 0.0 dummy generators, and only does this once instead.
# added a new very large generator to the end of the merit order so that demand cannot exceed supply
# v24:
# added easiur environmental damages for each generator. These damages will be tracked through the dispatch object into the results. These damages are not included in the generation cost of the power plants, so they do not influence the power plant dispatch solution.
# fixed a minor bug that caused leap years to throw an error (changed "0" to "52" in df_orispl_unit['t'] = 52)
# now tracking gas and coal mmBtu consumption in the results
# now tracking production cost in the historic data. Production cost = mmtbu * fuel_price + mwh * vom where fuel_price and vom have the same assumptions as used in the dispatch model
# added the hist_downtime boolean to generatorData. If hist_downtime = True, this allows you to use historical data to define the weekly capacity, heat rate, etc., which lets generators be unavailable if they didn't produce any power during the historical week. This setting is useful for validation against historical data. If hist_downtime = False, then for weeks when a generator is turned off, or its capacity, heat rate, or emissions rates are outside of the 99th or 1st percentile of the historical data, then we assign the previously observed weekly value to that generator. In this way we ignore maintenance but also allow for generators that did not show up in the dispatch historically to be dispatchable in scenario analysis (e.g. if demand/prices are low, you might not see many GTs online - not because of maintenance but because of economics - but their capacity is still available for dispatch even though they didn't produce anything historically.)
# note that the dispatch goes a lot slower now. maybe 2x slower. Not exactly sure why, it could be from the addition of easiur damages and gas,coal,oil mmbtu in the output, but I wouldn't expect these to take twice as long. Future versions of this code might benefit from some rewrites to try and improve the code's efficiency and transparency.
# updated returnMarginalGenerator to be 8x faster for floats. 
# updated returnTotalCost (and other returnTotal...) to be ~90x faster (yes, 90).
# the last 2 updates combined let us generate a TRE bidStack in 6.5 seconds instead of 50 seconds = ~8x faster. It reduced TRE dispatch from ~60 minutes to ~7 minutes.
# v25:
# updated bs.returnFullTotalValue to be 15x faster using similar methods as v24. This has a tiny amount of error associated with it (less than 0.5% annually for most output variables). If we use the old returnFullTotalValue function with the new returnFullMarginalValue, the error goes to zero, but the solve time increases from 2.5 minutes to 4 minutes. The error is small enough that I will leave as is, but we can always remove it later if something in the results seems strange. Perhaps the linear interpolation is not quite accurate and there is something more nuanced going on. Or maybe I wasn't previously calculating it correctly. 
# updated bs.returnFullMarginalValue to be 5x faster using similar methods as v24. This reduced dispatch solve time by another 60%, brining the TRE dispatch down to 2.5 minutes. 
# v26:
# added an approximation of the minimum downtime constraint
# fixed an issue where year_online was being left off of many generators because the unit portion of the orispl_unit string isn't always consistent between egrid_unt and egrid_gen
# realized that WECC states had MN instead of MT and excluded NV
# fixed a problem where plants with nan fuel price data (i.e. plants in CEMS but not in EIA923) where not having their fuel prices correctly filled
# changed VOM to be a range based on generator age according to the data and discusison in the NREL Western Interconnection Study Phase 2
# made a variety of fuel price updates:
#   'rc' fuel price is now calculated as the price for 'sub' X 1.15. Refined coal has some additional processing that makes it more expensive.
#   'lig' fuel missing price data is now calculated as the national EIA923 average (there are not many LIG plants, so most regions don't have enough data to be helpful)
#   'ng' fuel missing price data is populated according to purchase type (contract, spot, or tolling) where tolling is assumed to be a bit cheaper. In this program tolling ng prices are popualated using 0.90 X spot prices
# v27:
# fixed an issue where coal that was being turned down in response to the min_downtime constraint was not contributing to the marginal generation
# updated some fuel and vom assumptions:
#   'rc' fuel price is now calculated as the price for 'sub' X 1.11. Refined coal has some additional processing that makes it more expensive.
#   'ng' fuel missing price data is populated according to purchase type (contract, spot, or tolling) where tolling prices (which EIA923 has no information on) are calculated as contract prices. Any nan prices (plants in CEMS that aren't in EIA923) are populated with spot, contract, and tolling prices
#   cleanGeneratorData now derates CHP plants according to their electricity : gross ratio
"""

import pandas
import matplotlib.pylab
import scipy
import scipy.interpolate
import datetime
import math
import copy
from bisect import bisect_left
import numpy
import cvxpy as cp


class generatorData(object):
    def __init__(self, nerc, egrid_fname, eia923_fname, ferc714_fname='', ferc714IDs_fname='', cems_folder='', easiur_fname='', include_easiur_damages=True, year=2017, fuel_commodity_prices_excel_dir='', hist_downtime = True, coal_min_downtime = 12, cems_validation_run=False, testing_single_function=False, nan_price_cems_sorting=False, tz_aware=False):
        """ 
        Translates the CEMS, eGrid, FERC, and EIA data into a dataframe for feeding into the bidStack class
        ---
        nerc : nerc region of interest (e.g. 'TRE', 'MRO', etc.)
        egrid_fname : a .xlsx file name for the eGrid generator data
        eia923_fname : filename of eia form 923
        ferc714_fname : filename of nerc form 714 hourly system lambda 
        ferc714IDs_fname : filename that matches nerc 714 respondent IDs with nerc regions
        easiur_fname : filename containing easiur damages ($/tonne) for each power plant orispl
        include_easiur_damages : if True, then easiur damages will be added to the generator data. If False, we will skip that step.
        year : year that we're looking at (e.g. 2017)
        fuel_commodity_prices_excel_dir : filename of national EIA fuel prices (in case EIA923 data is empty)
        hist_downtime : if True, will use each generator's maximum weekly mw for weekly capacity. if False, will use maximum annual mw.
        coal_min_downtime : hours that coal plants must remain off if they shut down
        cems_validation_run : if True, then we are trying to validate the output against CEMS data and will only look at power plants included in the CEMS data
        """
        #read in the data. This is a bit slow right now because it reads in more data than needed, but it is simple and straightforward
        self.nerc = nerc
        if year == 2019:
            egrid_year_str = str(19)
        elif year == 2020:
            egrid_year_str = str(20) # though they have started publishing every year, they have not published 2020 yet so we use 2019 to help with the 2020 run
        elif year >= 2014:
            egrid_year_str = str(math.floor((year / 2.0)) * 2)[2:4] #eGrid is only every other year so we have to use eGrid 2016 to help with a 2017 run, for example
        else:
            egrid_year_str = str(14) #egrid data before 2014 does not have unit level data, so use 2014. We risk missing a few generators that retired between 'year' and 2014.
        print('Reading in unit level data from eGRID...')
        self.egrid_unt = pandas.read_excel(egrid_fname, 'UNT'+egrid_year_str, skiprows=[0]) 
        print('Reading in generator level data from eGRID...')
        self.egrid_gen = pandas.read_excel(egrid_fname, 'GEN'+egrid_year_str, skiprows=[0])
        print('Reading in plant level data from eGRID...')
        self.egrid_plnt = pandas.read_excel(egrid_fname, 'PLNT'+egrid_year_str, skiprows=[0])
        print('Reading in data from EIA Form 923...')
        eia923 = pandas.read_excel(eia923_fname, 'Page 5 Fuel Receipts and Costs', skiprows=[0,1,2,3]) 
        eia923 = eia923.rename(columns={'Plant Id': 'orispl'})
        self.eia923 = eia923
        eia923_1 = pandas.read_excel(eia923_fname, 'Page 1 Generation and Fuel Data', skiprows=[0,1,2,3,4]) 
        eia923_1 = eia923_1.rename(columns={'Plant Id': 'orispl'})
        self.eia923_1 = eia923_1
        print('Reading in data from FERC Form 714...')
        self.ferc714 = pandas.read_csv(ferc714_fname)
        self.ferc714_ids = pandas.read_csv(ferc714IDs_fname)
        self.cems_folder = cems_folder
        self.fuel_commodity_prices = pandas.read_excel(fuel_commodity_prices_excel_dir, str(year))
        self.cems_validation_run = cems_validation_run
        self.hist_downtime = hist_downtime
        self.coal_min_downtime = coal_min_downtime
        self.year = year
        self.tz_aware = tz_aware
        if not testing_single_function:
            self.cleanGeneratorData()
            self.addGenVom()
            print('calc fuel prices')
            self.calcFuelPrices(nan_price_cems_sorting=nan_price_cems_sorting)
            if include_easiur_damages:
                self.easiur_per_plant = pandas.read_csv(easiur_fname)
                self.easiurDamages()
            self.addGenMinOut()
            self.addDummies()
            print('calc demand data')
            self.calcDemandData()
            self.addElecPriceToDemandData()
            self.demandTimeSeries()
            self.calcMdtCoalEvents()
        

    def cleanGeneratorData(self):
        """ 
        Converts the eGrid and CEMS data into a dataframe usable by the bidStack class.
        ---
        Creates
        self.df : has 1 row per generator unit or plant. columns describe emissions, heat rate, capacity, fuel, grid region, etc. This dataframe will be used to describe the generator fleet and merit order.
        self.df_cems : has 1 row per hour of the year per generator unit or plant. columns describe energy generated, emissions, and grid region. This dataframe will be used to describe the historical hourly demand, dispatch, and emissions
        """
        #copy in the egrid data and merge it together. In the next few lines we use the eGRID excel file to bring in unit level data for fuel consumption and emissions, generator level data for capacity and generation, and plant level data for fuel type and grid region. Then we compile it together to get an initial selection of data that defines each generator.
        print('Cleaning eGRID Data...')
        #unit-level data
        df = self.egrid_unt.copy(deep=True)
        #rename columns
        df = df[['PNAME', 'ORISPL', 'UNITID', 'PRMVR', 'FUELU1', 'HTIAN', 'NOXAN', 'SO2AN', 'CO2AN', 'HRSOP']]
        df.columns = ['gen', 'orispl', 'unit', 'prime_mover', 'fuel', 'mmbtu_ann', 'nox_ann', 'so2_ann', 'co2_ann', 'hours_on']
        df['orispl_unit'] = df.orispl.map(str) + '_' + df.unit.map(str) #orispl_unit is a unique tag for each generator unit
        #plant-level data: contains fuel, fuel_type, balancing authority, nerc region, and egrid subregion data
        df_plnt = self.egrid_plnt.copy(deep=True)
        #drop nan fuel
        for i in df[df.fuel.isna()].index:
            df.loc[i, 'fuel'] = df_plnt.loc[df_plnt[df_plnt['ORISPL']==df.loc[i, 'orispl']].index, 'PLPRMFL'].values[0]
        df = df[~df.fuel.isna()]
        #gen-level data: contains MW capacity and MWh annual generation data
        df_gen = self.egrid_gen.copy(deep=True) 
        df_gen['orispl_unit'] = df_gen['ORISPL'].map(str) + '_' + df_gen['GENID'].map(str) #orispl_unit is a unique tag for each generator unit
        df_gen_long = df_gen[['ORISPL', 'NAMEPCAP', 'GENNTAN', 'GENYRONL', 'orispl_unit', 'PRMVR', 'FUELG1']].copy()
        df_gen_long.columns = ['orispl', 'mw', 'mwh_ann', 'year_online', 'orispl_unit', 'prime_mover', 'fuel']
        df_gen = df_gen[['NAMEPCAP', 'GENNTAN', 'GENYRONL', 'orispl_unit']]
        df_gen.columns = ['mw', 'mwh_ann', 'year_online', 'orispl_unit']
        #fuel
        df_plnt_fuel = df_plnt[['PLPRMFL', 'PLFUELCT']]
        df_plnt_fuel = df_plnt_fuel.drop_duplicates('PLPRMFL')
        df_plnt_fuel.PLFUELCT = df_plnt_fuel.PLFUELCT.str.lower()
        df_plnt_fuel.columns = ['fuel', 'fuel_type']
        #geography
        df_plnt = df_plnt[['ORISPL', 'BACODE', 'NERC', 'SUBRGN']]
        df_plnt.columns = ['orispl', 'ba', 'nerc', 'egrid']
        #merge these egrid data together at the unit-level
        df = df.merge(df_gen, how='left', on='orispl_unit')  
        df = df.merge(df_plnt, how='left', on='orispl')  
        df = df.merge(df_plnt_fuel, how='left', on='fuel')  
        #keep only the units in the nerc region we're analyzing
        df = df[df.nerc == self.nerc]
        #calculate the emissions rates
        df['co2'] = scipy.divide(df.co2_ann,df.mwh_ann) * 907.185 #tons to kg
        df['so2'] = scipy.divide(df.so2_ann,df.mwh_ann) * 907.185 #tons to kg
        df['nox'] = scipy.divide(df.nox_ann,df.mwh_ann) * 907.185 #tons to kg
        #for empty years, look at orispl in egrid_gen instead of orispl_unit 
        df.loc[df.year_online.isna(), 'year_online'] = df[df.year_online.isna()].loc[:, ['orispl', 'prime_mover', 'fuel']].merge(df_gen_long[['orispl', 'prime_mover', 'fuel', 'year_online']].groupby(['orispl', 'prime_mover', 'fuel'], as_index=False).agg('mean'), on=['orispl', 'prime_mover', 'fuel'])['year_online']
        #for any remaining empty years, assume self.year (i.e. that they are brand new)
        df.loc[df.year_online.isna(), 'year_online'] = scipy.zeros_like(df.loc[df.year_online.isna(), 'year_online']) + self.year
        ###
        #now sort through and compile CEMS data. The goal is to use CEMS data to characterize each generator unit. So if CEMS has enough information to describe a generator unit we will over-write the eGRID data. If not, we will use the eGRID data instead. (CEMS data is expected to be more accurate because it has actual hourly performance of the generator units that we can use to calculate their operational characteristics. eGRID is reported on an annual basis and might be averaged out in different ways than we would prefer.)
        print('Compiling CEMS data...')
        #dictionary of which states are in which nerc region (b/c CEMS file downloads have the state in the filename)
        states = {'FRCC': ['fl'], 
                  'WECC': ['ca','or','wa', 'nv','mt','id','wy','ut','co','az','nm','tx'],
                  'SPP' : ['nm','ks','tx','ok','la','ar','mo'],
                  'RFC' : ['wi','mi','il','in','oh','ky','wv','va','md','pa','nj'],
                  'NPCC' : ['ny','ct','de','ri','ma','vt','nh','me'],
                  'SERC' : ['mo','ar','tx','la','ms','tn','ky','il','va','al','fl','ga','sc','nc'],
                  'MRO': ['ia','il','mi','mn','mo','mt','nd','ne','sd','wi','wy'], 
                  'TRE': ['ok','tx']}
        tz_mapping = {'ca':0,'or':0,'wa':0, 'nv':0,'mt':1,'id':1,'wy':1,'ut':1,'co':1,'az':1,'nm':1,'tx':1}
        #compile the different months of CEMS files into one dataframe, df_cems. (CEMS data is downloaded by state and by month, so compiling a year of data for ERCOT / TRE, for example, requires reading in 12 Texas .csv files and 12 Oklahoma .csv files)   
        df_cems = pandas.DataFrame()
        for s in states[self.nerc]:
            for m in ['01','02','03','04','05','06','07','08','09','10','11', '12']:
                print(s + ': ' + m)
                df_cems_add = pandas.read_csv(self.cems_folder + '/%s/%s%s%s.csv'%(str(self.year),str(self.year),s,m))
                df_cems_add = df_cems_add[['ORISPL_CODE', 'UNITID', 'OP_DATE','OP_HOUR','GLOAD (MW)', 'SO2_MASS (lbs)', 'NOX_MASS (lbs)', 'CO2_MASS (tons)', 'HEAT_INPUT (mmBtu)']].dropna()
                df_cems_add.columns=['orispl', 'unit', 'date','hour','mwh', 'so2_tot', 'nox_tot', 'co2_tot', 'mmbtu']
                
                if self.tz_aware:
                    if tz_mapping[s] == 1: # shift onto California Time
                        tmp = df_cems_add.copy(deep=True)
                        tmp['date'] = pandas.to_datetime(tmp['date'])
                        inds1 = tmp.loc[tmp['hour'].isin(numpy.arange(1, 24))].index
                        inds2 = tmp.loc[tmp['hour']==0].index
                        tmp2 = tmp.loc[(tmp['hour']==23)&(tmp['date']==datetime.datetime(self.year, 12, 31))].copy(deep=True)
                        tmp.loc[inds1, 'hour'] -= 1 # it is one hour earlier in California
                        tmp.loc[inds2, 'date'] -= datetime.timedelta(days=1)
                        tmp.loc[inds2, 'hour'] = 23
                        tmp = pandas.concat((tmp, tmp2), ignore_index=True) # fill in first hour of following year with last hour of this year
                        df_cems_add = tmp.loc[tmp['date'].dt.year == self.year].copy(deep=True)
                        df_cems_add['date'] = df_cems_add['date'].dt.strftime("%m-%d-%Y")
                
                df_cems = pandas.concat([df_cems, df_cems_add])
        #create the 'orispl_unit' column, which combines orispl and unit into a unique tag for each generation unit
        df_cems['orispl_unit'] = df_cems['orispl'].map(str) + '_' + df_cems['unit'].map(str)
        #bring in geography data and only keep generators within self.nerc
        df_cems = df_cems.merge(df_plnt, how='left', on='orispl')  
#         df_cems = df_cems.merge(df_plnt, left_index=True, how='left', on='orispl')  
        df_cems = df_cems[df_cems['nerc']==self.nerc] 
        #convert emissions to kg
        df_cems.co2_tot = df_cems.co2_tot * 907.185 #tons to kg
        df_cems.so2_tot = df_cems.so2_tot * 0.454 #lbs to kg
        df_cems.nox_tot = df_cems.nox_tot * 0.454 #lbs to kg
        #calculate the hourly heat and emissions rates. Later we will take the medians over each week to define the generators weekly heat and emissions rates.
        df_cems['heat_rate'] = df_cems.mmbtu / df_cems.mwh
        df_cems['co2'] = df_cems.co2_tot / df_cems.mwh
        df_cems['so2'] = df_cems.so2_tot / df_cems.mwh
        df_cems['nox'] = df_cems.nox_tot / df_cems.mwh
        df_cems.replace([scipy.inf, -scipy.inf], scipy.nan, inplace=True) #don't want inf messing up median calculations
        #drop any bogus data. For example, the smallest mmbtu we would expect to see is 25MW(smallest unit) * 0.4(smallest minimum output) * 6.0 (smallest heat rate) = 60 mmbtu. Any entries with less than 60 mmbtu fuel or less than 6.0 heat rate, let's get rid of that row of data.
        df_cems = df_cems[(df_cems.heat_rate >= 6.0) & (df_cems.mmbtu >= 60)]
        #calculate emissions rates and heat rate for each week and each generator
        #rather than parsing the dates (which takes forever because this is such a big dataframe) we can create month and day columns for slicing the data based on time of year
        df_orispl_unit = df_cems.copy(deep=True)
        df_orispl_unit.date = df_orispl_unit.date.str.replace('/','-')
        temp = pandas.DataFrame(df_orispl_unit.date.str.split('-').tolist(), columns=['month', 'day', 'year'], index=df_orispl_unit.index)
        print('Start CEMS cleaning')
        idxs_replace = np.where(temp.day=='Dec')[0]
        for row in temp.index[idxs_replace]:
            if row%1000 == 0:
                print(row)
            temp.day[row] = df_orispl_unit.date[row][:df_orispl_unit.date[row].index('-')]
            temp.month[row] = '12'
            temp.year[row] = '2020'

        print('End CEMS cleaning')
        temp=temp.astype(float)
        df_orispl_unit['monthday'] = temp.year*10000 + temp.month*100 + temp.day
        ###
        #loop through the weeks, slice the data, and find the average heat rates and emissions rates
        #first, add a column 't' that says which week of the simulation we are in
        df_orispl_unit['t'] = 52
        for t in scipy.arange(52)+1:
            start = (datetime.datetime.strptime(str(self.year) + '-01-01', '%Y-%m-%d') + datetime.timedelta(days=7.05*(t-1)-1)).strftime('%Y-%m-%d') 
            end = (datetime.datetime.strptime(str(self.year) + '-01-01', '%Y-%m-%d') + datetime.timedelta(days=7.05*(t)-1)).strftime('%Y-%m-%d') 
            start_monthday = float(start[0:4])*10000 + float(start[5:7])*100 + float(start[8:])
            end_monthday = float(end[0:4])*10000 + float(end[5:7])*100 + float(end[8:])
            #slice the data for the days corresponding to the time series period, t
            df_orispl_unit.loc[(df_orispl_unit.monthday >= start_monthday) & (df_orispl_unit.monthday < end_monthday), 't'] = t          
        #remove outlier emissions and heat rates. These happen at hours where a generator's output is very low (e.g. less than 10 MWh). To remove these, we will remove any datapoints where mwh < 10.0 and heat_rate < 30.0 (0.5% percentiles of the 2014 TRE data).
        df_orispl_unit = df_orispl_unit[(df_orispl_unit.mwh >= 10.0) & (df_orispl_unit.heat_rate <= 30.0)]    
        #aggregate by orispl_unit and t to get the heat rate, emissions rates, and capacity for each unit at each t
        temp_2 = df_orispl_unit.groupby(['orispl_unit', 't'], as_index=False)[['heat_rate', 'co2', 'so2', 'nox']].agg('median')[['orispl_unit', 't', 'heat_rate', 'co2', 'so2', 'nox']].copy(deep=True)
        temp_2['mw'] = df_orispl_unit.groupby(['orispl_unit', 't'], as_index=False)['mwh'].agg('max')['mwh'].copy(deep=True)
        #condense df_orispl_unit down to where we just have 1 row for each unique orispl_unit
        df_orispl_unit = df_orispl_unit.groupby('orispl_unit', as_index=False)[['orispl', 'ba', 'nerc', 'egrid', 'mwh']].agg('max')[['orispl_unit', 'orispl', 'ba', 'nerc', 'egrid', 'mwh']]
        df_orispl_unit.rename(columns={'mwh':'mw'}, inplace=True)
        for c in ['heat_rate', 'co2', 'so2', 'nox', 'mw']:
            temp_3 = temp_2.set_index(['orispl_unit', 't'])[c].unstack().reset_index()
            temp_3.columns = list(['orispl_unit']) + ([c + str(a) for a in scipy.arange(52)+1])
            if not self.hist_downtime:
                #remove any outlier values in the 1st or 99th percentiles
                max_array = temp_3.copy().drop(columns='orispl_unit').quantile(0.99, axis=1) 
                min_array = temp_3.copy().drop(columns='orispl_unit').quantile(0.01, axis=1)
                median_array = temp_3.copy().drop(columns='orispl_unit').median(axis=1)
                for i in temp_3.index: 
                    test = temp_3.drop(columns='orispl_unit').iloc[i]
                    test[test > max_array[i]] = scipy.NaN
                    test[test < min_array[i]] = scipy.NaN
                    test = list(test) #had a hard time putting the results back into temp_3 without using a list
                    #if the first entry in test is nan, we want to fill that with the median value so that we can use ffill later
                    if math.isnan(test[0]):
                        test[0] = median_array[i]
                    test.insert(0, temp_3.iloc[i].orispl_unit)
                    temp_3.iloc[i] = test
            #for any nan values (assuming these are offline generators without any output data), fill nans with a large heat_rate that will move the generator towards the end of the merit order and large-ish emissions rate, so if the generator is dispatched in the model it will jack up prices but emissions won't be heavily affected (note, previously I just replaced all nans with 99999, but I was concerned that this might lead to a few hours of the year with extremely high emissions numbers that threw off the data)
            M = float(scipy.where(c=='heat_rate', 50.0, scipy.where(c=='co2', 1500.0, scipy.where(c=='so2', 4.0, scipy.where(c=='nox', 3.0, scipy.where(c=='mw', 0.0, 99.0)))))) #M here defines the heat rate and emissions data we will give to generators that were not online in the historical data
            #if we are using hist_downtime, then replace scipy.NaN with M. That way offline generators can still be dispatched, but they will have high cost and high emissions.
            if self.hist_downtime:
                temp_3 = temp_3.fillna(M)
            #if we are not using hist_downtime, then use ffill to populate the scipy.NaN values. This allows us to use the last observed value for the generator to populate data that we don't have for it. For example, if generator G had a heat rate of 8.5 during time t-1, but we don't have data for time t, then we assume that generator G has a heat rate of 8.5 for t. When we do this, we can begin to include generators that might be available for dispatch but were not turned on because prices were too low. However, we also remove any chance of capturing legitimate maintenance downtime that would impact the historical data. So, for validation purposes, we probably want to have hist_downtime = True. For future scenario analysis, we probably want to have hist_downtime = False.
            if not self.hist_downtime:
                temp_3 = temp_3.fillna(method='ffill', axis=1) # was erroroneously filling down columns in this version of python/pandas
#                 temp_3.iloc[0] = temp_3.iloc[0].fillna(method='ffill') #for some reason the first row was not doing fillna(ffill)
            #merge temp_3 with df_orispl_unit. Now we have weekly heat rates, emissions rates, and capacities for each generator. These values depend on whether we are including hist_downtime
            df_orispl_unit = df_orispl_unit.merge(temp_3, on='orispl_unit', how='left')
        #merge df_orispl_unit into df. Now we have a dataframe with weekly heat rate and emissions rates for any plants in CEMS with that data. There will be some nan values in df for those weekly columns (e.g. 'heat_rate1', 'co223', etc. that we will want to fill with annual averages from eGrid for now
        orispl_units_egrid = df.orispl_unit.unique()
        orispl_units_cems = df_orispl_unit.orispl_unit.unique()
        df_leftovers = df[df.orispl_unit.isin(scipy.setdiff1d(orispl_units_egrid, orispl_units_cems))]
        #if we're doing a cems validation run, we only want to include generators that are in the CEMS data
        if self.cems_validation_run:
            df_leftovers = df_leftovers[df_leftovers.orispl_unit.isin(orispl_units_cems)]
        #remove any outliers - fuel is solar, wind, waste-heat, purchased steam, or other, less than 25MW capacity, less than 88 operating hours (1% CF), mw = nan, mmbtu = nan
        df_leftovers = df_leftovers[(df_leftovers.fuel!='SUN') & (df_leftovers.fuel!='WND') & (df_leftovers.fuel!='WH') & (df_leftovers.fuel!='OTH') & (df_leftovers.fuel!='PUR') & (df_leftovers.mw >=25) & (df_leftovers.hours_on >=88) & (~df_leftovers.mw.isna()) & (~df_leftovers.mmbtu_ann.isna())]
        #remove any outliers that have 0 emissions (except for nuclear)
        df_leftovers = df_leftovers[~((df_leftovers.fuel!='NUC') & (df_leftovers.nox_ann.isna()))]
        df_leftovers['cf'] = df_leftovers.mwh_ann / (df_leftovers.mw *8760.)
        #drop anything with capacity factor less than 1%
        df_leftovers = df_leftovers[df_leftovers.cf >= 0.01]
        df_leftovers.fillna(0.0, inplace=True)
        df_leftovers['heat_rate'] = df_leftovers.mmbtu_ann / df_leftovers.mwh_ann
        #add in the weekly time columns for heat rate and emissions rates. In this case we will just apply the annual average to each column, but we still need those columns to be able to concatenate back with df_orispl_unit and have our complete set of generator data
        for e in ['heat_rate', 'co2', 'so2', 'nox', 'mw']:
            for t in scipy.arange(52)+1:
                if e == 'mw':
                    if self.hist_downtime:
                        df_leftovers[e + str(t)] = df_leftovers[e]
                    if not self.hist_downtime:
                        df_leftovers[e + str(t)] = df_leftovers[e].quantile(0.99)
                else: 
                    df_leftovers[e + str(t)] = df_leftovers[e]
        df_leftovers.drop(columns = ['gen', 'unit', 'prime_mover', 'fuel', 'mmbtu_ann', 'nox_ann', 'so2_ann', 'co2_ann', 'mwh_ann', 'fuel_type', 'co2', 'so2', 'nox', 'cf', 'heat_rate', 'hours_on', 'year_online'], inplace=True)   
        #concat df_leftovers and df_orispl_unit
        df_orispl_unit = pandas.concat([df_orispl_unit, df_leftovers])     
        #use df to get prime_mover, fuel, and fuel_type for each orispl_unit
        df_fuel = df[df.orispl_unit.isin(df_orispl_unit.orispl_unit.unique())][['orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'year_online']]
        df_fuel.fuel = df_fuel.fuel.str.lower()
        df_fuel.fuel_type = df_fuel.fuel_type.str.lower()
        df_fuel.prime_mover = df_fuel.prime_mover.str.lower()
        df_orispl_unit = df_orispl_unit.merge(df_fuel, on='orispl_unit', how='left')
        #if we are using, for example, 2017 CEMS and 2016 eGrid, there may be some powerplants without fuel, fuel_type, prime_mover, and year_online data. Lets assume 'ng', 'gas', 'ct', and 2017 for these units based on trends on what was built in 2017
        df_orispl_unit.loc[df_orispl_unit.fuel.isna(), ['fuel', 'fuel_type']] = ['ng', 'gas']
        df_orispl_unit.loc[df_orispl_unit.prime_mover.isna(), 'prime_mover'] = 'ct'
        df_orispl_unit.loc[df_orispl_unit.year_online.isna(), 'year_online'] = 2017
        #also change 'og' to fuel_type 'gas' instead of 'ofsl' (other fossil fuel)
        df_orispl_unit.loc[df_orispl_unit.fuel=='og', ['fuel', 'fuel_type']] = ['og', 'gas']
        df_orispl_unit.fillna(0.0, inplace=True)
        #add in some columns to aid in calculating the fuel mix
        for f_type in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
            df_orispl_unit['is_'+f_type.lower()] = (df_orispl_unit.fuel_type==f_type).astype(int)
        ###
        #derate any CHP units according to their ratio of electric fuel consumption : total fuel consumption
        #now use EIA Form 923 to flag any CHP units and calculate their ratio of total fuel : fuel used for electricity. We will use those ratios to de-rate the mw and emissions of any generators that have a CHP-flagged orispl
        #calculate the elec_ratio that is used for CHP derating
        chp_derate_df = self.eia923_1.copy(deep=True)
        chp_derate_df = chp_derate_df[(chp_derate_df.orispl.isin(df_orispl_unit.orispl)) & (chp_derate_df['Combined Heat And\nPower Plant']=='Y')].replace('.', 0.0)
        chp_derate_df = chp_derate_df[['orispl', 'Reported\nFuel Type Code', 'Elec_Quantity\nJanuary', 'Elec_Quantity\nFebruary', 'Elec_Quantity\nMarch', 'Elec_Quantity\nApril', 'Elec_Quantity\nMay', 'Elec_Quantity\nJune', 'Elec_Quantity\nJuly', 'Elec_Quantity\nAugust', 'Elec_Quantity\nSeptember', 'Elec_Quantity\nOctober', 'Elec_Quantity\nNovember', 'Elec_Quantity\nDecember', 'Quantity\nJanuary', 'Quantity\nFebruary', 'Quantity\nMarch', 'Quantity\nApril', 'Quantity\nMay', 'Quantity\nJune', 'Quantity\nJuly', 'Quantity\nAugust', 'Quantity\nSeptember', 'Quantity\nOctober', 'Quantity\nNovember', 'Quantity\nDecember']].groupby(['orispl', 'Reported\nFuel Type Code'], as_index=False).agg('sum')
        chp_derate_df['elec_ratio'] = (chp_derate_df[['Elec_Quantity\nJanuary', 'Elec_Quantity\nFebruary', 'Elec_Quantity\nMarch', 'Elec_Quantity\nApril', 'Elec_Quantity\nMay', 'Elec_Quantity\nJune', 'Elec_Quantity\nJuly', 'Elec_Quantity\nAugust', 'Elec_Quantity\nSeptember', 'Elec_Quantity\nOctober', 'Elec_Quantity\nNovember', 'Elec_Quantity\nDecember']].sum(axis=1) / chp_derate_df[['Quantity\nJanuary', 'Quantity\nFebruary', 'Quantity\nMarch', 'Quantity\nApril', 'Quantity\nMay', 'Quantity\nJune', 'Quantity\nJuly', 'Quantity\nAugust', 'Quantity\nSeptember', 'Quantity\nOctober', 'Quantity\nNovember', 'Quantity\nDecember']].sum(axis=1))
        chp_derate_df = chp_derate_df[['orispl', 'Reported\nFuel Type Code', 'elec_ratio']].dropna()
        chp_derate_df.columns = ['orispl', 'fuel', 'elec_ratio']    
        chp_derate_df.fuel = chp_derate_df.fuel.str.lower()
        mw_cols = ['mw','mw1','mw2','mw3','mw4','mw5','mw6','mw7','mw8','mw9','mw10','mw11','mw12','mw13','mw14','mw15','mw16','mw17','mw18','mw19','mw20','mw21','mw22','mw23','mw24','mw25','mw26','mw27','mw28','mw29','mw30','mw31','mw32','mw33','mw34','mw35','mw36','mw37','mw38','mw39','mw40','mw41','mw42','mw43','mw44','mw45','mw46','mw47','mw48','mw49','mw50', 'mw51', 'mw52']
        chp_derate_df = df_orispl_unit.merge(chp_derate_df, how='right', on=['orispl', 'fuel'])[mw_cols + ['orispl', 'fuel', 'elec_ratio', 'orispl_unit']]
        chp_derate_df[mw_cols] = chp_derate_df[mw_cols].multiply(chp_derate_df.elec_ratio, axis='index')
        chp_derate_df.dropna(inplace=True)
        #merge updated mw columns back into df_orispl_unit
        #update the chp_derate_df index to match df_orispl_unit
        chp_derate_df.index = df_orispl_unit[df_orispl_unit.orispl_unit.isin(chp_derate_df.orispl_unit)].index       
        df_orispl_unit.update(chp_derate_df[mw_cols])
        #replace the global dataframes
        self.df_cems = df_cems
        self.df = df_orispl_unit


    def calcFuelPrices(self, nan_price_cems_sorting=False):
        """ 
        let RC be a high-ish price (1.1 * SUB)
        let LIG be based on national averages
        let NG be based on purchase type, where T takes C purchase type values and nan takes C, S, & T purchase type values
        ---
        Adds one column for each week of the year to self.df that contain fuel prices for each generation unit
        """   
        #we use eia923, where generators report their fuel purchases
        df = self.eia923.copy(deep=True)
        df = df[['YEAR','MONTH','orispl','ENERGY_SOURCE','FUEL_GROUP','QUANTITY','FUEL_COST', 'Purchase Type']]
        df.columns = ['year', 'month', 'orispl' , 'fuel', 'fuel_type', 'quantity', 'fuel_price', 'purchase_type']
        df.fuel = df.fuel.str.lower()       
        #clean up prices
        df.loc[df.fuel_price=='.', 'fuel_price'] = scipy.nan
        df.fuel_price = df.fuel_price.astype('float')/100.
        df = df.reset_index()    
        #find unique monthly prices per orispl and fuel type
        #create empty dataframe to hold the results
        df3 = self.df_cems.copy(deep=True)
        df2 = self.df.copy(deep=True)[['fuel','orispl','orispl_unit']]
        orispl_prices = pandas.DataFrame(columns=['orispl_unit', 'orispl', 'fuel', 1,2,3,4,5,6,7,8,9,10,11,12, 'quantity'])
        orispl_prices[['orispl_unit','orispl','fuel']] = df2[['orispl_unit', 'orispl', 'fuel']]
        #populate the results by looping through the orispl_units to see if they have EIA923 fuel price data
        for o_u in orispl_prices.orispl_unit.unique():
            #grab 'fuel' and 'orispl'
            f = orispl_prices.loc[orispl_prices.orispl_unit==o_u].iloc[0]['fuel']
            o = orispl_prices.loc[orispl_prices.orispl_unit==o_u].iloc[0]['orispl']
            #find the weighted average monthly fuel price matching 'f' and 'o'
            temp = df[(df.orispl==o) & (df.fuel==f)][['month', 'quantity', 'fuel_price']]
            if len(temp) != 0:
                temp['weighted'] = scipy.multiply(temp.quantity, temp.fuel_price)
                temp = temp.groupby(['month'], as_index=False).sum()[['month', 'quantity', 'weighted']]
                temp['fuel_price'] = scipy.divide(temp.weighted, temp.quantity)
                temp_prices = pandas.DataFrame({'month': scipy.arange(12)+1})
                temp_prices = temp_prices.merge(temp[['month', 'fuel_price']], on='month', how='left')
                temp_prices.loc[temp_prices.fuel_price.isna(), 'fuel_price'] = temp_prices.fuel_price.median()
                #add the monthly fuel prices into orispl_prices
                orispl_prices.loc[orispl_prices.orispl_unit==o_u, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel'])] = scipy.append(scipy.array(temp_prices.fuel_price),temp.quantity.sum())
        
        #add in additional purchasing information for slicing that we can remove later on
        orispl_prices = orispl_prices.merge(df[['orispl' , 'fuel', 'purchase_type']].drop_duplicates(subset=['orispl', 'fuel'], keep='first'), on=['orispl', 'fuel'], how='left')           
                
        #for any fuels that we have non-zero region level EIA923 data, apply those monthly fuel price profiles to other generators with the same fuel type but that do not have EIA923 fuel price data
        f_iter = list(orispl_prices[orispl_prices[1] != 0].dropna().fuel.unique())
        if 'rc' in orispl_prices.fuel.unique():
            f_iter.append('rc')
        for f in f_iter:
            orispl_prices_filled = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices[1] != 0.0)].dropna().drop_duplicates(subset='orispl', keep='first').sort_values('quantity', ascending=0)
            #orispl_prices_empty = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices[1].isna())]
            orispl_prices_empty = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices[1]==0)].dropna(subset=['quantity']) #plants with some EIA923 data but no prices
            orispl_prices_nan = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices['quantity'].isna())] #plants with no EIA923 data
            multiplier = 1.00

            # added: if sorted orispl giving prices by quantity, can also sort orsipl receiving prices by size using cems data
            if nan_price_cems_sorting:
                cems_totals = df3[df3['orispl_unit'].isin(orispl_prices_nan['orispl_unit'].values)].groupby('orispl_unit').agg('sum')
                orispl_prices_nan['cems_total'] = scipy.nan
                for i in orispl_prices_nan.index:
                    orispl_prices_nan.loc[i, 'cems_total'] = cems_totals.loc[orispl_prices_nan.loc[i, 'orispl_unit'], 'mwh']
                orispl_prices_nan = orispl_prices_nan.sort_values(by='cems_total', ascending=False).reset_index(drop=True)


        #if lignite, use the national fuel-quantity-weighted median
            if f == 'lig':
                #grab the 5th - 95th percentile prices
                temp = df[(df.fuel==f) & (df.fuel_price.notna())][['month', 'quantity', 'fuel_price', 'purchase_type']]
                temp = temp[(temp.fuel_price >= temp.fuel_price.quantile(0.05)) & (temp.fuel_price <= temp.fuel_price.quantile(0.95))]
                #weight the remaining prices according to quantity purchased
                temp['weighted'] = scipy.multiply(temp.quantity, temp.fuel_price)
                temp = temp.groupby(['month'], as_index=False).sum()[['month', 'quantity', 'weighted']]
                temp['fuel_price'] = scipy.divide(temp.weighted, temp.quantity)
                #build a dataframe that we can insert into orispl_prices
                temp_prices = pandas.DataFrame({'month': scipy.arange(12)+1})
                temp_prices = temp_prices.merge(temp[['month', 'fuel_price']], on='month', how='left')
                temp_prices.loc[temp_prices.fuel_price.isna(), 'fuel_price'] = temp_prices.fuel_price.median()
                #update orispl_prices for any units in orispl_prices_empty or orispl_prices_nan
                orispl_prices.loc[(orispl_prices.fuel==f) & ((orispl_prices.orispl.isin(orispl_prices_empty.orispl)) | (orispl_prices.orispl.isin(orispl_prices_nan.orispl))), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'purchase_type'])] = scipy.append(scipy.array(temp_prices.fuel_price),temp.quantity.sum())
        
        #if natural gas, sort by supplier type (contract, tolling, spot, or other)
            elif f =='ng':        
                orispl_prices_filled_0 = orispl_prices_filled.copy()
                orispl_prices_empty_0 = orispl_prices_empty.copy()
                #loop through the different purchase types and update any empties
                for pt in ['T', 'S', 'C']:  
                    orispl_prices_filled = orispl_prices_filled_0[orispl_prices_filled_0.purchase_type==pt]
                    orispl_prices_empty = orispl_prices_empty_0[orispl_prices_empty_0.purchase_type==pt]
                    multiplier = 1.00
                    #if pt == tolling prices, use a cheaper form of spot prices
                    if pt == 'T':
                        orispl_prices_filled = orispl_prices_filled_0[orispl_prices_filled_0.purchase_type=='S']
                        multiplier = 0.90
                    #of the plants with EIA923 data that we are assigning to plants without eia923 data, we will use the plant with the highest energy production first, assigning its fuel price profile to one of the generators that does not have EIA923 data. We will move on to plant with the next highest energy production and so on, uniformly distributing the available EIA923 fuel price profiles to generators without fuel price data
                    loop = 0
                    loop_len = len(orispl_prices_filled) - 1
                    for o in orispl_prices_empty.orispl.unique():
                        orispl_prices.loc[(orispl_prices.orispl==o) & (orispl_prices.fuel==f), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])] = scipy.array(orispl_prices_filled[orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])].iloc[loop]) * multiplier
                        #keep looping through the generators with eia923 price data until we have used all of their fuel price profiles, then start again from the beginning of the loop with the plant with the highest energy production
                        if loop < loop_len:
                            loop += 1
                        else:
                            loop = 0                
                #for nan prices (those without any EIA923 information) use Spot, Contract, and Tolling Prices (i.e. all of the non-nan prices) 
                #update orispl_prices_filled to include the updated empty prices
                orispl_prices_filled_new = orispl_prices[(orispl_prices.fuel==f) & (orispl_prices[1] != 0.0)].dropna().drop_duplicates(subset='orispl', keep='first').sort_values('quantity', ascending=0)

                #loop through the filled prices and use them for nan prices
                loop = 0
                loop_len = len(orispl_prices_filled_new) - 1
                for o in orispl_prices_nan.orispl.unique():
                    orispl_prices.loc[(orispl_prices.orispl==o) & (orispl_prices.fuel==f), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])] = scipy.array(orispl_prices_filled_new[orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])].iloc[loop])
                    #keep looping through the generators with eia923 price data until we have used all of their fuel price profiles, then start again from the beginning of the loop with the plant with the highest energy production
                    if loop < loop_len:
                        loop += 1
                    else:
                        loop = 0     
            #otherwise            
            else:
                multiplier = 1.00
                #if refined coal, use subbit prices * 1.15
                if f =='rc':
                    orispl_prices_filled = orispl_prices[(orispl_prices.fuel=='sub') & (orispl_prices[1] != 0.0)].dropna().drop_duplicates(subset='orispl', keep='first').sort_values('quantity', ascending=0)
                    multiplier = 1.1
                loop = 0
                loop_len = len(orispl_prices_filled) - 1
                #of the plants with EIA923 data that we are assigning to plants without eia923 data, we will use the plant with the highest energy production first, assigning its fuel price profile to one of the generators that does not have EIA923 data. We will move on to plant with the next highest energy production and so on, uniformly distributing the available EIA923 fuel price profiles to generators without fuel price data
                for o in scipy.concatenate((orispl_prices_empty.orispl.unique(),orispl_prices_nan.orispl.unique())):
                    orispl_prices.loc[(orispl_prices.orispl==o) & (orispl_prices.fuel==f), orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])] = scipy.array(orispl_prices_filled[orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type'])].iloc[loop]) * multiplier
                    #keep looping through the generators with eia923 price data until we have used all of their fuel price profiles, then start again from the beginning of the loop with the plant with the highest energy production
                    if loop < loop_len:
                        loop += 1
                    else:
                        loop = 0
        
        #and now we still have some nan values for fuel types that had no nerc_region eia923 data. We'll start with the national median for the EIA923 data.
        f_array = scipy.intersect1d(orispl_prices[orispl_prices[1].isna()].fuel.unique(), df.fuel[~df.fuel.isna()].unique())
        for f in f_array: 
            temp = df[df.fuel==f][['month', 'quantity', 'fuel_price']]
            temp['weighted'] = scipy.multiply(temp.quantity, temp.fuel_price)
            temp = temp.groupby(['month'], as_index=False).sum()[['month', 'quantity', 'weighted']]
            temp['fuel_price'] = scipy.divide(temp.weighted, temp.quantity)
            temp_prices = pandas.DataFrame({'month': scipy.arange(12)+1})
            temp_prices = temp_prices.merge(temp[['month', 'fuel_price']], on='month', how='left')
            temp_prices.loc[temp_prices.fuel_price.isna(), 'fuel_price'] = temp_prices.fuel_price.median()
            orispl_prices.loc[orispl_prices.fuel==f, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'purchase_type'])] = scipy.append(scipy.array(temp_prices.fuel_price),temp.quantity.sum())
        #for any fuels that don't have EIA923 data at all (for all regions) we will use commodity price approximations from an excel file
        #first we need to change orispl_prices from months to weeks
        orispl_prices.columns = ['orispl_unit', 'orispl', 'fuel', 1, 5, 9, 14, 18, 22, 27, 31, 36, 40, 44, 48, 'quantity', 'purchase_type']
        #scipy.array(orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type']))
        test = orispl_prices.copy(deep=True)[['orispl_unit', 'orispl', 'fuel']]
        month_weeks = scipy.array(orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel', 'quantity', 'purchase_type']))
        for c in scipy.arange(52)+1:
            if c in month_weeks:
                test['fuel_price'+ str(c)] = orispl_prices[c]
            else:
                test['fuel_price'+ str(c)] = test['fuel_price'+str(c-1)]
        orispl_prices = test.copy(deep=True)
        #now we add in the weekly fuel commodity prices
        prices_fuel_commodity = self.fuel_commodity_prices
        f_array = orispl_prices[orispl_prices['fuel_price1'].isna()].fuel.unique()
        for f in f_array:
            l = len(orispl_prices.loc[orispl_prices.fuel==f, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel'])])
            orispl_prices.loc[orispl_prices.fuel==f, orispl_prices.columns.difference(['orispl_unit', 'orispl', 'fuel'])] = scipy.tile(prices_fuel_commodity[f], (l,1))
        #now we have orispl_prices, which has a weekly fuel price for each orispl_unit based mostly on EIA923 data with some commodity, national-level data from EIA to supplement
        #now merge the fuel price columns into self.df
        orispl_prices.drop(['orispl', 'fuel'], axis=1, inplace=True)
               
        #save
        self.df = self.df.merge(orispl_prices, on='orispl_unit', how='left')


    def easiurDamages(self):
        """ 
        Adds EASIUR environmental damages for SO2 and NOx emissions for each power plant.
        ---
        Adds one column for each week of the year to self.df that contains environmental damages in $/MWh for each generation unit calculated using the EASIURE method
        """   
        print('Adding environmental damages...')
        #clean the easiur data
        df = self.easiur_per_plant.copy(deep=True)
        df = df[['ORISPL','SO2 Winter 150m','SO2 Spring 150m','SO2 Summer 150m','SO2 Fall 150m','NOX Winter 150m','NOX Spring 150m','NOX Summer 150m','NOX Fall 150m']]
        df.columns = ['orispl', 'so2_dmg_win', 'so2_dmg_spr' , 'so2_dmg_sum', 'so2_dmg_fal', 'nox_dmg_win', 'nox_dmg_spr' , 'nox_dmg_sum', 'nox_dmg_fal']        
        #create empty dataframe to hold the results
        df2 = self.df.copy(deep=True)
        #for each week, calculate the $/MWh damages for each generator based on its emissions rate (kg/MWh) and easiur damages ($/tonne)
        for c in scipy.arange(52)+1:
            season = scipy.where(((c>49) | (c<=10)), 'win', scipy.where(((c>10) & (c<=23)), 'spr', scipy.where(((c>23) & (c<=36)), 'sum', scipy.where(((c>36) & (c<=49)), 'fal', 'na')))) #define the season string
            df2['dmg' + str(c)] = (df2['so2' + str(c)] * df['so2' + '_dmg_' + str(season)] + df2['nox' + str(c)] * df['nox' + '_dmg_' + str(season)]) / 1e3
        #use the results to redefine the main generator DataFrame
        self.df = df2


    def addGenMinOut(self):
        """ 
        Adds fuel price and vom costs to the generator dataframe 'self.df'
        ---
        """
        df = self.df.copy(deep=True)
        #define min_out, based on the NREL Black & Veatch report (2012)
        min_out_coal = 0.4
        min_out_ngcc = 0.5
        min_out_ngst = min_out_coal #assume the same as coal boiler
        min_out_nggt = 0.5 
        min_out_oilst = min_out_coal #assume the same as coal boiler
        min_out_oilgt = min_out_nggt #assume the same as gas turbine
        min_out_nuc = 0.5
        min_out_bio = 0.4
        df['min_out_multiplier'] = scipy.where(df.fuel_type=='oil', scipy.where(df.prime_mover=='st', min_out_oilst, min_out_oilgt), scipy.where(df.fuel_type=='biomass',min_out_bio, scipy.where(df.fuel_type=='coal',min_out_coal, scipy.where(df.fuel_type=='nuclear',min_out_nuc, scipy.where(df.fuel_type=='gas', scipy.where(df.prime_mover=='gt', min_out_nggt, scipy.where(df.prime_mover=='st', min_out_ngst, min_out_ngcc)), 0.10)))))
        df['min_out'] = df.mw * df.min_out_multiplier
        self.df = df          
        
        
    def addGenVom(self):
        """ 
        Adds vom costs to the generator dataframe 'self.df'
        ---
        """
        df = self.df.copy(deep=True)
        #define vom, based on the ranges of VOM values from pg.12, fig 5 of NREL The Western Wind and Solar Integration Study Phase 2" report. We assume, based on that study, that older units have higher values and newer units have lower values according to a linear relationship between the following coordinates:
        vom_range_coal_bit = [1.5, 5]
        vom_range_coal_sub = [1.5, 5]
        vom_range_coal = [1.5, 5]
        age_range_coal = [1955, 2013]
        vom_range_ngcc = [0.5, 1.5]
        age_range_ngcc = [1990, 2013]
        vom_range_nggt = [0.5, 2.0]
        age_range_nggt = [1970, 2013]
        vom_range_ngst = [0.5, 6.0]
        age_range_ngst = [1950, 2013]

        def vom_calculator(fuelType, fuel, primeMover, yearOnline):
            if fuelType=='coal':
                if fuel == 'bit':
                    return vom_range_coal_bit[0] + (vom_range_coal_bit[1]-vom_range_coal_bit[0])/(age_range_coal[1]-age_range_coal[0]) * (self.year - yearOnline)
                elif fuel == 'sub':
                    return vom_range_coal_sub[0] + (vom_range_coal_sub[1]-vom_range_coal_sub[0])/(age_range_coal[1]-age_range_coal[0]) * (self.year - yearOnline)
                else:
                    return vom_range_coal[0] + (vom_range_coal[1]-vom_range_coal[0])/(age_range_coal[1]-age_range_coal[0]) * (self.year - yearOnline)
            if fuelType!='coal':
                if (primeMover=='ct') | (primeMover=='cc'):
                    return vom_range_ngcc[0] + (vom_range_ngcc[1]-vom_range_ngcc[0])/(age_range_ngcc[1]-age_range_ngcc[0]) * (self.year - yearOnline)
                if primeMover=='gt':
                    return vom_range_nggt[0] + (vom_range_nggt[1]-vom_range_nggt[0])/(age_range_nggt[1]-age_range_nggt[0]) * (self.year - yearOnline)
                if primeMover=='st':
                    return vom_range_ngst[0] + (vom_range_ngst[1]-vom_range_ngst[0])/(age_range_ngst[1]-age_range_ngst[0]) * (self.year - yearOnline)
        
        df['vom'] = df.apply(lambda x: vom_calculator(x['fuel_type'], x['fuel'], x['prime_mover'], x['year_online']), axis=1)
        self.df = df


    def addDummies(self):
        """ 
        Adds dummy "coal_0" and "ngcc_0" generators to df
        ---
        """
        df = self.df.copy(deep=True)
        #coal_0
        df.loc[len(df)] = df.loc[0]
        df.loc[len(df)-1, self.df.columns.drop(['ba', 'nerc', 'egrid'])] = df.loc[0, df.columns.drop(['ba', 'nerc', 'egrid'])] * 0
        df.loc[len(df)-1,['orispl', 'orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'min_out_multiplier', 'min_out', 'is_coal']] = ['coal_0', 'coal_0', 'sub', 'coal', 'st', 0.0, 0.0, 1]
        #ngcc_0
        df.loc[len(df)] = df.loc[0]
        df.loc[len(df)-1, self.df.columns.drop(['ba', 'nerc', 'egrid'])] = df.loc[0, df.columns.drop(['ba', 'nerc', 'egrid'])] * 0
        df.loc[len(df)-1,['orispl', 'orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'min_out_multiplier', 'min_out', 'is_gas']] = ['ngcc_0', 'ngcc_0', 'ng', 'gas', 'ct', 0.0, 0.0, 1]
        self.df = df
            
     
    def calcDemandData(self):
        """ 
        Uses CEMS data to calculate net demand (i.e. total fossil generation), total emissions, and each generator type's contribution to the generation mix
        ---
        Creates
        self.hist_dispatch : one row per hour of the year, columns for net demand, total emissions, operating cost of the marginal generator, and the contribution of different fuels to the total energy production
        """
        print('Calculating demand data from CEMS...')
        #re-compile the cems data adding in fuel and fuel type
        df = self.df_cems.copy(deep=True)
        merge_orispl_unit = self.df.copy(deep=True)[['orispl_unit', 'fuel', 'fuel_type']]
        merge_orispl = self.df.copy(deep=True)[['orispl', 'fuel', 'fuel_type']].drop_duplicates('orispl')
        df = df.merge(merge_orispl_unit, how='left', on=['orispl_unit']) 
        df.loc[df.fuel.isna(), 'fuel'] = scipy.array(df[df.fuel.isna()].merge(merge_orispl, how='left', on=['orispl']).fuel_y)
        df.loc[df.fuel_type.isna(), 'fuel_type'] = scipy.array(df[df.fuel_type.isna()].merge(merge_orispl, how='left', on=['orispl']).fuel_type_y)
        #build the hist_dispatch dataframe
        #start with the datetime column
        start_date_str = (self.df_cems.date.min()[-4:] + '-' + self.df_cems.date.min()[:5] + ' 00:00')
        date_hour_count = len(self.df_cems.date.unique())*24#+1
        hist_dispatch = pandas.DataFrame(scipy.array([pandas.Timestamp(start_date_str) + datetime.timedelta(hours=i) for i in range(date_hour_count)]), columns=['datetime'])
        #add columns by aggregating df by date + hour
        hist_dispatch['demand'] = df.groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['co2_tot'] = df.groupby(['date','hour'], as_index=False).sum().co2_tot # * 2000
        hist_dispatch['so2_tot'] = df.groupby(['date','hour'], as_index=False).sum().so2_tot
        hist_dispatch['nox_tot'] = df.groupby(['date','hour'], as_index=False).sum().nox_tot
        hist_dispatch['coal_mix'] = df[(df.fuel_type=='coal') | (df.fuel=='SGC')].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['gas_mix'] = df[df.fuel_type=='gas'].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['oil_mix'] = df[df.fuel_type=='oil'].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['biomass_mix'] = df[(df.fuel_type=='biomass') | (df.fuel=='obs') | (df.fuel=='wds') | (df.fuel=='blq') | (df.fuel=='msw') | (df.fuel=='lfg') | (df.fuel=='ab') | (df.fuel=='obg') | (df.fuel=='obl') | (df.fuel=='slw')].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['geothermal_mix'] = df[(df.fuel_type=='geothermal') | (df.fuel=='geo')].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['hydro_mix'] = df[(df.fuel_type=='hydro') | (df.fuel=='wat')].groupby(['date','hour'], as_index=False).sum().mwh
        hist_dispatch['nuclear_mix'] = df[df.fuel=='nuc'].groupby(['date','hour'], as_index=False).sum().mwh
        #hist_dispatch['production_cost'] = df[['date', 'hour', 'production_cost']].groupby(['date','hour'], as_index=False).sum().production_cost
        hist_dispatch.fillna(0, inplace=True)
        #fill in last line to equal the previous line
        #hist_dispatch.loc[(len(hist_dispatch)-1)] = hist_dispatch.loc[(len(hist_dispatch)-2)]
        hist_dispatch = hist_dispatch.fillna(0)
        self.hist_dispatch = hist_dispatch


    def addElecPriceToDemandData(self):
        """ 
        Calculates the historical electricity price for the nerc region, adding it as a new column to the demand data
        ---
        """
        print('Adding historical electricity prices...')
        #We will use FERC 714 data, where balancing authorities and similar entities report their locational marginal prices. This script pulls in those price for every reporting entity in the nerc region and takes the max price across the BAs/entities for each hour.
        df = self.ferc714.copy(deep=True)
        df_ids = self.ferc714_ids.copy(deep=True)
        nerc_region = self.nerc
        year = self.year
        df_ids_bas = list(df_ids[df_ids.nerc == nerc_region].respondent_id.values)
        #aggregate the price data by mean price per hour for any balancing authorities within the nerc region
        df_bas = df[df.respondent_id.isin(df_ids_bas) & (df.report_yr==year)][['lambda_date', 'respondent_id', 'hour01', 'hour02', 'hour03', 'hour04', 'hour05', 'hour06', 'hour07', 'hour08', 'hour09', 'hour10', 'hour11', 'hour12', 'hour13', 'hour14', 'hour15', 'hour16', 'hour17', 'hour18', 'hour19', 'hour20', 'hour21', 'hour22', 'hour23', 'hour24']]
        df_bas.drop(['respondent_id'], axis=1, inplace=True)
        df_bas.columns = ['date',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24]
        df_bas_temp = pandas.melt(df_bas, id_vars=['date'])
        df_bas_temp.date = df_bas_temp.date.str[0:-7] + (df_bas_temp.variable - 1).astype(str) + ':00'
        df_bas_temp['time'] = df_bas_temp.variable.astype(str) + ':00'
        df_bas_temp['datetime'] = pandas.to_datetime(df_bas_temp.date)
        df_bas_temp.drop(columns=['date', 'variable', 'time'], inplace=True)
        #aggregate by datetime
        df_bas_temp = df_bas_temp.groupby('datetime', as_index=False).max()
        df_bas_temp.columns = ['datetime', 'price']        
        #add the price column to self.hist_dispatch
        self.hist_dispatch['gen_cost_marg'] = df_bas_temp.price


    def demandTimeSeries(self):
        """ 
        Re-formats and slices self.hist_dispatch to produce a demand time series to be used by the dispatch object
        ---
        Creates
        self.demand_data : row for each hour. columns for datetime and demand
        """
        print('Creating "demand_data" time series...')
        #demand using CEMS data
        demand_data = self.hist_dispatch.copy(deep=True)
        demand_data.datetime = pandas.to_datetime(demand_data.datetime)
        self.demand_data = demand_data[['datetime', 'demand']]
    
    
    def cemsBoxPlot(self, plot_col):
        """ 
        Creates a box plot of the hourly CEMS data for each unique orispl_unit for the given column
        ---
        plot_col: 'co2', 'heat_rate', etc.
        """
        print('Creating "demand_data" time series...')
        #copy the CEMS data
        cems_copy = self.df_cems.copy(deep=True)
        #each uniqe unit tag
        ounique = cems_copy.orispl_unit.unique()
       #empty data frame for results
        result = pandas.DataFrame({'orispl_unit': ounique, plot_col+'_5': scipy.zeros_like(ounique), plot_col+'_25': scipy.zeros_like(ounique), plot_col+'_50': scipy.zeros_like(ounique), plot_col+'_75': scipy.zeros_like(ounique), plot_col+'_95': scipy.zeros_like(ounique), 'data_points': scipy.zeros_like(ounique)})
        #for each unique unit calculate the 5th, 25th, median, 75th, and 95th percentile data
        print('Calculating quantiles...')
        for o in ounique:
            cems_e_test = cems_copy.loc[cems_copy.orispl_unit==o, plot_col]
            if len(cems_e_test) != 0:
                result.loc[result.orispl_unit==o, plot_col+'_5'] = cems_e_test.quantile(0.05)
                result.loc[result.orispl_unit==o, plot_col+'_25'] = cems_e_test.quantile(0.25)
                result.loc[result.orispl_unit==o, plot_col+'_50'] = cems_e_test.median()
                result.loc[result.orispl_unit==o, plot_col+'_75'] = cems_e_test.quantile(0.75)
                result.loc[result.orispl_unit==o, plot_col+'_95'] = cems_e_test.quantile(0.95)
                result.loc[result.orispl_unit==o, 'data_points'] = len(cems_e_test)
        #sort the results for plotting
        result = result.sort_values(by=[plot_col+'_50']).reset_index()
        #plot the results to be like a box plot
        x = scipy.arange(len(ounique))
        f, ax = matplotlib.pylab.subplots(1, figsize = (7,35))
        ax.scatter(result[plot_col+'_5'], x, marker='.', color='k', s=5)
        ax.scatter(result[plot_col+'_25'], x, marker='|', color='k', s=40)
        ax.scatter(result[plot_col+'_50'], x, marker='s', color='r', s=15)
        ax.scatter(result[plot_col+'_75'], x, marker='|', color='k', s=40)
        ax.scatter(result[plot_col+'_95'], x, marker='.', color='k', s=5)
        xmax = scipy.where(plot_col=='co2', 2000, scipy.where(plot_col=='so2', 5, scipy.where(plot_col=='nox', 3, 20)))
        for a in x:
            ax.plot([result.loc[a,plot_col+'_5'],result.loc[a,plot_col+'_95']], [a,a], lw=1, c='black', alpha=0.25)
            matplotlib.pylab.text(-xmax*0.2, a, result.loc[a, 'orispl_unit'])
        for v in [xmax*0.25, xmax*0.5, xmax*0.75]:
            ax.axvline(v, c='black', lw=1, alpha=0.5, ls=':')
        ax.set_xlim(0,xmax)
        ax.set_xticks([xmax*0.25, xmax*0.5, xmax*0.75, xmax])
        ax.get_yaxis().set_ticks([])
        if plot_col == 'heat_rate':
            ax.set_xlabel('Heat Rate [mmBtu/MWh]')
        else:
            ax.set_xlabel(plot_col.title() + ' Emissions Rate [kg/MWh]')
        return f


    def calcMdtCoalEvents(self):
        """ 
        Creates a dataframe of the start, end, and demand_threshold for each event in the demand data where we would expect a coal plant's minimum downtime constraint to kick in
        ---
        """                      
        mdt_coal_events = self.demand_data.copy()
        mdt_coal_events['indices'] = mdt_coal_events.index
        #find the integral from x to x+
        mdt_coal_events['integral_x_xt'] = mdt_coal_events.demand[::-1].rolling(window=self.coal_min_downtime+1).sum()[::-1]
        #find the integral of a flat horizontal line extending from x
        mdt_coal_events['integral_x'] = mdt_coal_events.demand * (self.coal_min_downtime+1)
        #find the integral under the minimum of the flat horizontal line and the demand curve
        def d_forward_convex_integral(mdt_index):
            try:
                return scipy.minimum(scipy.repeat(mdt_coal_events.demand[mdt_index], (self.coal_min_downtime+1)), mdt_coal_events.demand[mdt_index:(mdt_index+self.coal_min_downtime+1)]).sum()
            except:
                return mdt_coal_events.demand[mdt_index]   
        mdt_coal_events['integral_x_xt_below_x'] = mdt_coal_events.indices.apply(d_forward_convex_integral)
        #find the integral of the convex portion below x_xt
        mdt_coal_events['integral_convex_portion_btwn_x_xt'] = mdt_coal_events['integral_x'] - mdt_coal_events['integral_x_xt_below_x']
        #keep the convex integral only if x < 1.05*x+
        def d_keep_convex(mdt_index):
            try:
                return int(mdt_coal_events.demand[mdt_index] <= 1.05*mdt_coal_events.demand[mdt_index + self.coal_min_downtime]) * mdt_coal_events.integral_convex_portion_btwn_x_xt[mdt_index]
            except:
                return mdt_coal_events.integral_convex_portion_btwn_x_xt[mdt_index]   
        mdt_coal_events['integral_convex_filtered'] = mdt_coal_events.indices.apply(d_keep_convex)
        #mdt_coal_events['integral_convex_filtered'] = mdt_coal_events['integral_convex_filtered'].replace(0, scipy.nan)
        #keep any local maximums of the filtered convex integral
        mdt_coal_events['local_maximum'] = ((mdt_coal_events.integral_convex_filtered== mdt_coal_events.integral_convex_filtered.rolling(window=int(self.coal_min_downtime/2+1), center=True).max()) & (mdt_coal_events.integral_convex_filtered != 0) & (mdt_coal_events.integral_x >= mdt_coal_events.integral_x_xt))
        #spread the maximum out over the min downtime window
        mdt_coal_events = mdt_coal_events[mdt_coal_events.local_maximum]
        mdt_coal_events['demand_threshold'] = mdt_coal_events.demand
        mdt_coal_events['start'] = mdt_coal_events.datetime
        mdt_coal_events['end'] = mdt_coal_events.start + pandas.DateOffset(hours=self.coal_min_downtime)
        mdt_coal_events = mdt_coal_events[['start', 'end', 'demand_threshold']]
        self.mdt_coal_events = mdt_coal_events     


class generatorDataShort(object):
    """Create object containing subset of full GeneratorData object information."""

    def __init__(self, gen_data_full):

        self.year = gen_data_full.year
        self.nerc = gen_data_full.nerc
        self.hist_dispatch = gen_data_full.hist_dispatch
        self.demand_data = gen_data_full.demand_data
        self.mdt_coal_events = gen_data_full.mdt_coal_events
        self.df = gen_data_full.df


class bidStack(object):
    def __init__(self, gen_data_short, co2_dol_per_kg=0.0, so2_dol_per_kg=0.0, nox_dol_per_kg=0.0, coal_dol_per_mmbtu=0.0, coal_capacity_derate = 0.0, time=1, dropNucHydroGeo=False, include_min_output=True, initialization=True, coal_mdt_demand_threshold = 0.0, mdt_weight=0.50, include_easiur=True):
        """ 
        1) Bring in the generator data created by the "generatorData" class.
        2) Calculate the generation cost for each generator and sort the generators by generation cost. Default emissions prices [$/kg] are 0.00 for all emissions.
        ---
        gen_data_short : a generatorData object
        co2 / so2 / nox_dol_per_kg : a tax on each amount of emissions produced by each generator. Impacts each generator's generation cost
        coal_dol_per_mmbtu : a tax (+) or subsidy (-) on coal fuel prices in $/mmbtu. Impacts each generator's generation cost
        coal_capacity_derate : fraction that we want to derate all coal capacity (e.g. 0.20 multiplies each coal plant's capacity by (1-0.20))
        time : number denoting which time period we are interested in. Default is weeks, so time=15 would look at the 15th week of heat rate, emissions rates, and fuel prices
        dropNucHydroGeo : if True, nuclear, hydro, and geothermal plants will be removed from the bidstack (e.g. to match CEMS data)
        include_min_output : if True, will include a representation of generators' minimum output constraints that impacts the marginal generators in the dispatch. So, a "True" value here is closer to the real world.
        initialization : if True, the bs object is being defined for the first time. This will trigger the generation of a dummy 0.0 demand generator to bookend the bottom of the merit order (in calcGenCost function) after which initialization will be set to False
        """
        self.year = gen_data_short.year
        self.nerc = gen_data_short.nerc
        self.hist_dispatch = gen_data_short.hist_dispatch
        self.mdt_coal_events = gen_data_short.mdt_coal_events
        self.coal_mdt_demand_threshold = coal_mdt_demand_threshold
        self.mdt_weight = mdt_weight
        self.df_0 = gen_data_short.df
        self.df = self.df_0.copy(deep=True)
        self.co2_dol_per_kg = co2_dol_per_kg
        self.so2_dol_per_kg = so2_dol_per_kg
        self.nox_dol_per_kg = nox_dol_per_kg
        self.coal_dol_per_mmbtu = coal_dol_per_mmbtu
        self.coal_capacity_derate = coal_capacity_derate
        self.time = time
        self.include_min_output = include_min_output
        self.initialization = initialization
        self.include_easiur = include_easiur
        
        if dropNucHydroGeo:
            self.dropNuclearHydroGeo()
        self.addFuelColor()
        self.processData()
      
        
    def updateDf(self, new_data_frame):
        self.df_0 = new_data_frame
        self.df = self.df_0.copy(deep=True)
        self.processData()


    def dropNuclearHydroGeo(self):
        """ 
        Removes nuclear, hydro, and geothermal plants from self.df_0 (since they don't show up in CEMS)
        ---
        """
        self.df_0 = self.df_0[(self.df_0.fuel!='nuc') & (self.df_0.fuel!='wat') & (self.df_0.fuel!='geo')]


    def updateEmissionsAndFuelTaxes(self, co2_price_new, so2_price_new, nox_price_new, coal_price_new):
        """ Updates self. emissions prices (in $/kg) and self.coal_dol_per_mmbtu (in $/mmbtu)
        ---
        """
        self.co2_dol_per_kg = co2_price_new
        self.so2_dol_per_kg = so2_price_new
        self.nox_dol_per_kg = nox_price_new   
        self.coal_dol_per_mmbtu = coal_price_new
    
    
    def processData(self):
        """ runs a few of the internal functions. There are couple of places in the class that run these functions in this order, so it made sense to just locate this set of function runs in a single location
        ---
        """
        self.calcGenCost()
        self.createTotalInterpolationFunctions()
        self.createMarginalPiecewise()
        self.calcFullMeritOrder()
        self.createMarginalPiecewise() #create this again after FullMeritOrder so that it includes the new full_####_marg columns
        self.createTotalInterpolationFunctionsFull()
    
    
    def updateTime(self, t_new):
        """ Updates self.time
        ---
        """
        self.time = t_new
        self.processData()
    
    
    def addFuelColor(self):
        """ Assign a fuel type for each fuel and a color for each fuel type to be used in charts
        ---
        creates 'fuel_type' and 'fuel_color' columns
        """
        c = {'gas':'#888888', 'coal':'#bf5b17', 'oil':'#252525' , 'nuclear':'#984ea3', 'hydro':'#386cb0', 'biomass':'#7fc97f', 'geothermal':'#e31a1c', 'ofsl': '#c994c7'}
        self.df_0['fuel_color'] = '#bcbddc'
        for c_key in c.keys():
            self.df_0.loc[self.df_0.fuel_type == c_key, 'fuel_color'] = c[c_key]            
     
           
    def calcGenCost(self):
        """ Calculate average costs that are function of generator data, fuel cost, and emissions prices.
        gen_cost ($/MWh) = (heat_rate * "fuel"_price) + (co2 * co2_price) + (so2 * so2_price) + (nox * nox_price) + vom 
        """
        df = self.df_0.copy(deep=True)
        #pre-processing:
            #adjust coal fuel prices by the "coal_dol_per_mmbtu" input
        df.loc[df.fuel_type=='coal', 'fuel_price' + str(self.time)] = scipy.maximum(0, df.loc[df.fuel_type=='coal', 'fuel_price' + str(self.time)] + self.coal_dol_per_mmbtu)
            #adjust coal capacity by the "coal_capacity_derate" input
        df.loc[df.fuel_type=='coal', 'mw' + str(self.time)] = df.loc[df.fuel_type=='coal', 'mw' + str(self.time)] * (1.0 -  self.coal_capacity_derate)
        #calculate the generation cost:
        df['fuel_cost'] = df['heat_rate' + str(self.time)] * df['fuel_price' + str(self.time)] 
        df['co2_cost'] = df['co2' + str(self.time)] * self.co2_dol_per_kg 
        df['so2_cost'] = df['so2' + str(self.time)] * self.so2_dol_per_kg 
        df['nox_cost'] = df['nox' + str(self.time)] * self.nox_dol_per_kg 
        df['gen_cost'] = scipy.maximum(0.01, df.fuel_cost + df.co2_cost + df.so2_cost + df.nox_cost + df.vom)
        #add a zero generator so that the bid stack goes all the way down to zero. This is important for calculating information for the marginal generator when the marginal generator is the first one in the bid stack.
        if self.include_easiur:
            df['dmg_easiur'] = df['dmg' + str(self.time)]
        #if self.initialization:
        df = df.append(df.loc[0]*0) 
        df = df.append(df.iloc[-1])
        #self.initialization = False
        df.sort_values('gen_cost', inplace=True)
        #move coal_0 and ngcc_0 to the front of the merit order regardless of their gen cost
        coal_0_ind = df[df.orispl_unit=='coal_0'].index[0]
        ngcc_0_ind = df[df.orispl_unit=='ngcc_0'].index[0]
        df = pandas.concat([df.iloc[[0],:], df[df.orispl_unit=='coal_0'], df[df.orispl_unit=='ngcc_0'], df.drop([0, coal_0_ind, ngcc_0_ind], axis=0)], axis=0)
        df.reset_index(drop=True, inplace=True)
        df['demand'] = df['mw' + str(self.time)].cumsum()	
#         df.loc[len(df)-1, 'demand'] = df.loc[len(df)-1, 'demand'] + 1000000 #creates a very large generator at the end of the merit order so that demand cannot be greater than supply
        df['f'] = df['demand']
        df['s'] = scipy.append(0, scipy.array(df.f[0:-1]))
        df['a'] = scipy.maximum(df.s - df.min_out*(1/0.10), 1.0)       
        #add a very large demand for the last row
        self.df = df
        
        
        
    def createMarginalPiecewise(self):
        """ Creates a piecewsise dataframe of the generator data. We can then interpolate this data frame for marginal data instead of querying.
        """
        test = self.df.copy()      
        test_shift = test.copy()  
        test_shift.loc[:,'demand'] = test_shift.loc[:,'demand'] + 0.1  
        test.index = test.index * 2
        test_shift.index = test_shift.index * 2 + 1
        df_marg_piecewise = pandas.concat([test, test_shift]).sort_index()
        df_marg_piecewise.loc[:,'demand'] = pandas.concat([df_marg_piecewise.demand[0:1], df_marg_piecewise.demand[0:-1]]).reset_index(drop=True)
        df_marg_piecewise.loc[:,'demand'] = df_marg_piecewise.loc[:,'demand'] - 0.1
        self.df_marg_piecewise = df_marg_piecewise


    def returnMarginalGenerator(self, demand, return_type):
        """ Returns marginal data by interpolating self.df_marg_piecewise, which is much faster than the returnMarginalGenerator function below.
        ---
        demand : [MW]
        return_type : column header of self.df being returned (e.g. 'gen', 'fuel_type', 'gen_cost', etc.)
        """
        try: #try interpolation as it's much faster. 
            try: #for columns with a time value at the end (i.e. nox30)
                return scipy.interp(demand, self.df_marg_piecewise['demand'], scipy.array(self.df_marg_piecewise[return_type + str(self.time)], dtype='float64'))
            except: #for columns without a time value at the end (i.e. gen_cost)
                return scipy.interp(demand, self.df_marg_piecewise['demand'], scipy.array(self.df_marg_piecewise[return_type], dtype='float64'))   
        except: #interpolation will only work for floats, so we use querying below otherwise (~8x slower)
            ind = scipy.minimum(self.df.index[self.df.demand <= demand][-1], len(self.df)-2)
            return self.df[return_type][ind+1]
	
					
    def createTotalInterpolationFunctions(self):
        """ Creates interpolation functions for the total data (i.e. total cost, total emissions, etc.) depending on total demand. Then the returnTotalCost, returnTotal###, ..., functions use these interpolations rather than querying the dataframes as in previous versions. This reduces solve time by ~90x.
        """       
        test = self.df.copy()      
        test.demand = test.demand.astype(float)
        test['mw' + str(self.time)] = test['mw' + str(self.time)].astype(float)
        #cost
        self.f_totalCost = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['gen_cost']).cumsum())
        #emissions and health damages
        self.f_totalCO2 = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['co2' + str(self.time)]).cumsum())
        self.f_totalSO2 = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['so2' + str(self.time)]).cumsum())
        self.f_totalNOX = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['nox' + str(self.time)]).cumsum())
        if self.include_easiur:
            self.f_totalDmg = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['dmg' + str(self.time)]).cumsum())
        #for coal units only
        self.f_totalCO2_Coal = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['co2' + str(self.time)] * test['is_coal']).cumsum())
        self.f_totalSO2_Coal = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['so2' + str(self.time)] * test['is_coal']).cumsum())
        self.f_totalNOX_Coal = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['nox' + str(self.time)] * test['is_coal']).cumsum())
        if self.include_easiur:
            self.f_totalDmg_Coal = scipy.interpolate.interp1d(test.demand, (test['mw' + str(self.time)] * test['dmg' + str(self.time)] * test['is_coal']).cumsum())       
        #fuel mix
        self.f_totalGas = scipy.interpolate.interp1d(test.demand, (test['is_gas'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalCoal = scipy.interpolate.interp1d(test.demand, (test['is_coal'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalOil = scipy.interpolate.interp1d(test.demand, (test['is_oil'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalNuclear = scipy.interpolate.interp1d(test.demand, (test['is_nuclear'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalHydro = scipy.interpolate.interp1d(test.demand, (test['is_hydro'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalGeothermal = scipy.interpolate.interp1d(test.demand, (test['is_geothermal'] * test['mw' + str(self.time)]).cumsum())
        self.f_totalBiomass = scipy.interpolate.interp1d(test.demand, (test['is_biomass'] * test['mw' + str(self.time)]).cumsum())
        #fuel consumption
        self.f_totalConsGas = scipy.interpolate.interp1d(test.demand, (test['is_gas'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsCoal = scipy.interpolate.interp1d(test.demand, (test['is_coal'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsOil = scipy.interpolate.interp1d(test.demand, (test['is_oil'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsNuclear = scipy.interpolate.interp1d(test.demand, (test['is_nuclear'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsHydro = scipy.interpolate.interp1d(test.demand, (test['is_hydro'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsGeothermal = scipy.interpolate.interp1d(test.demand, (test['is_geothermal'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
        self.f_totalConsBiomass = scipy.interpolate.interp1d(test.demand, (test['is_biomass'] * test['heat_rate' + str(self.time)] * test['mw' + str(self.time)]).cumsum())
                
					
    def returnTotalCost(self, demand):
        """ Given demand input, return the integral of the bid stack generation cost (i.e. the total operating cost of the online power plants).
        ---
        demand : [MW]
        return : integral value of the bid stack cost = total operating costs of the online generator fleet [$].
        """
        return self.f_totalCost(demand)
      
       
    def returnTotalEmissions(self, demand, emissions_type):
        """ Given demand and emissions_type inputs, return the integral of the bid stack emissions (i.e. the total emissions of the online power plants).
        ---
        demand : [MW]
        emissions_type : 'co2', 'so2', 'nox', etc.
        return : integral value of the bid stack emissions = total emissions of the online generator fleet [lbs].
        """
        if emissions_type == 'co2':
            return self.f_totalCO2(demand)
        if emissions_type == 'so2':
            return self.f_totalSO2(demand)
        if emissions_type == 'nox':
            return self.f_totalNOX(demand)
            
            
    def returnTotalEmissions_Coal(self, demand, emissions_type):
        """ Given demand and emissions_type inputs, return the integral of the bid stack emissions (i.e. the total emissions of the online power plants).
        ---
        demand : [MW]
        emissions_type : 'co2', 'so2', 'nox', etc.
        return : integral value of the bid stack emissions = total emissions of the online generator fleet [lbs].
        """
        if emissions_type == 'co2':
            return self.f_totalCO2_Coal(demand)
        if emissions_type == 'so2':
            return self.f_totalSO2_Coal(demand)
        if emissions_type == 'nox':
            return self.f_totalNOX_Coal(demand)
    
    
    def returnTotalEasiurDamages(self, demand):
        """ Given demand input, return the integral of the bid stack EASIUR damages (i.e. the total environmental damages of the online power plants).
        ---
        demand : [MW]
        return : integral value of the bid environmental damages = total damages of the online generator fleet [$].
        """
        return self.f_totalDmg(demand)
        
    
    def returnTotalEasiurDamages_Coal(self, demand):
        """ Given demand input, return the integral of the bid stack EASIUR damages (i.e. the total environmental damages of the online power plants).
        ---
        demand : [MW]
        return : integral value of the bid environmental damages = total damages of the online generator fleet [$].
        """
        return self.f_totalDmg_Coal(demand)
    
    
    def returnTotalFuelMix(self, demand, is_fuel_type):
        """ Given demand and is_fuel_type inputs, return the total MW of online generation of is_fuel_type.
        ---
        demand : [MW]
        is_fuel_type : 'is_coal', etc.
        return : total amount of online generation of type is_fuel_type
        """
        if is_fuel_type == 'is_gas':
            return self.f_totalGas(demand) 
        if is_fuel_type == 'is_coal':
            return self.f_totalCoal(demand)
        if is_fuel_type == 'is_oil':
            return self.f_totalOil(demand)
        if is_fuel_type == 'is_nuclear':
            return self.f_totalNuclear(demand)
        if is_fuel_type == 'is_hydro':
            return self.f_totalHydro(demand)
        if is_fuel_type == 'is_geothermal':
            return self.f_totalGeothermal(demand)
        if is_fuel_type == 'is_biomass':
            return self.f_totalBiomass(demand)
    
    
    def returnTotalFuelConsumption(self, demand, is_fuel_type):
        """ Given demand and is_fuel_type inputs, return the total MW of online generation of is_fuel_type.
        ---
        demand : [MW]
        is_fuel_type : 'is_coal', etc.
        return : total amount of fuel consumption of type is_fuel_type
        """
        if is_fuel_type == 'is_gas':
            return self.f_totalConsGas(demand) 
        if is_fuel_type == 'is_coal':
            return self.f_totalConsCoal(demand)
        if is_fuel_type == 'is_oil':
            return self.f_totalConsOil(demand)
        if is_fuel_type == 'is_nuclear':
            return self.f_totalConsNuclear(demand)
        if is_fuel_type == 'is_hydro':
            return self.f_totalConsHydro(demand)
        if is_fuel_type == 'is_geothermal':
            return self.f_totalConsGeothermal(demand)
        if is_fuel_type == 'is_biomass':
            return self.f_totalConsBiomass(demand)
      
    
    def calcFullMeritOrder(self):
        """ Calculates the base_ and marg_ co2, so2, nox, and coal_mix, where "base_" represents the online "base load" that does not change with marginal changes in demand and "marg_" represents the marginal portion of the merit order that does change with marginal changes in demand. The calculation of base_ and marg_ changes depending on whether the minimum output constraint (the include_min_output variable) is being used. In general, "base" is a value (e.g. 'full_gen_cost_tot_base' has units [$], and 'full_co2_base' has units [kg]) while "marg" is a rate (e.g. 'full_gen_cost_tot_marg' has units [$/MWh], and 'full_co2_marg' has units [kg/MWh]). When the dispatch object solves the dispatch, it calculates the total emissions for one time period as 'full_co2_base' + 'full_co2_marg' * (marginal generation MWh) to end up with units of [kg].
        ---
        """
        df = self.df.copy(deep=True)
        binary_demand_is_below_demand_threshold = (scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal'))) > 0).values.astype(int)
        weight_marginal_unit = (1-self.mdt_weight) + self.mdt_weight*(1-binary_demand_is_below_demand_threshold)
        weight_mindowntime_units = 1 - weight_marginal_unit
        #INCLUDING MIN OUTPUT
        if self.include_min_output:
            
            #total production cost
            df['full_gen_cost_tot_base'] = 0.1*df.a.apply(self.returnTotalCost) + 0.9*df.s.apply(self.returnTotalCost) + df.s.apply(self.returnMarginalGenerator, args=('gen_cost',)) * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base production cost [$]
            df['full_gen_cost_tot_marg'] = ((df.s.apply(self.returnTotalCost) - df.a.apply(self.returnTotalCost)) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=('gen_cost',)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0) #calculate the marginal base production cost [$/MWh]
            
            #emissions
            for e in ['co2', 'so2', 'nox']:
                df['full_' + e + '_base'] = 0.1*df.a.apply(self.returnTotalEmissions, args=(e,)) + 0.9*df.s.apply(self.returnTotalEmissions, args=(e,)) + df.s.apply(self.returnMarginalGenerator, args=(e,)) * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base emissions [kg]
                #scipy.multiply(MEF of normal generation, weight of normal genearation) + scipy.multiply(MEF of mdt_reserves, weight of mdt_reserves) where MEF of normal generation is the calculation that happens without accounting for mdt, weight of normal generation is ((f-s) / ((f-s)) + mdt_reserves) and MEF of mdt_reserves is total_value_mdt_emissions / total_mw_mdt_reserves
                df['full_' + e + '_marg'] = scipy.multiply(  ((df.s.apply(self.returnTotalEmissions, args=(e,)) - df.a.apply(self.returnTotalEmissions, args=(e,))) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=(e,)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0)  ,    weight_marginal_unit  ) + scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalEmissions_Coal, args=(e,)) - self.returnTotalEmissions_Coal(self.coal_mdt_demand_threshold, e)))  ,  scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0)  ,  weight_mindowntime_units  )
            
            #emissions damages
            if self.include_easiur:
                df['full_dmg_easiur_base'] = 0.1*df.a.apply(self.returnTotalEasiurDamages) + 0.9*df.s.apply(self.returnTotalEasiurDamages) + df.s.apply(self.returnMarginalGenerator, args=('dmg_easiur',)) * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base easiur damages [$]
                #scipy.multiply(MEF of normal generation, weight of normal genearation) + scipy.multiply(MEF of mdt_reserves, weight of mdt_reserves) where MEF of normal generation is the calculation that happens without accounting for mdt, weight of normal generation is ((f-s) / ((f-s)) + mdt_reserves) and MEF of mdt_reserves is total_value_mdt_emissions / total_mw_mdt_reserves
                df['full_dmg_easiur_marg'] = scipy.multiply(  ((df.s.apply(self.returnTotalEasiurDamages) - df.a.apply(self.returnTotalEasiurDamages)) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=('dmg_easiur',)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0)  ,  weight_marginal_unit  ) + scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalEasiurDamages_Coal) - self.returnTotalEasiurDamages_Coal(self.coal_mdt_demand_threshold)))  ,  scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0)  ,  weight_mindowntime_units  )
            
            #fuel mix
            for fl in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
            #for fl in ['gas', 'coal', 'oil']:
                df['full_' + fl + '_mix_base'] = 0.1*df.a.apply(self.returnTotalFuelMix, args=(('is_'+fl),)) + 0.9*df.s.apply(self.returnTotalFuelMix, args=(('is_'+fl),)) + self.df['is_'+fl] * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base coal_mix [MWh]
                #scipy.multiply(dmgs of normal generation, weight of normal genearation) + scipy.multiply(dmgs of mdt_reserves, weight of mdt_reserves) where dmgs of normal generation is the calculation that happens without accounting for mdt, weight of normal generation is ((f-s) / ((f-s)) + mdt_reserves) and dmgs of mdt_reserves is total_value_mdt_reserves / total_mw_mdt_reserves
                fuel_multiplier = scipy.where(fl=='coal', 1.0, 0.0)
                df['full_' + fl + '_mix_marg'] = scipy.multiply(  ((df.s.apply(self.returnTotalFuelMix, args=(('is_'+fl),)) - df.a.apply(self.returnTotalFuelMix, args=(('is_'+fl),))) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=(('is_'+fl),)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0)  ,  weight_marginal_unit  )  +  scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal'))), scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0) * fuel_multiplier  ,  weight_mindowntime_units  )
            
            #fuel consumption
            for fl in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
                df['full_' + fl + '_consumption_base'] = 0.1*df.a.apply(self.returnTotalFuelConsumption, args=(('is_'+fl),)) + 0.9*df.s.apply(self.returnTotalFuelConsumption, args=(('is_'+fl),)) + self.df['is_'+fl] * df.s.apply(self.returnMarginalGenerator, args=('heat_rate',)) * df.s.apply(self.returnMarginalGenerator, args=('min_out',)) #calculate the base fuel consumption [mmBtu]
                #scipy.multiply(mmbtu/mw of normal generation, weight of normal genearation) + scipy.multiply(mmbtu/mw of mdt_reserves, weight of mdt_reserves) where mmbtu/mw of normal generation is the calculation that happens without accounting for mdt, weight of normal generation is ((f-s) / ((f-s)) + mdt_reserves) and mmbtu/mw of mdt_reserves is total_value_mdt_reserves / total_mw_mdt_reserves
                fuel_multiplier = scipy.where(fl=='coal', 1.0, 0.0)
                df['full_' + fl + '_consumption_marg'] = scipy.multiply(  ((df.s.apply(self.returnTotalFuelConsumption, args=('is_'+fl,)) - df.a.apply(self.returnTotalFuelConsumption, args=('is_'+fl,))) / (df.s-df.a) * (df.min_out/(df.f-df.s)) + df.s.apply(self.returnMarginalGenerator, args=('is_'+fl,)) * df.s.apply(self.returnMarginalGenerator, args=('heat_rate',)) * (1 -(df.min_out/(df.f-df.s)))).fillna(0.0)  ,  weight_marginal_unit  )  +  scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalFuelConsumption, args=(('is_coal'),)) - self.returnTotalFuelConsumption(self.coal_mdt_demand_threshold, 'is_coal'))), scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0) * fuel_multiplier ,  weight_mindowntime_units  )         
        
        #EXCLUDING MIN OUTPUT
        if not self.include_min_output:
            #total production cost
            df['full_gen_cost_tot_base'] = df.s.apply(self.returnTotalCost) #calculate the base production cost, which is now the full load production cost of the generators in the merit order below the marginal unit [$]
            df['full_gen_cost_tot_marg'] = df.s.apply(self.returnMarginalGenerator, args=('gen_cost',)) #calculate the marginal production cost, which is now just the generation cost of the marginal generator [$/MWh]
            #emissions
            for e in ['co2', 'so2', 'nox']:
                df['full_' + e + '_base'] = df.s.apply(self.returnTotalEmissions, args=(e,)) #calculate the base emissions, which is now the full load emissions of the generators in the merit order below the marginal unit [kg]
                df['full_' + e + '_marg'] = scipy.multiply(  df.s.apply(self.returnMarginalGenerator, args=(e,))  ,   weight_marginal_unit  ) + scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalEmissions_Coal, args=(e,)) - self.returnTotalEmissions_Coal(self.coal_mdt_demand_threshold, e)))  ,  scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0)  ,  weight_mindowntime_units  )
            #emissions damages
            if self.include_easiur:
                df['full_dmg_easiur_base'] = df.s.apply(self.returnTotalEasiurDamages) #calculate the total Easiur damages
                df['full_dmg_easiur_marg'] = scipy.multiply(  df.s.apply(self.returnMarginalGenerator, args=('dmg_easiur',))  ,  weight_marginal_unit  ) + scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalEasiurDamages_Coal) - self.returnTotalEasiurDamages_Coal(self.coal_mdt_demand_threshold)))  ,  scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0)  ,  weight_mindowntime_units  )
            #fuel mix
            for fl in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
                df['full_' + fl + '_mix_base'] = df.s.apply(self.returnTotalFuelMix, args=(('is_'+fl),)) #calculate the base fuel_mix, which is now the full load coal mix of the generators in the merit order below the marginal unit [MWh]
                fuel_multiplier = scipy.where(fl=='coal', 1.0, 0.0)
                df['full_' + fl + '_mix_marg'] = scipy.multiply(  df.s.apply(self.returnMarginalGenerator, args=(('is_'+fl),))  ,  weight_marginal_unit  )  +  scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal'))), scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0) * fuel_multiplier  ,  weight_mindowntime_units  )
            #fuel consumption
            for fl in ['gas', 'coal', 'oil', 'nuclear', 'hydro', 'geothermal', 'biomass']:
                df['full_' + fl + '_consumption_base'] = df.s.apply(self.returnTotalFuelConsumption, args=(('is_'+fl),)) #calculate the base fuel_consumption, which is now the fuel consumption of the generators in the merit order below the marginal unit [MWh]
                fuel_multiplier = scipy.where(fl=='coal', 1.0, 0.0)
                df['full_' + fl + '_consumption_marg'] = scipy.multiply(  df.s.apply(self.returnMarginalGenerator, args=('is_'+fl,)) * df.s.apply(self.returnMarginalGenerator, args=('heat_rate',))  ,  weight_marginal_unit  )  +  scipy.multiply(  scipy.divide(scipy.maximum(0, - (df.f.apply(self.returnTotalFuelConsumption, args=(('is_coal'),)) - self.returnTotalFuelConsumption(self.coal_mdt_demand_threshold, 'is_coal'))), scipy.maximum(0, - (df.f.apply(self.returnTotalFuelMix, args=(('is_coal'),)) - self.returnTotalFuelMix(self.coal_mdt_demand_threshold, 'is_coal')))).fillna(0.0).replace(scipy.inf, 0.0) * fuel_multiplier ,  weight_mindowntime_units  )   
        #update the master dataframe df
        self.df = df


    def returnFullMarginalValue(self, demand, col_type):
        """ Given demand and col_type inputs, return the col_type (i.e. 'co2' for marginal co2 emissions rate or 'coal_mix' for coal share of the generation) of the marginal units in the Full model (the Full model includes the minimum output constraint).
        ---
        demand : [MW]
        col_type : 'co2', 'so2', 'nox', 'coal_mix', etc.
        return : full_"emissions_type"_marg as calculated in the Full model: calcFullMeritOrder
        """
        return self.returnMarginalGenerator(demand, 'full_' + col_type + '_marg')


    def createTotalInterpolationFunctionsFull(self):
        """ Creates interpolation functions for the full total data (i.e. total cost, total emissions, etc.) depending on total demand.
        """       
        test = self.df.copy()      
        #cost
        self.f_totalCostFull = scipy.interpolate.interp1d(test.demand, test['full_gen_cost_tot_base'] + (test['demand'] - test['s']) * test['full_gen_cost_tot_marg'])  
        #emissions and health damages
        self.f_totalCO2Full = scipy.interpolate.interp1d(test.demand, test['full_co2_base'] + (test['demand'] - test['s']) * test['full_co2_marg'])
        self.f_totalSO2Full = scipy.interpolate.interp1d(test.demand, test['full_so2_base'] + (test['demand'] - test['s']) * test['full_so2_marg'])
        self.f_totalNOXFull = scipy.interpolate.interp1d(test.demand, test['full_nox_base'] + (test['demand'] - test['s']) * test['full_nox_marg'])
        if self.include_easiur:
            self.f_totalDmgFull = scipy.interpolate.interp1d(test.demand, test['full_dmg_easiur_base'] + (test['demand'] - test['s']) * test['full_dmg_easiur_marg'])
        #fuel mix
        self.f_totalGasFull = scipy.interpolate.interp1d(test.demand, test['full_gas_mix_base'] + (test['demand'] - test['s']) * test['full_gas_mix_marg'])
        self.f_totalCoalFull = scipy.interpolate.interp1d(test.demand, test['full_coal_mix_base'] + (test['demand'] - test['s']) * test['full_coal_mix_marg'])
        self.f_totalOilFull = scipy.interpolate.interp1d(test.demand, test['full_oil_mix_base'] + (test['demand'] - test['s']) * test['full_oil_mix_marg'])
        self.f_totalNuclearFull = scipy.interpolate.interp1d(test.demand, test['full_nuclear_mix_base'] + (test['demand'] - test['s']) * test['full_nuclear_mix_marg'])
        self.f_totalHydroFull = scipy.interpolate.interp1d(test.demand, test['full_hydro_mix_base'] + (test['demand'] - test['s']) * test['full_hydro_mix_marg'])
        self.f_totalGeothermalFull = scipy.interpolate.interp1d(test.demand, test['full_geothermal_mix_base'] + (test['demand'] - test['s']) * test['full_geothermal_mix_marg'])
        self.f_totalBiomassFull = scipy.interpolate.interp1d(test.demand, test['full_biomass_mix_base'] + (test['demand'] - test['s']) * test['full_biomass_mix_marg'])
        #fuel consumption
        self.f_totalConsGasFull = scipy.interpolate.interp1d(test.demand, test['full_gas_consumption_base'] + (test['demand'] - test['s']) * test['full_gas_consumption_marg'])
        self.f_totalConsCoalFull = scipy.interpolate.interp1d(test.demand, test['full_coal_consumption_base'] + (test['demand'] - test['s']) * test['full_coal_consumption_marg'])
        self.f_totalConsOilFull = scipy.interpolate.interp1d(test.demand, test['full_oil_consumption_base'] + (test['demand'] - test['s']) * test['full_oil_consumption_marg'])
        self.f_totalConsNuclearFull = scipy.interpolate.interp1d(test.demand, test['full_nuclear_consumption_base'] + (test['demand'] - test['s']) * test['full_nuclear_consumption_marg'])
        self.f_totalConsHydroFull = scipy.interpolate.interp1d(test.demand, test['full_hydro_consumption_base'] + (test['demand'] - test['s']) * test['full_hydro_consumption_marg'])
        self.f_totalConsGeothermalFull = scipy.interpolate.interp1d(test.demand, test['full_geothermal_consumption_base'] + (test['demand'] - test['s']) * test['full_geothermal_consumption_marg'])
        self.f_totalConsBiomassFull = scipy.interpolate.interp1d(test.demand, test['full_biomass_consumption_base'] + (test['demand'] - test['s']) * test['full_biomass_consumption_marg'])


    def returnFullTotalValue(self, demand, col_type):
        """ Given demand and col_type inputs, return the total column of the online power plants in the Full model (the Full model includes the minimum output constraint).
        ---
        demand : [MW]
        col_type : 'co2', 'so2', 'nox', 'coal_mix', etc.
        return : total emissions = base emissions (marginal unit) + marginal emissions (marginal unit) * (D - s)
        """
        if col_type == 'gen_cost_tot':
            return self.f_totalCostFull(demand)       
        if col_type == 'co2':
            return self.f_totalCO2Full(demand)
        if col_type == 'so2':
            return self.f_totalSO2Full(demand)
        if col_type == 'nox':
            return self.f_totalNOXFull(demand)
        if col_type == 'dmg_easiur':
            return self.f_totalDmgFull(demand)
        if col_type == 'gas_mix':
            return self.f_totalGasFull(demand) 
        if col_type == 'coal_mix':
            return self.f_totalCoalFull(demand)
        if col_type == 'oil_mix':
            return self.f_totalOilFull(demand)
        if col_type == 'nuclear_mix':
            return self.f_totalNuclearFull(demand)
        if col_type == 'hydro_mix':
            return self.f_totalHydroFull(demand)
        if col_type == 'geothermal_mix':
            return self.f_totalGeothermalFull(demand)
        if col_type == 'biomass_mix':
            return self.f_totalBiomassFull(demand)
        if col_type == 'gas_consumption':
            return self.f_totalConsGasFull(demand) 
        if col_type == 'coal_consumption':
            return self.f_totalConsCoalFull(demand)
        if col_type == 'oil_consumption':
            return self.f_totalConsOilFull(demand)
        if col_type == 'nuclear_consumption':
            return self.f_totalConsNuclearFull(demand)
        if col_type == 'hydro_consumption':
            return self.f_totalConsHydroFull(demand)
        if col_type == 'geothermal_consumption':
            return self.f_totalConsGeothermalFull(demand)
        if col_type == 'biomass_consumption':
            return self.f_totalConsBiomassFull(demand)
    
    
    def plotBidStack(self, df_column, plot_type, fig_dim = (4,4), production_cost_only=True):
        """ Given a name for the df_column, plots a bid stack with demand on the x-axis and the df_column data on the y-axis. For example bidStack.plotBidStack('gen_cost', 'bar') would output the traditional merit order curve.
        ---
        df_column : column header from df, e.g. 'gen_cost', 'co2', 'so2', etc.
        plot_type : 'bar' or 'line'
        production_cost_only : if True, the dispatch cost will exclude carbon, so2 and nox taxes. if False, all costs will be included
        fig_dim = figure dimensions (#,#)
        return : bid stack plot
        """	
        #color for any generators without a fuel_color entry
        empty_color = '#dd1c77'
        #create color array for the emissions cost
        color_2 = self.df.fuel_color.replace('', empty_color)
        color_2 = color_2.replace('#888888', '#F0F0F0')
        color_2 = color_2.replace('#bf5b17', '#E0E0E0')
        color_2 = color_2.replace('#7fc97f', '#D8D8D8')
        color_2 = color_2.replace('#252525', '#D0D0D0')
        color_2 = color_2.replace('#dd1c77', '#C0C0C0')
        color_2 = color_2.replace('#bcbddc', '#E0E0E0')
        #set up the y data
        y_data_e = self.df.gen_cost * 0 #emissions bar chart. Default is zero unless not production_cost_only                    
        if df_column == 'gen_cost':
            y_lab = 'Generation Cost [$/MWh]'
            y_data = self.df[df_column] - (self.df.co2_cost + self.df.so2_cost + self.df.nox_cost) #cost excluding emissions taxes
            if not production_cost_only:
                y_data_e = self.df[df_column]
        if df_column == 'co2':
            y_lab = 'CO$_2$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        if df_column == 'so2':
            y_lab = 'SO$_2$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        if df_column == 'nox':
            y_lab = 'NO$_x$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        #create the data to be stacked on y_data to show the cost of the emission tax
        matplotlib.pylab.clf()
        f = matplotlib.pylab.figure(figsize=fig_dim)
        ax = f.add_subplot(111)
        if plot_type == 'line':
            ax.plot( self.df.demand/1000, y_data, linewidth=2.5)
        elif plot_type == 'bar':
            ax.bar(self.df.demand/1000, height=y_data_e, width=-scipy.maximum(0.2, self.df['mw' + str(self.time)]/1000), color=color_2, align='edge'), ax.bar(self.df.demand/1000, height=y_data, width=-scipy.maximum(0.2, self.df['mw' + str(self.time)]/1000), color=self.df.fuel_color.replace('', empty_color), align='edge')
            ##add legend above chart
            #color_legend = []
            #for c in self.df.fuel_color.unique():
            #    color_legend.append(matplotlib.patches.Patch(color=c, label=self.df.fuel_type[self.df.fuel_color==c].iloc[0]))
            #ax.legend(handles=color_legend, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fancybox=True, shadow=True)
        else:
            print('***Error: enter valid argument for plot_type')
            pass
        matplotlib.pylab.ylim(ymax=y_data.quantile(0.98)) #take the 98th percentile for the y limits.
        #ax.set_xlim(self.hist_dispatch.demand.quantile(0.025)*0.001, self.hist_dispatch.demand.quantile(0.975)*0.001) #take the 2.5th and 97.5th percentiles for the x limits
        ax.set_xlim(0, self.hist_dispatch.demand.quantile(0.975)*0.001) #take 0 and the 97.5th percentiles for the x limits
        if self.nerc == 'MRO':
            ax.set_xticks((10,15,20))
        if self.nerc == 'TRE':
            ax.set_xticks((20,30,40,50))
        if self.nerc == 'FRCC':
            ax.set_xticks((15,20,25,30))
        if self.nerc == 'WECC':
            ax.set_xticks((20,30,40,50,60))
        if self.nerc == 'SERC':
            ax.set_xticks((0,25,50,75,100))
        if df_column == 'gen_cost':
            if production_cost_only:
                ax.set_ylim(0, 65)
                ax.set_yticks((0, 15, 30, 45, 60))
            if not production_cost_only:
                ax.set_ylim(0, 160)
                ax.set_yticks((0, 30, 60, 90, 120, 150))    
        if df_column == 'co2':
            ax.set_ylim(0, 1300)
            ax.set_yticks((250, 500, 750, 1000, 1250))
        matplotlib.pylab.xlabel('Generation [GW]')
        matplotlib.pylab.ylabel(y_lab)
        matplotlib.pylab.tight_layout()
        matplotlib.pylab.show()
        return f
    
    
    def plotBidStackMultiColor(self, df_column, plot_type, fig_dim = (4,4), production_cost_only=True):    
        bs_df_fuel_color = self.df.copy()
        
        c = {'ng': {'cc': '#377eb8', 'ct': '#377eb8', 'gt': '#4daf4a', 'st': '#984ea3'}, 'sub': {'st': '#e41a1c'}, 'lig': {'st': '#ffff33'}, 'bit': {'st': '#ff7f00'}, 'rc': {'st': '#252525'}}
                    
        bs_df_fuel_color['fuel_color'] = '#bcbddc'
        for c_key in c.keys():
            for p_key in c[c_key].keys():
                bs_df_fuel_color.loc[(bs_df_fuel_color.fuel == c_key) & (bs_df_fuel_color.prime_mover == p_key), 'fuel_color'] = c[c_key][p_key]
        
        #color for any generators without a fuel_color entry
        empty_color = '#dd1c77'
        #create color array for the emissions cost
        color_2 = bs_df_fuel_color.fuel_color.replace('', empty_color)
        #color_2 = color_2.replace('#888888', '#F0F0F0')
        #color_2 = color_2.replace('#bf5b17', '#E0E0E0')
        #color_2 = color_2.replace('#7fc97f', '#D8D8D8')
        #color_2 = color_2.replace('#252525', '#D0D0D0')
        #color_2 = color_2.replace('#dd1c77', '#C0C0C0')
        #color_2 = color_2.replace('#bcbddc', '#E0E0E0')
        #set up the y data
        y_data_e = self.df.gen_cost * 0 #emissions bar chart. Default is zero unless not production_cost_only
        if df_column == 'gen_cost':
            y_lab = 'Generation Cost [$/MWh]'
            y_data = self.df[df_column] - (self.df.co2_cost + self.df.so2_cost + self.df.nox_cost) #cost excluding emissions taxes
            if not production_cost_only:
                y_data_e = self.df[df_column]
        if df_column == 'co2':
            y_lab = 'CO$_2$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        if df_column == 'so2':
            y_lab = 'SO$_2$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        if df_column == 'nox':
            y_lab = 'NO$_x$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        #create the data to be stacked on y_data to show the cost of the emission tax
        matplotlib.pylab.clf()
        f = matplotlib.pylab.figure(figsize=fig_dim)
        ax = f.add_subplot(111)
        if plot_type == 'line':
            ax.plot( self.df.demand/1000, y_data, linewidth=2.5)
        elif plot_type == 'bar':
            ax.bar(self.df.demand/1000, height=y_data_e, width=-scipy.maximum(0.2, self.df['mw' + str(self.time)]/1000), color=color_2, align='edge'), ax.bar(self.df.demand/1000, height=y_data, width=-scipy.maximum(0.2, self.df['mw' + str(self.time)]/1000), color=color_2, align='edge')
            ##add legend above chart
            color_legend = []
            for c in bs_df_fuel_color.fuel_color.unique():
                color_legend.append(matplotlib.patches.Patch(color=c, label=bs_df_fuel_color.fuel[bs_df_fuel_color.fuel_color==c].iloc[0] + '_' + bs_df_fuel_color.prime_mover[bs_df_fuel_color.fuel_color==c].iloc[0]))
            ax.legend(handles=color_legend, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fancybox=True, shadow=True)
        else:
            print('***Error: enter valid argument for plot_type')
            pass
        matplotlib.pylab.ylim(ymax=y_data.quantile(0.98)) #take the 98th percentile for the y limits.
        #ax.set_xlim(bs.hist_dispatch.demand.quantile(0.025)*0.001, bs.hist_dispatch.demand.quantile(0.975)*0.001) #take the 2.5th and 97.5th percentiles for the x limits
        ax.set_xlim(0, self.hist_dispatch.demand.quantile(0.975)*0.001) #take 0 and the 97.5th percentiles for the x limits
        if df_column == 'gen_cost':
            if production_cost_only:
                ax.set_ylim(0, 65)
                ax.set_yticks((0, 15, 30, 45, 60))
            if not production_cost_only:
                ax.set_ylim(0, 160)
                ax.set_yticks((0, 30, 60, 90, 120, 150))    
        if df_column == 'co2':
            ax.set_ylim(0, 1300)
            ax.set_yticks((250, 500, 750, 1000, 1250))
        matplotlib.pylab.xlabel('Generation [GW]')
        matplotlib.pylab.ylabel(y_lab)
        matplotlib.pylab.tight_layout()
        matplotlib.pylab.show()
        return f
    
    
    def plotBidStackMultiColor_Coal_NGCC_NGGT_NGOther(self, df_column, plot_type, fig_dim = (4,4), production_cost_only=True):    
        bs_df_fuel_color = self.df.copy()
        
        c = {'ng': {'cc': '#1b9e77', 'ct': '#1b9e77', 'gt': '#fc8d62', 'st': '#8da0cb'}, 'sub': {'st': '#252525'}, 'lig': {'st': '#252525'}, 'bit': {'st': '#252525'}, 'rc': {'st': '#252525'}}
                    
        bs_df_fuel_color['fuel_color'] = '#bcbddc'
        for c_key in c.keys():
            for p_key in c[c_key].keys():
                bs_df_fuel_color.loc[(bs_df_fuel_color.fuel == c_key) & (bs_df_fuel_color.prime_mover == p_key), 'fuel_color'] = c[c_key][p_key]
        
        #color for any generators without a fuel_color entry
        empty_color = '#dd1c77'
        #hold the colors
        color_2 = bs_df_fuel_color.fuel_color.replace('', empty_color)
        #create color array for the emissions cost
        color_3 = self.df.fuel_color.replace('', empty_color)
        color_3 = color_3.replace('#888888', '#F0F0F0')
        color_3 = color_3.replace('#bf5b17', '#E0E0E0')
        color_3 = color_3.replace('#7fc97f', '#D8D8D8')
        color_3 = color_3.replace('#252525', '#D0D0D0')
        color_3 = color_3.replace('#dd1c77', '#C0C0C0')
        color_3 = color_3.replace('#bcbddc', '#E0E0E0')
        #set up the y data
        y_data_e = self.df.gen_cost * 0 #emissions bar chart. Default is zero unless not production_cost_only                    
        if df_column == 'gen_cost':
            y_lab = 'Generation Cost [$/MWh]'
            y_data = self.df[df_column] - (self.df.co2_cost + self.df.so2_cost + self.df.nox_cost) #cost excluding emissions taxes
            if not production_cost_only:
                y_data_e = self.df[df_column]
        if df_column == 'co2':
            y_lab = 'CO$_2$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        if df_column == 'so2':
            y_lab = 'SO$_2$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        if df_column == 'nox':
            y_lab = 'NO$_x$ Emissions [kg/MWh]'
            y_data = self.df[df_column + str(self.time)]
        #create the data to be stacked on y_data to show the cost of the emission tax
        matplotlib.pylab.clf()
        f = matplotlib.pylab.figure(figsize=fig_dim)
        ax = f.add_subplot(111)
        if plot_type == 'line':
            ax.plot( self.df.demand/1000, y_data, linewidth=2.5)
        elif plot_type == 'bar':
            ax.bar(self.df.demand/1000, height=y_data_e, width=-scipy.maximum(0.2, self.df['mw' + str(self.time)]/1000), color=color_3, align='edge'), ax.bar(self.df.demand/1000, height=y_data, width=-scipy.maximum(0.2, self.df['mw' + str(self.time)]/1000), color=color_2, align='edge')
            ##add legend above chart
            #color_legend = []
            #for c in self.df.fuel_color.unique():
            #    color_legend.append(matplotlib.patches.Patch(color=c, label=self.df.fuel_type[self.df.fuel_color==c].iloc[0]))
            #ax.legend(handles=color_legend, bbox_to_anchor=(0.5, 1.2), loc='upper center', ncol=3, fancybox=True, shadow=True)
        else:
            print('***Error: enter valid argument for plot_type')
            pass
        matplotlib.pylab.ylim(ymax=y_data.quantile(0.98)) #take the 98th percentile for the y limits.
        #ax.set_xlim(self.hist_dispatch.demand.quantile(0.025)*0.001, self.hist_dispatch.demand.quantile(0.975)*0.001) #take the 2.5th and 97.5th percentiles for the x limits
        ax.set_xlim(0, self.hist_dispatch.demand.quantile(0.975)*0.001) #take 0 and the 97.5th percentiles for the x limits
        if self.nerc == 'MRO':
            ax.set_xticks((10,15,20))
        if self.nerc == 'TRE':
            ax.set_xticks((20,30,40,50))
        if self.nerc == 'FRCC':
            ax.set_xticks((15,20,25,30))
        if self.nerc == 'WECC':
            ax.set_xticks((20,30,40,50,60))
        if self.nerc == 'SERC':
            ax.set_xticks((0,25,50,75,100))
        if self.nerc == 'SPP':
            ax.set_xlim(0, 40)
            ax.set_xticks((0,10, 20, 30, 40))
        if df_column == 'gen_cost':
            if production_cost_only:
                ax.set_ylim(0, 65)
                ax.set_yticks((0, 15, 30, 45, 60))
            if not production_cost_only:
                ax.set_ylim(0, 120)
                ax.set_yticks((0, 30, 60, 90, 120))    
        if df_column == 'co2':
            ax.set_ylim(0, 1300)
            ax.set_yticks((250, 500, 750, 1000, 1250))
        matplotlib.pylab.xlabel('Generation [GW]')
        matplotlib.pylab.ylabel(y_lab)
        matplotlib.pylab.tight_layout()
        matplotlib.pylab.show()
        return f
    
    

class dispatch(object):
    def __init__(self, bid_stack_object, demand_df, time_array=0, include_storage=False, return_generator_limits=False):
        """ Read in bid stack object and the demand data. Solve the dispatch by projecting the bid stack onto the demand time series, updating the bid stack object regularly according to the time_array
        ---
        gen_data_object : a object defined by class generatorData
        bid_stack_object : a bid stack object defined by class bidStack
        demand_df : a dataframe with the demand data 
        time_array : a scipy array containing the time intervals that we are changing fuel price etc. for. E.g. if we are doing weeks, then time_array=scipy.arange(52) + 1 to get an array of (1, 2, 3, ..., 51, 52)
        """
        self.bs = bid_stack_object
        self.df = demand_df
        self.time_array = time_array
        self.addDFColumns()

        self.include_storage = include_storage
        self.storage_df = pandas.DataFrame({'datetime':demand_df['datetime'].values, 'demand':numpy.zeros((len(demand_df), ))})
#         self.storage_df['demand'] = 0

        self.return_generator_limits = return_generator_limits
        if return_generator_limits:
            self.gen_limits = pandas.DataFrame({'datetime':demand_df['datetime'].values, 'limit':numpy.zeros((len(demand_df), )), 'limit2':numpy.zeros((len(demand_df),))})
               
    def addDFColumns(self):
        """ Add additional columns to self.df to hold the results of the dispatch. New cells initially filled with zeros
        ---
        """
        indx = self.df.index
        cols = scipy.array(('gen_cost_marg', 'gen_cost_tot', 'co2_marg', 'co2_tot', 'so2_marg', 'so2_tot', 'nox_marg', 'nox_tot', 'dmg_easiur', 'biomass_mix', 'coal_mix', 'gas_mix', 'geothermal_mix', 'hydro_mix', 'nuclear_mix', 'oil_mix', 'marg_gen', 'coal_mix_marg', 'marg_gen_fuel_type', 'mmbtu_coal', 'mmbtu_gas', 'mmbtu_oil'))
        dfExtension = pandas.DataFrame(index=indx, columns=cols).fillna(0)
        self.df = pandas.concat([self.df, dfExtension], axis=1)


    def calcDispatchSlice(self, bstack, start_date=0, end_date=0):
        """ For each datum in demand time series (e.g. each hour) between start_date and end_date calculate the dispatch
        ---
        bstack: an object created using the simple_dispatch.bidStack class
        start_datetime : string of format '2014-01-31' i.e. 'yyyy-mm-dd'. If argument == 0, uses start date of demand time series
        end_datetime : string of format '2014-01-31' i.e. 'yyyy-mm-dd'. If argument == 0, uses end date of demand time series
        """
        if start_date==0:
            start_date = self.df.datetime.min()
        else:
            start_date = pandas._libs.tslib.Timestamp(start_date)
        if end_date==0:
            end_date = self.df.datetime.max()
        else:
            end_date = pandas._libs.tslib.Timestamp(end_date)

        df_slice = self.df[(self.df.datetime >= pandas._libs.tslib.Timestamp(start_date)) & (self.df.datetime < pandas._libs.tslib.Timestamp(end_date))].copy(deep=True)
        if self.return_generator_limits:
            print('Limit value:', bstack.df_marg_piecewise.demand.max())
            self.gen_limits.loc[df_slice.index, 'limit'] = bstack.df_marg_piecewise.demand.max()
            for i in df_slice.index:
                if self.gen_limits.loc[i, 'limit2'] == 0:
                    self.gen_limits.loc[i, 'limit2'] = bstack.df_marg_piecewise.demand.max()
                else:
                    self.gen_limits.loc[i, 'limit2'] = numpy.minimum(self.gen_limits.loc[i, 'limit2'], bstack.df_marg_piecewise.demand.max())
#             if self.gen_limits.loc[df_slice.index, 'limit'].min() != self.gen_limits.loc[df_slice.index, 'limit2'].min():
#                 print('limit1 here:', self.gen_limits.loc[df_slice.index, 'limit'].min(), 'limit2 here:', self.gen_limits.loc[df_slice.index, 'limit2'].min())
        if self.include_storage:
            clipval = bstack.df_marg_piecewise.demand.max()-1
#             print('Clipval:', clipval)
            if df_slice.demand.max() > clipval:
                inds = df_slice[df_slice['demand'] > clipval].index
                self.storage_df.loc[inds, 'demand'] = self.storage_df.loc[inds, 'demand'] + (df_slice.loc[inds, 'demand'].values-clipval)
#                 print('Maxval in storage_df:', self.storage_df.loc[inds, 'demand'].max())
                df_slice['demand'] = numpy.clip(df_slice['demand'], 0, clipval)

        # #slice of self.df within the desired dates

        #calculate the dispatch for the slice by applying the return###### functions of the bstack object
        df_slice['gen_cost_marg'] = df_slice.demand.apply(bstack.returnMarginalGenerator, args=('gen_cost',)) #generation cost of the marginal generator ($/MWh)
        df_slice['gen_cost_tot'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('gen_cost_tot',)) #generation cost of the total generation fleet ($)
        for e in ['co2', 'so2', 'nox']:
            df_slice[e + '_marg'] = df_slice.demand.apply(bstack.returnFullMarginalValue, args=(e,)) #emissions rate (kg/MWh) of marginal generators
            df_slice[e + '_tot'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=(e,)) #total emissions (kg) of online generators
        if self.bs.include_easiur:
            df_slice['dmg_easiur'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('dmg_easiur',)) #total easiur damages ($)
        for f in ['gas', 'oil', 'coal', 'nuclear', 'biomass', 'geothermal', 'hydro']:
            df_slice[f + '_mix'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=(f+'_mix',))
        df_slice['coal_mix_marg'] = df_slice.demand.apply(bstack.returnFullMarginalValue, args=('coal_mix',))
        df_slice['marg_gen_fuel_type'] = df_slice.demand.apply(bstack.returnMarginalGenerator, args=('fuel_type',))
        df_slice['mmbtu_coal'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('coal_consumption',)) #total coal mmBtu
        df_slice['mmbtu_gas'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('gas_consumption',)) #total gas mmBtu
        df_slice['mmbtu_oil'] = df_slice.demand.apply(bstack.returnFullTotalValue, args=('oil_consumption',)) #total oil mmBtu
        self.df[(self.df.datetime >= pandas._libs.tslib.Timestamp(start_date)) & (self.df.datetime < pandas._libs.tslib.Timestamp(end_date))] = df_slice


    def createDfMdtCoal(self, demand_threshold, time_t):
        """ For a given demand threshold, creates a new version of the generator data that approximates the minimum down time constraint for coal plants
        ---
        demand_threshold: the system demand below which some coal plants will turn down to minimum rather than turning off
        returns a dataframe of the same format as gd.df but updated so the coal generators in the merit order below demand_threshold have their capacities reduced by their minimum output, their minimum output changed to zero, and the sum of their minimum outputs applied to the capacity of coal_0, where coal_0 also takes the weighted average of their heat rates, emissions, rates, etc. Note that this new dataframe only contains the updated coal plants, but not the complete gd.df information (i.e. for gas plants and higher cost coal plants), but it can be incorporated back into the original df (i.e. bs.df_0) using the pandas update command.
        """
        #set the t (time i.e. week) object
        t = time_t
        #get the orispl_unit information for the generators you need to adjust
        coal_mdt_orispl_unit_list = list(self.bs.df[(self.bs.df.fuel_type=='coal') & (self.bs.df.demand <= demand_threshold)].orispl_unit.copy().values)
        coal_mdt_gd_idx = self.bs.df_0[self.bs.df_0.orispl_unit.isin(coal_mdt_orispl_unit_list)].index
        
        #create a new set of generator data where there is a large coal unit at the very bottom representing the baseload of the coal generators if they do not turn down below their minimum output, and all of the coal generators have their capacity reduced to (1-min_output).         
        if self.bs.include_easiur:
            df_mdt_coal = self.bs.df_0[self.bs.df_0.orispl_unit.isin(coal_mdt_orispl_unit_list)][['orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'vom', 'min_out_multiplier', 'min_out', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t, 'dmg%i'%t]].copy()
        else:
            df_mdt_coal = self.bs.df_0[self.bs.df_0.orispl_unit.isin(coal_mdt_orispl_unit_list)][['orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'vom', 'min_out_multiplier', 'min_out', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t]].copy()
        df_mdt_coal = df_mdt_coal[df_mdt_coal.orispl_unit != 'coal_0']
        #create a pandas Series that will hold the large dummy coal unit that represents coal base load
        df_mdt_coal_base = df_mdt_coal.copy().iloc[0]
        df_mdt_coal_base[['orispl_unit', 'fuel', 'fuel_type', 'prime_mover', 'min_out_multiplier', 'min_out']] = ['coal_0', 'sub', 'coal', 'st', 0.0, 0.0]
        #columns for the week we are currently solving
        if self.bs.include_easiur:
            t_columns = ['orispl_unit', 'fuel_type', 'prime_mover', 'vom', 'min_out_multiplier', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t, 'dmg%i'%t]
        else:
            t_columns = ['orispl_unit', 'fuel_type', 'prime_mover', 'vom', 'min_out_multiplier', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t]
        df_mdt_coal_base_temp = df_mdt_coal[t_columns].copy()       
        #the capacity of the dummy coal unit will be the sum of the minimum output of all the coal units      
        df_mdt_coal_base_temp.loc[:,'mw%i'%t] = df_mdt_coal_base_temp['mw%i'%t] * df_mdt_coal_base_temp.min_out_multiplier
        #the vom, co2, so2, nox, heat_rate, fuel_price, and dmg of the dummy unit will equal the weighted average of the other coal plants
        weighted_cols = df_mdt_coal_base_temp.columns.drop(['orispl_unit', 'fuel_type', 'prime_mover', 'min_out_multiplier', 'mw%i'%t])
        df_mdt_coal_base_temp[weighted_cols] = df_mdt_coal_base_temp[weighted_cols].multiply(df_mdt_coal_base_temp['mw%i'%t], axis='index') / df_mdt_coal_base_temp['mw%i'%t].sum()
        df_mdt_coal_base_temp = df_mdt_coal_base_temp.sum(axis=0)
        #update df_mdt_coal_base with df_mdt_coal_base_temp, which holds the weighted average characteristics of the other coal plants
        if self.bs.include_easiur:
            df_mdt_coal_base[['vom', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t, 'dmg%i'%t]] = df_mdt_coal_base_temp[['vom', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t, 'dmg%i'%t]]
        else:
            df_mdt_coal_base[['vom', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t]] = df_mdt_coal_base_temp[['vom', 'co2%i'%t, 'so2%i'%t, 'nox%i'%t, 'heat_rate%i'%t, 'mw%i'%t, 'fuel_price%i'%t]]
        #reduce the capacity of the other coal plants by their minimum outputs (since their minimum outputs are now a part of coal_0)
        df_mdt_coal.loc[df_mdt_coal.fuel_type == 'coal','mw%i'%t] = df_mdt_coal[df_mdt_coal.fuel_type == 'coal'][['mw%i'%t]].multiply((1-df_mdt_coal[df_mdt_coal.fuel_type == 'coal'].min_out_multiplier), axis='index')
        #add coal_0 to df_mdt_coal    
        df_mdt_coal = df_mdt_coal.append(df_mdt_coal_base, ignore_index = True)
        #change the minimum output of the coal plants to 0.0
        df_mdt_coal.loc[df_mdt_coal.fuel_type == 'coal',['min_out_multiplier', 'min_out']] = [0.0, 0.0]
        #update the index to match the original bidStack
        df_mdt_coal.index = coal_mdt_gd_idx
        return df_mdt_coal
    
    
    def calcMdtCoalEventsT(self, start_datetime, end_datetime, coal_merit_order_input_df):
        """ For a given demand threshold, creates a new version of the generator data that approximates the minimum down time constraint for coal plants
        ---
        demand_threshold: the system demand below which some coal plants will turn down to minimum rather than turning off
        returns a dataframe of the same format as gd.df but updated so the coal generators in the merit order below demand_threshold have their capacities reduced by their minimum output, their minimum output changed to zero, and the sum of their minimum outputs applied to the capacity of coal_0, where coal_0 also takes the weighted average of their heat rates, emissions, rates, etc.
        """
        #the function below returns the demand value of the merit_order_input_df that is just above the demand_input_scalar
        def bisect_column(demand_input_scalar, merit_order_input_df):
            try: 
                out = coal_merit_order_input_df.iloc[bisect_left(list(coal_merit_order_input_df.demand),demand_input_scalar)].demand   
        #if demand_threshold exceeds the highest coal_merit_order.demand value (i.e. all of min output constraints are binding for coal)
            except:
                out = coal_merit_order_input_df.iloc[-1].demand
            return out  
        #bring in the coal mdt events calculated in generatorData        
        mdt_coal_events_t = self.bs.mdt_coal_events.copy()
        #slice the coal mdt events based on the current start/end section of the dispatch solution
        mdt_coal_events_t = mdt_coal_events_t[(mdt_coal_events_t.end >= start_datetime) & (mdt_coal_events_t.start <= end_datetime)]
        #translate the demand_thresholds into the next highest demand data in the merit_order_input_df. This will allow us to reduce the number of bidStacks we need to generate. E.g. if two days have demand thresholds of 35200 and 35250 but the next highest demand in the coal merit order is 36000, then both of these days can use the 36000 mdt_bidStack, and we can recalculate the bidStack once instead of twice. 
        mdt_coal_events_t.loc[:,'demand_threshold'] = mdt_coal_events_t.demand_threshold.apply(bisect_column, args=(coal_merit_order_input_df,))
        return mdt_coal_events_t
    
              
    def calcDispatchAll(self):
        """ Runs calcDispatchSlice for each time slice in the fuel_prices_over_time dataframe, creating a new bidstack each time. So, fuel_prices_over_time contains multipliers (e.g. 0.95 or 1.14) for each fuel type (e.g. ng, lig, nuc) for different slices of time (e.g. start_date = '2014-01-07' and end_date = '2014-01-14'). We use these multipliers to change the fuel prices seen by each generator in the bidStack object. After changing each generator's fuel prices (using bidStack.updateFuelPrices), we re-calculate the bidStack merit order (using bidStack.calcGenCost), and then calculate the dispatch for the slice of time defined by the fuel price multipliers. This way, instead of calculating the dispatch over the whole year, we can calculate it in chunks of time (e.g. weeks) where each chunk of time has different fuel prices for the generators. 
        Right now the only thing changing per chunk of time is the fuel prices based on trends in national commodity prices. Future versions might try and do regional price trends and add things like maintenance downtime or other seasonal factors.
        ---
        fills in the self.df dataframe one time slice at a time
        """
        #run the whole solution if self.fuel_prices_over_time isn't being used
        if scipy.shape(self.time_array) == (): #might be a more robust way to do this. Would like to say if ### == 0, but doing that when ### is a dataframe gives an error
            self.calcDispatchSlice(self.bs)
        #otherwise, run the dispatch in time slices, updating the bid stack each slice
        else:
            for t in self.time_array:
                print(str(round(round(t/float(len(self.time_array)),3)*100,3)) + '% Complete')
                #update the bidStack object to the current week
                self.bs.updateTime(t)
                #calculate the dispatch for the time slice over which the updated fuel prices are relevant
                start = (datetime.datetime.strptime(str(self.bs.year) + '-01-01', '%Y-%m-%d') + datetime.timedelta(days=7.05*(t-1)-1)).strftime('%Y-%m-%d') 
                end = (datetime.datetime.strptime(str(self.bs.year) + '-01-01', '%Y-%m-%d') + datetime.timedelta(days=7.05*(t)-1)).strftime('%Y-%m-%d') 
                #note that calcDispatchSlice updates self.df, so there is no need to do it in this calcDispatchAll function
                self.calcDispatchSlice(self.bs, start_date=start ,end_date=end)
                #coal minimum downtime
                #recalculate the dispatch for times when generatorData pre-processing estimates that the minimum downtime constraint for coal plants would trigger
                #define the coal merit order
                coal_merit_order = self.bs.df[(self.bs.df.fuel_type == 'coal')][['orispl_unit', 'demand']]
                #slice and bin the coal minimum downtime events
                events_mdt_coal_t = self.calcMdtCoalEventsT(start, end, coal_merit_order)  
                #create a dictionary for holding the updated bidStacks, which change depending on the demand_threshold                
                bs_mdt_dict = {}
                #for each unique demand_threshold
                for dt in events_mdt_coal_t.demand_threshold.unique():
                    #create an updated version of gd.df
                    gd_df_mdt_temp = self.bs.df_0.copy()
                    gd_df_mdt_temp.update(self.createDfMdtCoal(dt, t))
                    #use that updated gd.df to create an updated bidStack object, and store it in the bs_mdt_dict
                    bs_temp = copy.deepcopy(self.bs)
                    bs_temp.coal_mdt_demand_threshold = dt
                    bs_temp.updateDf(gd_df_mdt_temp)
                    bs_mdt_dict.update({dt:bs_temp})
                #for each minimum downtime event, recalculate the dispatch by inputting the bs_mdt_dict bidStacks into calcDispatchSlice to override the existing dp.df results datafram
                for i, e in events_mdt_coal_t.iterrows():
                    self.calcDispatchSlice(bs_mdt_dict[e.demand_threshold], start_date=e.start ,end_date=e.end)
                
