import pickle
from simple_dispatch import generatorData
from simple_dispatch import generatorDataShort
from simple_dispatch import GridModel
import time
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)



"""
Initialize the grid model by creating a generator data object and running the dispatch for 2020 and 2030.
"""
# Create generator data object


#create generator data object
run_year = 2020

ferc714_part2_schedule6_csv = 'GridInputData/Part 2 Schedule 6 - Balancing Authority Hourly System Lambda.csv' # yes
ferc714IDs_csv='GridInputData/Respondent IDs Cleaned.csv' 
cems_folder_path ='GridInputData/CEMS'
fuel_commodity_prices_xlsx = 'GridInputData/fuel_default_prices.xlsx' 
egrid_data_xlsx = 'GridInputData/egrid'+str(run_year)+'_data.xlsx'
eia923_schedule5_xlsx = 'GridInputData/EIA923_Schedules_2_3_4_5_M_12_'+str(run_year)+'_Final_Revision.xlsx' # yes

nerc_region = 'WECC'

tic = time.time()
gd = generatorData(nerc_region, egrid_fname=egrid_data_xlsx, eia923_fname=eia923_schedule5_xlsx, ferc714IDs_fname=ferc714IDs_csv, ferc714_fname=ferc714_part2_schedule6_csv, cems_folder=cems_folder_path, easiur_fname=None, include_easiur_damages=False, year=run_year, fuel_commodity_prices_excel_dir=fuel_commodity_prices_xlsx, hist_downtime=False, coal_min_downtime = 12, cems_validation_run=False,
                tz_aware=True)   
toc = time.time()
print('Finished in '+str(np.round((toc-tic)/60, 2))+' minutes. Saving.')

gd_short = generatorDataShort(gd)
pickle.dump(gd_short, open('Data/generator_data_short_WECC_'+str(run_year)+'.obj', 'wb'))
print('Saved.')
