import pickle
from simple_dispatch import generatorData
from simple_dispatch import generatorDataShort
from simple_dispatch import GridModel
import time
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


"""
Update the grid demand for the years 2020 and 2030.
"""

#update grid demand 2020:
grid1 = GridModel(year=2020)

grid1.run_dispatch(save_str = 'Figures/Results/2020') 

#update grid demand 2030:
grid2 = GridModel(year=2030, reference_year=2020)

grid2.run_dispatch(save_str = 'Figures/Results/2030') 

print('Completed updating dispatch for 2020 and 2030.')

