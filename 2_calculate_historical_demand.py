from simple_dispatch import GridModel

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#update grid demand 2020:
grid1 = GridModel(year=2020)

grid1.run_dispatch(save_str = 'Figures/Results/2020') 

#update grid demand 2030:
grid2 = GridModel(year=2030, reference_year=2020)

grid2.run_dispatch(save_str = 'Figures/Results/2030') 