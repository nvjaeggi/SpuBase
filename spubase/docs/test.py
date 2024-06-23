#%% md

#%%
from spubase.data_access import Particles
import pandas as pd
#%% md
# Initialize SpuBase
#%%
SpuBase = Particles(verbose=True, show_plot=False)  # create object (acts as 'self' and is used to call globals)
#%% md
#### Setup usrface mineral fractions
#%%
SpuBase.casename = 'Feldspar'  # name that is used for output folder and files

mineraldict = {'Ab':0.33, 'An':0.33, 'Or':0.33} # mineral short forms and their fractions given as dictionary
minfrac = pd.DataFrame.from_dict(mineraldict, orient='index', columns=['frac'])
#%% md
#### Optional Inputs
#%%
# SpuBase.update_file_format('pdf') # plot format, choose between 'pdf', 'png', 'svg', 'tiff'
# SpuBase.update_impactor('H', comp_frac=[1.00, 0.00])  # either 1 keV H ('H'), 4 keV He ('He') or both ('SW'); comp_frac: different mixture of H, He (Significantly effects yield and angular distribution!)
# SpuBase.sulfur_diffusion = False  # turns off S diffusion in sulfides (gets turned off automatically if impactor != 'SW')
# SpuBase.v_esc = 2380 # m/s  escape velocity of irradiated body
# SpuBase.return_amu_ion = True  # plot total mass yield in amu/ion instead of atomic yields
# SpuBase.is_summed_up = False  # return result for each individual species separately instead of summing components
#%% md
#### Change ouput directory to DATABASE/output/casename
#%%
SpuBase.update_directory()
#%% md
#### Pass surface composition to SpuBase and call data
#%%
SpuBase.surfcomp(minfrac.T, form='vol%')  # other options are 'wt%' or 'mol%'
SpuBase.minfrac_a
#%% md

#%% md
#### Create DataFrame based on input
#%%
SpuBase.dataseries()
#%%
SpuBase.yield_df  # yield gets written into the output file together with the particle angular and energy fit parameter
#%% md
#### Plot data
#%%
fig, ax = SpuBase.plot_yield()
fig
#%% md
#### Set up plotting parameters for particle data
#%%
SpuBase.dist_angle = 45 # set distribution angle (default: 45°)
SpuBase.sputtered_particles_data()
#%%
fig_adist, ax_adist = SpuBase.plot_dist('angular', title='')
fig_adist
#%%
fig_edist, ax_edist = SpuBase.plot_dist('energy', title='')
fig_edist
#%%
fig_plume, ax_plume = SpuBase.plot_dist('plume', title='')
fig_plume
#%%

#%%

#%%
