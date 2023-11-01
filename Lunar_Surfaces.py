#%%
from SpuBase import Particles
import pandas as pd

#%%
impactor = 'SW' # either 1 keV H ('H'), 4 keV He ('He') or default: both ('SW')
SpuBase = Particles(verbose=True, show_plot=False)  # create object (acts as 'self' and is used to call globals)
SpuBase.update_file_format('pdf') # plot format, choose between 'pdf', 'png', 'svg', 'tiff'

distribution = ['angular', 'energy']
plot_angles = [45]

#%%

oxide_comp_df = pd.read_csv('input/Lunar_compositions.csv', index_col='sample')


for comp in [oxide_comp_df.index.tolist()[i] for i in [0,1]]: #,2,3,4]]:
    print(f'\n###### {comp} #####')

    SpuBase.casename = comp

    at_l = oxide_comp_df.columns.tolist()
    print(at_l)
    at_frac = oxide_comp_df.loc[comp].tolist()
    minfrac = SpuBase.cipw_norm(at_l, at_frac)

    SpuBase.surfcomp(minfrac.T)

    #%% md
    #### Output directory

    # %%
    SpuBase.update_directory()

    #%% md
    #### Create DataFrame based on input
    SpuBase.dataseries()

    #%% md
    #### Plot up data

    fig, ax = SpuBase.plot_yield()

    #%%
    if not SpuBase.return_amu_ion:
        for angle in plot_angles:
            SpuBase.dist_angle = angle # set distribution angle (default: 45°)
            SpuBase.sputtered_particles_data()

            for d_type in distribution:
                SpuBase.plot_dist(d_type, minfrac_scaling=True, title='')
