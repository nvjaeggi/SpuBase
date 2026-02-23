"""Sputter Database (spubase) class to access data stored in /data

See the LICENSE file for licensing information.
"""

import os
import sys
import math
import warnings
import itertools
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate

from scipy.optimize import OptimizeWarning
from .src._fitting_sb import lobe_function, eckstein, cosfit, fit_eq, thompson, cosfit_integrate




def fxn():
    warnings.warn("runtime", RuntimeWarning)
    warnings.warn("optimize", OptimizeWarning)


def frho(phi, theta, exp_mn):
    return ((1.0 + np.cos(2.0 * theta)) / 2.0) ** (
                np.cos(phi) ** 2.0 * exp_mn[0] + np.sin(phi) ** 2.0 * exp_mn[1])
    # Set the limits of integration


def closest_entry(input_list, input_value):

    input_list.sort()

    difference = lambda input_list: abs(input_list - input_value)

    min_angle = min(input_list, key=difference)
    min_angle_index = list(input_list).index(min_angle)

    return min_angle_index, min_angle

class Particles:
    def __init__(self, verbose=False, show_plot=False):

        self.maindir = os.path.dirname(os.path.realpath(__file__))
        self.tabledir = os.path.join(self.maindir, 'tables')
        
        # --- DATA MODEL ---
        self.sfx = None
        self.update_model('sbb', 'x')
        # binding model - surface and bulk binding model (sbb)
        # component model - oxides (x)
        self.impactor = 'SW'  # default is solar wind 'SW', alternatives are 'H' and 'He'
        self.sulfur_diffusion = True

        self.sw_comp = [0.96, 0.04]  # solar wind composition. Default is 96% H+ and 4% He++
        self.update_impactor(self.impactor, self.sw_comp)
        self.ekin = {'H': 1000, 'He': 4000}  # dictionary of impactor energies (eV) in database
        self.energy_impactor = None  # e_kin determined by database
        self.mass_impactor = None  # mass determined by database

        self.v_esc = None  # 4300 m/s (Mercury); 2380 m/s (Moon)

        self.is_summed_up = True  # summs up particle data and returns single fit for composition
        self.return_amu_ion = False

        # --- OUTPUT ---
        self.yield_df = None  # mineral sputter yield information
        self.particledata_df = None  # mineral particle information for angular and energy distribution
        self.refitparticledata_df = None  # mineral particle information summed up element-wise and re-fitted
        self.yield_system_dfdf = None  # dataframe with nested dataframes containing particle data for each mineral

        # --- DEBUGGING ---
        self.show_plot = show_plot
        self.isDebug = False
        self.isVerbose = verbose  # SpuBase tells you what it's currently doing in detail
        self.force_reload = False  # if False, reads fit parameters from existing file if any

        # --- PLOTTING ---
        self.logplot = False  # sets y-axis of plots to be a log 10 scale
        self.plot_inset = False  # adds an inset for the energy distributions
        self.file_format = None  
        self.update_file_format('png')

        self.rho_minerals_df = None
        self.minfrac_df_volume = None  # mineral fraction in volume%
        self.minfrac_df_weight = None  # mineral fraction in wt%
        self.minfrac_df_volume_CIPW = None  # mineral fraction in weight% with default CIPW mineralogy (one Ol, one Px)
        self.rho_system_df = None
        self.plotting_key = None  # color palette used for seaborn plots
        self.adist_df = None  # dataframe with angular distribution data
        self.edist_df = None  # dataframe with energy distribution data
        self.plume_df = None  # dataframe with plume distribution data
        self.edist_loss_df = None  # dataframe with loss fraction data relative to escape velocity
        self.species_a = None  # array of elements in system
        self.mineral_a = None  # array of minerals in system
        self.minfrac_a = None  # array of mineral fractions

        self.min_not_included = ['Cor', 'Ap']
        self.dist_angle = 45  # default incident angle for outputs is 45°

        self.angles_a = [0, 10, 20, 30, 45, 48, 51, 54, 57, 60, 63, 66, 69, 72, 74, 76, 78, 80, 82, 84, 86, 88, 89]
        self.angles_expanded_a = np.linspace(min(self.angles_a), max(self.angles_a),
                                             num=max(self.angles_a)+1, dtype='int')

        self.params = ['theta_k', 'theta_tilt', 'theta_m', 'theta_n',
                       'plume_k', 'plume_tilt', 'plume_m', 'plume_n',
                       'energy_k', 'energy_e']

        # dictionaries and table data
        self.mindict_df_atoms = pd.read_csv(os.path.join(self.tabledir, 'minerals_atoms.txt'), header=0, index_col=0)
        table_dir_atoms = os.path.join(self.tabledir, "table1.txt")  # adapted SDTrimSP table for at properties
        self.at_density_df_atoms = pd.read_csv(table_dir_atoms, sep="\s+", usecols=['atomic_density'], index_col=0,
                                               skiprows=10, encoding='latin-1')
        table_dir_minerals = os.path.join(self.tabledir, "rho_minerals.txt")
        self.wt_density_df_minerals = pd.read_csv(table_dir_minerals, sep="\s+", header=0,
                                                  index_col=0, encoding='latin-1')
        self.wt_density_df_minerals_dic = self.wt_density_df_minerals.T.to_dict('index')['g/cm3']
        self.mindict_df_oxides = pd.read_csv(os.path.join(self.tabledir, 'minerals_oxides.txt'), header=0, index_col=0)
        table_dir_oxides = os.path.join(self.tabledir, "table_compound.txt")  # adapted SDTrimSP table for ox properties
        self.oxides_df = pd.read_csv(table_dir_oxides, sep="\s+", index_col=0, skiprows=5, encoding='latin-1')
        self.at_density_df_oxides = self.oxides_df['atomic_density']

        self.amu_elements = pd.read_csv(os.path.join(self.tabledir, 'amu_elements.txt'),
                                        index_col=0, header=0, delim_whitespace=True)
        self.amu_oxides = pd.read_csv(os.path.join(self.tabledir, 'amu_oxides.txt'),
                                      index_col=0, header=0, delim_whitespace=True)
        self.amu_minerals = pd.read_csv(os.path.join(self.tabledir, 'amu_minerals.txt'),
                                        index_col=0, header=0, delim_whitespace=True)

        self.amu_dic = self.amu_elements.T.to_dict('index')['amu']
        self.amu_oxides_dic = self.amu_oxides.T.to_dict('index')['amu']
        self.amu_minerals_dic = self.amu_minerals.T.to_dict('index')['amu']

        # --- DIRECTORIES ---
        self.casename = ''
        self.outdir = self.maindir
        curdir = os.path.abspath(os.curdir)
        self.DBdir = os.path.abspath(curdir)
        if self.isVerbose:
            print(f'DB directory: {self.DBdir}')

        self.sw_comparison = False  # todo: remove this check if not needed anymore

    # Imported methods
    from .src._fitting_sb import eckstein_fit_data
    from .src._cipw_sb import cipw_norm, surface_composition
    from .src._plotting_sb import plot_yield, create_color_palette, plot_dist

    def update_file_format(self, form='png'):
        self.file_format = form

    def update_directory(self):
        out_folder = os.path.join('output', self.casename)
        self.outdir = os.path.abspath(os.path.join(self.maindir, out_folder))

        dirs_create = [self.outdir]

        for loc in dirs_create:
            try:
                os.mkdir(loc)
            except:
                if self.isVerbose:
                    print(f"Directory already exists:\n{loc}")
            else:
                print(f"Output:\n{loc}\n(new)")

    def update_model(self, sbm, cpm):
        self.sfx = sbm + cpm
        print(f"Model: {self.sfx}")

    def update_impactor(self, impactor, comp_frac=None):
        self.impactor = impactor
        if self.impactor != 'SW':
            self.sulfur_diffusion = False
            print(f'Sulfur diffusion was ignored because {self.impactor} != "SW"')

        if comp_frac is not None and sum(comp_frac) == 1:
            self.sw_comp = comp_frac
            print(f'SW fraction is {self.sw_comp[0]:2.0%} H+ and {self.sw_comp[1]:2.0%} He2+')
            if comp_frac[0] != 0.96:
                print(f'SW composition deviates from database of [0.96, 0.04]!'
                      f'\nData of H and He are summed up instead.'
                      f'\n!!!This does affect yields (+/- 15% divergence) '
                      f'and angular distribution data (+/- 30% on tilt angle)!!!')

        elif comp_frac is not None and sum(comp_frac) != 1:
            print(f'The sum of the fractions must be 1!\nFraction was not updated from default of {self.sw_comp}')

        elif comp_frac is None:
            print(f'impactor changed to {self.impactor}')
            return

    def get_binary_e(self, target_species, spec_impactor=None):
        energy_impactor = self.energy_impactor
        mass_impactor = self.mass_impactor

        if spec_impactor and spec_impactor != 'SW':
            # average energy and mass of impactor,
            # given the solar wind ion composition
            energy_impactor = [*map(self.ekin.get, spec_impactor)][0]
            mass_impactor = [*map(self.amu_dic.get, spec_impactor)][0]

        mass_target = [*map(self.amu_dic.get, [target_species])]
        if target_species == 'amu/ion':
            mass_target = [0]
        binary_e = energy_impactor * 4 * mass_impactor * mass_target[0] * (
                mass_impactor + mass_target[0]) ** (-2)
        return binary_e

    def data_series(self):
        """
        Creates a dataframe with nested dataframes
        Structure:
             | Species 1                 | Species 2                 | ...
        alpha| Min1 | Min2 | Min3 |  ... | Min1 | Min2 | Min3 |  ... | ...
        0    |
        1    |
        2    |
        """
        self.yield_system_dfdf = pd.DataFrame(columns=self.species_a)

        mindict_df_atoms = self.mindict_df_atoms.loc[:, self.mindict_df_atoms.columns != 'total']

        mineral_suffix = self.mineral_a
        dist_fit_params = [param + '_' + sfx for sfx in mineral_suffix for param in self.params]
        dist_fit_header = np.concatenate([['alpha'], dist_fit_params])

        yield_df = pd.DataFrame(data=self.angles_expanded_a, columns=['alpha'])
        yield_df = yield_df.set_index('alpha')

        # if amu info is to be plotted, then the elements in the species_a are replaced by amu from here on out
        if self.return_amu_ion:
            self.species_a = ['amu/ion']
            # todo: add amu/ion to the species_a instead and then differentiate when calling?

        # create DF with a single column made up of the self.species_a components
        particle_data_df = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting

        data_dfdf, eckstein_dfdf = self.eckstein_fit_data()

        for ss, species in enumerate(self.species_a):
            collect_yield = np.zeros(len(self.angles_expanded_a))  # total yield at each step
            element_yield = pd.DataFrame(columns=self.mineral_a)

            particle_info = self.angles_expanded_a
            for ff, mineral in enumerate(self.mineral_a):

                # --- FIX Missing Entries whilst writing fitting params into dataframe ---
                # set a 'equal zero' in eckstein form if species is not present,
                # otherwise fetch eckstein components of species

                mineral_comp = mindict_df_atoms.loc[mineral]
                species_list = mineral_comp.loc[mineral_comp != 0].index.tolist()
                # species_list += ['amu/ion']  # todo: make sure that amu/ion is added to the output dataframe!

                species_is_in_mineral = species in species_list  # find out if the species is present in the dataset
                if self.return_amu_ion:
                    species_is_in_mineral = True

                iyield, iadist, iplume, iedist = Particles.particle_data_impactor_refit(self,
                                                                                        species,
                                                                                        data_dfdf[mineral][0],
                                                                                        species_is_in_mineral,
                                                                                        eckstein_dfdf[mineral][0],
                                                                                        self.minfrac_a[ff])

                element_yield[self.mineral_a[ff]] = iyield  # set element yield for mineral
                collect_yield = np.add(collect_yield, iyield)
                particle_info = np.c_[particle_info, iadist, iplume, iedist]  # adds up along second axis

            iparticledata_df = pd.DataFrame(particle_info, columns=dist_fit_header)
            yield_df[species] = collect_yield
            particle_data_df[species] = [iparticledata_df.loc[(iparticledata_df.loc[:,
                                                              iparticledata_df.columns != 'alpha'] != 0).any(axis=1)]]
            self.yield_system_dfdf[species] = [element_yield]
            self.particledata_df = particle_data_df

        yield_df = yield_df.fillna(0)
        self.yield_df = yield_df.loc[:, (yield_df != 0).any(axis=0)]  # drop all columns that are pure zeroes
        #self.yield_df = self.expand_series(self.yield_df)

        if self.is_summed_up and not self.return_amu_ion:
            self.particle_data_refit()

        if self.force_reload:
            print(
                f'Data for {self.casename} loaded\n'
            )
        else:
            print(
                  f'Data for {self.casename} created\n'
                  )

    def particle_data_impactor_refit(self, species, data_df, species_is_in_mineral, eckstein_df, iminfrac):

        # Initiate outputs
        iyield = []
        iadist = []
        iplume = []
        iedist = []

        if not self.sw_comparison:  # todo: remove this check and the content of the 'else'

            if 'SW' in self.impactor and self.sw_comp[0] == 0.96:
                impactors_a = ['SW']
                sw_comp = [1]

            elif 'SW' in self.impactor:
                impactors_a = ['H', 'He']
                sw_comp = self.sw_comp
            else:
                impactors_a = [self.impactor]
                sw_comp = [1]

        else:

            if 'SW' in self.impactor:
                impactors_a = ['H', 'He']
                sw_comp = self.sw_comp
            else:
                impactors_a = [self.impactor]
                sw_comp = [1]

        nr_datapoints = 500
        nr_xyz_points = 20
        theta_a = np.linspace(-np.pi/2, np.pi/2, num=nr_datapoints)
        energy_a = np.linspace(0, 500, num=nr_datapoints)

        theta_3d = np.linspace(0, np.pi/2, int(nr_xyz_points))  # polar angle
        phi_3d = np.linspace(-np.pi, np.pi, int(nr_xyz_points))
        plume_mesh = list(itertools.product(theta_3d, phi_3d))
        plume_df = pd.DataFrame(data=plume_mesh, columns=['theta', 'phi'])

        for aa, angle in enumerate(self.angles_expanded_a):

            iyield_mb = 0

            iadist_params = [0, 0, 0, 0]
            iplume_params = [0, 0, 0, 0]
            iedist_params = [0, 0]

            a_isumpart = np.zeros(len(theta_a))
            p_isumpart = np.zeros(len(plume_mesh))
            e_isumpart = np.zeros(len(energy_a))
            for ii, impactor in enumerate(impactors_a):
                if species_is_in_mineral:
                    # write component into dataframe column
                    eckparam = eckstein_df[impactor][0]
                    iyield_mb += sw_comp[ii] * eckstein(angle, *eckparam[species])

                    if angle in self.angles_a and not self.return_amu_ion:
                        data = data_df[impactor][0]
                        theta_k = data[f'thetaK_{species}'].loc[angle]
                        theta_tilt = data[f'thetaTilt_{species}'].loc[angle]
                        theta_m = data[f'thetaM_{species}'].loc[angle]
                        theta_n = data[f'thetaN_{species}'].loc[angle]
                        plume_k = data[f'3dK_{species}'].loc[angle]
                        plume_tilt = data[f'3dTilt_{species}'].loc[angle]
                        plume_m = data[f'3dM_{species}'].loc[angle]
                        plume_n = data[f'3dN_{species}'].loc[angle]
                        energy_k = data[f'energyK_{species}'].loc[angle]
                        energy_e = data[f'energyE_{species}'].loc[angle]

                        iadist_params = [theta_k, theta_tilt, theta_m, theta_n]
                        iplume_params = [plume_k, plume_tilt, plume_m, plume_n]
                        iedist_params = [energy_k, energy_e]

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fxn()
                            binary_e = self.get_binary_e(species, impactor)
                            adist_particles = np.array(cosfit(theta_a, *iadist_params)) * sw_comp[ii]
                            plume_particles = lobe_function(iplume_params, plume_df)['rho'] * sw_comp[ii]
                            edist_particles = np.array(thompson(energy_a, *iedist_params, binary_e)) * sw_comp[ii]
                        a_isumpart = np.add(a_isumpart, np.nan_to_num(adist_particles, copy=False))
                        p_isumpart = np.add(p_isumpart, np.nan_to_num(plume_particles, copy=False))
                        e_isumpart = np.add(e_isumpart, np.nan_to_num(edist_particles, copy=False))

            if len(impactors_a) > 1 and angle in self.angles_a and species_is_in_mineral and not self.return_amu_ion:
                theta_df = pd.DataFrame([theta_a, a_isumpart]).T
                energy_df = pd.DataFrame([energy_a, e_isumpart]).T
                plume_df['rho'] = p_isumpart
                init_guess = iplume_params
                init_guess[0] = plume_df['rho'].max()

                iadist_params, a_pcov = fit_eq(theta_df, eq=cosfit)

                iplume_params = fit_eq(plume_df, eq=lobe_function, init_guess=init_guess)
                iedist_params, e_pcov = fit_eq(energy_df, eq=thompson, binary_e=binary_e)

            iyield.append(iyield_mb)
            iadist.append(iadist_params)
            iplume.append(iplume_params)
            iedist.append(iedist_params)

        iyield = iminfrac * np.array(iyield)  # multiply calculated yields with mineral fraction

        return iyield, iadist, iplume, iedist

    def particle_data_refit(self):
        # For each impact angle
        # Create data points for each element with mineral-specific fit parameters
        # Scale the data points with the mineral-specific element yield
        # Sum up the data points of one element between all minerals
        # Fit the summed up datapoints and store the fit parameters
        # The output is a nested dataframe with element columns on the first layer
        # and fit parameter dataframes on the second layer

        # creat suffix for exported data
        name_comp_dict = self.minfrac_df_volume['frac'].apply(lambda x: '{:,.2f}'.format(x)).to_dict()
        icomp_sfx = []
        [icomp_sfx.extend([k, v]) for k, v in name_comp_dict.items()]
        comp_sfx = ''.join(icomp_sfx)

        export_dir = os.path.join(self.outdir, f'{self.impactor}_{self.casename}_particle_data_{comp_sfx}.txt')
        self.refitparticledata_df = pd.DataFrame(columns=self.species_a)

        mindict_df_atoms = self.mindict_df_atoms.loc[:, self.mindict_df_atoms.columns != 'total']
        if os.path.isfile(export_dir) and self.is_summed_up and not self.force_reload :

            single_file_df = pd.read_csv(export_dir, sep=',')

            for species in self.species_a:
                species_params = [entry for entry in single_file_df.columns if entry.endswith(species)]

                self.refitparticledata_df[species] = [single_file_df[species_params[1:]]]
                self.refitparticledata_df[species][0].columns = self.params
                if len(self.refitparticledata_df[species][0].index) < 80:
                    self.refitparticledata_df[species][0].index = self.angles_a

            return

        nr_datapoints = 500
        nr_xyz_points = 20
        theta_a = np.linspace(-np.pi/2, np.pi/2, num=nr_datapoints)
        energy_a = np.linspace(0, 500, num=nr_datapoints)

        theta_3d = np.linspace(0, np.pi/2, int(nr_xyz_points))  # polar angle
        phi_3d = np.linspace(-np.pi, np.pi, int(nr_xyz_points))
        plume_mesh = list(itertools.product(theta_3d, phi_3d))
        plume_df = pd.DataFrame(data=plume_mesh, columns=['theta', 'phi'])

        print("\nSum up particles from minerals and perform a re-fit\n"
              "Progress:")
        for ss, species in enumerate(self.species_a):

            total_progress = 100 / len(self.species_a)
            print(f'{total_progress * ss:0.1f}% {species}')

            isummed = pd.DataFrame(columns=['alpha', *self.params])
            isummed['alpha'] = self.angles_a
            isummed.set_index('alpha', inplace=True)

            fitparam_df = self.particledata_df[species][0]

            if 'alpha' in fitparam_df.columns.tolist():
                fitparam_df.set_index('alpha', inplace=True)

            for aa, angle in enumerate(self.angles_a):
                if (aa+1) % 5 == 0:
                    print(f'{total_progress  * ss + total_progress/len(self.angles_a) * aa:0.1f}%')
                a_isumpart = np.zeros(len(theta_a))
                e_isumpart = np.zeros(len(energy_a))
                p_isumpart = np.zeros(len(plume_mesh))
                for mm, mineral in enumerate(self.mineral_a):
                    mineral_comp = mindict_df_atoms.loc[mineral]
                    species_list = mineral_comp.loc[mineral_comp != 0].index.tolist()
                    # species_list += ['amu/ion']  # todo: make sure that amu/ion is added to the output dataframe!
                    species_is_in_mineral = species in species_list  # find out if the species is present in the dataset

                    if species_is_in_mineral:
                        param_minsfx = [param + f'_{mineral}' for param in self.params]
                        fitparams = fitparam_df[param_minsfx]  # all fit parameters of the given mineral
                        iadist_params = fitparams[param_minsfx[0:4]].iloc[int(aa)]
                        iplume_params = fitparams[param_minsfx[4:8]].iloc[int(aa)]
                        iedist_params = fitparams[param_minsfx[8:]].iloc[int(aa)]
                        binary_e = self.get_binary_e(species)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fxn()
                            mineral_element_yield = self.yield_system_dfdf[species][0][mineral].iloc[aa]
                            adist_particles = np.array(cosfit(theta_a, *iadist_params)) * mineral_element_yield
                            pdist_particles = lobe_function(iplume_params, plume_df)['rho'] * mineral_element_yield
                            edist_particles = np.array(thompson(energy_a, *iedist_params, binary_e)) * mineral_element_yield
                        a_isumpart = np.add(a_isumpart, np.nan_to_num(adist_particles, copy=False))
                        e_isumpart = np.add(e_isumpart, np.nan_to_num(edist_particles, copy=False))
                        p_isumpart = np.add(p_isumpart, np.nan_to_num(pdist_particles, copy=False))
                    # elif angle == 0 and not species_is_in_mineral:
                    #     print(f'{species} not in {mineral}')


                if len(self.mineral_a) > 1:
                    theta_df = pd.DataFrame([theta_a, a_isumpart]).T
                    energy_df = pd.DataFrame([energy_a, e_isumpart]).T
                    plume_df['rho'] = p_isumpart
                    init_guess = iplume_params
                    init_guess.iloc[0] = plume_df['rho'].max()
                    iadist_params, a_pcov = fit_eq(theta_df, eq=cosfit)  # angle fit
                    iplume_params = fit_eq(plume_df, eq=lobe_function, init_guess=init_guess)
                    iedist_params, e_pcov = fit_eq(energy_df, eq=thompson, binary_e=binary_e)  # energy fit

                isummed.loc[int(angle)] = np.concatenate((iadist_params, iplume_params, iedist_params), axis=None)

                if self.isDebug:
                    """
                    DEBUG: print summed data and fit
                    """
                    if angle in self.angles_a[::10]:
                        fig_adist_polar = plt.figure(dpi=300, figsize=(2*4, 1*4))
                        ax_adist = fig_adist_polar.add_subplot(1, 2, 1, projection="polar")
                        ## data:
                        plot_yield_df = pd.DataFrame(index=theta_a,
                                                     data=a_isumpart,
                                                     columns=['amu/ion'])
                        plot_yield_df = plot_yield_df/plot_yield_df.max()

                        sns.lineplot(ax=ax_adist,
                                     data=plot_yield_df,
                                     palette=['#898989'],
                                     alpha=1.0)

                        ## fit:
                        plot_yieldFit_df = pd.DataFrame(index=(np.linspace(np.pi / 2, -np.pi / 2, 50)),
                                                           data=(cosfit(np.linspace(np.pi / 2,
                                                                                  -np.pi / 2,
                                                                                  50),
                                                                      *iadist_params)),
                                                           columns=[species])
                        plot_yieldFit_df = plot_yieldFit_df/plot_yieldFit_df.max()
                        sns.lineplot(ax=ax_adist,
                                     data=plot_yieldFit_df,
                                     palette=['#C85200'],
                                     alpha=1.0)
                        fig_adist_polar.savefig(self.outdir + f'adist_{species}_debug-{angle}.{self.file_format}',
                                                  dpi=300, format=self.file_format)
            self.refitparticledata_df[species] = [isummed]

            # expand 23 angles to 89 by interpolating fit parameters and re-fitting scaling parameters
            self.refitparticledata_df[species] = [self.expand_series(self.refitparticledata_df[species].iloc[0], species)]
        """
        Export data for each angle separately
        """
        # for angle in self.angles_a:
        #     refitparticledata_df = pd.DataFrame(index=self.species_a, columns=self.params)
        #     refitparticledata_df.index.name = 'element'
        #
        #     for species in self.species_a:
        #         refitparticledata_df.loc[species] = self.refitparticledata_df[species][0].loc[angle]
        #     refitparticledata_df['at/ion'] = self.yield_df.T[angle]
        #
        #     output_df = refitparticledata_df.copy(deep=True)
        #     output_format = ['{:.3f}', '{:.3E}', '{:.3f}', '{:.3f}', '{:.3E}', '{:.3f}', '{:.3E}']
        #     for cc, column in enumerate(output_df.columns.tolist()):
        #         output_df[column] = output_df[column].map(lambda x: output_format[cc].format(x))
        #
        #     output_df.to_csv(
        #         os.path.join(self.outdir, f'{self.impactor}_{self.casename}_refit_particle_data-{int(angle)}.txt'),
        #         sep=';')

        """
        Export data in single file
        """
        params_spec_sfx = [[param + '_' + species for param in self.params] for species in self.species_a]
        params_spec_sfx = list(np.append(self.species_a, params_spec_sfx))

        single_file_df = pd.DataFrame(index=self.refitparticledata_df[self.species_a[0]][0].index,
                                      columns=params_spec_sfx)
        single_file_df.index.name = 'alpha'

        single_file_df[self.species_a] = self.yield_df

        for species in self.species_a:
            species_params = [entry for entry in params_spec_sfx if entry.endswith(species)]
            single_file_df[species_params[1:]] = self.refitparticledata_df[species][0].values

        single_file_df = single_file_df.apply(pd.to_numeric, downcast='float').fillna(0)
        single_file_df.index = pd.to_numeric(single_file_df.index, downcast='integer')

        single_file_df.to_csv(export_dir, sep=',', float_format='{:.3E}'.format)
        if self.isVerbose:
            print(f'Data exported as .csv to {export_dir}')

    def sputtered_particles_info(self, angles_l, species_l, param=''):
        for angle in angles_l:
            for species in species_l:
                df_of_interest = self.particledata_df[species][0].loc[angle]

                if param == '':
                    print(f' %%%%%%%%%%%%%%%% {species}  %%%%%%%%%%%%%%%%')
                    print(df_of_interest)
                elif param in ('theta_', 'energy_e'):
                    data_of_interest = df_of_interest[df_of_interest.index.str.contains(param)]
                    print(f'\n%%%%%%%%%%%%%%% {species} @ {angle}° %%%%%%%%%%%%%%%%\n')
                    if param == 'theta_tilt':
                        print(f'Peak of sputtered {species} angular distribution'
                              '(angle/° relative to surface normal, away from impinging ion)\n')
                        print(abs(data_of_interest / np.pi * 180))

                    elif param == 'energy_e':
                        print(f'Peak of sputtered {species} energy distribution'
                              '(in eV)\n')
                        print(data_of_interest)

                    else:
                        print(data_of_interest)

                else:
                    print(f'{param} not found in data of interest. Pass either none \'\', \'Phi\' or \'energy_bind\'')
        return df_of_interest

    def sputtered_particles_data(self,
                                 dist_angle=None,
                                 energy_a=None):  # returns scaled function for particles from dist and adist
        if dist_angle != None:
            self.dist_angle = dist_angle

        # if self.isMassBalanced:
        #     info_df = particles.mass_balance_part_info(self, self.dist_angle)

        elif self.is_summed_up:
            info_df = self.refitparticledata_df

        else:
            info_df = self.particledata_df

        if ['amu/ion'] == self.species_a:
            print('Not supported if \'return_amu_ion = True\'')
            sys.exit()

        e_max = 100
        nr_datapoints = 1000
        nr_xyz_points = 50
        if energy_a is None:
            energy_a = np.linspace(0.1, e_max,
                                   num=nr_datapoints)  # if no energy range is defined, 0.1 -> 100 eV is used
            energy_bin = e_max / nr_datapoints
        theta_a = np.linspace(-np.pi/2, np.pi/2, num=nr_datapoints)  # full angular distribution is always necessary
        theta_bin = 2 * np.pi / nr_datapoints

        edist_df: pd.DataFrame = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting
        edist_loss_df: pd.DataFrame = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting
        adist_df: pd.DataFrame = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting
        plume_df: pd.DataFrame = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting

        theta_3d = np.linspace(0, np.pi/2, int(nr_xyz_points))  # polar angle
        phi_3d = np.linspace(-np.pi, np.pi, int(nr_xyz_points))
        xyz_bin = np.pi/2 / nr_xyz_points + 2*np.pi / nr_xyz_points
        plume_mesh = list(itertools.product(theta_3d, phi_3d))
        if self.v_esc:
            print(
                  f'Loss fraction\n'
                  f'(> {self.v_esc} m/s)'
                  )
            print()
        for ss, species in enumerate(self.species_a):
            iout_df_energy: pd.DataFrame = pd.DataFrame(data=energy_a, columns=['energy'])
            iout_df_energy_loss: pd.DataFrame = pd.DataFrame(columns=[species], index=['loss'])
            iout_df_plume: pd.DataFrame = pd.DataFrame(data=plume_mesh, columns=['theta', 'phi'])
            iout_df_polar: pd.DataFrame = pd.DataFrame(data=theta_a, columns=['alpha'])

            binary_e = self.get_binary_e(
                species)  # maximum energy that can be transferred from impactor to species in BC

            if self.v_esc:
                v_esc = float(self.v_esc)
                amu = np.vectorize(self.amu_dic.get)(species)  # amu of species
                escape_e = (amu * v_esc ** 2) * 0.5 / 6.022e+26 * 6.242e+18  # calc escape energy in eV

            if self.is_summed_up:
                parameters = info_df[species][0].loc[self.dist_angle]
                if self.isDebug:
                    print(species)
                    print(*parameters)
                theta_k, theta_tilt, theta_m, theta_n, plume_k, plume_tilt, plume_m, plume_n, energy_k, energy_e = parameters.values.tolist()


                iout_df_polar[species] = cosfit(iout_df_polar['alpha'], theta_k, theta_tilt, theta_m, theta_n)
                iout_df_polar[species] = iout_df_polar[species].mul(theta_bin)


                iout_df_plume[species] = lobe_function([plume_k, plume_tilt, plume_m, plume_n], iout_df_plume)['rho']
                iout_df_plume[species] = iout_df_plume[species].mul(xyz_bin)

                iout_df_energy[species] = thompson(iout_df_energy['energy'], energy_k, energy_e, binary_e)
                iout_df_energy[species] = iout_df_energy[species].mul(energy_bin)


                if self.v_esc:
                    iout_df_energy_loss[species], _ = integrate.quad(thompson, escape_e, binary_e,
                                                            args=(energy_k, energy_e, binary_e))
                    print(f'{species}: {iout_df_energy_loss[species].values.tolist()[0]:0.2%}')
                else:
                    iout_df_energy_loss[species] = 'n.d'

                if self.isDebug:
                    k_theta_leq, k_theta_leq_error = integrate.quad(
                        lambda x: 1 / theta_k * np.power(np.cos((-abs(theta_tilt) - x) / (1 - 2 / np.pi * -abs(theta_tilt))),
                                                       theta_m),
                        -np.pi / 2,
                        -abs(theta_tilt))

                    k_theta_geq, k_theta_geq_error = integrate.quad(
                        lambda x: 1 / theta_k * np.power(np.cos((x - theta_tilt) / (1 - 2 / np.pi * theta_tilt)), theta_n),
                        theta_tilt,
                        np.pi / 2)

                    k_angle = k_theta_leq + k_theta_geq
                    k_energy, _ = integrate.quad(thompson, 0, binary_e, args=(energy_k, energy_e, binary_e))
                    # k_adist, k_adist_error = integrate.quad(lambda x: self.cosfit(x, theta_k, theta_tilt, theta_m, theta_n))
                    # k_edist, k_edist_error = integrate.quad(lambda x: self.thompson(x, energy_k, energy_e, binary_e))
                    print(k_angle)
                    print(k_energy)

            else:
                parameters = info_df[species][0].loc[self.dist_angle]
                for mineral in self.mineral_a:

                    theta_k, theta_tilt, theta_m, theta_n, plume_k, plume_tilt, plume_m, plume_n, energy_k, energy_e = \
                        parameters[parameters.index.str.contains(f'{mineral}(?!.)', regex=True)]

                    iout_df_polar[species + '_' + mineral] = cosfit(iout_df_polar['alpha'],
                                                                    theta_k, theta_tilt, theta_m, theta_n)
                    iout_df_polar[species + '_' + mineral] = iout_df_polar[species + '_' + mineral].mul(theta_bin)

                    iout_df_plume[species + '_' + mineral] = lobe_function([plume_k, plume_tilt, plume_m, plume_n],
                                                                           iout_df_plume)['rho']
                    iout_df_plume[species + '_' + mineral] = iout_df_plume[species + '_' + mineral].mul(xyz_bin)

                    iout_df_energy[species + '_' + mineral] = thompson(iout_df_energy['energy'],
                                                                       energy_k, energy_e, binary_e)
                    iout_df_energy[species + '_' + mineral] = iout_df_energy[species + '_' + mineral].mul(energy_bin)


                    if self.v_esc:
                        iout_df_energy_loss[species + '_' + mineral] = \
                            (energy_e * (energy_e + 2 * escape_e)) / (4 * (energy_e + escape_e) ** 2)
                    else:
                        iout_df_energy_loss[species + '_' + mineral] = 'n.d'

                    if self.isDebug:
                        print(mineral)
                        print(theta_k)
                        print(energy_k)
                        print(plume_k)
                        print(f"Integrated angular distribution:"
                              f"{sum(iout_df_polar[species + '_' + mineral].fillna(0))} (before flux)")
                        print(f"Integrated energy distribution:"
                              f"{sum(iout_df_energy[species + '_' + mineral].fillna(0))} (before flux)")
                        print(f"Loss of energy distribution:"
                              f"{sum(iout_df_energy_loss[species + '_' + mineral].fillna(0))} (before flux)")

            iout_df_polar = iout_df_polar.set_index('alpha').fillna(0)
            iout_df_energy = iout_df_energy.set_index('energy').fillna(0)
            iout_df_plume = iout_df_plume.set_index(['phi', 'theta']).fillna(0)

            edist_df[species] = [iout_df_energy]
            edist_loss_df[species] = [iout_df_energy_loss]
            adist_df[species] = [iout_df_polar]
            plume_df[species] = [iout_df_plume]

        self.edist_df = edist_df
        self.edist_loss_df = edist_loss_df
        self.adist_df = adist_df
        self.plume_df = plume_df

    def expand_series(self, df, species=None):
        ##########################################
        # add interpolated data
        ##########################################

        angles_range = self.angles_expanded_a

        entries = [0] * len(angles_range)

        df.insert(0, 'alpha', df.index)

        for angle in self.angles_a:
            entries[angle] = df.loc[df['alpha'] == angle].to_numpy()[0]

        angles_added = self.angles_a
        help_serie = df.copy(deep=True)

        while len(angles_added) < 80:
            for angle in angles_added[:-1]:

                idx_low, angle_low = closest_entry(angles_added, angle)
                idx_hi = idx_low + 1
                angle_high = angles_added[idx_hi]

                target_angle = int(math.floor((angle_high + angle_low) / 2))
                if target_angle not in angles_added:
                    entries[int(target_angle)] = (help_serie.iloc[idx_hi].values + help_serie.iloc[idx_low].values) / 2
                    entries[int(target_angle)][0] = int(target_angle)
                    angles_added = np.insert(angles_added, 0, int(target_angle))

                help_entries = [[entry for entry in entries[idx]] for idx in angles_added]
                help_serie = pd.DataFrame(help_entries, columns=list(help_serie.columns.values))
        expanded_serie = pd.DataFrame(entries, columns=help_serie.columns.values)

        if species:
            ''' Re-calculate scaling factor '''
            for row in expanded_serie.index:

                '''
                Define maximum transferred energy in binary collision
                '''

                e_bc = self.get_binary_e(species, spec_impactor=self.impactor)

                theta_tilt = expanded_serie[self.params[1]].loc[row]
                theta_m = expanded_serie[self.params[2]].loc[row]
                theta_n = expanded_serie[self.params[3]].loc[row]
                plume_tilt = expanded_serie[self.params[5]].loc[row]  # not used, only determined in fit
                plume_m = expanded_serie[self.params[6]].loc[row]
                plume_n = expanded_serie[self.params[7]].loc[row]
                energy_e = expanded_serie[self.params[9]].loc[row]

                theta_k_leq_tilt, theta_k_geq_tilt = cosfit_integrate(theta_tilt, theta_m, theta_n)

                # before = expanded_serie['phiK_'+species].loc[row]
                expanded_serie[self.params[0]].loc[row] = theta_k_leq_tilt + theta_k_geq_tilt
                phi_lower = -np.pi
                phi_upper = np.pi
                theta_lower = 0
                theta_upper = np.pi / 2
                expanded_serie[self.params[4]].loc[row], _ = integrate.dblquad(frho,
                                                                               theta_lower,
                                                                               theta_upper,
                                                                               lambda phi: phi_lower,
                                                                               lambda phi: phi_upper,
                                                                               args=([plume_m, plume_n],)
                                                                               )

                expanded_serie[self.params[8]].loc[species], _ = integrate.quad(
                    lambda E: E / (E + energy_e) ** 3 * (1 - ((E + energy_e) / e_bc) ** (1 / 2)),
                    0.1,
                    e_bc)
        expanded_serie.drop('alpha', inplace=True, axis=1)
        return expanded_serie

    def system_density(self):

        minfrac_wt = self.minfrac_df_weight  # use mass wt% to perform mass balance of output system density in g/cm^3
        amu_df = self.amu_minerals
        mindict_df_atoms = self.mindict_df_atoms.loc[:, self.mindict_df_atoms.columns != 'total']

        an = 0.60221367  # [mol/cm**3]

        def rho_mineral_atoms(mineral_name):
            min_df_i = mindict_df_atoms.loc[mineral_name]
            sum_i = 0
            for element in min_df_i.index:
                sum_i += min_df_i[element] / self.at_density_df_atoms.loc[element]

            atomic_density = sum_i ** -1  # [at./A^3]

            # atomic-density_[at./A^3] / an * Molare-Masse[g/mol] / N = density_[g/cm^3]
            number_atoms = self.mindict_df_atoms['total'].loc[
                mineral]  # round(sum(min_df_i / min_df_i[min_df_i > .01].min()))  # Number of atoms in mineral
            # fraction = min_df_i[min_df_i > .01].min()/min_df_i[min_df_i > .01].max()

            mass_density = amu_df.loc[mineral].values[0] / an / number_atoms * atomic_density.values[0]

            return mass_density, atomic_density.values[0]

        def rho_mineral_oxide(mineral):

            min_df_i = self.mindict_df_oxides.loc[mineral]
            sum_i = 0
            comp_i = 0

            for compound in min_df_i[min_df_i[:] > 0].index:
                comp_i += min_df_i[compound] * self.oxides_df['ctot'].loc[compound]
                sum_i += (min_df_i[compound] / min_df_i[min_df_i > .01].sum()) / self.at_density_df_oxides.loc[compound]

            atomic_density = sum_i ** (-1)  # [at./A^3]
            number_atoms = comp_i  # Number of atoms in mineral

            # mass density = atomic-density_[at./A^3] / an * Molare-Masse[g/mol] / N = density_[g/cm^3]
            mass_density = amu_df.loc[mineral].values[0] / an / number_atoms * atomic_density

            return mass_density, atomic_density

        mineral_l = self.mindict_df_atoms.index.tolist()  # minfrac.index.tolist()
        rho_df = pd.DataFrame(index=mineral_l, columns=['gcm3', 'atA3'])

        for mineral in mineral_l:
            if self.sfx[-1] == 'x':
                rho_df['gcm3'].loc[mineral], rho_df['atA3'].loc[mineral] = rho_mineral_oxide(mineral)
                type = 'oxide densities'

                if self.isDebug:
                    print(
                        f"{mineral}\t{rho_df['gcm3'].loc[mineral]:.2f} [g./cm^3]\t"
                        f"{rho_df['atA3'].loc[mineral]:.4f} [at./A^3]"
                    )

            else:
                rho_df['gcm3'].loc[mineral], rho_df['atA3'].loc[mineral] = rho_mineral_atoms(mineral)
                type = 'atomic densities'

                print(
                    f"{mineral}\t{rho_df['gcm3'].loc[mineral]:.2f} [g./cm^3]\t"
                    f"{rho_df['atA3'].loc[mineral]:.4f} [at./A^3]"
                )

        self.rho_minerals_df = rho_df
        self.rho_system_df = (rho_df['gcm3'].loc[minfrac_wt.index] * minfrac_wt['frac']).sum()
        print(f'System density from {type}: {self.rho_system_df:0.3f} g/cm-3')
