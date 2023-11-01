"""Sputter Database (SpuBase) class

See the LICENSE file for licensing information.
"""

import os
import re
import sys
import math
import warnings

import pandas

import pylab
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy.optimize import OptimizeWarning
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# import argparse
from pandas import DataFrame


def invpairdiff(lst):
    f_lst = lst.astype(float)  # floats are required for "** (-1)" tp work
    length = len(lst)
    total = np.ones(length)
    for i in range(length - 1):
        # adding the alternate numbers
        total[i] = (f_lst[i + 1] - f_lst[i]) ** (-1)
    return total


def safe_title_trans(problematic_title):
    table = str.maketrans({'\u2192': '_',
                           '>': '',
                           '-': '_',
                           '+': '_',
                           '.': None,
                           '(': '_',
                           ')': None,
                           ' ': '',
                           '/': '_'})
    safe_title = problematic_title.translate(table)
    return safe_title


def peak(x, y):
    max_int = np.max(y)
    center_idx = np.where(y == max_int)[0]
    return x[center_idx]


def rad2deg(x_rad):
    return x_rad * 180 / np.pi


def deg2rad(x_rad):
    return x_rad / 180 * np.pi


def fxn():
    warnings.warn("runtime", RuntimeWarning)
    warnings.warn("optimize", OptimizeWarning)


def sum_up_same(df, cp_target):
    summed_yield_df = df.copy(deep=True)

    cp_no_suffix = [s.split('_')[0] for s in cp_target]

    cp_no_suffix = np.unique(cp_no_suffix)

    for element in np.unique(cp_no_suffix):
        summed_yield_df[element] = summed_yield_df[[col for col in summed_yield_df.columns
                                                    if col.startswith(element)]].sum(axis=1)

    cp_sfx = [s for s in cp_target if (s not in cp_no_suffix)]

    summed_yield_df.drop(cp_sfx, inplace=True, axis=1)
    summed_yield_df.drop('alpha', inplace=True, axis=1)
    cp_target = cp_no_suffix

    return summed_yield_df, cp_target


def thompson(e, energy_k, energy_e, binary_e):
    # with energy e and the constants
    # energy_k (PDF scaling factor),
    # energy_e (binding energy), and
    # binary_e (maximum binary collision energy)
    n_part = e / (e + energy_e) ** 3 * (1 - ((e + energy_e) / binary_e) ** (1 / 2))
    if energy_k > 0:
        n_part /= energy_k
    else:
        n_part = 0
    return n_part


def eckstein(angle, b, c, f, y0):
    angle0 = 90  # in degrees
    return y0 * (np.cos((angle / angle0 * np.pi / 2) ** c)) ** (-f) * np.exp(
        (b * (1 - 1 / np.cos((angle / angle0 * np.pi / 2) ** c))))


def cosfit(angle, cos_k, cos_tilt, cos_m, cos_n):
    # for x > cos_tilt
    leqcos_tilt = np.power(np.abs(np.cos((cos_tilt - angle) / (1 - 2 / np.pi * cos_tilt))), cos_m)
    geqcos_tilt = np.power(np.abs(np.cos((angle - cos_tilt) / (1 - 2 / np.pi * cos_tilt))), cos_n)
    # for x < cos_tilt
    nr_part = np.heaviside(cos_tilt - angle, 0.5) * leqcos_tilt + \
              np.heaviside(angle - cos_tilt, 0.5) * geqcos_tilt
    if cos_k > 0:
        nr_part /= cos_k
    else:
        nr_part = 0
    return nr_part


def fit_eq(xdata, ydata, eq, binary_e=1e5):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
        # mask all values that are too close to the origin (makes it nigh impossible to fit)
        if eq == cosfit:
            maskforfit = [-1.4 < x < 0.8 for x in xdata]
            xdata = xdata[maskforfit]
            ydata = ydata[maskforfit]
            popt, pcov = curve_fit(f=eq,
                                   xdata=xdata,
                                   ydata=ydata,
                                   # bounds=([-np.pi/3,0.8, 0.8], [np.pi/3, 3,3]),
                                   p0=[xdata.max(), deg2rad(-5), 2, 1],
                                   # p0=[-40/deg_from_rad,2,1],
                                   # sigma=ydata**-2,
                                   # method='trf',
                                   maxfev=500)

            """
            Determine normalization factor
            """
            phi_tilt = popt[1]
            phi_m = popt[2]
            phi_n = popt[3]
            k_leq, k_leq_error = integrate.quad(
                lambda x: np.power(np.cos((-abs(phi_tilt) - x) / (1 - 2 / np.pi * -abs(phi_tilt))), phi_m),
                -np.pi / 2, -abs(phi_tilt))

            k_geq, k_geq_error = integrate.quad(
                lambda x: np.power(np.cos((x - phi_tilt) / (1 - 2 / np.pi * phi_tilt)), phi_n), phi_tilt,
                np.pi / 2)
            popt[0] = k_leq + k_geq

        elif eq == thompson:
            popt, pcov = curve_fit(eq, xdata, ydata, p0=[0.5,
                                                         xdata[np.where(ydata == ydata.max())][0],
                                                         binary_e],
                                   maxfev=1000)
            energy_e = popt[1]
            # popt[0] = integrate.quad(eq, 0, args=(1, energy_e, binary_e))
            popt[0], _ = integrate.quad(lambda E: E / (E + energy_e) ** 3
                                                  * (1 - ((E + energy_e) / binary_e) ** (1 / 2)),
                                        0.1,
                                        binary_e)

            popt = popt[:2]  # drop binary_e again
        return popt, pcov


class Particles:
    def __init__(self, verbose=False, show_plot=False):

        # --- DATA MODEL ---
        self.sfx = None
        self.update_model('sbb', 'x')
        # binding model - surface and bulk binding model (sbb)
        # component model - oxides (x)
        self.impactor = 'SW'  # default is solar wind 'SW', alternatives are 'H' and 'He'
        if self.impactor == 'SW':
            self.sulfur_diffusion = True
            print(f'Sulfur diffusion: {self.sulfur_diffusion}')
        self.sw_comp = [0.96, 0.04]  # solar wind composition. Default is 96% H+ and 4% He++
        self.update_impactor(self.impactor, self.sw_comp)
        self.ekin = {'H': 1000, 'He': 4000}  # dictionary of impactor energies (eV) in database
        self.energy_impactor = None  # e_kin determined by database
        self.mass_impactor = None  # mass determined by database

        self.isSummedUp = True  # summs up particle data and returns single fit for composition
        self.return_amu_ion = False

        # --- OUTPUT ---
        self.yield_df = None  # mineral sputter yield information
        self.particledata_df = None  # mineral particle information for angular and energy distribution
        self.refitparticledata_df = None  # mineral particle information summed up element-wise and re-fitted
        self.yield_system_dfdf = None  # dataframe with nested dataframes containing particle data for each mineral

        # --- DEBUGGING ---
        self.show_plot = show_plot
        self.isDebug = False
        self.isVerbose = verbose

        # --- PLOTTING ---
        self.logplot = False
        self.plot_inset = False
        self.file_format = None
        self.update_file_format('png')

        self.rho_minerals_df = None
        self.minfrac_df_molar = None  # mineral fraction in mole% (Ol, Px and Plag are separated into end-members)
        self.minfrac_df_weight = None  # mineral fraction in weight%
        self.minfrac_df_weight_CIPW = None  # mineral fraction in weight% with default CIPW mineralogy (one Ol, one Px)
        self.rho_system_df = None
        self.plotting_key = None  # color palette used for seaborn plots
        self.adist_df = None  # dataframe with angular distribution data
        self.edist_df = None  # dataframe with energy distribution data
        self.edist_loss_df = None  # dataframe with loss fraction data relative to escape velocity
        self.v_esc = None  # 4300 m/s (Mercury); 2380 m/s (Moon)
        self.species_a = None  # array of elements in system
        self.mineral_a = None  # array of minerals in system
        self.minfrac_a = None  # array of mineral fractions

        self.min_not_included = ['Cor']
        self.dist_angle = 45  # default incident angle for outputs is 45°

        self.angles_a = list(np.linspace(0.0, 88.0, num=89))
        self.params = ['phi_k', 'phi_tilt', 'phi_m', 'phi_n', 'energy_k', 'energy_e']

        # dictionaries and table data
        self.mindict_df_atoms = pd.read_csv('tables/minerals_atoms.txt', header=0, index_col=0)
        table_dir_atoms = "tables/table1.txt"  # table for atomic properties; adapted from standard SDTrimSP table
        self.at_density_df_atoms = pd.read_csv(table_dir_atoms, sep="\s+", usecols=['atomic_density'], index_col=0,
                                               skiprows=10, encoding='latin-1')
        table_dir_minerals = "tables/rho_minerals.txt"
        self.wt_density_df_minerals = pd.read_csv(table_dir_minerals, sep="\s+", header=0,
                                                  index_col=0, encoding='latin-1')
        self.wt_density_df_minerals_dic = self.wt_density_df_minerals.T.to_dict('index')['g/cm3']
        self.mindict_df_oxides = pd.read_csv('tables/minerals_oxides.txt', header=0, index_col=0)
        table_dir_oxides = "tables/table_compound.txt"  # table for oxide properties from standard SDTrimSP table
        self.oxides_df = pd.read_csv(table_dir_oxides, sep="\s+", index_col=0, skiprows=5, encoding='latin-1')
        self.at_density_df_oxides = self.oxides_df['atomic_density']

        self.amu_elements = pd.read_csv('tables/amu_elements.txt', index_col=0, header=0, delim_whitespace=True)
        self.amu_oxides = pd.read_csv('tables/amu_oxides.txt', index_col=0, header=0, delim_whitespace=True)
        self.amu_minerals = pd.read_csv('tables/amu_minerals.txt', index_col=0, header=0, delim_whitespace=True)

        self.amu_dic = self.amu_elements.T.to_dict('index')['amu']
        self.amu_oxides_dic = self.amu_oxides.T.to_dict('index')['amu']
        self.amu_minerals_dic = self.amu_minerals.T.to_dict('index')['amu']

        # --- DIRECTORIES ---
        self.casename = ''
        self.maindir = os.path.abspath(os.curdir)
        self.outdir = self.maindir
        curdir = os.path.abspath(os.curdir)
        self.DBdir = os.path.abspath(curdir)
        if self.isVerbose:
            print(f'DB directory: {self.DBdir}')

        self.sw_comparison = False  # todo: remove this check if not needed anymore

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
        self.impactor = impactor
        if comp_frac is not None and sum(comp_frac) == 1:
            self.sw_comp = comp_frac
            print(f'SW fraction is {self.sw_comp[0]:2.0%} H+ and {self.sw_comp[1]:2.0%} He2+')
            if comp_frac[0] != 0.96:
                print(f'SW composition deviates from database of [0.96, 0.04]!'
                      f'\nData of H and He are summed up instead.'
                      f'\n!!!This does affect yields (+/- 15% divergence)'
                      f'and angular distribution data (+/- 30% on tilt angle)!!!')
        elif comp_frac is not None and sum(comp_frac) != 1:
            print(f'The sum of the fractions must be 1!\nFraction was not updated from default of {self.sw_comp}')
        elif comp_frac is None:
            print(f'impactor changed to {self.impactor}')
            return

    def sfx_for_plot(self, drop_prefix=True):
        """
        Replaces short forms with human-readable descriptions. Used in legends and titles.
        """
        numbers = re.compile(r'(\d+)')

        sfx = [self.sfx]
        sfx = [sf.replace("exp_", "") for sf in sfx]
        sfx = [sf.replace("sbbxd", "_HB-CD") for sf in sfx]
        sfx = [sf.replace("sbba", "_HB") for sf in sfx]
        sfx = [sf.replace("sbad", "_SB-D") for sf in sfx]
        sfx = [sf.replace("sbbxf", "_HB-C") for sf in sfx]
        sfx = [sf.replace("sbbx", "_HB-C") for sf in sfx]

        sfx = [sf.replace("bba", "_BB") for sf in sfx]
        sfx = [sf.replace("bbx", "_BB-C") for sf in sfx]

        sfx = [sf.replace("sba", "_SB") for sf in sfx]
        sfx = [sf.replace("sbx", "_SB-C") for sf in sfx]
        sfx = [sf.replace("sz2", "SDTrimSP:Szabo+2020") for sf in sfx]
        sfx = [sf.replace("mor", "SDTrimSP:Morrissey+2022") for sf in sfx]
        sfx = [sf.replace("+", "_et_al._") for sf in sfx]

        sfx = [numbers.sub(r'(\1)', sf) for sf in sfx]  # puts brackets around number

        sfx = [sf.replace("sbas", "_SB_(static)") for sf in sfx]
        sfx = [sf.replace("sbxs", "_SB-C_(static)") for sf in sfx]

        sfx = [sf.replace("_SB", "SDTrimSP:SB") for sf in sfx]
        sfx = [sf.replace("_BB", "SDTrimSP:BB") for sf in sfx]
        sfx = [sf.replace("_HB", "SDTrimSP:HB") for sf in sfx]
        sfx = [sf.replace("SDTrimSP_", "SDTrimSP:") for sf in sfx]
        sfx = [sf.replace("TRIM", "TRIM:") for sf in sfx]
        sfx = [sf.replace("_", " ") for sf in sfx]
        if drop_prefix:
            sfx = [sf.replace("SDTrimSP:", "") for sf in sfx]
            sfx = [sf.replace("TRIM:", "") for sf in sfx]
        return sfx[0]

    def get_binary_e(self, target_species, spec_impc=None):
        if spec_impc == 'SW' or spec_impc is None:
            energy_impactor = self.energy_impactor
            mass_impactor = self.mass_impactor

        elif spec_impc:
            # average energy and mass of impactor,
            # given the solar wind ion composition
            energy_impactor = [*map(self.ekin.get, spec_impc)][0]
            mass_impactor = [*map(self.amu_dic.get, spec_impc)][0]

        mass_target = [*map(self.amu_dic.get, [target_species])]
        if target_species == 'amu/ion':
            mass_target = [0]
        binary_e = energy_impactor * 4 * mass_impactor * mass_target[0] * (
                mass_impactor + mass_target[0]) ** (-2)
        return binary_e

    def surfcomp(self, comp_df=None, form=None):
        """
        Determine atomic composition based on mineral composition
        """

        if self.minfrac_df_weight is None:
            # Cleans up passed composition dataframe by dropping zero columns
            # and any mineral that is not part of the databse.
            comp_df = comp_df.loc[:, (comp_df != 0).any(axis=0)]
            comp_df.drop(self.min_not_included, inplace=True, errors='ignore', axis=1)

            if form == 'mol%':
                minfrac_wt = comp_df.mul(pd.Series(self.amu_minerals_dic))
                minfrac_wt.dropna(inplace=True, axis=1)
                minfrac_wt = minfrac_wt.T
                minfrac_wt /= minfrac_wt.sum(axis=0).values[0]
                self.minfrac_df_weight = minfrac_wt
                self.minfrac_df_molar = comp_df.T
            elif form == 'wt%':
                minfrac_mol = comp_df.div(pd.Series(self.amu_minerals_dic))
                minfrac_mol.dropna(inplace=True, axis=1)
                minfrac_mol = minfrac_mol.T
                minfrac_mol /= minfrac_mol.sum(axis=0).values[0]
                self.minfrac_df_weight = minfrac_mol
                self.minfrac_df_molar = comp_df.T

            else:
                print('Weight percent mineral fractions were assumed.\nPass form="mol%" instead for molar fractions\n')
                minfrac_mol = comp_df.div(pd.Series(self.amu_minerals_dic))
                minfrac_mol.dropna(inplace=True, axis=1)
                minfrac_mol = minfrac_mol.T
                minfrac_mol /= minfrac_mol.sum(axis=0).values[0]
                self.minfrac_df_weight = minfrac_mol
                self.minfrac_df_molar = comp_df.T
        else:
            comp_df = self.minfrac_df_weight.T
            self.minfrac_df_weight = self.minfrac_df_weight.loc[:, (self.minfrac_df_weight != 0).any(axis=0)]
            self.minfrac_df_weight.drop(self.min_not_included, inplace=True, errors='ignore', axis=1)

        # Creates separate array for mineral names and mineral fractions
        try:
            self.mineral_a = [col for col in comp_df.columns]
        except:
            self.mineral_a = [col for col in comp_df.columns]

        self.minfrac_a = self.minfrac_df_molar.T.values.tolist()[0]

        # Get mineral dictionary that connects elements to minerals
        mindict_df_atoms = self.mindict_df_atoms.loc[:, self.mindict_df_atoms.columns != 'total']
        out = []
        for mm, mineral in enumerate(self.mineral_a):
            out.append(mindict_df_atoms.loc[mineral].mul(self.minfrac_a[mm]).values.tolist())
        mincomp_df = pd.DataFrame(out, columns=list(mindict_df_atoms), index=self.mineral_a)
        sumcomp_df = mincomp_df.sum()
        sumcomp_df = sumcomp_df[~(sumcomp_df == 0)]  # drop all rows that are zero

        # Print mineral composition and fraction of total mineralogy present within database
        if self.isVerbose:
            print(f'Sum of mineral fractions available in DataBase is {sumcomp_df.sum():0.2f}/1.00\n')

        # obtain all non-zero elements that are present
        self.species_a = sumcomp_df.index.values.tolist()

    def eckstein_fit_data(self):

        if 'SW' in self.impactor:
            impactors_a = ['H', 'He']
            sw_comp = self.sw_comp
        else:
            impactors_a = [self.impactor]
            sw_comp = [1]

        # average energy and mass of impactor, given the solar wind ion composition
        self.energy_impactor = sum([*map(self.ekin.get, impactors_a)] * np.array(sw_comp))
        self.mass_impactor = sum([*map(self.amu_dic.get, impactors_a)] * np.array(sw_comp))

        if not self.sw_comparison:  # todo: remove this check
            if 'SW' in self.impactor and self.sw_comp[0] == 0.96:
                impactors_a = ['SW']
                sw_comp = [1]

        eckstein_dfdf = pd.DataFrame(columns=self.mineral_a)
        data_dfdf = pd.DataFrame(columns=self.mineral_a)
        for mm, mineral in enumerate(self.mineral_a):
            if self.isDebug:
                print(f'{mineral}')
            data_df = pd.DataFrame(columns=impactors_a)
            eckstein_df = pd.DataFrame(columns=impactors_a)
            sfx = self.sfx
            if self.sulfur_diffusion and mineral in ['Tro', 'Nng', 'Abd', 'Bzn', 'Was', 'Old', 'Dbr']:
                sfx = 'sbbxd'

            for ii, impactor in enumerate(impactors_a):
                input_dir = os.path.join(self.maindir, 'data', sfx)
                data_dir = os.path.join(input_dir, f'{impactor}_{mineral}_{sfx}_part_info.txt')

                data = pd.read_csv(data_dir, encoding='utf-8')
                data.set_index('alpha', drop=False, inplace=True)
                data.fillna(0, inplace=True)

                """
                Eckstein fit of yield for all components
                """
                columns = list(data.columns)
                cp_target = columns[columns.index('alpha') + 1:columns.index('amu/ion') + 1]

                angles = list(data['alpha'])

                element_yield_df, cp_aux = sum_up_same(data, cp_target)

                eckstein_param_df = pandas.DataFrame(columns=cp_aux)
                for element in cp_aux:

                    if len(angles) == 1:
                        break
                    x_values = element_yield_df.index.values
                    y_values = element_yield_df[element].values
                    x_weight = invpairdiff(x_values)
                    y0 = y_values[6]  # yield at normal incidence is rarely in equilibrium and thus stoichiometric

                    # curve fit with weights (sigma) based on distance between points
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fxn()
                        anti_overflow_fac = 1e6
                        popt, _ = curve_fit(lambda angle, b_var, c_var, f_var:
                                            eckstein(angle, b_var, c_var, f_var, anti_overflow_fac * y0),
                                            x_values,
                                            anti_overflow_fac * y_values,
                                            sigma=x_weight)
                        b, c, f = popt
                        if self.isDebug:
                            print(f'{element}: {popt}')
                        eckstein_param_df[element] = [b, c, f, y0]

                data_df[impactor] = [data]
                eckstein_df[impactor] = [eckstein_param_df]

            eckstein_dfdf[mineral] = [eckstein_df]
            data_dfdf[mineral] = [data_df]

        return data_dfdf, eckstein_dfdf

    def dataseries(self):
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

        dist_fit_params = ['phi_k', 'phi_tilt', 'phi_m', 'phi_n', 'energy_k', 'energy_e']
        mineral_suffix = self.mineral_a
        dist_fit_params = [param + '_' + sfx for sfx in mineral_suffix for param in dist_fit_params]
        dist_fit_header = np.concatenate([['alpha'], dist_fit_params])

        yield_df = pd.DataFrame(data=self.angles_a, columns=['alpha'])
        yield_df = yield_df.set_index('alpha')

        # if amu info is to be plotted, then the elements in the species_a are replaced by amu from here on out
        if self.return_amu_ion:
            self.species_a = ['amu/ion']
            # todo: add amu/ion to the species_a instead and then differentiate when calling?

        # create DF with a single column made up of the self.species_a components
        particledata_df = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting

        data_dfdf, eckstein_dfdf = self.eckstein_fit_data()

        for ss, species in enumerate(self.species_a):
            collect_yield = np.zeros(len(self.angles_a))  # total yield at each step
            element_yield = pd.DataFrame(columns=self.mineral_a)

            particle_info = self.angles_a
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

                iyield, iadist, iedist = Particles.particle_data_impactor_refit(self,
                                                                                species,
                                                                                data_dfdf[mineral][0],
                                                                                species_is_in_mineral,
                                                                                eckstein_dfdf[mineral][0],
                                                                                self.minfrac_a[ff])

                element_yield[self.mineral_a[ff]] = iyield  # set element yield for mineral
                collect_yield = np.add(collect_yield, iyield)
                particle_info = np.c_[particle_info, iadist, iedist]  # adds up along second axis

            iparticledata_df = pd.DataFrame(particle_info, columns=dist_fit_header)
            yield_df[species] = collect_yield
            particledata_df[species] = [iparticledata_df]
            self.yield_system_dfdf[species] = [element_yield]
            self.particledata_df = particledata_df

        yield_df = yield_df.fillna(0)
        self.yield_df = yield_df.loc[:, (yield_df != 0).any(axis=0)]  # drop all columns that are pure zeroes

        if self.isSummedUp and not self.return_amu_ion:
            self.particle_data_refit()

        print(f'################# Data for {self.casename} created  #################')

    def particle_data_impactor_refit(self, species, data_df, species_is_in_mineral, eckstein_df, iminfrac):

        # Initiate outputs
        iyield = []
        iadist = []
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
        phi_a = np.linspace(-np.pi, np.pi, num=nr_datapoints)
        energy_a = np.linspace(0, 500, num=nr_datapoints)

        for aa, angle in enumerate(self.angles_a):

            iyield_mb = 0
            iadist_params = [0, 0, 0, 0]
            iedist_params = [0, 0]

            a_isumpart = np.zeros(len(phi_a))
            e_isumpart = np.zeros(len(energy_a))
            for ii, impactor in enumerate(impactors_a):
                if species_is_in_mineral:
                    # write component into dataframe column
                    eckparam = eckstein_df[impactor][0]
                    iyield_mb += sw_comp[ii] * eckstein(angle, *eckparam[species])
                    if not self.return_amu_ion:
                        data = data_df[impactor][0]
                        phi_k = data[f'phiK_{species}'].loc[angle]
                        phi_tilt = data[f'phiTilt_{species}'].loc[angle]
                        phi_m = data[f'phiM_{species}'].loc[angle]
                        phi_n = data[f'phiN_{species}'].loc[angle]
                        energy_k = data[f'energyK_{species}'].loc[angle]
                        energy_e = data[f'energyE_{species}'].loc[angle]
                        adist_params = [phi_k, phi_tilt, phi_m, phi_n]
                        edist_params = [energy_k, energy_e]

                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            fxn()
                            binary_e = self.get_binary_e(species, impactor)
                            adist_particles = np.array(cosfit(phi_a, *adist_params)) * sw_comp[ii]
                            edist_particles = np.array(thompson(energy_a, *edist_params, binary_e)) * sw_comp[ii]

                        a_isumpart = np.add(a_isumpart, np.nan_to_num(adist_particles, copy=False))
                        e_isumpart = np.add(e_isumpart, np.nan_to_num(edist_particles, copy=False))

            if species_is_in_mineral and not self.return_amu_ion:
                iadist_params, a_pcov = fit_eq(phi_a, a_isumpart, eq=cosfit)
                iedist_params, e_pcov = fit_eq(energy_a, e_isumpart, eq=thompson, binary_e=binary_e)

            iyield.append(iyield_mb)
            iadist.append(iadist_params)
            iedist.append(iedist_params)

        iyield = iminfrac * np.array(iyield)  # multiply calculated yields with mineral fraction

        return iyield, iadist, iedist

    def particle_data_refit(self):
        # For each impact angle
        # Create data points for each element with mineral-specific fit parameters
        # Scale the data points with the mineral-specific element yield
        # Sum up the data points of one element between all minerals
        # Fit the summed up datapoints and store the fit parameters
        # The output is a nested dataframe with element columns on the first layer
        # and fit parameter dataframes on the second layer

        nr_datapoints = 500
        phi_a = np.linspace(-np.pi, np.pi, num=nr_datapoints)
        energy_a = np.linspace(0, 500, num=nr_datapoints)
        self.refitparticledata_df = pd.DataFrame(columns=self.species_a)
        print("Sum up particles from minerals and perform a re-fit\n"
              "Process:")
        for ss, species in enumerate(self.species_a):
            print(f'{100 / len(self.species_a) * (ss+1):0.1f}% {species}')
            isummed = pd.DataFrame(columns=['alpha', *self.params])
            isummed['alpha'] = self.angles_a
            isummed.set_index('alpha', inplace=True)

            fitparam_df = self.particledata_df[species][0]

            if 'alpha' in fitparam_df.columns.tolist():
                fitparam_df.set_index('alpha', inplace=True)

            for aa, angle in enumerate(self.angles_a):

                a_isumpart = np.zeros(len(phi_a))
                e_isumpart = np.zeros(len(energy_a))
                for mm, mineral in enumerate(self.mineral_a):
                    param_minsfx = [param + f'_{mineral}' for param in self.params]
                    fitparams = fitparam_df[param_minsfx]  # all fit parameters of the given mineral
                    adist_params = fitparams[param_minsfx[0:4]].iloc[int(aa)]
                    edist_params = fitparams[param_minsfx[4:]].iloc[int(aa)]

                    binary_e = self.get_binary_e(species)
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fxn()
                        mineral_element_yield = self.yield_system_dfdf[species][0][mineral].iloc[aa]
                        adist_particles = np.array(cosfit(phi_a, *adist_params)) * mineral_element_yield
                        edist_particles = np.array(thompson(energy_a, *edist_params, binary_e)) * mineral_element_yield
                    a_isumpart = np.add(a_isumpart, np.nan_to_num(adist_particles, copy=False))
                    e_isumpart = np.add(e_isumpart, np.nan_to_num(edist_particles, copy=False))

                a_popt, a_pcov = fit_eq(phi_a, a_isumpart, eq=cosfit)  # angle fit
                e_popt, e_pcov = fit_eq(energy_a, e_isumpart, eq=thompson, binary_e=binary_e)  # energy fit

                isummed.loc[angle] = np.append(a_popt, e_popt)

                if self.isDebug:
                    """
                    DEBUG: print summed data and fit
                    """
                    debug = False
                    if debug and angle in self.angles_a[::10]:
                        fig_adist_azimuth = plt.figure(dpi=300, figsize=(2*4, 1*4))
                        ax_adist = fig_adist_azimuth.add_subplot(1, 2, 1, projection="polar")
                        ## data:
                        plot_yield_df = pd.DataFrame(index=phi_a,
                                                    data=a_isumpart,
                                                    columns=['amu/ion'])
                        plot_yield_df = plot_yield_df/plot_yield_df.max()
                        sns.lineplot(ax=ax_adist,
                                     data=plot_yield_df,
                                     palette=['#898989'],
                                     alpha=1.0)

                        ## fit:
                        plot_yieldFit_df = pd.DataFrame(index=(np.linspace(np.pi / 2, -np.pi / 2, 50)),
                                                           data=(self.cosfit(np.linspace(np.pi / 2,
                                                                                  -np.pi / 2,
                                                                                  50),
                                                                      *a_popt)),
                                                           columns=[species])
                        plot_yieldFit_df = plot_yieldFit_df/plot_yieldFit_df.max()
                        sns.lineplot(ax=ax_adist,
                                     data=plot_yieldFit_df,
                                     palette=['#C85200'],
                                     alpha=1.0)
                        fig_adist_azimuth.savefig(self.outdir + f'adist_{species}_debug-{angle}.{self.file_format}',
                                                  dpi=300, format=self.file_format)
            self.refitparticledata_df[species] = [isummed]

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

        single_file_df = pd.DataFrame(index=self.angles_a, columns=params_spec_sfx)
        single_file_df.index.name = 'alpha'
        single_file_df[self.species_a] = self.yield_df

        for species in self.species_a:
            species_params = [entry for entry in params_spec_sfx if entry.endswith(species)]
            single_file_df[species_params[1:]] = self.refitparticledata_df[species][0].values

        single_file_df = single_file_df.apply(pd.to_numeric, downcast='float').fillna(0)
        single_file_df.index = pd.to_numeric(single_file_df.index, downcast='integer')

        export_path = os.path.join(self.outdir, f'{self.impactor}_{self.casename}_refit_particle_data.txt')
        single_file_df.to_csv(export_path,sep=',', float_format='{:.3E}'.format)
        if self.isVerbose:
            print(f'Data exported as .csv to {export_path}')

    def ColorPalette(self):
        """ Create color palette for each species and line styles for each oxide"""

        # todo: make a color scheme with suffixes

        elements = self.species_a
        mineral_suffixes = ['_' + val for val in self.mineral_a]
        mineral_suffixes.append('')
        element_mineral = [val + msfx for val in elements for msfx in mineral_suffixes]
        element_suffixes = tuple((len(self.mineral_a) + 1) * [''])
        elements = [val + esfx for val in elements for esfx in element_suffixes]

        self.plotting_key = pd.DataFrame(elements, columns=['element'])
        self.plotting_key['index'] = element_mineral  # new column with element + mineral name
        self.plotting_key['color'] = elements  # new column with color
        self.plotting_key['line'] = [(4, 1)] * len(elements)

        cpalette = sns.color_palette("colorblind", len(self.species_a))  # the only qualitative colormap with >8 colors
        cmap = np.array(cpalette.as_hex())

        # cmap = np.delete(cmap, -2)  # drop yellow as it doesn't plot well
        # cmap = np.roll(cmap, 2)  # roll grey to the front to ensure O is displayed in grey
        # cmap = np.append(cmap, '#000000')

        species_dict = dict(zip(self.species_a, cmap))  # dictionary linking element to color

        self.plotting_key.replace({"color": species_dict}, inplace=True)  # apply species_dict
        self.plotting_key.set_index("index", inplace=True, drop=True)  # replace index with elements, dropping column

    def sputtered_particles_info(self, angles_l, species_l, param=''):
        for angle in angles_l:
            for species in species_l:
                df_of_interest = self.particledata_df[species][0].loc[angle]

                if param == '':
                    print(f' %%%%%%%%%%%%%%%% {species}  %%%%%%%%%%%%%%%%')
                    print(df_of_interest)
                elif param in ('phi_', 'energy_e'):
                    data_of_interest = df_of_interest[df_of_interest.index.str.contains(param)]
                    print(f'\n%%%%%%%%%%%%%%% {species} @ {angle}° %%%%%%%%%%%%%%%%\n')
                    if param == 'phi_tilt':
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

        elif self.isSummedUp:
            info_df = self.refitparticledata_df

        else:
            info_df = self.particledata_df

        if ['amu/ion'] == self.species_a:
            print('Not supported for \'amu/ion\'')
            sys.exit()

        e_max = 100
        nr_datapoints = 1000
        if energy_a is None:
            energy_a = np.linspace(0.1, e_max,
                                   num=nr_datapoints)  # if no energy range is defined, 0.1 -> 100 eV are returned
            energy_bin = e_max / nr_datapoints
        phi_a = np.linspace(-np.pi, np.pi, num=nr_datapoints)  # full angular distribution is always necessary
        phi_bin = 2 * np.pi / nr_datapoints

        edist_df: DataFrame = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting
        edist_loss_df: DataFrame = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting
        adist_df: DataFrame = pd.DataFrame(columns=self.species_a)  # returns bullshit for amu setting
        if self.v_esc:
            print(f'#### Loss Fraction #####')
            print(f'Fraction exceeding v_esc of {self.v_esc} m/s:')
        for ss, species in enumerate(self.species_a):
            iout_df_energy: DataFrame = pd.DataFrame(data=energy_a, columns=['energy'])
            iout_df_energy_loss: DataFrame = pd.DataFrame(columns=[species], index=['loss'])
            iout_df_angle: DataFrame = pd.DataFrame(data=phi_a, columns=['alpha'])

            binary_e = self.get_binary_e(
                species)  # maximum energy that can be transferred from impactor to species in BC

            if self.v_esc:
                v_esc = float(self.v_esc)
                amu = np.vectorize(self.amu_dic.get)(species)  # amu of species
                escape_e = (amu * v_esc ** 2) * 0.5 / 6.022e+26 * 6.242e+18  # calc escape energy in eV

            if self.isSummedUp:
                parameters = info_df[species][0].loc[self.dist_angle]
                if self.isDebug:
                    print(species)
                    print(*parameters)
                phi_k, phi_tilt, phi_m, phi_n, energy_k, energy_e = parameters.values.tolist()

                iout_df_angle[species] = iout_df_angle['alpha'].apply(lambda x: phi_bin * cosfit(x,
                                                                                                 phi_k,
                                                                                                 phi_tilt,
                                                                                                 phi_m,
                                                                                                 phi_n))

                iout_df_energy[species] = iout_df_energy['energy'].apply(lambda x: energy_bin * thompson(x,
                                                                                                         energy_k,
                                                                                                         energy_e,
                                                                                                         binary_e))

                if self.v_esc:
                    iout_df_energy_loss[species], _ = integrate.quad(thompson, escape_e, binary_e,
                                                            args=(energy_k, energy_e, binary_e))
                    print(f'{species}: {iout_df_energy_loss[species].values.tolist()[0]:0.2%}')
                else:
                    iout_df_energy_loss[species] = 'n.d'

                if self.isDebug:
                    k_phi_leq, k_phi_leq_error = integrate.quad(
                        lambda x: 1 / phi_k * np.power(np.cos((-abs(phi_tilt) - x) / (1 - 2 / np.pi * -abs(phi_tilt))),
                                                       phi_m),
                        -np.pi / 2,
                        -abs(phi_tilt))

                    k_phi_geq, k_phi_geq_error = integrate.quad(
                        lambda x: 1 / phi_k * np.power(np.cos((x - phi_tilt) / (1 - 2 / np.pi * phi_tilt)), phi_n),
                        phi_tilt,
                        np.pi / 2)

                    k_angle = k_phi_leq + k_phi_geq
                    k_energy, _ = integrate.quad(thompson, 0, binary_e, args=(energy_k, energy_e, binary_e))
                    # k_adist, k_adist_error = integrate.quad(lambda x: self.cosfit(x, phi_k, phi_tilt, phi_m, phi_n))
                    # k_edist, k_edist_error = integrate.quad(lambda x: self.thompson(x, energy_k, energy_e, binary_e))
                    print(k_angle)
                    print(k_energy)

            else:
                parameters = info_df[species][0].loc[self.dist_angle]
                for mineral in self.mineral_a:

                    phi_k, phi_tilt, phi_m, phi_n, energy_k, energy_e = \
                        parameters[parameters.index.str.contains(f'{mineral}(?!.)', regex=True)]

                    iout_df_angle[species + '_' + mineral] = \
                        iout_df_angle['alpha'].apply(lambda x: phi_bin * cosfit(x, phi_k, phi_tilt, phi_m, phi_n))

                    iout_df_energy[species + '_' + mineral] = \
                        iout_df_energy['energy'].apply(lambda x: energy_bin * thompson(x, energy_k,
                                                                                       energy_e, binary_e))

                    if self.v_esc:
                        iout_df_energy_loss[species + '_' + mineral] = \
                            (energy_e * (energy_e + 2 * escape_e)) / (4 * (energy_e + escape_e) ** 2)
                    else:
                        iout_df_energy_loss[species + '_' + mineral] = 'n.d'

                    if self.isDebug:
                        print(mineral)
                        print(phi_k)
                        print(energy_k)
                        print(f"Integrated angular distribution:"
                              f"{sum(iout_df_angle[species + '_' + mineral].fillna(0))} (before flux)")
                        print(f"Integrated energy distribution:"
                              f"{sum(iout_df_energy[species + '_' + mineral].fillna(0))} (before flux)")
                        print(f"Loss of energy distribution:"
                              f"{sum(iout_df_energy_loss[species + '_' + mineral].fillna(0))} (before flux)")

            iout_df_angle = iout_df_angle.set_index('alpha').fillna(0)
            iout_df_energy = iout_df_energy.set_index('energy').fillna(0)

            edist_df[species] = [iout_df_energy]
            edist_loss_df[species] = [iout_df_energy_loss]
            adist_df[species] = [iout_df_angle]

        self.edist_df = edist_df
        self.edist_loss_df = edist_loss_df
        self.adist_df = adist_df

    def plot_dist(self, dist='energy', species_l=None, ion_flux=1, e_lims=None, minfrac_scaling=True, title=None):
        if self.isVerbose:
            print(f'######## Plotting {dist} distribution ########')
        if species_l == None:
            species_l = self.species_a
        """minfrac_scaling is to show all distribution with scaling by the mineral fraction of the surface"""
        params = {'legend.fontsize': 'large',
                  'figure.titlesize': 'x-large',
                  'axes.labelsize': 'large',
                  'axes.titlesize': 'medium',
                  'xtick.labelsize': 'large',
                  'ytick.labelsize': 'large'}
        pylab.rcParams.update(params)

        def times1e5(x, pos=None):
            x = x * 1e5
            return int(round(x, 0))

        def logscale(x, pos=None):
            x = np.log10(x)
            return int(round(x, 0))

        yFormat = FuncFormatter(times1e5)

        self.ColorPalette()  # create color palette based on the minerals present

        if self.impactor == 'SW':
            impactor = f'H$^+_{{{self.sw_comp[0]}}}$ He$^{{2+}}_{{{self.sw_comp[1]}}}$'
        else:
            impactor = self.impactor

        if title is not None:
            plottitle = title
        else:
            plottitle = f'{self.impactor} \u2192 {self.casename}'

        title_impactor_angle = r'$\alpha_{in}$ = ' + str(self.dist_angle)

        fig_dist = plt.figure(figsize=(6, 4.5), dpi=300)

        if dist == 'energy':
            ax_dist = fig_dist.add_subplot(1, 1, 1)
            dist_df = self.edist_df
            # if title == '':
            #     title = f'Energy distribution'
            if ion_flux != 1:
                ax_dist.set_title(f'10$^{{{math.log(ion_flux, 10):.2g}}}\\times$ {impactor} ' + r'm$^2$s$^{-1}$')
            else:
                ax_dist.set_title(title_impactor_angle)

        elif dist == 'angular':
            dist_df = self.adist_df
            ax_dist = fig_dist.add_subplot(1, 1, 1, projection="polar")
            x_cut = 0.08
            y_cut = -0.33
            ax_dist.set_position([x_cut, -0.25, 1 - 2 * x_cut, 1 - 2 * y_cut])
            ax_dist.set_title(title_impactor_angle)
            # if ion_flux != 1:
            #     ax_dist.set_title(
            #     f'10$^{{{math.log(ion_flux,10):.2g}}}\\times$ {impactor} ' + r'm$^2$s$^{-1}$', y=.83)#, fontsize=10)
            ax_dist.set_thetamin(90)
            ax_dist.set_xlabel('at/ion')
            ax_dist.set_thetamax(-90)
            ax_dist.set_theta_zero_location("N")

        else:
            print('Pass either \'energy\' or \'angular\' as distribution (dist) type.')
            sys.exit()

        """
        Return quantitative results.
        """
        # Fraction is multiplied with
        # - total ion ion_flux,
        # - element sputter yield, and
        # - mineral surface fraction
        if self.isSummedUp:
            minfrac_scaling = False  # self.yield_df is already scaled by mineral fraction

        i_ymax = 0
        labels = []
        loss_text = ['Loss fraction']
        quant_dist = pd.DataFrame(columns=self.species_a)
        for ss, species in enumerate(species_l):
            if self.isSummedUp:
                quant_dist_i = dist_df[species][0].apply(
                    lambda x: x * ion_flux * self.yield_df[species].loc[self.dist_angle])
            else:
                quant_dist_i = dist_df[species].apply(
                    lambda x: x * ion_flux * self.yield_df[species].loc[self.dist_angle])

            quant_dist[species] = quant_dist_i

            if self.isSummedUp:
                peak_value = peak(quant_dist_i.index.values, quant_dist_i.iloc[:, 0])[0]  # get global maxima

                if dist == 'energy':
                    """ adds peak ENERGY value to label"""
                    peak_label = r'$E_{{{}}}$'.format(species) + \
                                 f' = {peak_value:2.1f}'.format(species) + ' eV'
                    """Add escape velocity lines and loss fractions"""

                else:
                    """ adds peak ANGLE value to label"""
                    peak_value = rad2deg(peak_value)
                    peak_label = r'$\phi_{{{}}}$'.format(species) + \
                                 f' = {peak_value:2.1f}'.format(species) + r'$^\circ$'

                labels.append(f'{peak_label.rjust(2)}')

                # todo: re-implement loss fraction in plot
                # if self.v_esc:
                #     lossfrac = self.edist_loss_df.iloc[0][species].values.tolist()[0][0]
                #     loss_text.append(f'{species}: {lossfrac:.2%}')

        if self.isSummedUp:

            sns.lineplot(ax=ax_dist,
                         data=quant_dist,
                         palette=dict(self.plotting_key['color']),
                         dashes=True,  # dict(self.plotting_key['line']),
                         alpha=1.0)
            if self.plot_inset and dist == 'energy' and not self.logplot:
                ax_inset = inset_axes(ax_dist, width=1.4, height=1,
                                      bbox_to_anchor=(0.68, 0.22),
                                      bbox_transform=ax_dist.transAxes,
                                      loc=3,
                                      borderpad=0)
                sns.lineplot(ax=ax_inset,
                             data=quant_dist,
                             palette=dict(self.plotting_key['color']),
                             dashes=True,  # dict(self.plotting_key['line']),
                             alpha=1.0)
                ax_inset.set_xlim(0, 4)
                ax_inset.set_ylim(0, 6e-5)
                ax_inset.set_xlabel('')
                ax_inset.yaxis.set_major_formatter(yFormat)
                ax_inset.xaxis.set_major_locator(plt.MaxNLocator(2))
                ax_inset.yaxis.set_major_locator(plt.MaxNLocator(3))
                ax_inset.get_legend().remove()

            if quant_dist.max().iloc[0] >= i_ymax and dist == 'energy':
                i_ymax = quant_dist.max().max()

        else:
            if minfrac_scaling:
                minfrac = self.minfrac_a
            else:
                minfrac = 1
            for col in quant_dist.columns:
                quant_dist[col][0] *= minfrac
                quant_dist[col][0].fillna(quant_dist[col][0].mode(), inplace=True)
                quant_dist[col][0] = quant_dist[col][0].loc[:, (quant_dist[col][0] != 0).any(axis=0)]
                sns.lineplot(ax=ax_dist,
                             data=quant_dist[col][0],
                             palette=dict(self.plotting_key['color']),
                             dashes=True,  # dict(self.plotting_key['line']),
                             alpha=1.0)
                if quant_dist[col][0].max()[0] >= i_ymax and dist == 'energy':
                    i_ymax = quant_dist[col][0].max().max()

        """
        Adjust axis limits depending on distribution type
        """
        if dist == 'energy':
            ax_dist.set_xlabel(r'emission energy [eV]', ha='center')
            ax_dist.set_xlim(0, 20)
            ax_dist.set_ylim(0, i_ymax * 1.1)
            if self.logplot:
                ax_dist.set_yscale('log')
            if e_lims:
                ax_dist.set_xlim(0, e_lims[0])
                ax_dist.set_ylim(0, e_lims[1])

            if ion_flux == 1:
                ax_dist.set_ylabel(r'particles [10$^{-5}$ at. ion$^{-1}$ eV$^{-1}$]')
                if not self.logplot:
                    ax_dist.yaxis.set_major_formatter(yFormat)


            else:
                ax_dist.set_ylabel(r'particles [at. ion$^{-1}$ eV$^{-1}$]')
            ax_dist.xaxis.set_major_locator(plt.MaxNLocator(4))

            # Adjust edist Legend
            if self.isSummedUp:
                handles, _ = ax_dist.get_legend_handles_labels()  # labels are pre-defined
            else:
                handles, labels = ax_dist.get_legend_handles_labels()
                labels = [label.replace('_', '$_\mathrm{') + '}$' for label in labels]

            if len(labels) > 0:
                ax_dist.legend(handles,
                               labels,
                               loc='upper right',
                               ncol=max(int(np.ceil(len(handles) / 6)), 1),
                               frameon=False)

        if dist == 'angular':

            arrow_length = ax_dist.get_ylim()[1]

            """
            Two annotation commands are necessary as the arrow won't plot properly as it takes the length of the text
            into account BEFORE plotting the arrow, resulting in a wrong positioning
            """
            ax_dist.annotate('', xy=[0, 0],
                             xytext=[self.dist_angle / 180 * np.pi, arrow_length],
                             xycoords='data',
                             arrowprops=dict(facecolor='black')  # , fontsize='large'
                             )
            ax_dist.tick_params(labelleft=False, labelright=True,
                                labeltop=False, labelbottom=True)
            if ion_flux == 1:
                ax_dist.set_xlabel(r'yield [10$^{-5}$ atoms ion$^{-1}$]')  # , fontsize=8)
                if not self.logplot:
                    ax_dist.yaxis.set_major_formatter(yFormat)
            else:
                ax_dist.set_xlabel(r'yield [atoms s$^{-1}$ cm$^{-2}$]')  # , fontsize=8)

            ax_dist.xaxis.set_label_coords(0.25, 0.235)
            ax_dist.yaxis.set_major_locator(plt.MaxNLocator(4))
            if self.logplot:
                ax_dist.set_rscale('symlog', linthresh=1e-5)
                ax_dist.set_rlim(0)

            """
            Adjust adist Legend
            """
            if self.isSummedUp:  # or self.isMassBalanced #deprecated
                handles, _ = ax_dist.get_legend_handles_labels()  # labels are pre-defined
            else:
                handles, labels = ax_dist.get_legend_handles_labels()  # labels are pre-defined
                labels = [label.replace('_', '$_\mathrm{') + '}$' for label in labels]

            yanchor = 0.00
            nrows = 3
            if self.isSummedUp:
                yanchor = 0.21
                nrows = 2

            ax_dist.legend(handles,
                           labels,
                           loc='upper center',
                           bbox_to_anchor=(0.5, yanchor),
                           ncol=min(max(int(np.ceil(len(handles) / nrows)), 2), 3),
                           frameon=False)

        plotname = f'{self.impactor} \u2192 {self.casename}'
        file_plotname = safe_title_trans(plotname)
        filename = file_plotname + \
                   '_' + \
                   dist[0] + 'dist' + \
                   '_' + \
                   '_'.join(species_l)
        filename = safe_title_trans(filename)
        if self.isSummedUp:
            summedsfx = '_sm'
        else:
            summedsfx = ''

        fig_dist.suptitle(plottitle, y=0.98)
        fig_dist.savefig(os.path.join(self.outdir, filename) + f'_{self.dist_angle}{summedsfx}.{self.file_format}',
                         dpi=300, format=self.file_format)  # , bbox_inches='tight')

        if self.show_plot:
            fig_dist.show()
        plt.close(fig_dist)
        return fig_dist, ax_dist

    def plot_yield(self, exp_H_data=None, exp_He_data=None, addTRIMdata=False):
        if self.isVerbose:
            print('######## Plotting yield ########')
        params = {'legend.fontsize': 'large',
                  'figure.titlesize': 'x-large',
                  'figure.figsize': (6, 4.5),
                  'axes.labelsize': 'large',
                  'axes.titlesize': 'medium',
                  'xtick.labelsize': 'large',
                  'ytick.labelsize': 'large'}
        pylab.rcParams.update(params)
        fig = plt.figure(dpi=300)
        self.ColorPalette()  # create plotting dictionary (only colors for now)
        plotname = f'{self.impactor} \u2192 {self.casename}'

        ax = sns.lineplot(data=self.yield_df,
                          palette=dict(self.plotting_key['color']),
                          dashes=True, # dict(self.plotting_key['line']),
                          alpha=1.0)

        ax.set_xlabel(r"angle [$^\circ$]")
        ax.set_ylabel(r"Y [at/ion]")
        ax.set_xlim(0, 90)

        if self.return_amu_ion:
            ax.set_ylim(0, self.yield_df.max().max() * 1.1)
            ax.set_ylabel(r"Y [amu/ion]")

            """
            Add experimental data
            """
            # todo: (REMOVE FOR RELEASE)
            if exp_H_data is not None and self.impactor == 'H':
                exp_data = exp_H_data
                exp_alpha = exp_data.index.values
                ax.errorbar(exp_alpha, exp_data['amu/ion'], fmt='o', yerr=exp_data['error'], color='k',
                            label='Experiment: Brötzner, unpub.')

            elif exp_He_data is not None and self.impactor == 'He':
                exp_data = exp_He_data
                exp_alpha = exp_data.index.values
                ax.errorbar(exp_alpha, exp_data['amu/ion'], fmt='o', yerr=exp_data['error'], color='k',
                            label='Experiment: Brötzner, unpub.')

        else:
            ax.set_ylim(1e-4, 1e-0)
            ax.set(yscale='log')
        fig.suptitle(plotname, y=0.97)  # , fontsize=16)

        mineral_str = 'Minerals:'
        for mineral in self.mineral_a:
            mineral_str = mineral_str + ' ' + mineral
        ax.set_title(mineral_str, y=0.985)  # , fontsize=10)

        """
        Adjust Legend
        """
        handles, labels = ax.get_legend_handles_labels()
        if self.return_amu_ion:
            model_label = self.sfx_for_plot(drop_prefix=False)
            rho_sys = self.rho_system_df.sum()
            labels[0] = f'{model_label},\nrho = {rho_sys:0.2f}' + ' g cm$^{-3}$'

        ax.legend(handles,
                  labels,
                  ncol=max(int(np.ceil(len(handles) / 5)), 1),
                  frameon=False)

        # %%
        filename = safe_title_trans(plotname)
        if self.return_amu_ion:
            filename += '_amu'
        fig.savefig(os.path.join(self.outdir, f'{filename}.{self.file_format}'),
                    dpi=300, format=self.file_format)  # , bbox_inches='tight')

        if self.show_plot:
            fig.show()
        plt.close(fig)

        return fig, ax

    def cipw_norm(self, comp_l, comp_frac_l, verboseCIPW=False):
        if self.isDebug:
            verboseCIPW = True

        amu_min_dict = self.amu_minerals_dic
        rho_min_dict = self.wt_density_df_minerals_dic
        mindict_df_atoms = self.mindict_df_atoms.iloc[:, self.mindict_df_atoms.columns != 'total']
        mineral_names = mindict_df_atoms.index.tolist()
        minfrac = pd.DataFrame(data=np.zeros(len(mineral_names)), columns=['frac'],
                               index=mineral_names).T

        comp_df = pd.DataFrame(data=comp_frac_l, columns=['at%'], index=comp_l)

        species_dict_dir = 'tables/species_dict.txt'
        species_df = pd.read_csv(species_dict_dir, header=0, delim_whitespace=True)
        species_dict = {species_df['metal'].values[i]: species_df['species'].values[i] for i in
                        range(len(species_df['species'].values))}
        species_cat_dict = {species_df['metal'].values[i]: species_df['cation'].values[i] for i in
                            range(len(species_df['species'].values))}

        """
        Check if  composition was passed as elements or as oxides
        """
        CIPW_elements = False
        element_l = ['O', 'Si', 'Ca', 'Mg']  # elements that should be passed as oxides
        el_test = [i for i in element_l if i in comp_l]
        if len(el_test) > 0:
            CIPW_elements = True
            CIPW_oxides = not CIPW_elements
            print(f'Composition given as oxides: {CIPW_oxides}')

        """
        2 divide oxide weights by their respective formula weights to give molar oxide proportions
        """
        if CIPW_elements:
            if comp_df.loc['O'].values <= 0.47:
                comp_df = comp_df.T.div(pd.Series(self.amu_dic))
                comp_df = comp_df.T
                print('Composition with < 47 % oxygen - wt% assumed')
            elif comp_df.loc['O'].values == 0:
                comp_df = comp_df.T.div(pd.Series(self.amu_dic))
                comp_df = comp_df.T
                print('Composition without oxygen - wt% assumed')
            else:
                print('Composition with > 47 % oxygen - at% assumed')
            comp_df = comp_df.T.div(pd.Series(species_cat_dict))
            comp_df.columns = comp_df.columns.map(species_dict.get)

        else:
            comp_df = comp_df.T.div(pd.Series(self.amu_oxides_dic))

        comp_df = comp_df / comp_df.sum(axis=1).iloc[0]

        comp_df.dropna(inplace=True, axis=1)  # drop all NAN columns

        """
        ** Sulfides: Add Mn, Cr, Ti, and Fe to Sulfur  
        """
        S_limit = 1e-5

        if verboseCIPW:
            print(f'#0\nS in columns? {"S" in comp_df.columns.values.tolist()}')
        if 'S' in comp_df.columns:
            if S_limit >= comp_df['S'].tolist()[0] > 0:
                print(f'Low amount of S ({comp_df["S"].values[0]:0.2e} < {S_limit}) omitted')

            elif comp_df['S'].tolist()[0] > S_limit:
                print(f'Transition metals are attributed to Sulfur ({comp_df["S"].values[0]:0.2e})')
                if verboseCIPW:
                    print(f'#1\nCrO in columns? {"CrO" in comp_df.columns.values.tolist()}')
                    print(f'Cr2O3 in columns? {"Cr2O3" in comp_df.columns.values.tolist()}')

                if 'CrO' in comp_df.columns.values.tolist():
                    if 2 * comp_df['FeO'].tolist()[0] >= comp_df['CrO'].tolist()[0]:
                        minfrac['Dbr'] = comp_df['CrO'].tolist()[0] / 2  # Dbr = FeCr2S4; occurance: Fe-meteorite
                        minfrac['Chr'] = minfrac['Dbr'].tolist()[0]  # Chr = FeCr2O4; occurance: Moon
                        comp_df['FeO'] = comp_df['FeO'] - comp_df['CrO'] / 2
                        comp_df['CrO'] = 0.00
                    else:
                        minfrac['Bzn'] = comp_df['CrO'].tolist()[0] / 3  # Bzn = Cr3S4; occurance: meteorites
                        comp_df['CrO'] = 0.00
                elif 'Cr2O3' in comp_df.columns.values.tolist():
                    if comp_df['FeO'].tolist()[0] >= comp_df['Cr2O3'].tolist()[0]:
                        minfrac['Dbr'] = comp_df['Cr2O3'].tolist()[0]  # Dbr = FeCr2S4; occurance: Fe-meteorite
                        minfrac['Chr'] = minfrac['Dbr'].tolist()[0]  # Chr = FeCr2O4; occurance: Moon
                        comp_df['FeO'] = comp_df['FeO'] - comp_df['Cr2O3']
                        comp_df['Cr2O3'] = 0.00
                    else:
                        minfrac['Bzn'] = comp_df['Cr2O3'].tolist()[0] * 2 / 3  # Bzn = Cr3S4
                        comp_df['Cr2O3'] = 0.00
                else:
                    minfrac['Bzn'] = 0.00
                    minfrac['Dbr'] = 0.00

                """
                ** Check if there is enough S to put Cr into sulfides, otherwise, keep only chromite
                """
                pS1 = comp_df['S'].tolist()[0] -\
                               minfrac['Bzn'].tolist()[0] * 3 - \
                               minfrac['Dbr'].tolist()[0] * 4

                if pS1 > 0:
                    minfrac['Chr'] = 0
                    comp_df['S'] = pS1
                else:
                    minfrac['Bzn'] = 0.00
                    minfrac['Dbr'] = 0.00
                    if verboseCIPW:
                        print(f'Not enough S to accomodate all Cr, accomodated into chromite (Chr) instead ')

                if verboseCIPW:
                    print(f'remaining sulfur: {comp_df["S"].tolist()[0]}')
                    print(f'remaining FeO: {comp_df["FeO"].tolist()[0]}')

                minerals = ['Abd', 'Tro', 'Was']
                oxides = ['MnO', 'FeO', 'TiO2']

                for mm, mineral in enumerate(minerals):
                    oxide = oxides[mm]
                    if comp_df['S'].tolist()[0] > 0:
                        if comp_df['S'].tolist()[0] >= comp_df[oxide].tolist()[0]:
                            minfrac[mineral] = comp_df[oxide].tolist()[0]
                            comp_df[oxide] = 0.00
                            comp_df['S'] = comp_df['S'].tolist()[0] - minfrac[mineral].tolist()[0]
                        else:
                            minfrac[mineral] = comp_df['S'].tolist()[0]
                            comp_df['S'] = 0.00
                            comp_df[oxide] = comp_df[oxide].tolist()[0] - minfrac[mineral].tolist()[0]

                if verboseCIPW:
                    print(f'remaining sulfur: {comp_df["S"].tolist()[0]} put into 1/4 Old and 3/4 Nng')
                if comp_df['S'].tolist()[0] > 0:
                    if comp_df['S'].tolist()[0] < min(comp_df['MgO'].tolist()[0] / 3 * 4,
                                                      comp_df['CaO'].tolist()[0] / 4):
                        minfrac['Nng'] = comp_df['S'].tolist()[0] / 3 * 4
                        minfrac['Old'] = comp_df['S'].tolist()[0] / 4
                        comp_df['MgO'] = comp_df['MgO'].tolist()[0] - minfrac['Nng'].tolist()[0]
                        comp_df['CaO'] = comp_df['CaO'].tolist()[0] - minfrac['Old'].tolist()[0]
                        comp_df['S'] = 0.00
        """
        3 Add MnO to FeO. 
        """

        if verboseCIPW:
            print(f'#3\nMnO in columns? {"MnO" in comp_df.columns.values.tolist()}')
        if 'MnO' in comp_df.columns:
            comp_df['FeO'] = comp_df['FeO'].tolist()[0] + comp_df['MnO'].tolist()[0]

        """
        4 Apatite: Multiply P2O5 by 3.33 and subtract this number from CaO.
        """
        if verboseCIPW:
            print(f'#4\nP2O5 in columns? {"P2O5" in comp_df.columns.values.tolist()}')
        if 'P2O5' in comp_df.columns.values.tolist():
            minfrac['Ap'] = comp_df['P2O5'].tolist()[0] * 2 / 3
            comp_df['CaO'] = comp_df['CaO'] - comp_df['P2O5'] * 3.33
            comp_df['P2O5'] = 0.00

        """
        5 Ilmenite: Subtract TiO2 from FeO. Put the TiO2 value in Ilmenite
        """
        if verboseCIPW:
            print(f'#5\nTiO2 in columns? {"TiO2" in comp_df.columns.values.tolist()}')
        if 'TiO2' in comp_df.columns.values.tolist():
            minfrac['Ilm'] = comp_df['TiO2'].tolist()[0]
            comp_df['FeO'] = comp_df['FeO'].tolist()[0] - comp_df['TiO2'].tolist()[0]
            comp_df['TiO2'] = 0.00

        """
        6 Magnetite: Subtract Fe2O3 from FeO. Put the Fe2O3 value in magnetite. Fe2O3 is now zero
        """
        # if 'Fe2O3' in comp_df.columns:
        #     minfrac['Mt'] = comp_df['Fe2O3'].tolist()[0]
        #     comp_df['FeO'] = comp_df['FeO'].tolist()[0] - comp_df['Fe2O3'].tolist()[0]
        #     comp_df['Fe2O3'] = 0.00

        """
        7 Orthoclase: Subtract K2O from Al2O3. Put the K2O value in orthoclase. K2O is now zero
        """
        minfrac['Or'] = comp_df['K2O'].values
        comp_df['Al2O3'] = comp_df['Al2O3'] - comp_df['K2O']
        comp_df['K2O'] = 0.00

        """
        8 Albite (provisional): Subtract Na2O from Al2O3. Put the Na2O value in albite. 
                                Retain the Na2O value for possible normative nepheline.
        """
        if comp_df['Na2O'].values > comp_df['Al2O3'].values:
            minfrac['Ab'] = comp_df['Al2O3'].values
            Na2O_surplus = comp_df['Na2O'] - comp_df['Al2O3']
            comp_df['Al2O3'] = 0
            print(f'There is a Na2O surplus of {Na2O_surplus.values[0]:.2%}!')
        else:
            minfrac['Ab'] = comp_df['Na2O'].values
            comp_df['Al2O3'] = comp_df['Al2O3'] - comp_df['Na2O']

        if verboseCIPW:
            print(f'#8\nAb = {minfrac["Ab"].tolist()[0]}')

        """
        9 Anorthite:                       
        A. If CaO is more than the remaining Al2O3, then subtract Al2O3 from CaO. Put all Al2O3 into Anorthite.
        
        B. If Al2O3 is more than CaO, then subtract CaO from Al2O3. Put all CaO into anorthite.
        """
        if verboseCIPW:
            print(f'#9\nCaO = {comp_df["CaO"].tolist()[0]}')
            print(f'#9\nAl2O3 = {comp_df["Al2O3"].tolist()[0]}')
        if comp_df['Al2O3'].tolist()[0] <= comp_df['CaO'].tolist()[0]:
            minfrac['An'] = comp_df['Al2O3'].tolist()[0]
            comp_df['CaO'] = comp_df['CaO'] - comp_df['Al2O3']
            comp_df['Al2O3'] = 0
            if verboseCIPW:
                print(f'#9A\nAn = {minfrac["An"].tolist()[0]}')

        else:
            minfrac['An'] = comp_df['CaO'].tolist()[0]
            comp_df['Al2O3'] = comp_df['Al2O3'] - comp_df['CaO']
            comp_df['CaO'] = 0
            if verboseCIPW:
                print(f'#9B\nAn = {minfrac["An"].tolist()[0]}')

        """
        10 Corundum: If Al2O3 is not zero, put the remaining Al2O3 into corundum.                        
        """
        if comp_df['Al2O3'].tolist()[0] > 0:
            minfrac['Cor'] = comp_df['Al2O3'].tolist()[0]
            comp_df['Al2O3'] = 0.0

            comp_df['CaO'] = comp_df['CaO'] - comp_df['Al2O3']

        """
        11 Calculate Magnesium Number Mg/(Mg+Fe)                       
        """
        if 'FeO' not in comp_df.columns.values.tolist():
            comp_df['FeO'] = 0.00
        Mg_nbr = comp_df['MgO'].tolist()[0] / (comp_df['MgO'].tolist()[0] + comp_df['FeO'].tolist()[0])

        """
        12. Calculate the mean formula weight of the remaining FeO and MgO. 
            This combined FeMg oxide, called FMO, will be used in subsequent calculations.                     
        """
        FMO_wt = (Mg_nbr * 40.3044) + ((1 - Mg_nbr) * 71.8464)

        # Add dictionary entries for final wt%
        amu_min_dict['Opx'] = 60.0843 + FMO_wt
        amu_min_dict['Ol'] = 60.0843 + 2 * FMO_wt
        # amu_min_dict['Di'] = 176.2480 + 2 * FMO_wt

        """
        13. Add FeO and MgO to make FMO              
        """

        FMO = comp_df['MgO'].tolist()[0] + comp_df['FeO'].tolist()[0]

        # """
        # 14. Diopside: If CaO is not zero, subtract CaO from FMO. Put all CaO into diopside. CaO is now zero.
        # """
        # if comp_df['CaO'].tolist()[0] > 0:
        #     minfrac['Di'] = comp_df['CaO'].tolist()[0]
        #     FMO = FMO - comp_df['CaO'].tolist()[0]
        #     comp_df['CaO'] = 0.0
        """
        14a. Diopside: If CaO is not zero, set diopside as min(CaO,MgO) and subtract from FMO.                   
        """
        if comp_df['CaO'].tolist()[0] > 0:
            minfrac['Di'] = min(comp_df['CaO'].tolist()[0], comp_df['MgO'].tolist()[0])
            FMO = FMO - minfrac['Di'].tolist()[0]
            comp_df['CaO'] = comp_df['CaO'].tolist()[0] - minfrac['Di'].tolist()[0]

        """
        14b. Diopside: If CaO is not zero, set diopside as min(CaO,MgO) and subtract from MgO.                   
        """
        if comp_df['CaO'].tolist()[0] > 0:
            minfrac['Wo'] = comp_df['CaO'].tolist()[0]
            comp_df['CaO'] = 0.0

        """
        15. Orthopyroxene (provisional): Put all remaining FMO into orthopyroxene. 
            Retain the FMO value for the possible normative olivine.
        """

        if FMO > 0:
            minfrac['Opx'] = FMO
            minfrac['En'] = FMO * Mg_nbr
            minfrac['Fs'] = FMO * (1 - Mg_nbr)

        """
        16. Calculate the amount of SiO2 needed for all of the normative silicates listed above, allotting SiO2 as follows:
            Orthoclase * 6 = needed SiO2 for each Orthoclase
            Albite * 6 = needed SiO2 for each Albite
            Anorthite * 2 = needed SiO2 for each Anorthite
            Diopside * 2 = needed SiO2 for each Diopside
            Orthopyroxene * 1 = needed SiO2 for each Hypersthene
        """

        SiO2_Or = minfrac['Or'] * 6
        SiO2_Ab = minfrac['Ab'] * 6
        SiO2_An = minfrac['An'] * 2
        SiO2_Di = minfrac['Di'] * 2
        SiO2_Wo = minfrac['Wo'] * 1
        SiO2_Opx = minfrac['Opx'] * 1

        """
        17. Sum the five SiO2 values just calculated, and call this number pSi1 for the first provisional SiO2.
        """
        pSi1 = SiO2_Or.values + SiO2_Ab.values + SiO2_An.values + SiO2_Di.values + SiO2_Opx.values + SiO2_Wo.values

        """
        18. Quartz: If there is enough silica to make all five minerals in the list in #16 then the
            rock is quartz-normative. Otherwise there is no quartz in the norm and silica to make the rest
            of the silicates must come from other sources.
            
            A. If pSi1 calculated in #16 is less than SiO2, then there is excess silica. Subtract pSi1 from
            SiO2, and put excess SiO2 in quartz. SiO2, nepheline, and olivine are now zero. Skip to #23.

            B. If pSi1 calculated in #16 is more than SiO2, then the rock is silica deficient. Proceed to #19.
        """

        if comp_df['SiO2'].values < pSi1:
            minfrac['Qz'] = 0.00
            """
            19. -> 20. Sum the four SiO2 values just calculated to get pSi2. Subtract pSi2 from SiO2 to get the
            of SiO2 available for olivine and orthopyroxene, called pSi3.
            
            A. If FMO is greater than or equal to 2 times pSi3, then put all FMO in Olivine. FMO and
            Orthopyroxene are now zero. Proceed to #21.
            
            B. If FMO is less than 2 times pSi3, then nepheline is zero. Calculate the amount of
            orhtopyroxene and olivine as follows:
                Orthopyroxne = ((2 * pSi3) - FMO)
                Olivine = (FMO - orthopyroxene)
                Skip to #23
            """
            pSi2 = SiO2_Or.tolist()[0] + SiO2_Ab.tolist()[0] + SiO2_An.tolist()[0] + SiO2_Di.tolist()[0] + \
                   SiO2_Wo.tolist()[0]
            pSi3 = comp_df['SiO2'].tolist()[0] - pSi2
            if FMO >= 2 * pSi3:
                if verboseCIPW: print(
                    'Total Fe+Mg is greater equal to two times the SiO2 remaining after forming Or, Ab, An and Di ')
                minfrac['Ol'] = FMO
                minfrac['Fo'] = minfrac['Ol'].tolist()[0] * FMO
                minfrac['Fa'] = minfrac['Ol'].tolist()[0] * (1 - FMO)
                minfrac['Opx'] = 0.00
                minfrac['En'] = 0.00
                minfrac['Fs'] = 0.00
                FMO = 0

                """
                21. Nepheline, albite (final): If you reached this step, then turning orthopyroxene into olivine
                in #20A did not yield enough silica to make orthoclase, albite, anorthite, diopside, and
                olivine.                 """

                SiO2_Ol = 0.5 * minfrac['Ol']

                """
                22. Sum the three SiO2 values just calculated to get pSi4. Subtract pSi4 from SiO2 to get
                    pSi5, which is the amount of SiO2 available for albite and nepheline.
                        Albite = (pSi5-(2*Na2O))/4
                        Nepheline = Na2O-Albite
                """
                pSi4 = SiO2_Or.tolist()[0] + SiO2_An.tolist()[0] + SiO2_Di.tolist()[0] + SiO2_Ol.tolist()[0] + \
                       SiO2_Wo.tolist()[0]
                pSi5 = comp_df['SiO2'].tolist()[0] - pSi4
                minfrac['Ab'] = (pSi5 - (2 * comp_df['Na2O'].tolist()[0])) / 4
                if verboseCIPW:
                    print(f'#22\nAb = {minfrac["Ab"].tolist()[0]}')
                minfrac['Nph'] = comp_df['Na2O'].tolist()[0] - minfrac['Ab'].tolist()[0]

            elif FMO < 2 * pSi3:
                if verboseCIPW:
                    print('Total Fe+Mg is less than two time the SiO2 remaining after forming Or, Ab, An and Di ')
                minfrac['Nph'] = 0.00
                minfrac['Opx'] = (2 * pSi3) - FMO
                minfrac['En'] = ((2 * pSi3) - FMO) * Mg_nbr
                minfrac['Fs'] = ((2 * pSi3) - FMO) * (1 - Mg_nbr)
                minfrac['Ol'] = FMO - minfrac['Opx'].tolist()[0]
                minfrac['Fo'] = minfrac['Ol'].tolist()[0] * FMO
                minfrac['Fa'] = minfrac['Ol'].tolist()[0] * (1 - FMO)

        elif pSi1 < comp_df['SiO2'].tolist()[0]:
            minfrac['Nph'] = 0
            minfrac['Ol'] = 0
            comp_df['SiO2'] = comp_df['SiO2'] - pSi1
            minfrac['Qz'] = comp_df['SiO2'].tolist()[0]
            comp_df['SiO2'] = 0.0
        """
        23. Multiply orthoclase, albite, and nepheline by two. Divide olivine by two
        """
        minfrac['Or'] = minfrac['Or'].tolist()[0] * 2
        minfrac['Ab'] = minfrac['Ab'].tolist()[0] * 2
        minfrac['Nph'] = minfrac['Nph'].tolist()[0] * 2
        minfrac['Ol'] = minfrac['Ol'].tolist()[0] / 2
        minfrac['Fo'] = minfrac['Ol'].tolist()[0] * FMO
        minfrac['Fa'] = minfrac['Ol'].tolist()[0] * (1 - FMO)
        if verboseCIPW:
            print(f'#23\nAb = {minfrac["Ab"].tolist()[0]}')
        """
        24. Calculate An number, which is the Ca/(Ca+Na) ratio in normative plagioclase:
        """

        An_nbr = comp_df['CaO'].tolist()[0] / (comp_df['CaO'].tolist()[0] + comp_df['Na2O'].tolist()[0])

        """
        25. Plagioclase: Add albite to anorthite to make plagioclase. Retain the albite value, anorthite is now zero.
        """
        minfrac['Plag'] = minfrac['Ab'].tolist()[0] + minfrac['An'].tolist()[0]
        if verboseCIPW:
            print(f'#25\nAb = {minfrac["Ab"].tolist()[0]}')
            print(f'An = {minfrac["An"].tolist()[0]}')
            print(minfrac['Plag'].tolist()[0])

        """
        25. Calculate the formula weight of plagioclase, using the An number value from #24
        """
        plag_wt = (An_nbr * 278.2093) + ((1 - An_nbr) * 262.2230)

        amu_min_dict['Plag'] = plag_wt

        """
        Obtain minfrac for comparison with CIPW (in wt%)
        """
        minfrac_CIPW_wt = minfrac.mul(pd.Series(amu_min_dict))
        minfrac_CIPW_wt = minfrac_CIPW_wt.fillna(0)
        minfrac_CIPW_wt.drop(['Ab', 'An', 'Fo', 'Fa', 'En', 'Fs', 'Wo'], inplace=True, axis=1)
        minfrac_CIPW_wt = minfrac_CIPW_wt.T
        minfrac_CIPW_wt = minfrac_CIPW_wt.loc[(minfrac_CIPW_wt != 0.0).any(axis=1)]
        minfrac_CIPW_wt = minfrac_CIPW_wt / minfrac_CIPW_wt.sum(axis=0).values[0]

        if verboseCIPW:
            print(f'\nCIPW weight fractions:\n{minfrac_CIPW_wt}')
            print(f'Total: {minfrac_CIPW_wt.sum()[0]:0.2f}')

        """
        Obtain minfrac for mass balancing density
        """
        minfrac_wt = minfrac.mul(pd.Series(amu_min_dict))
        minfrac_wt = minfrac_wt.fillna(0)
        minfrac_wt.drop(['Plag', 'Ol', 'Opx'], inplace=True, axis=1)
        minfrac_wt = minfrac_wt.T
        minfrac_wt = minfrac_wt.loc[(minfrac_wt != 0.0).any(axis=1)]
        minfrac_wt = minfrac_wt / minfrac_wt.sum(axis=0).values[0]

        print(f'\nWeight fractions:\n{minfrac_wt}')
        print(f'Total: {minfrac_wt.sum().iloc[0]:0.2f}')

        """
        Obtain mineralogy in vol%
        """

        minfrac_vol = minfrac.mul(pd.Series(amu_min_dict))
        minfrac_vol = minfrac_vol.div(pd.Series(rho_min_dict))
        minfrac_vol = minfrac_vol.fillna(0)
        minfrac_vol.drop(['Plag', 'Ol', 'Opx'], inplace=True, axis=1)
        minfrac_vol = minfrac_vol.T
        minfrac_vol = minfrac_vol.loc[(minfrac_vol != 0.0).any(axis=1)]
        minfrac_vol = minfrac_vol/minfrac_vol.sum(axis=0).values[0]
        if verboseCIPW:
            print(f'\nVolume fractions:\n{minfrac_vol}')
            print(f'Total: {minfrac_vol.sum()[0]:0.2f}')

        """
        Obtain molar fraction of minerals; drop Plag, Ol and Opx 
        """
        minfrac.drop(['Plag', 'Ol', 'Opx'], inplace=True, axis=1)
        minfrac = minfrac.T
        minfrac = minfrac.loc[(minfrac != 0.0).any(axis=1)]
        minfrac = minfrac / minfrac.sum(axis=0).values[0]

        print(f'\nMolar fractions\n {minfrac}')
        print(f'Total: {minfrac.sum().iloc[0]:0.2f}')

        self.minfrac_df_molar = minfrac
        self.minfrac_df_weight = minfrac_wt
        self.minfrac_df_weight_CIPW = minfrac_CIPW_wt

        return minfrac

    def systemdensity(self):

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
