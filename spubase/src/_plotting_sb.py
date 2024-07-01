# ===============================
# AUTHORS: Noah Jäggi, Adam K. Woodson
# CREATE DATE: 4. April 2024
# PURPOSE: Fit SDTrimSP angular data in 3D
# SPECIAL NOTES:
# ===============================
# Change History:
# 26/06/2024: Added plot3D
# ==================================

import re
import os
import sys
import pylab
import math

import pandas as pd
import seaborn as sns
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
import numpy as np


def rad2deg(x_rad):
    return x_rad * 180 / np.pi


def deg2rad(x_rad):
    return x_rad / 180 * np.pi


def times1e5(x, pos=None):
    x = x * 1e5
    return int(round(x, 0))

def times1e3(x, pos=None):
    x = x * 1e3
    return int(round(x, 0))


def peak(x, y):
    max_int = np.max(y)
    center_idx = np.where(y == max_int)[0]
    return x[center_idx]


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


def create_color_palette(self):
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


def plot_dist(self, dist='energy', species_l=None, ion_flux=1, e_lims=None, title=None, elev=15, azim=25-90):
    if self.isVerbose:
        print(
              f'Plotting {dist} distribution'
              )
    if species_l is None:
        species_l = self.species_a

    params = {'legend.fontsize': 'large',
              'figure.titlesize': 'x-large',
              'axes.labelsize': 'large',
              'axes.titlesize': 'medium',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    pylab.rcParams.update(params)

    # def logscale(x, pos=None):
    #     x = np.log10(x)
    #     return int(round(x, 0))

    y_format = FuncFormatter(times1e5)

    self.create_color_palette()  # create color palette based on the minerals present

    if self.impactor == 'SW':
        impactor = f'H$^+_{{{self.sw_comp[0]}}}$ He$^{{2+}}_{{{self.sw_comp[1]}}}$'
    else:
        impactor = self.impactor

    if title is not None:
        plottitle = title
    else:
        plottitle = f'{self.impactor} \u2192 {self.casename}'

    title_impactor_angle = r'$\alpha_{in}$ = ' + str(self.dist_angle)

    if dist == 'energy':
        fig_dist = plt.figure(figsize=(6, 4.5), dpi=300)
        ax_dist = fig_dist.add_subplot(1, 1, 1)
        dist_df = self.edist_df
        # if title == '':
        #     title = f'Energy distribution'
        if ion_flux != 1:
            ax_dist.set_title(f'10$^{{{math.log(ion_flux, 10):.2g}}}\\times$ {impactor} ' + r'm$^2$s$^{-1}$')
        else:
            ax_dist.set_title(title_impactor_angle)

    elif dist == 'angular':
        fig_dist = plt.figure(figsize=(6, 4.5), dpi=300)
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

    elif dist == 'plume':
        fig_dist = plt.figure(figsize=(6, 6), dpi=300)
        dist_df = self.plume_df
        ax_dist = fig_dist.add_subplot(1, 1, 1, projection='3d')

    elif dist == 'surface_plume':
        fig_dist = plt.figure(figsize=(6, 6), dpi=300)
        dist_df = self.plume_df
        ax_dist = fig_dist.add_subplot(1, 1, 1, projection='3d')

    else:
        print('Pass either \'energy\', \'angular\', \'plume\' or \'surface_plume\' as distribution (dist) type.')
        sys.exit()

    """
    Return quantitative results.
    """

    i_ymax = 0
    labels = []
    loss_text = ['Loss fraction']
    quant_dist = pd.DataFrame(columns=self.species_a)
    for ss, species in enumerate(species_l):
        if self.is_summed_up:
            quant_dist_i = dist_df[species][0].apply(
                lambda x: x * ion_flux * self.yield_df[species].loc[self.dist_angle])
        else:
            quant_dist_i = dist_df[species].apply(
                lambda x: x * ion_flux * self.yield_df[species].loc[self.dist_angle])

        quant_dist[species] = quant_dist_i

        if self.is_summed_up:
            peak_value = peak(quant_dist_i.index.values, quant_dist_i.iloc[:, 0])[0]  # get global maxima

            if dist == 'energy':
                """ adds peak ENERGY value to label"""
                peak_label = r'$E_{{{}}}$'.format(species) + \
                             f' = {peak_value:2.1f}' + ' eV'
                """Add escape velocity lines and loss fractions"""

            elif dist == 'angular':
                """ adds peak ANGLE value to label"""
                peak_value = rad2deg(peak_value)
                peak_label = r'$\theta_{{{}}}$'.format(species) + \
                             f' = {peak_value:2.1f}' + r'$^\circ$'

            else:
                peak_label = ''

            labels.append(f'{peak_label.rjust(2)}')

            # todo: re-implement loss fraction in plot
            # if self.v_esc:
            #     lossfrac = self.edist_loss_df.iloc[0][species].values.tolist()[0][0]
            #     loss_text.append(f'{species}: {lossfrac:.2%}')

    if self.is_summed_up:
        if dist == 'plume' or dist == 'surface_plume':
            quant_dist = quant_dist.reset_index(names=['phi', 'theta'])

        if dist == 'plume':
            sin_theta = np.sin(quant_dist.theta)
            cos_theta = np.cos(quant_dist.theta)
            sin_phi = np.sin(quant_dist.phi)
            cos_phi = np.cos(quant_dist.phi)

            for species in self.species_a:
                quant_dist['x'] = sin_theta * cos_phi
                quant_dist['y'] = sin_theta * sin_phi
                quant_dist['z'] = cos_theta
                quant_dist[['x', 'y', 'z']] = quant_dist[['x', 'y', 'z']].mul(quant_dist[species], axis='index')

                color = self.plotting_key['color'].loc[species]
                ax_dist.scatter(quant_dist['x'], quant_dist['y'], quant_dist['z'],
                                c=color, s=0.1, label=species)
                ax_dist.legend()

        elif dist == 'surface_plume':
            for species in self.species_a:
                color = self.plotting_key['color'].loc[species]
                ax_dist.scatter(quant_dist['theta'], quant_dist['phi'], quant_dist[species],
                                c=color, s=0.1, label=species)
        else:
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
            ax_inset.yaxis.set_major_formatter(y_format)
            ax_inset.xaxis.set_major_locator(plt.MaxNLocator(2))
            ax_inset.yaxis.set_major_locator(plt.MaxNLocator(3))
            ax_inset.get_legend().remove()

        if quant_dist.max().iloc[0] >= i_ymax and dist == 'energy':
            i_ymax = quant_dist.max().max()

    else:
        for col in quant_dist.columns:

            if dist == 'plume' or dist == 'surface_plume':
                # first, drop all columns that only contain zeroes (elements are mineral-specific)
                quant_dist[col][0] = quant_dist[col][0].loc[:, (quant_dist[col][0] != 0).any(axis=0)]
                min_elem_l = quant_dist[col][0].columns.to_list()
                quant_dist[col][0] = quant_dist[col][0].reset_index(names=['phi', 'theta'])

            if dist == 'surface_plume':
                for min_elem in min_elem_l:
                    ax_dist.scatter(quant_dist[col][0]['theta'],
                                    quant_dist[col][0]['phi'],
                                    quant_dist[col][0][min_elem],
                                    alpha=0.5, s=0.1, label=min_elem)

            elif dist == 'plume':
                sin_theta = np.sin(quant_dist[col][0].theta)
                cos_theta = np.cos(quant_dist[col][0].theta)
                sin_phi = np.sin(quant_dist[col][0].phi)
                cos_phi = np.cos(quant_dist[col][0].phi)

                for min_elem in min_elem_l:
                    quant_dist[col][0]['x'] = sin_theta * cos_phi
                    quant_dist[col][0]['y'] = sin_theta * sin_phi
                    quant_dist[col][0]['z'] = cos_theta
                    quant_dist[col][0][['x', 'y', 'z']] = quant_dist[col][0][['x', 'y', 'z']].\
                        mul(quant_dist[col][0][min_elem], axis='index')

                    ax_dist.scatter(quant_dist[col][0]['x'],
                                    quant_dist[col][0]['y'],
                                    quant_dist[col][0]['z'],
                                    s=0.1, label=min_elem)
                    ax_dist.legend()

            else:
                quant_dist[col][0].fillna(quant_dist[col][0].mode(), inplace=True)
                quant_dist[col][0] = quant_dist[col][0].loc[:, (quant_dist[col][0] != 0).any(axis=0)]
                sns.lineplot(ax=ax_dist,
                             data=quant_dist[col][0],
                             palette=dict(self.plotting_key['color']),
                             dashes=True,  # dict(self.plotting_key['line']),
                             alpha=1.0)
            if quant_dist[col][0].max().iloc[0] >= i_ymax and dist == 'energy':
                i_ymax = quant_dist[col][0].max().max()

    """
    Adjust axis limits depending on distribution type
    """
    if dist == 'plume':
        # Tweak the limits and add latex math labels.
        ax_dist.set_xlabel(r'x')
        ax_dist.set_ylabel(r'y')
        ax_dist.set_zlabel(r'z')

        ax_dist.set_zlim([0, None])

        ax_dist.view_init(elev, azim)

        ax_dist.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax_dist.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax_dist.zaxis.set_major_locator(plt.MaxNLocator(4))

        ax_dist.xaxis.set_major_formatter(FuncFormatter(times1e3))
        ax_dist.yaxis.set_major_formatter(FuncFormatter(times1e3))

        # Placement 0, 0 would be the bottom left, 1, 1 would be the top right.
        ax_dist.text2D(0.90, 0.85, r'$\times$10$^{-3}$', fontsize='x-large', transform=ax_dist.transAxes)

    if dist == 'surface_plume':
        ax_dist.set_xlabel(r'$\theta$')
        ax_dist.set_ylabel(r'$\phi$')
        ax_dist.set_zlabel(r'$\rho\times$10$^{-3}$')

        ax_dist.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax_dist.yaxis.set_major_locator(plt.MaxNLocator(4))
        ax_dist.zaxis.set_major_locator(plt.MaxNLocator(4))

    if dist == 'surface_plume' or dist == 'plume':
        ax_dist.tick_params(axis='x', which='major', pad=1)
        ax_dist.tick_params(axis='y', which='major', pad=1)
        ax_dist.tick_params(axis='z', which='major', pad=2)

        ax_dist.zaxis.set_major_formatter(FuncFormatter(times1e3))
        ax_dist.view_init(elev, azim)

        # Adjust plume legend

        if self.is_summed_up:  # or self.isMassBalanced #deprecated
            handles, labels = ax_dist.get_legend_handles_labels()  # labels are pre-defined
        else:
            handles, labels = ax_dist.get_legend_handles_labels()  # labels are pre-defined
            labels = [label.replace('_', '$_\mathrm{') + '}$' for label in labels]

        yanchor = 0.09
        nrows = 2

        if self.is_summed_up:
            yanchor = 0.04
            nrows = 2

        ax_dist.legend(handles,
                       labels,
                       loc='upper center',
                       bbox_to_anchor=(0.5, yanchor),
                       ncol=min(max(int(np.ceil(len(handles) / nrows)), 2), 6),
                       frameon=False,
                       markerscale=20,
                       handletextpad=-0.4,
                       columnspacing=0.5)

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
                ax_dist.yaxis.set_major_formatter(y_format)
        else:
            ax_dist.set_ylabel(r'particles [at. ion$^{-1}$ eV$^{-1}$]')
        ax_dist.xaxis.set_major_locator(plt.MaxNLocator(4))

        # Adjust edist Legend
        if self.is_summed_up:
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
                ax_dist.yaxis.set_major_formatter(y_format)
        else:
            ax_dist.set_xlabel(r'yield [atoms s$^{-1}$ cm$^{-2}$]')  # , fontsize=8)

        ax_dist.xaxis.set_label_coords(0.25, 0.235)
        ax_dist.yaxis.set_major_locator(plt.MaxNLocator(4))
        if self.logplot:
            ax_dist.set_rscale('symlog', linthresh=1e-5)
            ax_dist.set_rlim(0)

        # Adjust adist legend

        if self.is_summed_up:  # or self.isMassBalanced #deprecated
            handles, _ = ax_dist.get_legend_handles_labels()  # labels are pre-defined
        else:
            handles, labels = ax_dist.get_legend_handles_labels()  # labels are pre-defined
            labels = [label.replace('_', '$_\mathrm{') + '}$' for label in labels]

        yanchor = 0.18
        nrows = 3
        if self.is_summed_up:
            yanchor = 0.15
            nrows = 2

        ax_dist.legend(handles,
                       labels,
                       loc='upper center',
                       bbox_to_anchor=(0.5, yanchor),
                       ncol=min(max(int(np.ceil(len(handles) / nrows)), 2), 5),
                       frameon=False)

    plotname = f'{self.impactor} \u2192 {self.casename}'
    file_plotname = safe_title_trans(plotname)
    filename = file_plotname + \
               '_' + \
               dist[0] + 'dist' + \
               '_' + \
               '_'.join(species_l)
    filename = safe_title_trans(filename)
    if self.is_summed_up:
        summedsfx = '_sm'
    else:
        summedsfx = ''

    elevazim = ''.join(['e', str(elev), 'a', str(azim)])

    fig_dist.suptitle(plottitle, y=0.98)
    fig_dist.savefig(os.path.join(self.outdir, filename) + f'_{self.dist_angle}{summedsfx}_{elevazim}.{self.file_format}',
                     dpi=300, format=self.file_format)  # , bbox_inches='tight')

    if self.show_plot:
        fig_dist.show()
    plt.close(fig_dist)
    return fig_dist, ax_dist


def plot_yield(self, exp_H_data=None, exp_He_data=None):
    if self.isVerbose:
        print(
              'Plotting yield'
              )
    params = {'legend.fontsize': 'large',
              'figure.titlesize': 'x-large',
              'figure.figsize': (6, 4.5),
              'axes.labelsize': 'large',
              'axes.titlesize': 'medium',
              'xtick.labelsize': 'large',
              'ytick.labelsize': 'large'}
    pylab.rcParams.update(params)
    fig = plt.figure(dpi=300)
    self.create_color_palette()  # create plotting dictionary (only colors for now)
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
        modat_label = self.sfx_for_plot(drop_prefix=False)
        if self.rho_system_df:
            rho_sys = self.rho_system_df.sum()
            labels[0] = f'{modat_label},\nrho = {rho_sys:0.2f}' + ' g cm$^{-3}$'

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
