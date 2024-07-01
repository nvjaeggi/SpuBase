# ===============================
# AUTHORS: Noah Jäggi, Adam K. Woodson
# CREATE DATE: 4. April 2024
# PURPOSE: Fit SDTrimSP angular data in 3D
# SPECIAL NOTES:
# ===============================
# Change History:
# 26/06/2024: Added lobe function (Woodson)
# ==================================

import os
import warnings
import numpy as np
import pandas as pd
from ._plotting_sb import deg2rad
import scipy.integrate as integrate
from scipy.optimize import curve_fit
from scipy.optimize import minimize
from scipy.optimize import OptimizeWarning


def rotate_y(alpha):
    return np.array([[np.cos(alpha), 0, np.sin(alpha)],
                     [0, 1, 0],
                     [-np.sin(alpha), 0, np.cos(alpha)]])


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


def cosfit_integrate(theta_tilt, theta_m, theta_n):
    theta_k_leq_tilt, k_tilt_leq_error = integrate.quad(
        lambda x: np.power(np.cos((-abs(theta_tilt) - x) / (1 - 2 / np.pi * -abs(theta_tilt))), theta_m),
        -np.pi / 2, -abs(theta_tilt))

    theta_k_geq_tilt, k_tilt_geq_error = integrate.quad(
        lambda x: np.power(np.cos((x - theta_tilt) / (1 - 2 / np.pi * theta_tilt)), theta_n), theta_tilt,
        np.pi / 2)
    return theta_k_leq_tilt, theta_k_geq_tilt


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


def lobe_function(params, data_df):
    amplitude, theta_tilt, exponent_m, exponent_n = params

    sin_theta = np.sin(data_df.theta)
    cos_theta = np.cos(data_df.theta)
    sin_phi = np.sin(data_df.phi)
    cos_phi = np.cos(data_df.phi)

    output_df = data_df.copy(deep=True)
    output_df['x'] = sin_theta * cos_phi
    output_df['y'] = sin_theta * sin_phi
    output_df['z'] = cos_theta

    x_prime = rotate_y(theta_tilt).dot(np.array([1, 0, 0]))
    y_prime = np.array([0, 1, 0])
    z_prime = rotate_y(theta_tilt).dot(np.array([0, 0, 1]))

    prime_coords_dtheta_dphi_rho = []

    for row in output_df[['x', 'y', 'z']].to_numpy():
        q_vector = np.array([row[0], row[1], row[2]])

        xp = q_vector.dot(x_prime)
        yp = q_vector.dot(y_prime)
        zp = q_vector.dot(z_prime)

        dtheta = np.arccos(zp)

        dphi = np.arccos(xp / np.sqrt(xp * xp + yp * yp))

        rho = ((1.0 + np.cos(2.0 * dtheta)) / 2.0)\
            ** (np.cos(dphi) ** 2.0 * exponent_m + np.sin(dphi) ** 2.0 * exponent_n)
        rho *= amplitude * np.heaviside(row[2], 0) * np.heaviside(zp, 0)

        prime_coords_dtheta_dphi_rho.append([xp, yp, zp, dtheta, dphi, rho])

    output_df[['x_prime',
               'y_prime',
               'z_prime',
               'dtheta',
               'dphi',
               'rho']] = pd.DataFrame(prime_coords_dtheta_dphi_rho, index=output_df.index)

    # return points in cartesian and spherical coordinates
    output_df[['x', 'y', 'z']] = output_df[['x', 'y', 'z']].mul(output_df['rho'], axis='index')
    return output_df


def cost_function(params, df):
    fit_df = lobe_function(params, df)
    res = sum((np.array(fit_df['rho']) - np.array(df['rho']))**2)
    return res


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


def invpairdiff(lst):
    f_lst = lst.astype(float)  # floats are required for "** (-1)" tp work
    length = len(lst)
    total = np.ones(length)
    for i in range(length - 1):
        # adding the alternate numbers
        total[i] = (f_lst[i + 1] - f_lst[i]) ** (-1)
    return total


def fit_eq(df, eq, binary_e=1e5, init_guess=None):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fxn()
        # mask all values that are too close to the origin (makes it nigh impossible to fit)
        if eq == cosfit:
            maskmin = deg2rad(-80)
            maskmax = deg2rad(35)
            maskforfit = [maskmin < x < maskmax for x in df.iloc[:, 0]]
            xdata = df[maskforfit].iloc[:, 0]
            ydata = df[maskforfit].iloc[:, 1]
            popt, pcov = curve_fit(f=eq,
                                   xdata=xdata,
                                   ydata=ydata,
                                   # bounds=([-np.pi/3,0.8, 0.8], [np.pi/3, 3,3]),
                                   p0=[xdata.max(), deg2rad(-10), 1.5, 1.5],
                                   # p0=[-40/deg_from_rad,2,1],
                                   # sigma=ydata**-2,
                                   # method='trf',
                                   maxfev=50000)

            """
            Determine normalization factor
            """
            theta_tilt = popt[1]
            theta_m = popt[2]
            theta_n = popt[3]
            k_leq, k_geq = cosfit_integrate(theta_tilt, theta_m, theta_n)
            popt[0] = k_leq + k_geq
            return popt, pcov

        elif eq == thompson:
            xdata = df.iloc[:, 0].to_numpy()
            ydata = df.iloc[:, 1].to_numpy()
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
        elif eq == lobe_function:
            bounds = [(0, np.pi / 2), (0.0, np.pi/4), (1, 5), (1, 5)]
            result = minimize(cost_function,
                              bounds=bounds,
                              method='L-BFGS-B',
                              x0=init_guess,
                              args=(df,))
            solution = result['x']

            return solution


def eckstein_fit_data(self):
    if 'SW' in self.impactor:
        impactor_a = ['H', 'He']
        sw_comp = self.sw_comp
    else:
        impactor_a = [self.impactor]
        sw_comp = [1]

    # average energy and mass of impactor, given the solar wind ion composition
    self.energy_impactor = sum([*map(self.ekin.get, impactor_a)] * np.array(sw_comp))
    self.mass_impactor = sum([*map(self.amu_dic.get, impactor_a)] * np.array(sw_comp))

    if not self.sw_comparison:  # todo: remove this check
        if 'SW' in self.impactor and self.sw_comp[0] == 0.96:
            impactor_a = ['SW']

    eckstein_dfdf = pd.DataFrame(columns=self.mineral_a)
    data_dfdf = pd.DataFrame(columns=self.mineral_a)
    for mm, mineral in enumerate(self.mineral_a):
        if self.isDebug:
            print(f'{mineral}')
        data_df = pd.DataFrame(columns=impactor_a)
        eckstein_df = pd.DataFrame(columns=impactor_a)
        sfx = self.sfx
        if self.sulfur_diffusion and mineral in ['Tro', 'Nng', 'Abd', 'Bzn', 'Was', 'Old', 'Dbr']:
            sfx = 'sbbxd'

        for ii, impactor in enumerate(impactor_a):
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

            eckstein_param_df = pd.DataFrame(columns=cp_aux)
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
                    result = curve_fit(lambda angle, b_var, c_var, f_var:
                                       eckstein(angle, b_var, c_var, f_var, anti_overflow_fac * y0),
                                       x_values,
                                       anti_overflow_fac * y_values,
                                       sigma=x_weight)
                    popt, _ = result[:2]
                    b, c, f = popt
                    if self.isDebug:
                        print(f'{element}: {popt}')
                    eckstein_param_df[element] = [b, c, f, y0]

            data_df[impactor] = [data]
            eckstein_df[impactor] = [eckstein_param_df]

        eckstein_dfdf[mineral] = [eckstein_df]
        data_dfdf[mineral] = [data_df]

    return data_dfdf, eckstein_dfdf


