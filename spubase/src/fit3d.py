# ===============================
# AUTHORS: Noah Jäggi, Adam K. Woodson
# CREATE DATE: 4. April 2024
# PURPOSE: Fit SDTrimSP angular data in 3D
# SPECIAL NOTES:
# ===============================
# Change History:
#
# ==================================

import math
import time
import sys
import pandas as pd
import numpy as np
from numpy import sin, cos, arccos, array
from numpy.linalg import norm
import matplotlib.pyplot as plt

def rotate_y(alpha):
    return np.array([[np.cos(alpha), 0, np.sin(alpha)],
                     [0, 1, 0],
                     [-np.sin(alpha), 0, np.cos(alpha)]])


def lobeFunction(params, data_df):
    amplitude, theta_tilt, exponent_m, exponent_n = params

    sin_theta = sin(data_df.theta)
    cos_theta = cos(data_df.theta)
    sin_phi = sin(data_df.phi)
    cos_phi = cos(data_df.phi)

    output_df = data_df.copy(deep=True)
    output_df['x'] = sin_theta * cos_phi
    output_df['y'] = sin_theta * sin_phi
    output_df['z'] = cos_theta

    X_prime = rotate_y(theta_tilt).dot(np.array([1, 0, 0]))
    Y_prime = np.array([0, 1, 0])
    Z_prime = rotate_y(theta_tilt).dot(np.array([0, 0, 1]))

    prime_coords_dtheta_dphi_rho = []

    for row in output_df[['x', 'y', 'z']].to_numpy():
        q_vector = np.array([row[0], row[1], row[2]])
        xp = q_vector.dot(X_prime)
        yp = q_vector.dot(Y_prime)
        zp = q_vector.dot(Z_prime)

        dtheta = arccos(zp)

        dphi = arccos(xp / math.sqrt(xp * xp + yp * yp))

        rho = ((1.0 + cos(2.0 * dtheta)) / 2.0) ** (cos(dphi) ** 2.0 * exponent_m + sin(dphi) ** 2.0 * exponent_n)
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


def costfunction(params, df):
    fit_df = lobeFunction(params, df)
    res = sum((np.array(fit_df['rho']) - np.array(df['rho']))**2)
    return res

def plot3D(df, plotname):
    fig3D = plt.figure(dpi=300, figsize=(6, 2.5))
    ax3D_spherical = fig3D.add_subplot(121, projection='3d')
    ax3D_cartesian = fig3D.add_subplot(122, projection='3d')

    # Plot the surface
    ax3D_spherical.plot_trisurf(df['theta'], df['phi'], df['rho'], cmap=plt.get_cmap('jet'), linewidth=0, antialiased=False, alpha=0.2)
    ax3D_spherical.scatter(df['theta'], df['phi'], df['rho'], c='black', s=1)

    ax3D_cartesian.scatter(df['x'], df['y'], df['z'], c='black', s=1)

    ## Tweak the limits and add latex math labels.
    ax3D_spherical.set_xlabel(r'$\theta$')
    ax3D_spherical.set_ylabel(r'$\phi$')
    ax3D_spherical.set_zlabel(r'$\rho$')
    ax3D_spherical.view_init(25, 45 - 90)
    ax3D_cartesian.set_xlabel(r'x')
    ax3D_cartesian.set_ylabel(r'y')
    ax3D_cartesian.set_zlabel(r'z')
    ax3D_cartesian.set_xlim(-0.5, 0.5)
    ax3D_cartesian.set_ylim(-0.5, 0.5)

    for ax in [ax3D_spherical, ax3D_cartesian]:
        ax.view_init(25, 45-90)
        ax.set_zlim(0, 0.9)

    fig3D.suptitle(plotname.replace('_', ' '))

    return fig3D