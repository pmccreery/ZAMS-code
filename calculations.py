from properties import *
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import pandas as pd

def density(P, T):
    """
    Calculates the density for a given pressure and temperature. Assumes pure gas and radiation pressure
    :param P: (float) pressure value
    :param T: (float) temperature value
    :return: (float) density
    """
    return (P - ((1 / 3) * a * T ** 4)) * (mu * mp) / (k * T)


def opacity_pressure(rho, T, g):
    """
    Calculates the pressure based on opacity value
    :param rho: (float) density value
    :param T: (float) temperature value
    :param g: (float) gravity value
    :return: (float) pressure value
    """
    return (2 / 3) * g / opacityValue(log_R, log_T, totTable, rho, T)


def gas_and_rad_pressure(rho, T):
    """
    Calculates the pressure based on gas and radiation pressure
    :param rho: (float) density value
    :param T: (float) temperature value
    :return: (float) pressure value
    """
    return (1 / 3) * a * T ** 4 + rho * Na * k * T / mu


def rho_solver(rho, T, g):
    """
    Calculates the fractional difference between opacity pressure and gas+radiation pressure
    :param rho: (float) density value
    :param T: (float) temperature value
    :param g: (float) gravity value
    :return: (float) fractional difference between opacity pressure and gas+radiation pressure
    """
    diff = 1 - (opacity_pressure(rho, T, g) / gas_and_rad_pressure(rho, T))
    return diff


def f11(rho, T):
    """
    Calculates the weak screening factor f_11
    :param rho: (float) density value
    :param T: (float) temperature value
    :return: (float) weak screening factor
    """
    T7 = T / 1e7
    if T7 < 1:
        psi = 1
    elif T7 > 3.5:
        psi = 1.4
    else:
        psi = .15 * T7 + .85
    return np.exp(5.92e-3 * (rho / T7 ** 3) ** (1 / 2)) * psi


def energy(rho, T):
    """
    Calculates the energy generation rate from the proton-proton chain and the CNO cycle
    :param rho: (float) density value
    :param T: (float) temperature value
    :return: (float) energy generation rate
    """
    T9 = T / 1e9
    g11 = 1 + (3.82 * T9) + (1.51 * T9 ** 2) + (.144 * T9 ** 3) - (.0114 * T9 ** 4)
    epp = 2.57e4 * f11(rho, T) * g11 * rho * (X ** 2) * (T9 ** (-2 / 3)) * np.exp(-3.381 / T9 ** (1 / 3))
    g141 = 1 - (2.00 * T9) + (3.41 * T9 ** 2) - (2.43 * T9 ** 3)
    ecno = 8.24e25 * g141 * XCNO * X * rho * T9 ** (-2 / 3) * np.exp(-15.231 * T9 ** (-1 / 3) - (T9 / .8) ** 2)
    return epp + ecno


def delrad(P, T, L, kappa, M):
    """
    Calculates the temperature gradient in radiative zones
    :param P: (float) pressure value
    :param T: (float) temperature value
    :param L: (float) luminosity value
    :param kappa: (float) opacity value
    :param M: (float) interior mass value
    :return: (float) temperature gradient
    """
    return (3 / (16 * np.pi * a * c)) * (P * kappa / (T ** 4)) * (L / (G * M))


def load1(Pc, Tc):
    """
    Loads the core's initial values for the IVP
    :param Pc: (float) pressure value at the core
    :param Tc: (float) temperature value at the core
    :return: (array) boundary values near the core for L, P, r, and T
    """
    rhoc = density(Pc, Tc)
    ec = energy(rhoc, Tc)
    kappac = opacityValue(log_R, log_T, totTable, rhoc, Tc)

    Li = ec * Mr
    Pi = Pc - (3 * G / (8 * np.pi)) * ((4 * np.pi * rhoc / 3) ** (4 / 3)) * Mr ** (2 / 3)
    ri = np.power(3 * Mr / (4 * np.pi * rhoc), 1 / 3)

    del_rad = (3 / (16 * np.pi * a * c)) * (Pc * kappac / (Tc ** 4)) * (Li / (G * Mr))
    del_ad = .4

    Ti = 0
    if del_ad >= del_rad:
        Ti = np.power(Tc ** 4 - (1 / (2 * a * c)) * ((3 / (4 * np.pi)) ** (2 / 3)) * kappac * ec * (rhoc ** (4 / 3)) * (
                Mr ** (2 / 3)), 1 / 4)
    elif del_rad > del_ad:
        lnT = np.log(Tc) - np.power(np.pi / 6, 1 / 3) * G * del_ad * np.power(rhoc, 4 / 3) * np.power(Mr, 2 / 3) / Pc
        Ti = np.exp(lnT)
    else:
        print("Something has gone wrong.")

    return [Li, Pi, ri, Ti]


def load2(Ls, Rs):
    """
    Loads the surface's initial values for the IVP
    :param Ls: (float) surface luminosity value
    :param Rs: (float) stellar radius value
    :return: (array) boundary values near the surface for L, P, r, and T
    """
    gs = G * M_star / (Rs ** 2)
    Ts = np.power(Ls / (4 * np.pi * sb * Rs ** 2), 1 / 4)
    rhos = brentq(rho_solver, 10 ** -12, 10 ** -6, args=(Ts, gs))
    kappas = opacityValue(log_R, log_T, totTable, rhos, Ts)
    Ps = 2 * gs / (3 * kappas)

    return [Ls, Ps, Rs, Ts]


def derivs(x, y):
    """
    Calculates the stellar structure equations' derivatives
    :param x: mass coordinate to calculate derivatives at
    :param y: array of parameter values for [L, P, r, T]
    :return: (array) contains the derivative values at mass point x
    """
    rho = density(y[1], y[3])
    kappa = opacityValue(log_R, log_T, totTable, rho, y[3])
    del_rad = (3 / (16 * np.pi * a * c)) * (y[1] * kappa / (y[3] ** 4)) * (y[0] / (G * x))
    del_ad = .4
    nabla = np.minimum(del_rad, del_ad)

    dldx = energy(rho, y[3])
    dpdx = - G * x / (4 * np.pi * y[2] ** 4)
    drdx = 1 / (4 * np.pi * y[2] ** 2 * rho)
    dtdx = - (G * x * y[3] / (4 * np.pi * (y[2] ** 4) * y[1])) * nabla

    return [dldx, dpdx, drdx, dtdx]


def plotting(y_o, y_i, save=False):
    """
    Plotting the results of the shooting method.
    :param y_o: solve_ivp result for outward integration
    :param y_i: solve_ivp result for inward integration
    :param save: boolean value for if the plot should be saved or not
    :return:
    """
    plt.figure(figsize=(7, 7))

    plt.plot(y_o.t / M_star, y_o.y[0] / np.max(y_i.y[0]), c='#009E73', linestyle='-', lw=3,
             label=r'$\mathcal{L}/\mathcal{L}_*$')
    plt.plot(y_i.t / M_star, y_i.y[0] / np.max(y_i.y[0]), c='#009E73', linestyle='-', lw=3)

    plt.plot(y_o.t / M_star, y_o.y[1] / np.max(y_o.y[1]), c='#D55E00', linestyle='--', lw=3, label=r'$P/P_c$')
    plt.plot(y_i.t / M_star, y_i.y[1] / np.max(y_o.y[1]), c='#D55E00', linestyle='--', lw=3)

    plt.plot(y_o.t / M_star, y_o.y[2] / np.max(y_i.y[2]), c='#CC79A7', linestyle='-.', lw=3, label=r'$r/R_*$')
    plt.plot(y_i.t / M_star, y_i.y[2] / np.max(y_i.y[2]), c='#CC79A7', linestyle='-.', lw=3)

    plt.plot(y_o.t / M_star, y_o.y[3] / np.max(y_o.y[3]), c='#56B4E9', linestyle=':', lw=3, label=r'$T/T_c$')
    plt.plot(y_i.t / M_star, y_i.y[3] / np.max(y_o.y[3]), c='#56B4E9', linestyle=':', lw=3)

    plt.axvspan(-.01, 1 / 5, facecolor='#f5f0aa')
    plt.axvspan(1 / 5, 1.01, facecolor='#0072B2')

    plt.legend(loc=(.72, .72), fontsize=14)

    plt.title('Converged Solutions')
    plt.ylabel('Normalized Value')
    plt.xlabel(r'Mass Coordinate ($\mathcal{M}_*$)')
    plt.xlim(-.01, 1.01)
    plt.ylim(-.01, 1.01)

    if save:
        plt.savefig('TotalPlot.png', dpi=800)