from scipy.integrate import solve_ivp
from calculations import *
from properties import *
import ray
from ray.util.multiprocessing import Pool

"""
This script defines the functions that do the shooting method and calculates the difference between the two integrations
"""


def single_run(params):
    """
    Integrates the stellar structure equations from the two boundaries (two initial value problems)

    Arguments:
        params (arr):  array carrying the value of the parameters (L, P, r, T)

    Returns:
        x_o (arr): array of mass coordinates the outward integration was evaluated at
        x_i (arr): array of mass coordinates the inward integration was evaluated at
        y_i (arr): array of parameters evaluated at the mass coordinates for the inward integration
        y_o (arr): array of parameters evaluated at the mass coordinates for the outward integration
    """
    Li, Pi, Ri, Ti = params
    y0_o = load1(Pi, Ti)  # load in boundary conditions
    y0_i = load2(Li, Ri)

    x_tot = np.linspace(Mr, M_star, num=2000000)

    x_o = x_tot[x_tot < M_cut]
    t_span_o = [x_o[0], x_o[-1]]

    x_i = np.flipud(x_tot[x_tot >= M_cut])
    t_span_i = [x_i[0], x_i[-1]]

    ray.init(num_cpus=2)  # multiprocessing
    pool = Pool(2)

    y_or = pool.apply_async(solve_ivp, args=(derivs, t_span_o, y0_o, 'RK45', x_o))
    y_ir = pool.apply_async(solve_ivp, args=(derivs, t_span_i, y0_i, 'RK45', x_i))

    y_o = y_or.get()
    y_i = y_ir.get()
    ray.shutdown()

    # y_o = solve_ivp(derivs, t_span_o, y0_o, 'RK45', x_o)
    # y_i = solve_ivp(derivs, t_span_i, y0_i, 'RK45', x_i)

    return x_o, x_i, y_i, y_o


def shootf(params):
    """
    Calculates the difference between the two integrations at the mass cutoff point

    Arguments:
        params (arr):  array carrying the value of the parameters (L, P, r, T)

    Returns:
        array that contains the difference between the inward and outward integrations at the meeting point (M_cut)
    """
    _, _, y_i, y_o = single_run(params)

    dL = (y_i.y[0, -1] - y_o.y[0, -1])
    dP = (y_i.y[1, -1] - y_o.y[1, -1])
    dR = (y_i.y[2, -1] - y_o.y[2, -1])
    dT = (y_i.y[3, -1] - y_o.y[3, -1])

    return np.array([dL, dP, dR, dT])
