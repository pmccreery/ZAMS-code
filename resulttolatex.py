from calculations import *

"""
This script takes the results of the converged solution and turns them into a latex table
"""


def round_to_sf(x, sf):
    """
    Rounds to significant digits
    :param x: (float) number to round
    :param sf: (int) number of significant digits required
    :return: number rounded to sf significant digits
    """
    return round(x, -int(np.floor(np.log10(abs(x)))) + (sf - 1))


def resulttolatex(y_o, y_i, df_return=False):
    """

    :param y_o: solve_ivp output for outward integration
    :param y_i: solve_ivp output for inward integration
    :param df_return: (boolean) if the dataframe should be returned (True) or the plain text for the latex table (False)
    :return: dataframe with values (if df_return == True)
    """

    x_tot = np.hstack((y_o.t, np.flipud(y_i.t)))
    r_tot = np.hstack((y_o.y[2], np.flipud(y_i.y[2])))
    inds = []
    r_want = np.linspace(0, 1, 60) * np.max(r_tot)
    for i in r_want:
        inds.append(np.argmin(abs(r_tot - i)))

    mass_vals = x_tot[inds]
    L_vals = np.hstack((y_o.y[0], np.flipud(y_i.y[0])))[inds]
    P_vals = np.hstack((y_o.y[1], np.flipud(y_i.y[1])))[inds]
    r_vals = np.hstack((y_o.y[2], np.flipud(y_i.y[2])))[inds]
    T_vals = np.hstack((y_o.y[3], np.flipud(y_i.y[3])))[inds]
    rho_vals = density(P_vals, T_vals)
    # e_vals = energy(rho_vals, T_vals)
    e_vals = []
    for r, T in zip(rho_vals, T_vals):
        e_vals.append(energy(r, T))
    e_vals = np.array(e_vals)
    ad_vals = np.ones(len(mass_vals)) * .4
    kappa_vals = opacityValue(log_R, log_T, totTable, rho_vals, T_vals)
    rad_vals = delrad(P_vals, T_vals, L_vals, kappa_vals, mass_vals)
    grad_vals = []
    radoconv = []
    for j, k in zip(ad_vals, rad_vals):
        grad_vals.append(np.min([j, k]))
        if np.min([j, k]) == 0.4:
            radoconv.append('Convective')
        else:
            radoconv.append('Radiative')
    grad_vals = np.array(grad_vals)
    radoconv = np.array(radoconv)

    d = {'Mass': mass_vals / M_star, 'Luminosity': L_vals / np.max(L_vals), 'log Pressure': np.log10(P_vals),
         'Radius': r_vals / np.max(r_vals), 'log Temperature': np.log10(T_vals), 'log Density': np.log10(rho_vals),
         'Energy Generation': e_vals, 'Adiabatic': ad_vals, 'Actual': grad_vals, 'Transport': radoconv}
    df = pd.DataFrame(d)
    df = df.drop_duplicates()

    strdf = df['Transport']
    df = df.drop('Transport', 1)

    # Function to round to a certain number of significant digits
    def round_to_significant_digits(df, digits):
        for col in df.columns:
            # Compute the order of magnitude for each value
            order_of_magnitude = np.floor(np.log10(np.abs(df[col])))
            # Compute the rounding factor
            rounding_factor = 10 ** (digits - 1 - order_of_magnitude)
            # Round the values using the rounding factor
            df[col] = np.round(df[col] * rounding_factor) / rounding_factor
        return df

    # Round data frame to 6 significant digits
    df = round_to_significant_digits(df, 6)
    df['Transport'] = strdf

    if df_return:
        return df
    else:
        print(df.to_latex(header=False, index=False))
