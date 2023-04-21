import numpy as np
from scipy.interpolate import griddata, RBFInterpolator
import pandas as pd

'''
This script defines two functions and stitches together two opacity tables.
'''


def txttodf(filename):
    """
    Takes the 126 OPAL opacity tables and turns them into a dictionary that stores dataframes for each table.

    Arguments:
        filename (str):  path to the OPAL opacity file

    Returns:
        dict (dict): dictionary of each table provided in the file
        log_R (arr): array of log R values in the opacity table
        log_T (arr): array of log T values in the opacity table
    """
    with open(filename) as f:
        lines = f.readlines()

    dict = {}
    for k in range(126):  # 126 tables in the file
        arr = []
        log_T = []
        table_name = 'Table ' + str(k + 1)
        for j in range(70):
            arr1 = []
            for i in lines[6 + 77 * k + j].split()[1:]: arr1.append(
                float(i))  # note the length of each table, separation between
            log_T.append(float(lines[6 + 77 * k + j].split()[0]))
            arr.append(np.array(arr1))
            dict[table_name] = pd.DataFrame(arr)

    log_R = []
    for i in lines[4].split()[1:]: log_R.append(float(i))

    return dict, log_R, log_T


colw = list(np.ones(19, dtype=int) * 7)
colw.insert(0, 6)
lowT = pd.read_fwf('opacities/A09photo.7.02.tron', header=2, widths=colw)
lowT.columns = [''] * 20

dict, log_R, log_T = txttodf('opacities/GN93hz.txt')
highT = dict['Table 73']
highT.insert(0, '', log_T)  # rearrange the headers, indices, labels
highT.columns = [''] * 20
highT = highT.iloc[16:]
highT = highT.iloc[::-1]
totTable = pd.concat([highT, lowT])  # concatenate
log_T = np.array(totTable.iloc[:, 0])
totTable = pd.concat([highT, lowT]).iloc[:, 1:]

totTable.to_csv('opacities/totalOpacity.txt', sep=' ', index=None)  # save the stitched opacities


def opacityValue(r_arr, t_arr, table, rho_i, T_i):
    """
    Interpolates the opacity tables at a given density and temperature

    Arguments:
        r_arr (arr): array of log_R values in the table
        t_arr (arr): array of log_T values in the table
        table (df): the table that needs interpolated
        rho_i (float): density that requires an opacity value at (linear)
        T_i (float): temperature that requires an opacity value at (linear)

    Returns:
        kappa (float): opacity value at (rho_i, T_i)
    """
    points = []
    values = []
    for i in range(len(r_arr)):
        for j in range(len(t_arr)):
            points.append([r_arr[i], t_arr[j]])
            values.append(table.iloc[j, i])
    log_R_i = np.log10(rho_i / (T_i / 1e6) ** 3)  # find value of R given rho_i and T_i

    kappa = 10 ** griddata(points, values, (log_R_i, np.log10(T_i)), method='linear')  # non-log (linear) kappa value
    return kappa
