from tableInterpolation import *
from constants import *

"""This script initializes our star, defining its mass, the guesses, and other stellar properties. This also defines 
the MESA parameters that we're comparing our model against (must run MESA first). 
"""

M_factor = 2.0 # 2 solar mass star

# USE ZAMS SOLAR ESTIMATE
M_sun = 1.989e+33
R_sun = 6.957e+10
L_sun = 3.839e+33
g_sun = G * M_sun / (R_sun ** 2)
rho_sun = M_sun / ((4 / 3) * np.pi * (R_sun ** 3))

M_star = M_sun * M_factor
L_star = L_sun * M_factor ** 3.9
R_star = R_sun * M_factor ** .2
g_star = G * M_star / (R_star ** 2)

Mr = 1e-8 * M_star
M_cut = M_star / 5

X = 0.70
Y = 0.28
Z = 0.02
XCNO = (2/3) * Z
mu = 4 / (3 + 5 * X)

opacityTable = totTable

Pcg = (3 / (8 * np.pi)) * G * (M_star ** 2) / (R_star ** 4)
Tcg = (1 / 2) * (mu / (Na * k)) * (G * M_star / R_star)
Lg = L_star
Rg = R_star

guess = np.array([Lg, Pcg, Rg, Tcg])

# MESA PROPERTIES
# FOR 2 SOLAR MASSES
L_expect = (10**1.207703)*L_sun
R_expect = (10**0.223076)*R_sun
Tc_expect = (10**7.320338)
rhoc_expect = (10**1.811656)
Pc_expect = (1/3)*a*(Tc_expect**4) + rhoc_expect*Na*k*Tc_expect/mu
MESA = np.array([L_expect, Pc_expect, R_expect, Tc_expect])
