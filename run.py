from shootf import *
from newton import *
from resulttolatex import *

"""
This script runs the Newton-Raphson method
"""

improved_guess = TPStep(shootf, guess)  # take the step to improve guess
converged_solution = newton_raphson(shootf, guess)  # NR

print('Percent Errors: {}'.format((MESA - converged_solution) / MESA * 100))  # Find percent differences
print(
    'Fractional Errors: {}'.format(shootf(converged_solution) / converged_solution))  # Fractional errors at the cutoff

# x_o, x_i, y_i, y_o = single_run(converged_solution)
# resulttolatex(y_o, y_i, df_return=False)
