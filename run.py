from shootf import *
from newton import *
from resulttolatex import *

improved_guess = TPStep(shootf, guess)
converged_solution = newton_raphson(shootf, guess)

print('Percent Errors: {}'.format((MESA-converged_solution)/MESA*100))
print('Fractional Errors: {}'.format(shootf(converged_solution)/converged_solution))

#x_o, x_i, y_i, y_o = single_run(converged_solution)
#resulttolatex(y_o, y_i, df_return=False)
