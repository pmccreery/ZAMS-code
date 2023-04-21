import numpy as np

def TPStep(f,x):
    fxi = f(x)
    dx_o = np.array([fxi[1], fxi[3]])
    x[1] += 2e15 * (dx_o[0] / x[1])
    x[3] += 2e7 * (dx_o[1] / x[3])
    return x

def new_guess(f, x, overshoot_frac=.1, thresh = .01):
    print('Guess: ', x)

    h = np.array([1e33, 1e16, 1e8, 1e4])
    n = len(x)
    J = np.zeros((n, n))
    fx = f(x)

    max_frac = np.max(np.abs(fx/x))
    if max_frac < thresh:
        print("We have convergence!")
        return x
    else:
        print('Fractional Difference: ', np.abs(fx / x))

        for i in range(n):
            x_copy = x.copy()
            x_copy[i] += h[i]
            J[:, i] = (f(x_copy) - fx) / h[i]

        x_new = x + (np.dot(np.linalg.inv(J), -fx) * overshoot_frac)
        return x_new


def newton_raphson(f, x):
    i = 0
    while i < 50:
        x_new = new_guess(f, x)
        if np.array(x).tolist() == np.array(x_new).tolist():
            return x
        else:
            x = x_new
    i += 1

    print('Convergence is taking over 50 steps. Stopping.')