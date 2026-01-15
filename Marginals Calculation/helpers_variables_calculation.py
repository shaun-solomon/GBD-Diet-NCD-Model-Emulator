import numpy as np
import scipy.optimize as optimize
from scipy.special import gamma


def run_for_beta(mu, sigma, u, l):
    """
    Calculate parameters (a and b) for the Beta distribution based on mean and standard deviation.
    mu (array-like): Mean values.
    sigma (array-like): Standard deviation values.
    u (array-like): Upper bounds for the distribution.
    l (array-like): Lower bounds for the distribution.
    Returns: Tuple (a, b, u, l) representing Beta distribution parameters.
    """
    vr = sigma ** 2
    epsilon = 10 ** (-10)

    selection_i = np.where((vr >= np.multiply((mu - l), (u - mu))) & (vr < np.multiply(mu, (u + l - mu))))[0]
    selection_ii = np.where((vr >= np.multiply((mu - l), (u - mu))) & (vr >= np.multiply(mu, (u + l - mu))))[0]
    selection_iii = np.where((vr < np.multiply((mu - l), (u - mu))))[0]

    if len(selection_i) == 0:
        delta = epsilon
        u[selection_iii] += delta
        l[selection_iii] -= delta
    else:
        delta = \
            beta_approx(mu[selection_i], vr[selection_i], u[selection_i],
                                   l[selection_i]) + epsilon
        u[selection_i] += delta
        u[selection_iii] += epsilon
        l[selection_i] -= delta
        l[selection_iii] -= epsilon

    l[selection_ii] = 0
    u[selection_ii] = vr[selection_ii] / mu[selection_ii] + mu[selection_ii] + epsilon

    mu_beta = (mu - l) / (u - l)
    vr_beta = vr / (u - l) ** 2

    a = np.abs(mu_beta ** 2 * (1 - mu_beta) / vr_beta - mu_beta)
    b = np.abs(mu_beta * (1 - mu_beta) ** 2 / vr_beta - 1 + mu_beta)

    a[a <= 1e-12] = 1e-12
    b[b <= 1e-12] = 1e-12
    return a, b, u, l


def beta_approx(mu, var, up, low):
    """
    Helper function to calculate parameters for the Beta distribution .
    mu (array-like): Mean values.
    var (array-like): Variance values.
    up (array-like): Upper bounds for the distribution.
    low (array-like): Lower bounds for the distribution.
    Returns: Parameter 'delta' for the Beta distribution.
    """
    return - (up - low) / 2 + np.power(((up - low) / 2) ** 2 + var - (mu - low) * (up - mu), 0.5)


def run_for_fisk(mu, sigma):
    """
    Calculate parameters (alpha and beta) for the Fisk (Log-Logistic) distribution based on mean and standard deviation.
    mu (array-like): Mean values.
    sigma (array-like): Standard deviation values.
    Returns: Tuple (alpha, beta) representing Fisk distribution parameters.
    """
    vr = sigma ** 2

    # calculate the variables for the weibull's and log-logistic (fisk) distribution
    # methods for optimization without constraint -> fisk
    optimization_methods = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']
    # 'BFGS', 'Nelder-Mead', 'Powell', 'CG', 'Newton-CG', 'L-BFGS-B', 'TNC', 'COBYLA', 'SLSQP',
    #                        'trust-constr', 'dogleg', 'trust-ncg', 'trust-exact', 'trust-krylov'
    for meth in optimization_methods:
        alpha_fisk, beta_fisk = calculate_parameters_fisk(mu, vr, meth)
        if np.all(abs(mu - (alpha_fisk * np.pi / beta_fisk) / np.sin(np.pi / beta_fisk)) < (0.05 * mu)):
            break
        else:
            if meth != 'trust-krylov':
                print('try next one as {} was not successful'.format(meth))
                continue
            else:
                print('Error: Optimization of log-logistic parameters failed!')
                exit()

    return alpha_fisk, beta_fisk


def fisk_func(variable, mu, var):
    """
    Fisk distribution function, used to solve the optimization.
    """
    x = 1 / variable
    return (np.sinc(x) - (1 + var / mu ** 2) / 2) ** 2


def calculate_parameters_fisk(mu, vr, meth='BFGS'):
    """
    Calculate parameters (alpha and beta) for the Fisk (Log-Logistic) distribution using optimization.
    mu (array-like): Mean values.
    vr (array-like): Variance values.
    meth (str): Optimization method.
    Returns: Tuple (alpha, beta) representing Fisk distribution parameters.
    """
    if mu < np.sqrt(vr):
        vr = (mu * 0.999) ** 2

    beta = optimize.minimize(fisk_func, np.array([np.pi / 2]), args=(mu, vr), method=meth,
                             bounds=optimize.Bounds(10**(-15), np.pi)).x[0]
    alpha = mu * np.sinc(1 / beta)
    return alpha, beta


def run_for_weibull(mu, sigma):
    """
    Calculate parameters (k and lambda) for the Weibull distribution based on mean and standard deviation.
    mu (array-like): Mean values.
    sigma (array-like): Standard deviation values.
    Returns: Tuple (k, lambda) representing Weibull distribution parameters.
    """
    vr = sigma ** 2
    optimization_methods_with_bounds = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']

    for meth in optimization_methods_with_bounds:
        k_weibull, lambda_weibull = calculate_parameters_weibull(mu, vr, weibull_func, meth)
        if abs(mu - lambda_weibull * gamma(1 + 1 / k_weibull)) < (0.01 * mu):
            break
        else:
            if meth != 'trust-constr':
                print('try next one as {} was not successful'.format(meth))
                continue
            else:
                print('Error: Optimization of weibull parameters failed!')
                exit()

    return k_weibull[0], lambda_weibull[0]


def helper_func_one(x):
    """
    Helper function used in the calculation of Weibull parameters for 'greater_one_weibull'.
    """
    return gamma(1/x + 1)/(gamma(2/x + 1) - gamma(1/x + 1)**2)**(1/2)


def greater_one_weibull(mean, std):
    """
    Calculate Weibull distribution parameters (k and lambda) for cases where mean/std ratio is greater than 1.
    mean (array-like): Mean values.
    std (array-like): Standard deviation values.
    Returns: Tuple (k, lambda) representing Weibull distribution parameters.
    """
    y = mean/std
    m = (helper_func_one(100) - 1)/99
    k = (y + m - 1)/m
    lamb = mean / gamma(1 + 1 / k)
    if np.any(abs(mean - lamb * gamma(1 + 1 / k)) > (0.01 * mean)):
        k = np.zeros_like(mean)
        lamb = np.zeros_like(mean)
        for run in np.arange(len(mean)):
            k[run], lamb[run] = run_for_weibull(mean[run], std[run])
    return k, lamb


def helper_func_two(x):
    """
    Helper function used in the calculation of Weibull parameters for 'smaller_one_eighth_weibull'.
    """
    return np.log(gamma(1 + 2*x)) - 2 * np.log(gamma(1 + x))


def smaller_one_eighth_weibull(mean, std):
    """
    Calculate Weibull distribution parameters (k and lambda) for cases where mean/std ratio is less than or equal to 1/8.
    mean (array-like): Mean values.
    std (array-like): Standard deviation values.
    Returns: Tuple (k, lambda) representing Weibull distribution parameters.
    """
    m = (helper_func_two(20) - helper_func_two(4)) / 16
    c = (5 * helper_func_two(4) - helper_func_two(20)) / 4
    y = mean/std
    k = m / (np.log(1 + 1/y**2) - c)
    lamb = mean / gamma(1 + 1 / k)
    if np.any(abs(mean - lamb * gamma(1 + 1 / k)) > (0.01 * mean)):
        print('problem for y <= 1/8')
        k = np.zeros_like(mean)
        lamb = np.zeros_like(mean)
        for run in np.arange(len(mean)):
            k[run], lamb[run] = run_for_weibull(mean[run], std[run])
    return k, lamb


def weibull_func(variable, mean, var):
    """
    PDF of the Weibull function.
    """
    k = variable
    return np.abs((gamma(1 + 2 / k) / (gamma(1 + 1 / k) ** 2) - 1) - (var / mean ** 2))


def calculate_parameters_weibull(mu, vr, func_to_solve, meth='Nelder-Mead'):
    """
    Calculate parameters (k and lambda) for the Weibull distribution using optimization.
    mu (array-like): Mean values.
    vr (array-like): Variance values.
    func_to_solve (function): Function to solve.
    meth (str): Optimization method.
    Returns: Tuple (k, lambda) representing Weibull distribution parameters.
    """
    k = optimize.minimize(func_to_solve, np.array(mu / np.sqrt(vr)), args=(mu, vr), method=meth,
                          bounds=[(0, np.inf)]).x # optimize.Bounds
    lamb = mu / gamma(1 + 1 / k)
    return k, lamb


def my_cons(x):
    f = np.zeros(2)
    f[0] = x + 0
    f[1] = np.inf - x
    return f


def run_for_invweibull(mu, sigma):
    """
    Calculate parameters (k and lambda) for the Inverse Weibull distribution based on mean and standard deviation.
    mu (array-like): Mean values.
    sigma (array-like): Standard deviation values.
    Returns: Tuple (k, lambda) representing Inverse Weibull distribution parameters.
    """
    vr = sigma ** 2
    optimization_methods_with_bounds = ['Nelder-Mead', 'Powell', 'L-BFGS-B', 'TNC', 'SLSQP', 'trust-constr']

    for meth in optimization_methods_with_bounds:
        k_invweibull, lambda_invweibull = calculate_parameters_invweibull(mu, vr, meth)
        if abs(mu - lambda_invweibull ** (-1) * gamma(1 - 1 / k_invweibull)) < (0.01 * mu):
            break
        else:
            if meth != 'trust-constr':
                print('try next one as {} was not successful'.format(meth))
                continue
            else:
                print('Error: Optimization of weibull parameters failed!')
                exit()

    return k_invweibull[0], lambda_invweibull[0]


def invweibull_func(variable, mean, var):
    """
    PDF of the inverse Weibull distribution.
    """
    k = variable
    return np.abs((gamma(1 - 2 / k) / (gamma(1 - 1 / k) ** 2) - 1) - (var / mean ** 2))


def calculate_parameters_invweibull(mu, vr, meth='Nelder-Mead'):
    """
    Calculate parameters (k and lambda) for the Inverse Weibull distribution using optimization.
    mu (array-like): Mean values.
    vr (array-like): Variance values.
    meth (str): Optimization method.
    Returns: Tuple (k, lambda) representing Inverse Weibull distribution parameters.
    """
    k = optimize.minimize(invweibull_func, np.array(2.5), args=(mu, vr), method=meth,
                          bounds=[(0, np.inf)]).x
    lamb = gamma(1 - 1 / k) / mu
    return k, lamb


def helper_func_invweibull_one(x):
    """
    Helper function used in the calculation of Inverse Weibull parameters for 'greater_4_5_inv_weibull'.
    """
    return gamma(1 - 1/x) * 1/np.power(gamma(1 - 2/x) - gamma(1 - 1/x)**2, 1/2)


def greater_4_5_inv_weibull(mu, std):
    """
    Calculate Inverse Weibull distribution parameters (k and lambda) for cases where mean/std ratio is greater than 4.5.
    mu (array-like): Mean values.
    std (array-like): Standard deviation values.
    Returns: Tuple (k, lambda) representing Inverse Weibull distribution parameters.
    """
    y = mu/std
    c = (100 * helper_func_invweibull_one(7) - 7 * helper_func_invweibull_one(100))/93
    k = 93 * (y - c) / (helper_func_invweibull_one(100) - helper_func_invweibull_one(7))
    lamb = gamma(1 - 1 / k) / mu
    if np.any(abs(mu - lamb ** (-1) * gamma(1 - 1 / k) > (0.01 * mu))):
        print('problem for y >= 4.5 inverse weibull')
        k = np.zeros_like(mu)
        lamb = np.zeros_like(mu)
        for run in np.arange(len(mu)):
            k[run], lamb[run] = run_for_invweibull(mu[run], std[run])
    return k, lamb


def helper_func_invweibull_two(x):
    """
    Helper function used in the calculation of Weibull parameters for 'smaller_one_eighth_weibull'.
    """
    return gamma(1 - 1/x)**2 / gamma(1 - 2/x)


def smaller_one_eighth_invweibull(mu, std):
    """
    Calculate inverse Weibull distribution parameters (k and lambda) for cases where mean/std ratio is less than or equal to 1/8.
    mean (array-like): Mean values.
    std (array-like): Standard deviation values.
    Returns: Tuple (k, lambda) representing inverse Weibull distribution parameters.
    """
    y = mu/std
    m = helper_func_invweibull_two(2.01) / 0.01
    k = (1 / (1 + 1/y**2) + 2 * m)/m
    lamb = gamma(1 - 1 / k) / mu
    if np.any(abs(mu - lamb ** (-1) * gamma(1 - 1 / k) > (0.01 * mu))):
        print('problem for y <= 1/8 inverse weibull')
        k = np.zeros_like(mu)
        lamb = np.zeros_like(mu)
        for run in np.arange(len(mu)):
            k[run], lamb[run] = run_for_invweibull(mu[run], std[run])
    return k, lamb


