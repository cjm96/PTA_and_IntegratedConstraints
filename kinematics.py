import numpy as np
from numpy import genfromtxt
from scipy import interpolate
from os.path import join, dirname


data_folder = join(dirname(__file__),'./data/')

# load the alpha-eta grids
alpha_def_tab = genfromtxt(data_folder + 'def_alpha_tab.csv', delimiter=',')
alpha_det_tab = genfromtxt(data_folder + 'det_alpha_tab.csv', delimiter=',')
eta_def_tab = genfromtxt(data_folder + 'def_eta_tab.csv', delimiter=',')
eta_det_tab = genfromtxt(data_folder + 'det_eta_tab.csv', delimiter=',')

# load the xi tables
xi_def_tab = genfromtxt(data_folder + 'def_xi_tab.csv', delimiter=',')
xi_det_tab = genfromtxt(data_folder + 'det_xi_tab.csv', delimiter=',')

# compute the interpolated xi
xi_def = interpolate.interp2d(eta_def_tab, alpha_def_tab, xi_def_tab, kind='cubic', bounds_error=False)
xi_det = interpolate.interp2d(eta_det_tab, alpha_det_tab, xi_det_tab, kind='cubic', bounds_error=False)

# load the boundary in the alpha plane between detonations and deflagrations
det2def_tab = genfromtxt(data_folder + 'det2def_trans.csv', delimiter=',')
det2def = interpolate.interp1d(det2def_tab[::,0], det2def_tab[::,1], kind='cubic', bounds_error=False)

# for alpha>alpha_inf the PT is in the runaway regime
def alpha_inf(eta):
    return eta + (1/3) * (1 - 0.85)

# define a bubble wall function
def v_w(eta, alpha):
    if alpha > alpha_inf(eta):
        return 1.
    elif alpha < det2def(eta):
        return np.minimum(xi_def(eta, alpha)[0], 1.)
    else:
        return np.minimum(xi_det(eta, alpha)[0], 1.)


# define the bubble efficiency
def k_phi(eta, alpha):
    if alpha < alpha_inf(eta):
        return 0.
    else:
        return 1 - alpha_inf(eta) / alpha

# define the sound-wave efficiency
def k_sw(eta, alpha):
    if alpha < alpha_inf(eta):

        v = np.abs(v_w(eta, alpha))

        kappa_A = (v*(6/5)) * (6.9*alpha) / (1.36 - 0.037*np.sqrt(alpha) + alpha)
        kappa_B = (alpha**(2/5)) / (0.017 + (0.997+alpha)**(2/5))
        kappa_C = np.sqrt(alpha) / (0.135 + np.sqrt(0.98 + alpha))
        kappa_D = alpha / (0.73 + 0.083*np.sqrt(alpha) + alpha)
        delta_kappa = - 0.9 * np.log(np.sqrt(alpha) / (1+np.sqrt(alpha)))
        xi_J = (np.sqrt(2*alpha/3 + alpha**2) + np.sqrt(1/3)) / (1 + alpha)
        c_s = 1 / np.sqrt(3)

        if v < c_s:
            return (c_s**(11/5) * kappa_A * kappa_B) / ((c_s**(11/5) - v**(11/5))*kappa_B + v*(c_s**(6/5))*kappa_A)

        elif v > xi_J:
            numerator = ((xi_J-1)**3) * (xi_J**(5/2)) * (v**(-5/2)) * kappa_C * kappa_D
            denominator = ((xi_J-1)**3 - (v-1)**3) * (xi_J**(5/2)) * kappa_C + ((v-1)**3) * kappa_D

            return numerator / denominator

        else:
            return kappa_B + (v - c_s)*delta_kappa + (((v-c_s)/(xi_J-c_s))**3) * (kappa_C - kappa_B - (xi_J-c_s)*delta_kappa)

    else:
        kappa_inf = alpha_inf(eta) / (0.73 + 0.083*np.sqrt(alpha_inf(eta)) + alpha_inf(eta))

        return kappa_inf * alpha_inf(eta) / alpha

# define the turbulence efficiency
def k_turb(eta, alpha):
    return 0.1 * k_sw(eta, alpha)

# check if the PT is in the runaway regime

def runaway_Q(log10_alpha, log10_eta):

    alpha = 10 ** log10_alpha
    eta = 10 ** log10_eta

    # alpha infinity
    alpha_inf = eta + (1/3) * (1-0.85)

    # the phase transition is runaway if alpha > alpha_inf
    if alpha > alpha_inf:
        return True

    else:
        return False
