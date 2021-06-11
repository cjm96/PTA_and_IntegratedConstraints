from os.path import join, dirname

from scipy import constants as scon
from scipy import interpolate
import numpy as np

from kinematics import *
from cosmo import *

# bubble spectrum
def h2_omega(f, log10_T, log10_H_on_beta, log10_alpha, log10_eta, contr='bubble', mod='Semi-analytic'):

    [T, H_on_beta, alpha, eta] = 10**np.array([log10_T, log10_H_on_beta, log10_alpha, log10_eta])

    vw = v_w(eta, alpha)
    gs = g_star(log10_T)

    ############
    ## bubble ##
    ############
    if contr == 'bubble':
        if mod=='Envelope':
            a = 3.
            b = 0.94
            c = 1.5

        elif mod=='Semi-analytic':
            a = 1.
            b = 2.61
            c = 1.5

        else:
            a = 0.7
            b = 2.3
            c = 1


        # velocity factor
        delta = 0.48 * vw**3 / (1 + 5.3 * vw**2 + 5 * vw**4)

        # efficiency factor
        k = k_phi(eta, alpha)

        [p, q] = [2 ,2]

        # spectral shape
        def S(x):
            return (a + b)**c / (b * x**(-a/c) + a * x**(b/c))**c

        # peak frequency at emission
        f_on_beta = 0.35/(1+0.07 * vw + 0.69 * vw**2)

    ############
    ## sound  ##
    ############
    if contr == 'sound':

        delta = 5.13 * 10**-1 * vw

        k = k_sw(eta, alpha)

        [p, q] = [2 ,1]

        def S(x):
            return x**3 * (7/(4 + 3 * x**2))**(7/2)

        f_on_beta = 5.36 * 10**-1 / vw

    ############
    ##  turb  ##
    ############
    if contr == 'turb':

        delta = 2.02 * 10 * vw

        k = k_turb(eta, alpha)

        [p, q] = [3/2 ,1]

        f_on_beta = 1.63 / vw

        def S(x):
            H_tilde = 16.5 * 10**-8 * T * (gs/100)**(1/6)
            return x**3 / ((1 + x)**(11/3) * (1 + 8 * np.pi * x * f0/H_tilde))

    # dilution coeff.
    R = 7.69 * 10**-5 * gs**(-1/3)

    # peak frequency today in Hz
    f0 = 1.13 * 10**-7 * f_on_beta * H_on_beta**-1 * T * (gs/10)**(1/6)

    return R * delta * (k * alpha / (1 + alpha))**p * H_on_beta**q * S(f/f0)


# bubble spectrum
def h2_omega_sum(f, log10_T, log10_H_on_beta, log10_alpha, log10_eta, mod='Semi-analytic'):
    ans = h2_omega(f, log10_T, log10_H_on_beta, log10_alpha, log10_eta, contr='bubble', mod=mod)
    ans += h2_omega(f, log10_T, log10_H_on_beta, log10_alpha, log10_eta, contr='sound', mod=mod)
    ans += h2_omega(f, log10_T, log10_H_on_beta, log10_alpha, log10_eta, contr='turb', mod=mod)
    return ans
