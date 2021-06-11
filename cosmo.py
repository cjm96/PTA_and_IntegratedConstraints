import numpy as np 
from os.path import join, dirname

data_folder = join(dirname(__file__),'./data/')

h = 0.674

# load the g_star grid
log10_T_table = np.log10(np.loadtxt(data_folder + 'g_star.txt')[:,0] / (1E3))
log10_g_star_table = np.log10(np.loadtxt(data_folder + 'g_star.txt')[:,1])

def g_star(log10_T):
    return 10**np.interp(log10_T, log10_T_table, log10_g_star_table)


g_eq = g_star(-7)