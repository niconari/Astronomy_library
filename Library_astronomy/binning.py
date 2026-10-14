from __future__ import print_function, division
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
import pdb
from PyAstronomy.pyasl import binningx0dt
import gls
from matplotlib import pyplot as plt
from matplotlib import ticker as tk
from matplotlib import axes as ax



def data_preparation_bin(rv,sigma_rv,time,sigma_clip=3):
    rv = np.array(rv)
    sigma_rv = np.array(sigma_rv)
    time = np.array(time)

    rv_ok = np.array(rv)
    sigma_rv_ok = np.array(sigma_rv)
    time_ok = np.array(time)

    lun = len(time_ok)

    time_int = []
    for i in range(lun):
        time_int.append(int(time_ok[i]))

    time_int = np.array(time_int)
    t = np.unique(time_int)
    t = np.append(t,np.amax(t) + 3.0)
    rv_corr, bin_edge,bin_numb = stats.binned_statistic(time_ok,rv_ok,'mean',t)
    time_corr, bin_edge,bin_numb = stats.binned_statistic(time_ok,time_ok,'mean',t)
    x0 = min(time_ok)//1
    time_ok = time_ok - x0
    dr,dt = binningx0dt(time_ok, rv_ok, sigma_rv_ok,x0=0.0,dt=1.0,useMeanX=True,removeEmpty=True)
    time_ok = dr[::,0]
    rv_ok = dr[::,1]
    sigma_rv_ok = dr[::,2]

    time_ok += x0
    time_ok = time_ok - time_corr
    print(time_ok)

    ind_ok = np.where(np.abs(rv_corr-np.median(rv_corr))<sigma_clip*np.std(rv_corr))[0]
    #ind_ok = np.where(np.abs(rv-np.median(rv[ind_ok]))<3*np.std(rv[ind_ok]))[0]
    rv_corr = np.array(rv_corr[ind_ok])
    sigma_rv_ok = np.array(sigma_rv_ok[ind_ok])
    time_corr = np.array(time_corr[ind_ok])

    ind_ok = np.where(sigma_rv_ok < 3*np.median(sigma_rv_ok))[0]
    #ind_ok = np.where(np.abs(rv-np.median(rv[ind_ok]))<3*np.std(rv[ind_ok]))[0]
    rv_corr = np.array(rv_corr[ind_ok])
    sigma_rv_ok = np.array(sigma_rv_ok[ind_ok])
    time_corr = np.array(time_corr[ind_ok])

    return rv_corr,sigma_rv_ok,time_corr

def harps_split(rv,sigma_rv,time):
    rv1 = []
    rv2 = []
    sigma_rv1 = []
    sigma_rv2 = []
    time_1 = []
    time_2 = []
    
    for i in range(len(time)):
        if time[i] < 2457176 :
                rv1.append(float(rv[i]))
                sigma_rv1.append(float(sigma_rv[i]))
                time_1.append(float(time[i]))
        else:
            rv2.append(float(rv[i]))
            sigma_rv2.append(float(sigma_rv[i]))
            time_2.append(float(time[i]))

    
    return rv1,sigma_rv1,time_1,rv2,sigma_rv2,time_2