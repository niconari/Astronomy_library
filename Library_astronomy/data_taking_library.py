from __future__ import print_function, division
import numpy as np
import matplotlib.pylab as plt
from scipy import stats
import scipy.stats as scst
import pdb
from PyAstronomy.pyasl import binningx0dt
import gls
from matplotlib import pyplot as plt
from matplotlib import ticker as tk
from matplotlib import axes as ax
from exoplanet_rv_function import *
from astropy.utils.data import get_pkg_data_filename
from astropy.utils.data import download_file
from astropy.io import fits
from scipy.integrate import simpson
from pathlib import Path
from numpy import trapz
import sys
sys.path.append('/home/nicola/Scrivania/Library_python3/')
#from emcee_modules_kp import *
from data_taking_library import *
from harps_tools import *
from extrac_files_re import *
import pdb
from PyAstronomy import pyasl
from pathlib import Path
import shutil
from datetime import datetime, timedelta
from scipy.stats import uniform, norm


''' DATA PREPARATION '''

# data preparation for rv, time -> data_preparation


def binning(y,x):
    xb = np.trunc(x)
    xu = np.unique(xb)
    x2 = np.zeros(len(xu))
    y2 = np.zeros(len(xu))
    for i in range(len(xu)):
        ind = np.where((x > xu[i]) & (x < xu[i] + 1))[0]
        x1 = x[ind]
        y1 = y[ind]
        x2[i] = np.mean(x1)
        y2[i] = np.mean(y1)
    return y2,x2

def panda_clip(y,sigma = 3.0):
    a,d,u = scst.sigmaclip(y,low=sigma,high=sigma)
    sigma1 = d < a 
    sigma2 = u > a
    cond = sigma1 & sigma2
    return cond

def data_preparation_bin(rv,sigma_rv,time,sigma_clip=3,bin_step=1.0):
    rv = np.array(rv)
    sigma_rv = np.array(sigma_rv)
    time = np.array(time)

    rv_ok = np.array(rv)
    sigma_rv_ok = np.array(sigma_rv)
    time_ok = np.array(time)
    x0 = min(time_ok)//1
    time_ok = time_ok - x0
    dr,dt = binningx0dt(time_ok, rv_ok, sigma_rv_ok,x0=0.0,dt=bin_step,useMeanX=True,removeEmpty=True)
    time_ok = dr[::,0]
    rv_ok = dr[::,1]
    sigma_rv_ok = dr[::,2]

    time_ok += x0

    if sigma_clip==7:
        return rv_ok,sigma_rv_ok,time_ok
    ind_ok = np.where(np.abs(rv_ok-np.median(rv_ok))<sigma_clip*np.std(rv_ok))[0]
    rv_ok = np.array(rv_ok[ind_ok])
    sigma_rv_ok = np.array(sigma_rv_ok[ind_ok])
    time_ok = np.array(time_ok[ind_ok])

    ind_ok = np.where(sigma_rv_ok < 3*np.median(sigma_rv_ok))[0]
    rv_ok = np.array(rv_ok[ind_ok])
    sigma_rv_ok = np.array(sigma_rv_ok[ind_ok])
    time_ok = np.array(time_ok[ind_ok])

    return rv_ok,sigma_rv_ok,time_ok

def data_preparation_bin_std(rv,sigma_rv,time,sigma_clip=3):
    rv = np.array(rv)
    sigma_rv = np.array(sigma_rv)
    time = np.array(time)

    rv_ok = np.array(rv)
    sigma_rv_ok = np.array(sigma_rv)
    time_int = time//1
    t = np.unique(time_int)
    rv_ok = []
    sigma_rv_ok = []
    time_ok = []
    for i in range(len(t)):
        ind = np.where((time > t[i]) & (time < t[i] + 1))[0]
        rv_ind = np.mean(rv[ind])
        sigma_rv_ind = np.std(rv[ind])
        time_ok_ind = np.mean(time[ind])
        rv_ok.append(rv_ind)
        sigma_rv_ok.append(sigma_rv_ind)
        time_ok.append(time_ok_ind)

    rv_ok = np.array(rv_ok)
    sigma_rv_ok = np.array(sigma_rv_ok)
    time_ok = np.array(time_ok)

    if sigma_clip==7:
        return rv_ok,sigma_rv_ok,time_ok
    ind_ok = np.where(np.abs(rv_ok-np.median(rv_ok))<sigma_clip*np.std(rv_ok))[0]
    rv_ok = np.array(rv_ok[ind_ok])
    sigma_rv_ok = np.array(sigma_rv_ok[ind_ok])
    time_ok = np.array(time_ok[ind_ok])

    ind_ok = np.where(sigma_rv_ok < 3*np.median(sigma_rv_ok))[0]
    rv_ok = np.array(rv_ok[ind_ok])
    sigma_rv_ok = np.array(sigma_rv_ok[ind_ok])
    time_ok = np.array(time_ok[ind_ok])

    rv_ok = np.array(rv_ok)
    sigma_rv_ok = np.array(sigma_rv_ok)
    time_ok = np.array(time_ok)

    return rv_ok,sigma_rv_ok,time_ok

def common_index(*arrays):
    # Calcola il np.floor per ogni array e salva gli indici originali
    floored_arrays = [np.floor(arr) for arr in arrays]
    
    # Trova i valori di np.floor comuni a tutti gli array
    common_floors = set(floored_arrays[0])
    for floored_array in floored_arrays[1:]:
        common_floors.intersection_update(floored_array)
    
    # Trova gli indici degli elementi che hanno i valori di np.floor comuni
    common_indices = []
    for floored_array in floored_arrays:
        common_indices.append([idx for idx, value in enumerate(floored_array) if value in common_floors])
    
    return common_indices


def data_bin(rv,sigma_rv,time,sigma_clip=3):
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
    sigma_rv_bin = np.zeros(len(t))
    rv_bin = np.zeros(len(t))
    time_bin = np.zeros(len(t))
    t = np.append(t,np.amax(t) + 3.0)
    rv_corr, bin_edge,bin_numb = stats.binned_statistic(time_ok,rv_ok,'mean',t)
    time_corr, bin_edge,bin_numb = stats.binned_statistic(time_ok,time_ok,'mean',t)
    #count_corr, bin_edge,bin_numb = stats.binned_statistic(time_ok,time_ok,'count',t)
    rv_err = rv_ok.copy()
    time_err = time_ok.copy()
    sigma_err = sigma_rv_ok.copy()
    x0 = min(time_ok)//1
    time_ok = time_ok - x0
    dr,dt = binningx0dt(time_ok, rv_ok, sigma_rv_ok,x0=0.0,dt=1.0,removeEmpty=True)
    time_ok = dr[::,0]
    rv_ok = dr[::,1]
    sigma_rv_ok = dr[::,2]

    time_ok += x0

    if sigma_clip ==7:
        return rv_corr,sigma_rv_ok,time_corr 

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


def index_ok(x, y, yerr, sigma_clip=5,err_clip=3):
    mask1 = np.abs(y - np.mean(y)) < sigma_clip * np.std(y)  # Select points within sigma_clip from the mean
    mask2 = yerr < err_clip * np.mean(yerr)  # Filter values with acceptable error
    ind_ok = np.where(mask1 & mask2)[0]  # Apply both conditions
    return ind_ok

            
    

''' READINGA DATA '''

#data reading for rv, time, sigma_rv -> reading
def reading(filename):
    f = open(filename,'r')
    rv = []
    sigma_rv = []
    time = []
    for line in f:
        if '#' not in line:
            rv.append(float(line.split()[1]))
            sigma_rv.append(float(line.split()[2]))
            time.append(float(line.split()[0]))
            
    return rv,sigma_rv,time

#data reading for rv, csx, time, dlw data -> reading_activity

# def samples(FILEPATH,ndim,mode=0):
#     if mode == 0:
#         samples = []

#         chains_name = FILEPATH+'SAMPLES.dat'
#         f = open(chains_name,'r')
#         for line in f:
#             if '#' not in line:
#                 samples.append([float(line.split()[i]) for i in range(ndim)])
#         # endif
#         # endfor
#         f.close()
#         samples = np.array(samples)
#     if mode == 1:
#         samples = []

#         chains_name = FILEPATH+'_SAMPLES_0.dat'
#         f = open(chains_name,'r')
#         for line in f:
#             if '#' not in line:
#                 samples.append([float(line.split()[i]) for i in range(ndim)])
#         # endif
#         # endfor
#         f.close()
#         samples = np.array(samples)
    
#     return(samples)

def samples2(FILEPATH,ndim,mode=0):
    if mode == 0:
        samples = []

        chains_name = FILEPATH+'SAMPLES.dat'
        f = open(chains_name,'r')
        for line in f:
            if '#' not in line:
                samples.append([float(line.split()[i]) for i in range(ndim)])
        # endif
        # endfor
        f.close()
        samples = np.array(samples)
    if mode == 1:
        samples = []

        chains_name = FILEPATH+'_SAMPLES.dat'
        f = open(chains_name,'r')
        for line in f:
            if '#' not in line:
                samples.append([float(line.split()[i]) for i in range(ndim)])
        # endif
        # endfor
        f.close()
        samples = np.array(samples)
    
    return(samples)

def periodogram(FILEPATH,x,y,yerr,inst,residuals=0,show=0,ar=[16,9],font=20,legend_font=20,ts_type='',min_time=2.0,max_time =None, jit=1,zoom=0,xlim=[0,1]):
    
    axd = plt.figure(constrained_layout=True,figsize=(ar[0],ar[1])).subplot_mosaic(
        """
        A
        """
        )
    y_jit = yerr.copy()
    if len(inst) > 1:
        for j in range(len(inst)):
            y_jit[inst[j]] = np.std(y[inst[j]])
    else:
        y_jit = np.std(y)


    if jit==1:
        y_err = np.sqrt(yerr**2 + y_jit**2)
    else:
        y_err=np.sqrt(yerr**2)
    print(np.mean(y_err))
    if max_time == None:
        max_time = np.amax(x) - np.amin(x)
    fbeg0 = 1.0/max_time
    fend0 = 1.0/min_time


    periodogram = gls.Gls((x[:],y[:],y_err[:]),fbeg=fbeg0,fend=fend0,verbose=True)
    pbest = 1/periodogram.hpstat["fbest"]
    maxPower = periodogram.pmax
    peakFAP = periodogram.FAP(maxPower)
    print('Maximum peak FAP: {0}'.format(peakFAP))
    
    ax = axd["A"]
    ax.set_xlabel(r'Period (d)', fontsize=font, color='black')
    ax.set_ylabel(r'PSD', fontsize=font, color='black')
    #ax.xaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
    #ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
    #ax.yaxis.set_minor_locator(tk.MultipleLocator(maxPower/6.0))
    ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.2f"))
    ax.yaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='in',)
    ax.xaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='out')
    ax.yaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='in')
    ax.xaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='out')
    ax.plot(1/periodogram.freq,periodogram.power,'k-',label=f'Pmax = {pbest:.2f}')
    ax.yaxis.set_tick_params(labelsize=font)
    ax.xaxis.set_tick_params(labelsize=font)
    ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.001),'b:',linewidth=4.0,label='FAP = 0.1 %')
    ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.01),'g-.',linewidth=2.0,label='FAP = 1 %')
    ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.1),'r--',linewidth=1.0,label='FAP = 10 %')
    #plt.plot([15.86,15.86],[0,0.65],'b--')   # Per evidenziare periodi significativi nel periodogramma
    #plt.plot([90.,90.],[0,0.65],'r--')   # Per evidenziare periodi significativi nel periodogramma
    ax.axis([1/fend0,np.amax(x)-np.amin(x),0,max(maxPower,periodogram.powerLevel(0.001))+0.02])
    ax.set_xscale('log')
    if min_time > 1:
        ax.set_xticks([10.0,100.0,1000.0],["10","100","1000"])
    else: 
        ax.set_xticks([1.0,10.0,100.0,1000.0],["1","10","100","1000"])
    #ax.set_title('FWHM periodogram')
    ax.set_ylabel('PSD')
    if zoom==1:
        ax.set_xlim(xlim[0],xlim[1])
    #ax.set_ylim(0.0, 0.10)
    ax.legend(fontsize=font,markerscale=1.0)
    if len(x)==2:
        axd = plt.figure(constrained_layout=True,figsize=(ar[0],ar[1])).subplot_mosaic(
            """
            A
            B
            """
            )
        for i in range(len(x)):
            y_jit = yerr[i].copy()
            if len(inst) > 1:
                for j in range(len(inst)):
                    y_jit[inst[j]] = np.std(y[i][inst[j]])

            else:
                y_jit = np.std(y[i])
            
            fbeg0 = 1.0/(np.amax(x[i])-np.amin(x[i]))
            fend0 = 1.0/2.0


            periodogram = gls.Gls((x[i][:],y[i][:],y_err[:]),fbeg=fbeg0,fend=fend0,verbose=True)
            pbest = 1/periodogram.hpstat["fbest"]
            maxPower = periodogram.pmax
            peakFAP = periodogram.FAP(maxPower)
            print('Maximum peak FAP: {0}'.format(peakFAP))
            if i==0:
                ax = axd["A"]
                ax.set_xlabel(r'Period (d)', fontsize=font, color='black')
                ax.set_ylabel(r'PSD', fontsize=font, color='black')
                #ax.xaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
                ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.2f"))
                ax.yaxis.set_minor_locator(tk.MultipleLocator(maxPower/6.0))
                ax.yaxis.set_major_locator(tk.MultipleLocator(maxPower/3.0))
                ax.yaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='in',)
                ax.xaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='out')
                ax.yaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='in')
                ax.xaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='out')
                ax.plot(1/periodogram.freq,periodogram.power,'k-',label=f'Pmax = {pbest:.2f}')
                ax.yaxis.set_tick_params(labelsize=font)
                ax.xaxis.set_tick_params(labelsize=font)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.001),'b:',linewidth=4.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.01),'g-.',linewidth=2.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.1),'r--',linewidth=1.0)
                #plt.plot([15.86,15.86],[0,0.65],'b--')   # Per evidenziare periodi significativi nel periodogramma
                #plt.plot([90.,90.],[0,0.65],'r--')   # Per evidenziare periodi significativi nel periodogramma
                ax.axis([1/fend0,np.amax(x[i])-np.amin(x[i]),0,max(maxPower,periodogram.powerLevel(0.001))+0.02])
                ax.set_xscale('log')
                ax.set_xticks([10.0,100.0,1000.0],["10","100","1000"])
                #ax.set_title('FWHM periodogram')
                ax.set_ylabel('PSD')
                #ax.set_ylim(0.0, 0.10)
                ax.legend(fontsize=font,markerscale=1.0)
            if i==1:
                ax = axd["B"]
                ax.set_xlabel(r'Period (d)', fontsize=font, color='black')
                ax.set_ylabel(r'PSD', fontsize=font, color='black')
                #ax.xaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
                ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.2f"))
                ax.yaxis.set_minor_locator(tk.MultipleLocator(maxPower/6.0))
                ax.yaxis.set_major_locator(tk.MultipleLocator(maxPower/3.0))
                ax.yaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='in',)
                ax.xaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='out')
                ax.yaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='in')
                ax.xaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='out')
                ax.plot(1/periodogram.freq,periodogram.power,'k-',label=f'Pmax = {pbest:.2f}')
                ax.yaxis.set_tick_params(labelsize=font)
                ax.xaxis.set_tick_params(labelsize=font)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.001),'b:',linewidth=4.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.01),'g-.',linewidth=2.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.1),'r--',linewidth=1.0)
                #plt.plot([15.86,15.86],[0,0.65],'b--')   # Per evidenziare periodi significativi nel periodogramma
                #plt.plot([90.,90.],[0,0.65],'r--')   # Per evidenziare periodi significativi nel periodogramma
                ax.axis([1/fend0,np.amax(x[i])-np.amin(x[i]),0,max(maxPower,periodogram.powerLevel(0.001))+0.02])
                ax.set_xscale('log')
                ax.set_xticks([10.0,100.0,1000.0],["10","100","1000"])
                #ax.set_title('FWHM periodogram')
                ax.set_ylabel('PSD')
                #ax.set_ylim(0.0, 0.10)
                ax.legend(fontsize=font,markerscale=1.0)
    if len(x)==3:
        axd = plt.figure(constrained_layout=True,figsize=(ar[0],ar[1])).subplot_mosaic(
            """
            A
            B
            C
            """
            )
        for i in range(len(x)):
            y_jit = yerr[i].copy()
            if len(inst) > 1:
                for j in range(len(inst)):
                    y_jit[inst[j]] = np.std(y[i][inst[j]])

            else:
                y_jit = np.std(y[i])
            y_jit = yerr[i].copy()
            
            fbeg0 = 1.0/(np.amax(x[i])-np.amin(x[i]))
            fend0 = 1.0/2.0


            periodogram = gls.Gls((x[i][:],y[i][:],y_err[i][:]),fbeg=fbeg0,fend=fend0,verbose=True)
            pbest = 1/periodogram.hpstat["fbest"]
            maxPower = periodogram.pmax
            peakFAP = periodogram.FAP(maxPower)
            print('Maximum peak FAP: {0}'.format(peakFAP))
            if i==0:
                ax = axd["A"]
                ax.set_xlabel(r'Period (d)', fontsize=font, color='black')
                ax.set_ylabel(r'PSD', fontsize=font, color='black')
                #ax.xaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
                ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.2f"))
                ax.yaxis.set_minor_locator(tk.MultipleLocator(maxPower/6.0))
                ax.yaxis.set_major_locator(tk.MultipleLocator(maxPower/3.0))
                ax.yaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='in',)
                ax.xaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='out')
                ax.yaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='in')
                ax.xaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='out')
                ax.plot(1/periodogram.freq,periodogram.power,'k-',label=f'Pmax = {pbest:.2f}')
                ax.yaxis.set_tick_params(labelsize=font)
                ax.xaxis.set_tick_params(labelsize=font)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.001),'b:',linewidth=4.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.01),'g-.',linewidth=2.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.1),'r--',linewidth=1.0)
                #plt.plot([15.86,15.86],[0,0.65],'b--')   # Per evidenziare periodi significativi nel periodogramma
                #plt.plot([90.,90.],[0,0.65],'r--')   # Per evidenziare periodi significativi nel periodogramma
                ax.axis([1/fend0,np.amax(x)-np.amin(x),0,max(maxPower,periodogram.powerLevel(0.001))+0.02])
                ax.set_xscale('log')
                ax.set_xticks([10.0,100.0,1000.0],["10","100","1000"])
                #ax.set_title('FWHM periodogram')
                ax.set_ylabel('PSD')
                #ax.set_ylim(0.0, 0.10)
                ax.legend(fontsize=font,markerscale=1.0)
            if i==1:
                ax = axd["B"]
                ax.set_xlabel(r'Period (d)', fontsize=font, color='black')
                ax.set_ylabel(r'PSD', fontsize=font, color='black')
                #ax.xaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
                ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.2f"))
                ax.yaxis.set_minor_locator(tk.MultipleLocator(maxPower/6.0))
                ax.yaxis.set_major_locator(tk.MultipleLocator(maxPower/3.0))
                ax.yaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='in',)
                ax.xaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='out')
                ax.yaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='in')
                ax.xaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='out')
                ax.plot(1/periodogram.freq,periodogram.power,'k-',label=f'Pmax = {pbest:.2f}')
                ax.yaxis.set_tick_params(labelsize=font)
                ax.xaxis.set_tick_params(labelsize=font)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.001),'b:',linewidth=4.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.01),'g-.',linewidth=2.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.1),'r--',linewidth=1.0)
                #plt.plot([15.86,15.86],[0,0.65],'b--')   # Per evidenziare periodi significativi nel periodogramma
                #plt.plot([90.,90.],[0,0.65],'r--')   # Per evidenziare periodi significativi nel periodogramma
                ax.axis([1/fend0,np.amax(x)-np.amin(x),0,max(maxPower,periodogram.powerLevel(0.001))+0.02])
                ax.set_xscale('log')
                ax.set_xticks([10.0,100.0,1000.0],["10","100","1000"])
                #ax.set_title('FWHM periodogram')
                ax.set_ylabel('PSD')
                #ax.set_ylim(0.0, 0.10)
                ax.legend(fontsize=font,markerscale=1.0)
            if i==2:
                ax = axd["C"]
                ax.set_xlabel(r'Period (d)', fontsize=font, color='black')
                ax.set_ylabel(r'PSD', fontsize=font, color='black')
                #ax.xaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
                ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.2f"))
                ax.yaxis.set_minor_locator(tk.MultipleLocator(maxPower/6.0))
                ax.yaxis.set_major_locator(tk.MultipleLocator(maxPower/3.0))
                ax.yaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='in',)
                ax.xaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='out')
                ax.yaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='in')
                ax.xaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='out')
                ax.plot(1/periodogram.freq,periodogram.power,'k-',label=f'Pmax = {pbest:.2f}')
                ax.yaxis.set_tick_params(labelsize=font)
                ax.xaxis.set_tick_params(labelsize=font)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.001),'b:',linewidth=4.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.01),'g-.',linewidth=2.0)
                ax.plot([np.amin(1/periodogram.freq),np.amax(1/periodogram.freq)],np.ones(2)*periodogram.powerLevel(0.1),'r--',linewidth=1.0)
                #plt.plot([15.86,15.86],[0,0.65],'b--')   # Per evidenziare periodi significativi nel periodogramma
                #plt.plot([90.,90.],[0,0.65],'r--')   # Per evidenziare periodi significativi nel periodogramma
                ax.axis([1/fend0,np.amax(x)-np.amin(x),0,max(maxPower,periodogram.powerLevel(0.001))+0.02])
                ax.set_xscale('log')
                ax.set_xticks([10.0,100.0,1000.0],["10","100","1000"])
                #ax.set_title('FWHM periodogram')
                ax.set_ylabel('PSD')
                #ax.set_ylim(0.0, 0.10)
                ax.legend(fontsize=font,markerscale=1.0)
    if residuals==0:
        plt.savefig(FILEPATH+f'_data_periodogram{ts_type}.pdf', bbox_inches='tight')
        plt.savefig(FILEPATH+f'_data_periodogram{ts_type}.png', bbox_inches='tight')
        if show==1:
            plt.show()
        plt.close()
    if residuals==1:
        plt.savefig(FILEPATH+f'_residuals_periodogram{ts_type}.pdf', bbox_inches='tight')
        plt.savefig(FILEPATH+f'_residuals_periodogram{ts_type}.png', bbox_inches='tight')
        if show==1:
            plt.show()
        plt.close()
    return 

def plot_model(x,y,yerr,x_plot,model,x_name,y_name,
               timeseries,inst,x_locator,y_locator,FILEPATH,
               show=0,ar=[12,9],colore='None',font=20,zoom=False,mark=0,
               jit=False,jit_value = [0,1],xlim=[0,1],legend_font=20,ts_type=''):

    if len(y_locator)!=2:
        print('Error, y locator lenghts must be 2')
        return
    if len(x_locator)!=2:
        print('Error, x locator lenghts must be 2')
        return
    if jit != False:
        yerr_jit = np.sqrt(yerr**2 + jit_value**2)
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
              "tab:pink","tab:gray","tab:olive","tab:cyan",]
    if mark==0:
        marker=8
    else:
        marker=mark
    axd = plt.figure(constrained_layout=True,figsize=(ar[0],ar[1])).subplot_mosaic(
    """
    A
    """
    )

    ax = axd["A"]
    ax.set_xlabel(x_name, fontsize=font, color='black')
    ax.set_ylabel(y_name, fontsize=font, color='black')
    #ax.xaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
    #ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.2f"))
    ax.yaxis.set_minor_locator(tk.MultipleLocator(y_locator[0]))
    ax.yaxis.set_major_locator(tk.MultipleLocator(y_locator[1]))
    ax.xaxis.set_minor_locator(tk.MultipleLocator(x_locator[0]))
    ax.xaxis.set_major_locator(tk.MultipleLocator(x_locator[1]))
    if zoom==True:
        ax.set_xlim(xlim[0],xlim[1])
    ax.yaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='in',)
    ax.xaxis.set_tick_params(labelsize=font,width=2.5, size=8,direction='in')
    ax.yaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='in')
    ax.xaxis.set_tick_params(labelsize=font,which='minor',width=1.0, size=4,direction='in')
    ax.yaxis.set_tick_params(labelsize=font)
    ax.xaxis.set_tick_params(labelsize=font)
    if jit == False:
        if len(timeseries) == 0:
            ax.errorbar(x,y,yerr,fmt='o',markersize=marker,label=inst)
        elif colore != 'None':
            for i in range(len(timeseries)):
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr[timeseries[i]],
                            fmt='o', 
                            markersize=marker,
                            color = colore,
                            label=inst[i])
        else:
            for i in range(len(timeseries)):
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr[timeseries[i]],
                            fmt='o', 
                            markersize=marker,
                            label=inst[i])
        ax.errorbar(x_plot,model,color='red',fmt='-',label='Model')
        ax.legend(fontsize=font,markerscale=1.0)
    else:
        if len(timeseries) == 0:
            ax.errorbar(x,y,yerr,elinewidth=2,fmt='o',markersize=marker,color=colors[0],label=inst)
            ax.errorbar(x,y,yerr_jit,fmt='o',elinewidth=1,alpha=0.7,markersize=marker,color=colors[0])
        elif colore != 'None':
            ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr[timeseries[i]],
                            elinewidth=2,
                            fmt='o', 
                            markersize=marker,
                            color = colore,
                            label=inst[i])
            ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr_jit[timeseries[i]],elinewidth=1,alpha=0.7,
                            fmt='o', 
                            markersize=marker,
                            color = colore)
        else:
            for i in range(len(timeseries)):
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr[timeseries[i]],
                            elinewidth=2,
                            fmt='o', 
                            markersize=marker,
                            color = colors[i],
                            label=inst[i])
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr_jit[timeseries[i]],elinewidth=1,alpha=0.7,
                            fmt='o', 
                            markersize=marker,
                            color = colors[i])
        ax.errorbar(x_plot,model,color='red',fmt='-',label='Model')
        ax.legend(fontsize=font,markerscale=1.0)
    if zoom==True:
        plt.savefig(FILEPATH+f'_model{ts_type}_zoom.pdf', bbox_inches='tight')
        plt.savefig(FILEPATH+f'_model{ts_type}_zoom.png', bbox_inches='tight')
    else:
        plt.savefig(FILEPATH+f'_model{ts_type}.pdf', bbox_inches='tight')
        plt.savefig(FILEPATH+f'_model{ts_type}.png', bbox_inches='tight')
    if show==1:
        plt.show()
    plt.close("all")
    return
def plot_data(x,y,yerr,x_name,y_name,timeseries,inst,x_locator,
              y_locator,FILEPATH,colore='None',residuals=0,show=0,ar=[12,9],
              font=20,mark=0,jit=False,jit_value = [0,1],legend_font=20,ts_type=''):
#def plot_data(x,y,yerr,x_name,y_name,timeseries,inst,x_locator,y_locator,FILEPATH,residuals=0,show=0,ar=[12,9],font=20,mark=0,legend_font=20,ts_type=''):

    if len(y_locator)!=2:
        print('Error, y locator lenghts must be 2')
        return
    if len(x_locator)!=2:
        print('Error, x locator lenghts must be 2')
        return
    if jit != False:
        yerr_jit = np.sqrt(yerr**2 + jit_value**2)
    colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown",
              "tab:pink","tab:gray","tab:olive","tab:cyan",]
    axd = plt.figure(constrained_layout=True,figsize=(ar[0],ar[1])).subplot_mosaic(
    """
    A
    """
    )
    if mark==0:
        marker=8
    else:
        marker=mark
    ax = axd["A"]
    ax.set_xlabel(x_name, fontsize=font, color='black')
    ax.set_ylabel(y_name, fontsize=font, color='black')
    #ax.xaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
    #ax.yaxis.set_major_formatter(tk.FormatStrFormatter("%.0f"))
    ax.yaxis.set_minor_locator(tk.MultipleLocator(y_locator[0]))
    ax.yaxis.set_major_locator(tk.MultipleLocator(y_locator[1]))
    ax.xaxis.set_minor_locator(tk.MultipleLocator(x_locator[0]))
    ax.xaxis.set_major_locator(tk.MultipleLocator(x_locator[1]))
    ax.yaxis.set_tick_params(labelsize=font,width=3.5, size=12,direction='in',)
    ax.xaxis.set_tick_params(labelsize=font,width=3.5, size=12,direction='in')
    ax.yaxis.set_tick_params(labelsize=font,which='minor',width=1.5, size=6,direction='in')
    ax.xaxis.set_tick_params(labelsize=font,which='minor',width=1.5, size=6,direction='in')
    ax.yaxis.set_tick_params(labelsize=font)
    ax.xaxis.set_tick_params(labelsize=font)
    if jit==False:
        if len(timeseries) == 0:
            ax.errorbar(x,y,yerr,fmt='o',markersize=marker,label=inst)
        elif colore != 'None':
            for i in range(len(timeseries)):
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr[timeseries[i]],
                            fmt='o', 
                            markersize=marker,
                            color = colore,
                            label=inst[i])
        else:
            for i in range(len(timeseries)):
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr[timeseries[i]],
                            fmt='o', 
                            markersize=marker,
                            label=inst[i])
    else:
        if len(timeseries) == 0:
            ax.errorbar(x,y,yerr,elinewidth=2,fmt='o',markersize=marker,color=colors[0],label=inst)
            ax.errorbar(x,y,yerr_jit,fmt='o',elinewidth=1,alpha=0.7,markersize=marker,color=colors[0])
        elif colore != 'None':
            for i in range(len(timeseries)):
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr[timeseries[i]],
                            elinewidth=2,
                            fmt='o', 
                            markersize=marker,
                            color = colore,
                            label=inst[i])
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr_jit[timeseries[i]],elinewidth=1,alpha=0.7,
                            fmt='o', 
                            markersize=marker,
                            color = colore)
        else:
            for i in range(len(timeseries)):
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr[timeseries[i]],
                            elinewidth=2,
                            fmt='o', 
                            markersize=marker,
                            color = colors[i],
                            label=inst[i])
                ax.errorbar(x[timeseries[i]],
                            y[timeseries[i]],
                            yerr_jit[timeseries[i]],elinewidth=1,alpha=0.7,
                            fmt='o', 
                            markersize=marker,
                            color = colors[i])
    ax.legend(fontsize=legend_font,markerscale=1.0)
    if residuals==0:
        plt.savefig(FILEPATH+f'_data{ts_type}.pdf', bbox_inches='tight')
        plt.savefig(FILEPATH+f'_data{ts_type}.png', bbox_inches='tight')
        if show==1:
            plt.show()
        plt.close("all")
        return
    if residuals==1:
        plt.savefig(FILEPATH+f'_residuals{ts_type}.pdf', bbox_inches='tight')
        plt.savefig(FILEPATH+f'_residuals{ts_type}.png', bbox_inches='tight')
        if show==1:
            plt.show()
        plt.close("all")
        return
    return

def ecc_func(samples_1,samples_2):
    ecc_cos = samples_1
    ecc_sin = samples_2
    ecc = ecc_cos**2 + ecc_sin**2
    omega = np.arctan2(ecc_sin,ecc_cos)
    return ecc,omega

def correct_sec(bjd,vrad,plx,mua,mub):
    """Corrects secular acceleration from RV time-series

    Args:
        bjd (array(floats)): dates
        vrad (array(floats)): vrad (km/s)
        plx (float): parallax (mas)
        mua (float): proper motion (mas/yr)
        mub (float): proper motion (mas/yr)

    Returns:
        array(floats), array: corrected vrad (km/s), secular acceleration (km/s/d)
    """

    dist=1./(plx/1000.)
    mua=mua/1000.
    mub=mub/1000.

    sec=((0.23* (dist/10.)*(mua**2. + mub**2.))/365.25)/1000.
    vrad = vrad - sec*(bjd - min(bjd))
    return vrad,sec


def create_subplot_figure(n_cols,i=16,j=9):
    # Calcola il numero di subplot richiesti
    n_subplots = n_cols * 2  # sempre due righe, quindi il numero di subplot è fisso

    # Genera la stringa del layout con due righe e `n_cols` colonne
    layout = "\n".join(
        "".join(chr(65 + i + j * n_cols) for i in range(n_cols))
        for j in range(2)
    )
    # Crea la figura con il layout dinamico
    fig, axd = plt.subplot_mosaic(
        layout,
        constrained_layout=True,
        figsize=(i, j),
        gridspec_kw={"height_ratios": [2, 1]}
    )
    return fig, axd

#print(create_subplot_figure(4))
### PHASE folded

def phase_folded(x,y,yerr,timeseries,inst,k,p,t0,e,omega,FILEPATH,show=False,loc=[0.5,2.0],binned=True,ij=[16,9]):
    lun = len(k)
    fig,axd = create_subplot_figure(lun,ij[0],ij[1])
    first_row = []
    second_row = []
    for t in range(lun):
        first_row.append(chr(65+t))
        second_row.append(chr(65+t+lun))

    tryplot_up = []
    tryplot_down = []
    for i in range(lun):
        y_plot = y.copy()
        for j in range(lun):
            if i != j:
                y_plot -= keplerian_t0(x,k[j],p[j],t0[j],e[j],omega[j])
        tryplot_up.append(max(y_plot))
        tryplot_down.append(min(y_plot))
    tryplot_up = np.array(tryplot_up)
    tryplot_down = np.array(tryplot_down)
    ylim_down = min(tryplot_down)
    ylim_up = max(tryplot_up)
    for i in range(lun):
        y_plot = y.copy()
        for j in range(lun):
            if i != j:
                y_plot -= keplerian_t0(x,k[j],p[j],t0[j],e[j],omega[j])
        x_plot = np.arange(min(x),max(x),0.01)
        y_fold = keplerian_t0(x_plot,k[i],p[i],t0[i],e[i],omega[i])
        x_plot_fold = foldAt(x_plot,p[i],t0[i])
        indic_fold = np.argsort(x_plot_fold)
        x_plot_fold = x_plot_fold[indic_fold]
        y_fold = y_fold[indic_fold]
        x_fold = foldAt(x,p[i],t0[i])

        y_res = y_plot.copy()
        y_res_2 = y_res*y_res
        bins = np.arange(0.,1.10,0.1)
        bin_mid = (bins[0:-1]+bins[1:])/2
        mean_bin_pl = scst.binned_statistic(x_fold,y_plot,'mean',bins)[0]
        rms_bin_pl = scst.binned_statistic(x_fold,y_plot,'std',bins)[0]
        N_points = scst.binned_statistic(x_fold,y_res,'count',bins)[0]
        
        lun_timeseries = len(timeseries)
        N_points = np.sqrt(N_points)
        alfa=[0.6]*lun_timeseries
        markcolor = ['none']*lun_timeseries
        rms_error = rms_bin_pl*1./N_points
        font=24

        ax = axd[first_row[i]]
        if i==0:
            ax.set_ylabel(r'RV (m/s)', fontsize=font, color='black')
        #ax.set_xlabel("Phase", fontsize=30, color='black')
        ax.set_ylim(ylim_down - 0.7,ylim_up + 0.7)
        ax.set_title(f'{p[i]:.1f} d planet',fontsize=font-4)
        ax.yaxis.set_minor_locator(tk.MultipleLocator(loc[0]))
        ax.yaxis.set_major_locator(tk.MultipleLocator(loc[1]))
        ax.xaxis.set_minor_locator(tk.MultipleLocator(0.1))
        ax.xaxis.set_major_locator(tk.MultipleLocator(0.5))
        ax.tick_params(labelsize=font,axis='y', which='major',width=2.5, size=8,direction='in')
        ax.tick_params(axis='y', which='minor', width=1, size=4,direction='in')
        ax.tick_params(axis='x', which='major',width=2.5, size=8,direction='in')
        ax.tick_params(axis='x', which='minor', width=1, size=4,direction='in')
        #ax.yaxis.set_tick_params(labelsize=20)
        #ax.xaxis.set_tick_params(labelsize=20)
        for l in range(len(timeseries)):
            ax.errorbar(x_fold[timeseries[l]],
                        y_plot[timeseries[l]],
                        yerr[timeseries[l]],
                        fmt='o', 
                        markersize=6,
                        alpha = alfa[l],
                        markerfacecolor=markcolor[l],
                        label=inst[l])
        ax.errorbar(x_plot_fold,y_fold,fmt='-', color='red',linewidth=2.0)
        if binned==True:
            ax.errorbar(bin_mid,mean_bin_pl,yerr=rms_error,fmt='o', color='red', markersize=12.0)
        if i==0:
            ax.legend(fontsize=16,markerscale=2.0)
        ax.tick_params(axis='x', labelbottom=False)
        if i != 0:
            ax.tick_params(axis='y', labelleft=False)

        r_rv = y - sum(keplerian_t0(x, k[n], p[n], t0[n], e[n], omega[n]) for n in range(lun))


        ax = axd[second_row[i]]
        if i == 0:
            ax.set_ylabel(r'Residuals RV (m/s)', fontsize=font, color='black')
        ax.set_xlabel("Phase", fontsize=font, color='black')
        ax.set_ylim(min(r_rv) - 0.7,max(r_rv) + 0.7)
        ax.yaxis.set_minor_locator(tk.MultipleLocator(loc[0]))
        ax.yaxis.set_major_locator(tk.MultipleLocator(loc[1]))
        ax.xaxis.set_minor_locator(tk.MultipleLocator(0.1))
        ax.xaxis.set_major_locator(tk.MultipleLocator(0.5))
        ax.tick_params(labelsize=font,axis='y', which='major',width=2.5, size=8,direction='in')
        ax.tick_params(axis='y', which='minor', width=1, size=4,direction='in')
        ax.tick_params(labelsize=font,axis='x', which='major',width=2.5, size=8,direction='in')
        ax.tick_params(axis='x', which='minor', width=1, size=4,direction='in')
        #ax.yaxis.set_tick_params(labelsize=20)
        #ax.xaxis.set_tick_params(labelsize=20)
        for l in range(len(timeseries)):
            ax.errorbar(x_fold[timeseries[l]],
                        r_rv[timeseries[l]],
                        yerr[timeseries[l]],
                        fmt='o', 
                        alpha=alfa[l],
                        markerfacecolor=markcolor[l],
                        markersize=6,
                        label=inst[l])
        ax.axhline(y=0.0,color='r',zorder=10,linewidth=2.0)
        if i != 0:
            ax.tick_params(axis='y', labelleft=False)
    fig.savefig(FILEPATH+'_phase_folded.pdf', bbox_inches='tight')
    fig.savefig(FILEPATH+'_phase_folded.png', bbox_inches='tight')
    if show==1:
        plt.show()
    plt.close("all")
    return


def ind_ok_data(rv,sigma_rv,time,sigma_clip=3):
    rv = np.array(rv)
    sigma_rv = np.array(sigma_rv)
    time = np.array(time)

    rv_ok = np.array(rv)
    sigma_rv_ok = np.array(sigma_rv)
    time_ok = np.array(time)
    x0 = min(time_ok)//1
    time_ok = time_ok - x0
    dr,dt = binningx0dt(time_ok, rv_ok, sigma_rv_ok,x0=0.0,dt=1.0,useMeanX=True,removeEmpty=True)
    time_ok = dr[::,0]
    rv_ok = dr[::,1]
    sigma_rv_ok = dr[::,2]

    time_ok += x0

    if sigma_clip==7:
        return rv_ok,sigma_rv_ok,time_ok
    ind_ok = np.where(np.abs(rv_ok-np.median(rv_ok))<sigma_clip*np.std(rv_ok))[0]
    rv_ok = np.array(rv_ok[ind_ok])
    sigma_rv_ok = np.array(sigma_rv_ok[ind_ok])
    time_ok = np.array(time_ok[ind_ok])

    ind_ok = indices = np.where((np.abs(rv_ok - np.median(rv_ok)) < sigma_clip * np.std(rv_ok)) &(sigma_rv_ok < 3 * np.median(sigma_rv_ok)))[0]
    return ind_ok


# timestamps and params

def time_stamp_reader(path_archive,inst,star,show=0,ephemeris=0,delete=1):  # sp spectral type
    if inst != 'HARPS' and inst != 'HARPSN' and inst != 'HARPS15' and inst != 'HARPS03':
        print('ERROR: The only instruments supported are : \n "HARPS" \n "HARPSN"')
        return
        
    extractor(path_archive,inst)
    path_ccf = 'CCF_{}/'.format(inst)
    path_s1d = 'S1D_{}/'.format(inst)
    file_header = 'file_header_s.txt'
    f_header = open(file_header,'w')
    for child in Path(path_s1d).iterdir():
        if child.is_file():
            f_header.write(f"{path_s1d}{child.name}\n")
    f_header.close()
    header_reader = open(file_header).readlines()

    for i in range(len(header_reader)):
        a = str(header_reader[i])
        a = a[:-1]
        header_reader[i] = a

    # header_data=fits.getheader('/home/nicola/Scrivania/HARPS/HD7134/SPECTRA_READER/CCF_HARPSN/HARPN.2014-09-19T03-48-43.419_ccf_G2_A.fits')
    # print(header_data)
    # exit()
    time_stamp = []
    night = []
    prog_id = []
    star_name = []
    for i in range(len(header_reader)):
        if inst == 'HARPS' or inst == 'HARPS03' or inst == 'HARPS15':
            header_data = fits.getheader(header_reader[i])
            if show:
                print(header_data)
                pdb.set_trace()
            # exit()
            # exit()
            pi = header_data["DATE-OBS"]
            prog_id.append(header_data["HIERARCH ESO OBS PROG ID"])
            star_name.append(header_data["HIERARCH ESO OBS TARG NAME"])
            before_T, after_T = pi.split('T')
            first_value = float(after_T[0])
            second_value = float(after_T[1])
            value = 10*first_value + second_value
            time_stamp.append(f"HARPS.{pi}")
            if value < 12:
                date = datetime.strptime(before_T, '%Y-%m-%d')
                
                # Add one day to the date 
                date_minus_one = date - timedelta(days=1)
                # Format new date as string 
                new_date_str = date_minus_one.strftime('%Y-%m-%d')
                night.append(new_date_str)
            else:
                night.append(before_T)

        if inst == 'HARPSN':
            header_data = fits.getheader(header_reader[i])
            if show:
                print(header_data)
                pdb.set_trace()
            
            pi = header_data["DATE-OBS"]
            prog_id.append(header_data["HIERARCH TNG OBS PROG ID"])
            star_name.append(header_data["HIERARCH TNG OBS TARG NAME"])
            before_T, after_T = pi.split('T')
            first_value = float(after_T[0])
            second_value = float(after_T[1])
            value = 10*first_value + second_value
            time_stamp.append(f"HARPSN.{pi}")
            if value < 12:
                date = datetime.strptime(before_T, '%Y-%m-%d')
                
                # Add one day to the date 
                date_minus_one = date - timedelta(days=1)
                # Format new date as string 
                new_date_str = date_minus_one.strftime('%Y-%m-%d')
                night.append(new_date_str)
            else:
                night.append(before_T)

    
    #time_stamp = np.unique(time_stamp)
    id_file = f'{star}_{inst}_timestamps.txt'
    name_file = open(id_file,'w')

    for i in range(len(time_stamp) + 1):
        if i==0:
            name_file.write('# Date-obs Night \n')
        else:
            name_file.write(f'{time_stamp[i-1]} {night[i-1]}\n')
    name_file.close()

    prog_id = np.unique(prog_id)
    star_name = np.unique(star_name)
    
    prog_file = f'{star}_{inst}_params.txt'
    id_file = open(prog_file,'w')

    for i in range(len(prog_id) + 1):
        if i==0:
            id_file.write('# Prog_ID \n')
        else:
            id_file.write(f'{prog_id[i-1]}\n')
    for i in range(len(star_name) + 1):
        if i==0:
            id_file.write('# Star_name \n')
        else:
            id_file.write(f'{star_name[i-1]}\n')
    
    if ephemeris==1:
        ephemeris_txt = f'{star}_ephemeris.txt'
        ra,dec,pmra,pmdec,rv,spt = np.loadtxt(ephemeris_txt,unpack=True,usecols=[0,1,2,3,4,5],skiprows=1,dtype=str)
        id_file.write('\n')
        id_file.write('# Ephemeris (Simbad)\n')
        id_file.write(f'RA = {ra} \n')
        id_file.write(f'DEC = {dec} \n')
        id_file.write(f'PMRA(mas/yr) = {pmra} \n')
        id_file.write(f'PMDEC(mas/yr) = {pmdec} \n')
        id_file.write(f'RV(km/s) = {rv} \n')
        id_file.write(f'Spectral_type = {spt} \n')


    
    name_file.close()
    if delete==1:

        main_folder = Path.cwd()

        # List of keywords to identify folders to delete
        keywords = ['CCF', 'BIS', 'S2D', 'S1D']

        # Find and delete folders containing the keywords in their name
        for folder in main_folder.rglob('*'):
            if folder.is_dir() and any(keyword in folder.name for keyword in keywords):
                shutil.rmtree(folder)
                print(f'Deleted folder: {folder}')
    return

def ccf_reader(path_archive,inst,star):  # sp spectral type
    if inst != 'HARPS' and inst != 'HARPSN' and inst != 'HARPS15' and inst != 'HARPS03':
        print('ERROR: The only instruments supported are : \n "HARPS" \n "HARPSN"')
        return
        
    extractor(path_archive,inst)
    path_ccf = 'BIS_{}/'.format(inst)
    file_header = 'file_header_s.txt'
    f_header = open(file_header,'w')
    for child in Path(path_ccf).iterdir():
        if child.is_file():
            f_header.write(f"{path_ccf}{child.name}\n")
    f_header.close()
    header_reader = open(file_header).readlines()

    for i in range(len(header_reader)):
        a = str(header_reader[i])
        a = a[:-1]
        header_reader[i] = a

    # header_data=fits.getheader('/home/nicola/Scrivania/HARPS/HD7134/SPECTRA_READER/CCF_HARPSN/HARPN.2014-09-19T03-48-43.419_ccf_G2_A.fits')
    # print(header_data)
    # exit()
    ccf_rv = []
    ccf_fwhm = []
    ccf_bis = []
    ccf_erv = []
    ccf_time = []
    ccf_mask = []
    ccf_contrast = []
    for i in range(len(header_reader)):
        if inst == 'HARPS' or inst == 'HARPS03' or inst == 'HARPS15':
            header_data = fits.getheader(header_reader[i])
            print(header_data)
            exit()
            ccf_mask.append(header_data["HIERARCH ESO DRS CCF MASK"])
            ccf_erv.append(header_data["HIERARCH ESO DRS DVRMS"] )
            ccf_time.append(header_data["HIERARCH ESO DRS BJD"])
            ccf_fwhm.append(header_data["HIERARCH ESO DRS CCF FWHM"])
            ccf_rv.append(header_data["HIERARCH ESO DRS CCF RV"])
            ccf_bis.append(header_data["HIERARCH ESO DRS BIS SPAN"])
            ccf_contrast.append(header_data["HIERARCH ESO DRS CCF CONTRAST"])
            
        if inst == 'HARPSN':
            header_data = fits.getheader(header_reader[i])
            
            
            ccf_mask.append(header_data["HIERARCH TNG DRS CCF MASK"])
            ccf_erv.append(header_data["HIERARCH TNG DRS DVRMS"] )
            ccf_time.append(header_data["HIERARCH TNG DRS BJD"])
            ccf_fwhm.append(header_data["HIERARCH TNG DRS CCF FWHM"])
            ccf_rv.append(header_data["HIERARCH TNG DRS CCF RV"])
            ccf_bis.append(header_data["HIERARCH TNG DRS BIS SPAN"])
            ccf_contrast.append(header_data["HIERARCH TNG DRS CCF CONTRAST"])
    
    #time_stamp = np.unique(time_stamp)

    id_file = f'{star}_{inst}_ccf.txt'
    prog_file = open(id_file,'w')
    argosort = np.argsort(ccf_time)
    ccf_time = [ccf_time[i] for i in argosort]
    ccf_rv = [ccf_rv[i] for i in argosort]
    ccf_bis = [ccf_bis[i] for i in argosort]
    ccf_fwhm = [ccf_fwhm[i] for i in argosort]
    ccf_contrast = [ccf_contrast[i] for i in argosort]
    ccf_erv = [ccf_erv[i] for i in argosort]
    ccf_mask = [ccf_mask[i] for i in argosort]

    for i in range(len(ccf_time) + 1):
        if i==0:
            prog_file.write('# BJD RV eRV FWHM BIS Contrast mask\n')
        else:
            prog_file.write(f'{ccf_time[i-1]} {ccf_rv[i-1]} {ccf_erv[i-1]} {ccf_fwhm[i-1]} {ccf_bis[i-1]} {ccf_contrast[i-1]} {ccf_mask[i-1]} \n')
    prog_file.close()

    main_folder = Path.cwd()
    # List of keywords to identify folders to delete
    keywords = ['CCF', 'BIS', 'S2D', 'S1D']

    # Find and delete folders containing the keywords in their name
    for folder in main_folder.rglob('*'):
        if folder.is_dir() and any(keyword in folder.name for keyword in keywords):
            shutil.rmtree(folder)
            print(f'Deleted folder: {folder}')
    return


def smw_index(path_archive,sp,inst,star,V0=0.0):  # sp spectral type
    if inst != 'HARPS' and inst != 'HARPSN' and inst != 'HARPS15' and inst != 'HARPS03':
        print('ERROR: The only instruments supported are : \n "HARPS" \n "HARPSN"')
        return
        
    extractor(path_archive,inst)
    path_ccf = 'CCF_{}/'.format(inst)
    path_s1d = 'S1D_{}/'.format(inst)
    file_header = 'file_header_s.txt'
    f_header = open(file_header,'w')
    for child in Path(path_ccf).iterdir():
        if child.is_file():
            f_header.write(f"{path_ccf}{child.name}\n")
    f_header.close()
    header_reader = open(file_header).readlines()

    for i in range(len(header_reader)):
        a = str(header_reader[i])
        a = a[:-1]
        header_reader[i] = a

    v0 = []
    if V0==0.0:
        for i in range(len(header_reader)):
            if inst == 'HARPS' or inst=='HARPS03' or inst == 'HARPS15':
                header_data = fits.getheader(header_reader[i])
                rad_v = header_data["HIERARCH ESO DRS CCF RVC"]
                v0.append(rad_v)
            if inst == 'HARPSN':
                header_data = fits.getheader(header_reader[i])
                rad_v = header_data["HIERARCH TNG DRS CCF RVC"]
                v0.append(rad_v)


        v0 = np.array(v0)

    file_header = 'file_header.txt'
    f_header = open(file_header,'w')
    for child in Path(path_s1d).iterdir():
        if child.is_file():
            f_header.write(f"{path_s1d}{child.name}\n")
    f_header.close()
    header_reader = open(file_header).readlines()

    for i in range(len(header_reader)):
        a = str(header_reader[i])
        a = a[:-1]
        header_reader[i] = a

    if V0 !=0.0:
        v0 = [V0]*len(header_reader)
        v0 = np.array(v0)

    t = []
    smw = []
    sigma_smw = []
    smw_phot = []
    sigma_smw_phot = []
    k = 0
    for i in range(len(header_reader)):
        file_s1d = f'{header_reader[i]}'

        spec       = fits.getdata(file_s1d)
        crval1     = fits.getval(file_s1d,'CRVAL1',0)
        cdelt1     = fits.getval(file_s1d,'CDELT1',0)
        if inst=='HARPS' or inst=='HARPS03' or inst =='HARPS15':
            bjd = fits.getval(file_s1d,'HIERARCH ESO DRS BJD',0)
        elif inst=='HARPSN':
            bjd = fits.getval(file_s1d,'HIERARCH TNG DRS BJD',0)
        wave       = np.linspace(crval1,crval1+len(spec)*cdelt1,len(spec))
        valid_data = np.where( (wave > 3800) & (wave < 6800))[0]
        wave       = wave[valid_data]
        flux       = spec[valid_data]
        err        = (np.sqrt(flux)**2 + 6**2)**0.5
        flux[np.where(np.isnan(flux) == True)] = 0
        err[np.where(np.isnan(err) == True)]   = 6

        sindex,sigma_sindex,sindex_phot,sigma_sindex_phot = measure_smw_s1d(wave,flux,err,v0[i],sp)
        t.append(bjd)
        smw.append(sindex)
        sigma_smw.append(sigma_sindex)
        smw_phot.append(sindex_phot)
        sigma_smw_phot.append(sigma_sindex_phot)
        k += 1
        print(k,sindex,sindex_phot)

    argosort = np.argsort(t)
    t = [t[i] for i in argosort]
    smw = [smw[i] for i in argosort]
    sigma_smw = [sigma_smw[i] for i in argosort] 
    smw_phot = [smw_phot[i] for i in argosort]
    sigma_smw_phot = [sigma_smw_phot[i] for i in argosort] 

    mjd = np.array(t)
    smw = np.array(smw)
    sigma_smw = np.array(sigma_smw)
    smw_phot = np.array(smw_phot)
    sigma_smw_phot = np.array(sigma_smw_phot)
    print(len(smw_phot),len(smw))

    smw_file = f'smw_index_{star}_{inst}.txt'
    s_mw = open(smw_file,'w')
    for i in range(len(mjd) + 1):
        if i == 0 :
            s_mw.write('BJD Smw e_Smw Smw_phot e_Smw_phot \n')
        else:
            s_mw.write('{} {} {} {} {} \n'.format(mjd[i-1],smw[i-1],sigma_smw[i-1],smw_phot[i-1],sigma_smw_phot[i-1]))
    s_mw.close()
    
    os.remove('file_header.txt')
    os.remove('file_header_s.txt')

    main_folder = Path.cwd()

    # List of keywords to identify folders to delete
    keywords = ['CCF', 'BIS', 'S2D', 'S1D']

    # Find and delete folders containing the keywords in their name
    for folder in main_folder.rglob('*'):
        if folder.is_dir() and any(keyword in folder.name for keyword in keywords):
            shutil.rmtree(folder)
            print(f'Deleted folder: {folder}')
    return

def ha_index(path_archive,sp,inst,star,V0=0):  # sp spectral type
    if inst != 'HARPS' and inst != 'HARPSN' and inst != 'HARPS03' and inst != 'HARPS15':
        print('ERROR: The only instruments supported are : \n "HARPS" \n "HARPSN"')
        return
        
    extractor(path_archive,inst)
    path_ccf = 'CCF_{}/'.format(inst)
    path_s1d = 'S1D_{}/'.format(inst)
    file_header = 'file_header_s.txt'
    f_header = open(file_header,'w')
    for child in Path(path_ccf).iterdir():
        if child.is_file():
            f_header.write(f"{path_ccf}{child.name}\n")
    f_header.close()
    header_reader = open(file_header).readlines()

    for i in range(len(header_reader)):
        a = str(header_reader[i])
        a = a[:-1]
        header_reader[i] = a

    v0 = []
    if V0==0.0:
        for i in range(len(header_reader)):
            if inst == 'HARPS' or inst=='HARPS03' or inst == 'HARPS15':
                header_data = fits.getheader(header_reader[i])
                rad_v = header_data["HIERARCH ESO DRS CCF RVC"]
                v0.append(rad_v)
            if inst == 'HARPSN':
                header_data = fits.getheader(header_reader[i])
                rad_v = header_data["HIERARCH TNG DRS CCF RVC"]
                v0.append(rad_v)


        v0 = np.array(v0)

    file_header = 'file_header.txt'
    f_header = open(file_header,'w')
    for child in Path(path_s1d).iterdir():
        if child.is_file():
            f_header.write(f"{path_s1d}{child.name}\n")
    f_header.close()
    header_reader = open(file_header).readlines()

    for i in range(len(header_reader)):
        a = str(header_reader[i])
        a = a[:-1]
        header_reader[i] = a

    if V0 !=0.0:
        v0 = [V0]*len(header_reader)
        v0 = np.array(v0)

    t = []
    ha = []
    sigma_ha = []
    ha_phot = []
    sigma_ha_phot = []
    k = 0
    for i in range(len(header_reader)):
        file_s1d = f'{header_reader[i]}'

        spec       = fits.getdata(file_s1d)
        crval1     = fits.getval(file_s1d,'CRVAL1',0)
        cdelt1     = fits.getval(file_s1d,'CDELT1',0)
        if inst=='HARPS' or inst =='HARPS03' or inst =='HARPS15':
            bjd = fits.getval(file_s1d,'HIERARCH ESO DRS BJD',0)
        elif inst=='HARPSN':
            bjd = fits.getval(file_s1d,'HIERARCH TNG DRS BJD',0)
        wave       = np.linspace(crval1,crval1+len(spec)*cdelt1,len(spec))
        valid_data = np.where( (wave > 3800) & (wave < 6800))[0]
        wave       = wave[valid_data]
        flux       = spec[valid_data]
        err        = (np.sqrt(flux)**2 + 6**2)**0.5
        flux[np.where(np.isnan(flux) == True)] = 0
        err[np.where(np.isnan(err) == True)]   = 6

        sindex,sigma_sindex,sindex_phot,sigma_sindex_phot = measure_ha_s1d(wave,flux,err,v0[i],sp)
        t.append(bjd)
        ha.append(sindex)
        sigma_ha.append(sigma_sindex)
        ha_phot.append(sindex_phot)
        sigma_ha_phot.append(sigma_sindex_phot)
        k += 1
        print(k,sindex,sindex_phot)
    
    argosort = np.argsort(t)
    t = [t[i] for i in argosort]
    ha = [ha[i] for i in argosort]
    sigma_ha = [sigma_ha[i] for i in argosort]
    ha_phot = [ha_phot[i] for i in argosort]
    sigma_ha_phot = [sigma_ha_phot[i] for i in argosort]
    mjd = np.array(t)
    ha = np.array(ha)
    sigma_ha = np.array(sigma_ha)
    ha_phot = np.array(ha_phot)
    sigma_ha_phot = np.array(sigma_ha_phot)
    print(len(ha_phot),len(ha))

    ha_file = 'ha_index_{}_{}.txt'.format(star,inst)
    s_mw = open(ha_file,'w')
    for i in range(len(mjd) + 1):
        if i == 0 :
            s_mw.write('MJD ha e_ha ha_phot e_ha_phot \n')
        else:
            s_mw.write('{} {} {} {} {} \n'.format(mjd[i-1],ha[i-1],sigma_ha[i-1],ha_phot[i-1],sigma_ha_phot[i-1]))
    s_mw.close()
    
    os.remove('file_header.txt')
    os.remove('file_header_s.txt')
    main_folder = Path.cwd()

    # List of keywords to identify folders to delete
    keywords = ['CCF', 'BIS', 'S2D', 'S1D']

    # Find and delete folders containing the keywords in their name
    for folder in main_folder.rglob('*'):
        if folder.is_dir() and any(keyword in folder.name for keyword in keywords):
            shutil.rmtree(folder)
            print(f'Deleted folder: {folder}')
    return
### Convert to scipy dists
def convert_to_scipy(dist_objects):
    """
    Converts a list of custom distribution objects into corresponding scipy.stats objects,
    ensuring equivalent parameters are preserved.

    Args:
        dist_objects (list): A list of custom distribution objects.

    Returns:
        list: A list of corresponding scipy.stats objects.
    """
    scipy_distributions = []
    
    for dist in dist_objects:
        # Directly assign loc and scale
        
        # Determine the type of distribution
        dist_class_name = dist.__class__.__name__.lower()
        if "uniform" in dist_class_name:
            loc = dist.lo
            scale = dist.scale
            scipy_distributions.append(uniform(loc=loc, scale=scale))
        elif "norm" in dist_class_name or "gaussian" in dist_class_name:
            loc = dist.loc
            scale = dist.scale
            scipy_distributions.append(norm(loc=loc, scale=scale))
        else:
            raise ValueError(f"Unsupported distribution type: {dist_class_name}")
    
    return scipy_distributions

