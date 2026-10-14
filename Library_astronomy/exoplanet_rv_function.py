import numpy as np
import matplotlib.pylab as plt
from PyAstronomy.pyasl import foldAt
import scipy.stats as scst
from numba import njit
PI = np.pi
TWO_PI = 2*np.pi
INV_TWO_PI = 1/TWO_PI
HALF_PI = 0.5*np.pi

def sine_model(x,K,P,T0):
  T0 = x[0] + P*T0
  sin_model = -K*np.sin(2*np.pi * (x - T0)/P)
  return sin_model

def sine_model_new(x,K,P,T0):
  T0 = x[-1] - P*T0
  sin_model = -K*np.sin(2*np.pi * (x - T0)/P)
  return sin_model


def sine_model_T0(x,K,P,T0):
  sin_model = -K*np.sin(2*np.pi * (x - T0)/P)
  return sin_model


def T0_calculator(T0,P,x):
  return x[0] + P*T0 + 2450000

@njit(cache=True)
def keplerian_t0(t,krv,per,t0,e,w):
    ma_offset = np.arctan2(np.sqrt(1.0-e*e) * np.sin(HALF_PI - w), e +np.cos(HALF_PI - w))
    ma_offset -= e*np.sin(ma_offset)
    Ma = np.mod(TWO_PI * (t - (t0 - ma_offset * per *INV_TWO_PI)) / per, TWO_PI)

    Ea = Ma.copy()
    for j in range(t.size):
        err = 0.05
        k = 0
        while abs(err) > 1e-4 and k<500:
            err   = Ea[j] - e*np.sin(Ea[j]) - Ma[j]
            Ea[j] = Ea[j] - err/(1.0-e*np.cos(Ea[j]))
            k += 1

    sta = np.sqrt(1.0-e*e) * np.sin(Ea)/(1.0-e*np.cos(Ea))
    cta = (np.cos(Ea)-e)/(1.0-e*np.cos(Ea))
    nu  = np.arctan2(sta, cta)

    return krv * (np.cos(nu + w) + e * np.cos(w))
@njit(cache=True)
def keplerian(t,krv,per,t0,e,w):
    t0 = t[-1] - per*t0
    ma_offset = np.arctan2(np.sqrt(1.0-e*e) * np.sin(HALF_PI - w), e +np.cos(HALF_PI - w))
    ma_offset -= e*np.sin(ma_offset)
    Ma = np.mod(TWO_PI * (t - (t0 - ma_offset * per *INV_TWO_PI)) / per, TWO_PI)

    Ea = Ma.copy()
    for j in range(t.size):
        err = 0.05
        k = 0
        while abs(err) > 1e-4 and k<500:
            err   = Ea[j] - e*np.sin(Ea[j]) - Ma[j]
            Ea[j] = Ea[j] - err/(1.0-e*np.cos(Ea[j]))
            k += 1

    sta = np.sqrt(1.0-e*e) * np.sin(Ea)/(1.0-e*np.cos(Ea))
    cta = (np.cos(Ea)-e)/(1.0-e*np.cos(Ea))
    nu  = np.arctan2(sta, cta)

    return krv * (np.cos(nu + w) + e * np.cos(w))

def exo_rv_model_T0(par,t):   # generic version for everyday use
  # par=[K,Per,T0,ecc,omega,slope] --> if ecc and omega
  # are missing set them to 0 (1), if slope missing doesn't
  # compute (2). To have slope but not eccentricity par should
  # be set  as [K,Per,T0,0,0,slope].
  # T0 is defined in BJD
  param = np.array(par)
  if len(param) <= 3:   # (1)
    param = np.concatenate((param,np.array([0,0])))
  timess = t
  ndata = timess.shape[0]
  model_rv = np.zeros(shape=ndata)
  phase = np.zeros(shape=ndata)
  wp = (5.0*np.pi/2.-param[4])
  arge0 = (np.sqrt(1-param[3]**2)*np.sin(wp),(param[3]+np.cos(wp))) # da 
  e0 = np.arctan2(arge0[0],arge0[1]) # qui 
  mean_an0 = e0 - param[3]*np.sin(e0) # a qui si passa dal tempo di passaggio al periastro al tempo di transito
  phase = foldAt(timess, param[1], T0=param[2])
  fma = phase*2.0*np.pi + mean_an0
  ee = fma+param[3]*np.sin(fma)
  for i in range(0,15):
    denom = 1-param[3]*np.cos(ee)
    disc = fma-ee+param[3]*np.sin(ee)
    ee = ee+disc/denom
    if np.max(np.abs(disc)) <= 2.0e-5:
        break
  cosv = (np.cos(ee)-param[3])/denom
  sinv = np.sin(ee)*np.sqrt(1-param[3]**2)/denom
  model_rv = param[0]*(np.cos(param[4])*cosv-np.sin(param[4])*sinv+param[3]*np.cos(param[4]))
  if len(param) == 6:   # (2)
    t0 = (timess[-1]+timess[0])/2.0
    model_rv += (timess-t0)*param[5]
  return model_rv


def exo_rv_model(par,t):   # generic version for everyday use
  # par=[K,Per,T0,ecc,omega,slope] --> if ecc and omega
  # are missing set them to 0 (1), if slope missing doesn't
  # compute (2). To have slope but not eccentricity par should
  # be set  as [K,Per,T0,0,0,slope].
  # T0 is defined in [0,1]
  param = np.array(par)
  if len(param) <= 3:   # (1)
    param = np.concatenate((param,np.array([0,0])))
  timess = t
  ndata = timess.shape[0]
  model_rv = np.zeros(shape=ndata)
  phase = np.zeros(shape=ndata)
  wp = (5.0*np.pi/2.-param[4])
  arge0 = (np.sqrt(1-param[3]**2)*np.sin(wp),(param[3]+np.cos(wp)))
  e0 = np.arctan2(arge0[0],arge0[1])
  mean_an0 = e0 - param[3]*np.sin(e0)
  phase = foldAt(timess, param[1], T0=param[2]*param[1])
  fma = phase*2.0*np.pi + mean_an0
  ee = fma+param[3]*np.sin(fma)
  for i in range(0,15):
    denom = 1-param[3]*np.cos(ee)
    disc = fma-ee+param[3]*np.sin(ee)
    ee = ee+disc/denom
    if np.max(np.abs(disc)) <= 2.0e-5:
        break
  cosv = (np.cos(ee)-param[3])/denom
  sinv = np.sin(ee)*np.sqrt(1-param[3]**2)/denom
  model_rv = param[0]*(np.cos(param[4])*cosv-np.sin(param[4])*sinv+param[3]*np.cos(param[4]))
  if len(param) == 6:   # (2)
    t0 = (timess[-1]+timess[0])/2.0
    model_rv += (timess-t0)*param[5]
  return model_rv

def exo_rv_model_2(par,t):   # exo_rv_model when I have to use different harmonics in activity indicators modeling
  param = np.array(par)
  if len(param) <= 3:   # (1)
    param = np.concatenate((param,np.array([0,0])))
  timess = t
  ndata = timess.shape[0]
  model_rv = np.zeros(shape=ndata)
  phase = np.zeros(shape=ndata)
  wp = (5.0*np.pi/2.-param[4])
  arge0 = (np.sqrt(1-param[3]**2)*np.sin(wp),(param[3]+np.cos(wp)))
  e0 = np.arctan2(arge0[0],arge0[1])
  mean_an0 = e0 - param[3]*np.sin(e0)
  phase = foldAt(timess, param[1], T0=param[2]*param[1]*2)
  fma = phase*2.0*np.pi + mean_an0
  ee = fma+param[3]*np.sin(fma)
  for i in range(0,15):
    denom = 1-param[3]*np.cos(ee)
    disc = fma-ee+param[3]*np.sin(ee)
    ee = ee+disc/denom
    if np.max(np.abs(disc)) <= 2.0e-5:
        break
  cosv = (np.cos(ee)-param[3])/denom
  sinv = np.sin(ee)*np.sqrt(1-param[3]**2)/denom
  model_rv = param[0]*(np.cos(param[4])*cosv-np.sin(param[4])*sinv+param[3]*np.cos(param[4]))
  if len(param) == 6:   # (2)
    t0 = (timess[-1]+timess[0])/2.0
    model_rv += (timess-t0)*param[5]
  return model_rv

# # 2459490.530017063 37732.1669239372 0.865913053933589 7.02883352780773 -0.00536833572841289 56.199393509523 2021-10-03T00:38:48.586
# # 2459490.5338465204 37730.9007736365 0.932469126943378 7.02655794274559 -0.00450512878396125 56.2396199028672 2021-10-03T00:44:19.482
# # 2459490.537671107 37732.5550262159 0.970222870471155 7.03071563608175 -0.00701734372074014 56.2113687688518 2021-10-03T00:49:49.959
# # 2459491.5267370334 37732.3367938888 1.23100511166667 7.02649731181022 -0.00678643205422702 56.3593801629659 2021-10-04T00:34:10.543
# # 2459491.530598778 37730.752582005 1.34663907184534 7.02082194396559 -0.00478065381755499 56.3724928506958 2021-10-04T00:39:41.230
# # 2459491.5342920194 37735.3648518378 1.57333627390171 7.01974482342564 -0.00411196794881619 56.3081760448984 2021-10-04T00:45:12.357
# # 2459492.48818548 37731.1239598242 0.618726422755681 7.04072264254373 -0.00649918585914833 56.1594436736107 2021-10-04T23:38:50.755
# # 2459492.4920472153 37732.164770366 0.640216680560988 7.03515424373222 -0.00892977937081696 56.1847262625978 2021-10-04T23:44:21.442
# # 2459492.4958420633 37731.0682551512 0.651599620523751 7.03541645753185 -0.00511368055703798 56.2121933483759 2021-10-04T23:49:52.349
# # 2459493.5012498796 37730.3349150103 0.796121880247821 7.02599893667357 -0.00635309920428995 56.2566467372476 2021-10-05T23:57:48.033
# # 2459493.5051491135 37732.2531598269 0.797816333001386 7.02490282364019 -0.0047251130583561 56.2781428792914 2021-10-06T00:03:18.960
# # 2459493.5089079714 37730.667105602 0.865552210051956 7.02223418847476 -0.00700393204396477 56.2579656503599 2021-10-06T00:08:49.756
# # 2459515.533420353 37730.0659477772 1.02834711022979 7.01786737426186 -0.0252404820958176 56.0784753764048 2021-10-28T00:47:10.168
# # 2459515.537248426 37727.6686521896 1.19695599937171 7.00531582720744 -0.0260751031628743 56.0868585635829 2021-10-28T00:52:40.944
# # 2459515.541076969 37729.824611204596 1.49880817342281 7.00258070046683 -0.0204444300772266 56.1444370563722 2021-10-28T00:58:11.761
# # 2459516.5108680595 37735.627920646 0.775503077437777 7.01516896034073 -0.020208345098581 56.128135968756 2021-10-29T00:14:49.417
# # 2459516.5146949845 37734.9044255962 0.802363574132322 7.00787290185813 -0.0183374184843501 56.17540039858 2021-10-29T00:20:20.094
# # 2459516.5185573287 37735.717348406506 0.802012545846117 7.00778927671034 -0.0222360126678183 56.211497053329 2021-10-29T00:25:50.831