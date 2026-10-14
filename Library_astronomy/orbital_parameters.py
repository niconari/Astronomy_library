import numpy as np
from matplotlib import pyplot as plt
import pdb
def sec_acc(d,prop,t):
    t = t - t[0]
    print(t)
    sa = 0.23*(d/10)*(prop/1000)**2*t/365.25
    return sa

def planet_mass_sini(K,P,Ecc,Mstar):
    
    ecc_part = np.sqrt(1-Ecc**2)
    period_part = P**(1./3)
    m_star_part = Mstar**(2./3)
    
    planet_mass = 4.919e-3*K*ecc_part*period_part*m_star_part    
    
    return planet_mass
# print(planet_mass_sini(0.98,3600,0.0,1)*317.8)
# pdb.set_trace()
#pdb.set_trace()
#print(planet_mass_sini(0.66,689,0.0,0.95)*317.8)
#print(planet_mass_sini(0.29,6.42,0.0,0.95)*317.8)

# print(planet_mass_sini(1.98,705.0,0.6,0.95)*317.8)
# print(planet_mass_sini(0.72,72.6,0.0,0.95)*317.8)
#print(planet_mass_sini(0.75,243,0.0,0.91)*317.8)
#exit()
#print(planet_mass_sini(1.5,127,0.0,0.1)*317.8)
#print(planet_mass_sini(1.0,337,0.0,0.79)*318)
# exit()
def insolation_flux(r,teff,au):
    r_term = (r)**2
    t_term = (teff/5778)**4
    au_term = (1.0/au)**2
    insolation = r_term * t_term * au_term # relative to Earth flux
    return insolation

r_array = np.random.normal(loc=0.93, scale=0.03, size=int(1e6))
t_array = np.random.normal(loc=5368, scale=64, size=int(1e6))
au_array_b = np.random.normal(loc=0.12570, scale=0.00053, size=int(1e6))
au_array_c = np.random.normal(loc=0.3625, scale=0.0016, size=int(1e6))

insolation_b = insolation_flux(r_array, t_array, au_array_b)
ins_b = np.percentile(insolation_b,50)
ins_plus_b = np.percentile(insolation_b,84) - ins_b
ins_minus_b = ins_b - np.percentile(insolation_b,16)

insolation_c = insolation_flux(r_array, t_array, au_array_c)
ins_c = np.percentile(insolation_c,50)
ins_plus_c = np.percentile(insolation_c,84) - ins_c
ins_minus_c = ins_c - np.percentile(insolation_c,16)

# print('Insolation flux for planet b: ',ins_b,'+/-',ins_plus_b,'- ',ins_minus_b)
# print('Insolation flux for planet c: ',ins_c,'+/-',ins_plus_c,'- ',ins_minus_c)
# pdb.set_trace()
def Teff(t_star,semi_major,r_star,albedo=0.3):
    r_star = r_star * 696340/ 149597870.7  # Convert radius from km to AU
    temp_eff = t_star*np.sqrt(r_star/(2*semi_major))*(1-albedo)**0.25
    return temp_eff


temperature_b = Teff(t_array,au_array_b,r_array,albedo=0.3)
temperature_c = Teff(t_array,au_array_c,r_array,albedo=0.3)
# temperature_d = Teff(t_array,au_array_d,r_array,albedo=0.3) 

temp_b = np.percentile(temperature_b,50)
temp_plus_b = np.percentile(temperature_b,84) - temp_b
temp_minus_b = temp_b - np.percentile(temperature_b,16)
temp_c = np.percentile(temperature_c,50)
temp_plus_c = np.percentile(temperature_c,84) - temp_c
temp_minus_c = temp_c - np.percentile(temperature_c,16)

# print('Effective temperature for planet b: ',temp_b,'+/-',temp_plus_b,'- ',temp_minus_b)
# print('Effective temperature for planet c: ',temp_c,'+/-',temp_plus_c,'- ',temp_minus_c)


#print(planet_mass_sini(1.48,27.56,0,0.12)*317.8)

def amplitude_from_mass(M,P,ecc,Mstar): # in MEarth units
    M = M/317.8
    ecc_part = np.sqrt(1-ecc**2)
    period_part = P**(1./3)
    m_star_part = Mstar**(2./3)

    amplitude = M/(4.919e-3*ecc_part*period_part*m_star_part)
    return amplitude

print(amplitude_from_mass(10,600,0.0,1))
# print(amplitude_from_mass(1,186,0.0,0.789))
# print(amplitude_from_mass(1,439,0.0,0.789))
# print(amplitude_from_mass(1,474,0.0,0.789))
# print(amplitude_from_mass(1,122,0.0,0.789))
# print(amplitude_from_mass(1,195,0.0,0.837))
# print(amplitude_from_mass(1,294,0.0,0.837))
# print(amplitude_from_mass(1,668,0.0,0.837))
# print(amplitude_from_mass(1,719,0.0,0.837))
def amplitude_from_mass_2(M,P,ecc,Mstar): # in MEarth units
    
    ecc_part = np.sqrt(1-ecc**2)
    period_part = (P/365.242199)**(-1./3)
    m_star_part = Mstar**(-2./3)

    amplitude = 28.4329*(M/317.8)*(1/ecc_part)*period_part*m_star_part
    return amplitude

# exit()
def planet_mass_err(K,P,Ecc,Mstar,kerr,perr,eccerr,mstarerr):
    
    ecc_part = np.sqrt(1-Ecc**2)
    period_part = P**(1./3)
    m_star_part = Mstar**(2./3)
    
    mass_error = np.sqrt((4.919e-3*ecc_part*period_part*m_star_part*kerr)**2 + ((2./3)*4.919e-3*K*ecc_part*period_part*Mstar**(-1./3)*mstarerr)**2 + ((1./3)*4.919e-3*K*ecc_part*m_star_part*P**(-2./3)*perr)**2 - (4.919e-3*K*ecc_part*period_part*m_star_part*(-Ecc/np.sqrt(1-Ecc**2))*eccerr)**2)
    
    return mass_error

def planet_distance(P,Mstar):
    
    p_year = P/365.25
    period_part = p_year**(2./3)
    star_part = Mstar**(1./3)
    planet_dista = period_part*star_part
    
    return planet_dista

def minor_axis(a, e):
    return a * np.sqrt(1 - e**2)

# print(minor_axis(1,0.55))
# exit()

def planet_distance_err(P,Mstar,perr,mstarerr):
    
    planet_distance_error = np.sqrt(((2./3)*(P/365.25)**(-1./3)*Mstar**(1./3)*(perr/365.25))**2+ ((P/365.25)**(2./3)*(1./3)*(Mstar)**(-2./3)*mstarerr)**2)
    
    return planet_distance_error
#+ ((P/365.25)**(2./3)*(1./3)*(Mstar)**(-2./3)*mstarrerr)**2

''' Calculation of Habitable Zone '''

def d_recent_venus(Teff,L):  #inner habitable zone (optimistic value)
    
    S0 = 1.7763
    a = 1.4335e-4
    b = 3.3954e-9
    c = -7.63642e-12
    d = -1.1950e-15
    Tstar = Teff - 5780
    
    Seff = S0 + a*Tstar + b*(Tstar**2) + c*(Tstar**3) + d*(Tstar**4)
    hz_recent_venus = np.sqrt(L/Seff)
    return hz_recent_venus

def d_runaway_greenhouse(Teff,L): #inner habitable zone
    
    S0 = 1.0385
    a = 1.2456e-4
    b = 1.4612e-8
    c = -7.6364e-12
    d = -1.7511e-15
    Tstar = Teff - 5780
    
    Seff = S0 + a*Tstar + b*(Tstar**2) + c*(Tstar**3) + d*(Tstar**4)
    hz_recent_venus = np.sqrt(L/Seff)
    return hz_recent_venus
    
#print(d_recent_venus(5400,0.91),d_runaway_greenhouse(5400,0.91))
def d_maximum_greenhouse(Teff,L): #outer habitable zone
    
    S0 = 0.3507
    a = 5.9578e-5
    b = 1.6707e-9
    c = -2.1709e-12
    d = -3.8282e-16
    Tstar = Teff - 5780
    
    Seff = S0 + a*Tstar + b*(Tstar**2) + c*(Tstar**3) + d*(Tstar**4)
    hz_recent_venus = np.sqrt(L/Seff)
    return hz_recent_venus
    
def d_early_mars(Teff,L): #outer habitable zone (optimistic value)
    
    S0 = 0.3179
    a = 5.4513e-5
    b = 1.5313e-9
    c = -2.7786e-12
    d = -4.8997e-16
    Tstar = Teff - 5780
    
    Seff = S0 + a*Tstar + b*(Tstar**2) + c*(Tstar**3) + d*(Tstar**4)
    hz_recent_venus = np.sqrt(L/Seff)
    return hz_recent_venus


t = np.random.normal(loc=5610, scale=50, size=1000000)
l = np.random.normal(loc=0.6769, scale=0.0015, size=1000000)
mstar = np.random.normal(loc=0.837, scale=0.029, size=1000000)
rec_ven = d_recent_venus(t,l)
run_green = d_runaway_greenhouse(t,l)
max_green = d_maximum_greenhouse(t,l)
ear_mars = d_early_mars(t,l)

# rv_med = np.percentile(rec_ven,50)
# rv_plus = np.percentile(rec_ven,84) - rv_med
# rv_minus = rv_med - np.percentile(rec_ven,16)

# rg_med = np.percentile(run_green,50)
# rg_plus = np.percentile(run_green,84) - rg_med
# rg_minus = rg_med - np.percentile(run_green,16)

# mg_med = np.percentile(max_green,50)
# mg_plus = np.percentile(max_green,84) - mg_med
# mg_minus = mg_med - np.percentile(max_green,16)

# em_med = np.percentile(ear_mars,50)
# em_plus = np.percentile(ear_mars,84) - em_med
# em_minus = em_med - np.percentile(ear_mars,16)

# print('Recent Venus HZ limit: ',rv_med,'+/-',rv_plus,'- ',rv_minus)
# print('Runaway Greenhouse HZ limit: ',rg_med,'+/-',rg_plus,'- ',rg_minus)
# print('Maximum Greenhouse HZ limit: ',mg_med,'+/-',mg_plus,'- ',mg_minus)
# print('Early Mars HZ limit: ',em_med,'+/-',em_plus,'- ',em_minus)

''' Period and linear and quadratic acceleration '''
    
def period(axis,Mstar):
    p = (axis**1.5)*(Mstar**(-0.5)) # axis in AU and Mstar in solar masses
    p *= 365.25
    return p

#print(period(0.2,1.0))
period_rv = period(rec_ven,mstar)
period_rg = period(run_green,mstar)
period_mg = period(max_green,mstar)
period_em = period(ear_mars,mstar)

prv_med = np.percentile(period_rv,50)
prv_plus = np.percentile(period_rv,84) - prv_med
prv_minus = prv_med - np.percentile(period_rv,16)

prg_med = np.percentile(period_rg,50)
prg_plus = np.percentile(period_rg,84) - prg_med
prg_minus = prg_med - np.percentile(period_rg,16)

pmg_med = np.percentile(period_mg,50)
pmg_plus = np.percentile(period_mg,84) - pmg_med
pmg_minus = pmg_med - np.percentile(period_mg,16)

pem_med = np.percentile(period_em,50)
pem_plus = np.percentile(period_em,84) - pem_med
pem_minus = pem_med - np.percentile(period_em,16)

# print('Period at Recent Venus HZ limit: ',prv_med,'+/-',prv_plus,'- ',prv_minus)
# print('Period at Runaway Greenhouse HZ limit: ',prg_med,'+/-',prg_plus,'- ',prg_minus)
# print('Period at Maximum Greenhouse HZ limit: ',pmg_med,'+/-',pmg_plus,'- ',pmg_minus)
# print('Period at Early Mars HZ limit: ',pem_med,'+/-',pem_plus,'- ',pem_minus)
# exit()
#print(period(0.73,0.91),period(0.96,0.91))
def trend(accel,timespan):
    K = accel*timespan
    
    K = accel*timespan
    P = 4*timespan/365.25
        
    return K,P

#x = trend(0.012085,7200)
#print x

def trend_2(accel_2,K):
    P = 2*np.pi*np.sqrt(K/accel_2)
    return P/365.25

def RHK(s_index,s_sigma,BV,FE_H):
    log10_ccf =  0.668*(BV*3) - 1.270*(BV**2) - 0.645*(BV) - 0.443
    R_phot = 1.48*0.0001*np.exp(-4.3658*BV)
    ccf = 10**log10_ccf
    rhk = 1.34*0.0001*ccf*s_index - R_phot
    metallicity_correction = (1 + (368*FE_H)/(9114 - 6827*BV +2638*(BV**2)))**4
    rhk = rhk*metallicity_correction
    log10_rhk = np.log10(rhk)
    sigma_rhk = np.abs((np.log10(2.718281828459045)/rhk)*1.34*0.0001*ccf*s_sigma) # Derivative is not on ln but in log10
    return log10_rhk,sigma_rhk
#x = trend_2(0.000005,71.5)
#print x

def RHK_2(s_index,s_sigma,BV,FE_H):
    log10_ccf =  0.668*(BV*3) - 1.270*(BV**2) - 0.645*(BV) - 0.443
    R_phot = 1.48*0.0001*np.exp(-4.3658*BV)
    ccf = 10**log10_ccf
    rhk = 1.34*0.0001*ccf*s_index - R_phot
    metallicity_correction = (1 + (368*FE_H)/(9114 - 6827*BV +2638*(BV**2)))**4
    rhk = rhk*metallicity_correction
    log10_rhk = np.log10(rhk)
    sigma_rhk = np.abs((np.log10(2.718281828459045)/rhk)*1.34*0.0001*ccf*s_sigma) # Derivative is not on ln but in log10
    return log10_rhk,sigma_rhk

def planet_radius(M): # rho > 3.3 gcm-3
    return 1.03*(M**0.29)

def planet_radius_low(M): # rho < 3.3 gcm-3
    return 0.70*M**0.63

def planet_radius2(M): # Two empirical regimes of the planetary mass-radius relation Dolev Bashi
    return M**0.55

#print(planet_radius(4))
rb = planet_radius(4.7)
rb_low = planet_radius_low(4.7)
rc = planet_radius(10.2)
rc_low = planet_radius_low(10.2)

#print(rb,rb_low,rc,rc_low)

# pdb.set_trace()
def planet_to_star_contrast(rp,a,albedo):
    rp = rp*(6371/149597870)
    return (1/np.pi)*albedo*(rp/a)**2


# print(planet_to_star_contrast(1.61,0.3209,0.3),planet_to_star_contrast(1.86,0.3209,0.3))
# print(planet_to_star_contrast(2.02,1.429,0.3),planet_to_star_contrast(3.02,1.429,0.3))


def angular_separation(dist,DIST): # dist in AU, DIST in pc
    ipo = DIST*206264.806
    alfa = dist/ipo
    mas = 1000*3600*alfa*180/np.pi
    return mas

# print(angular_separation(1.42,29.75),angular_separation(0.3209,29.75))

# print(planet_distance(700,0.95))
# print(angular_separation(planet_distance(700,0.95),29.791))
# print(planet_to_star_contrast(planet_radius_low(21.2),planet_distance(700,0.95),0.3))
# print(d_early_mars(5300,0.9))
# print(d_recent_venus(5300,0.9))
# print(planet_distance(240,0.9))
# print(angular_separation(planet_distance(240,0.9),27.8))
# print(planet_to_star_contrast(planet_radius(6.7),planet_distance(240,0.9),0.3))

#2460584 2460675
def transit_time_calculator(t0,t0_err,p,p_err,time):
    T0 = np.random.normal(loc=t0, scale=t0_err, size=1000000)
    P = np.random.normal(loc=p, scale=p_err, size=1000000)
    TT = []
    for i in range(100):
        TT.append(T0 + i*P)
    tt = []
    for i in range(len(TT)):
        tt.append(np.percentile(TT[i],[16,50,84],axis=0))
    tt = np.array(tt)
    t = tt[:,1] + 7000
    t_err = (tt[:,2] - tt[:,0])/2.0

    cond_t1 = time[0] - 2450000
    cond_t2 = time[1] - 2450000
    ind_ok = np.where((cond_t1 < t) & (t < cond_t2))[0]
    t = t[ind_ok] + 2450000
    t_err = t_err[ind_ok]
    print(t,t_err)
    # TT = T0 + P*n
    # tt = np.percentile(TT,[16,50,84],axis=0)
    # t = tt[1] + 7000
    # t_err = (tt[1] - tt[0])
    return 

def planet_density(mass,radius):
    volume = (4/3)*np.pi*((radius*6371e3)*100)**3  # radius in Earth radii converted to centimeters
    density = (mass*5.972e27)/volume  # mass in Earth masses converted to grams
    return density  # in kg/m^3

def eclipse(distance):
    R_sun = 6.957e8  # in meters
    distance_sun = 1.496e11
    angular_size_sun = distance_sun / R_sun* (206264806.24709636) # in milliarcseconds
    print(angular_size_sun)
    distance_m = distance *  1.496e11 # Convert distance from parsecs to meters
    radius = distance_m * (206264806.24709636)/angular_size_sun
    radius = radius/6.371e6  # Convert radius from meters to Earth radii
    return radius

#transit_time_calculator(823.52981042,0.00731904,40.67196145,0.00807572,[2460584,2460675])
# Hirano
# transit_time_calculator(782.8324,0.0048,40.6835,0.00307572,[2460584,2460675])
# # Noi
# transit_time_calculator(782.85549002,0.00664657,40.67420265,0.00455574,[2460584,2460675])