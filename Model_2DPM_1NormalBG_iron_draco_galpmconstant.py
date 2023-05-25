#!/usr/bin/env python
# coding: utf-8

# # Iron Draco
# This notebook presents the mixture model of 3 gaussians built for Iron Draco data. The data is taken from the S5 Collaboration. With quality cut, we obtained 371 stars with good measurements to feed the model. The mixture model is built with 16 parameters, including radial velocity, metallicity and proper motion parameters of the smcnod and a set of parameters for the background components. We fit a Gaussian mixture model to this data using `emcee`.

# In[1]:


# Imports
get_ipython().run_line_magic('matplotlib', 'inline')
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from scipy import optimize, stats
from astropy.table import Table
import emcee
import corner
from collections import OrderedDict
import time
from astropy import table 
from astropy.io import ascii
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from scipy.signal import find_peaks
#import uncertainties.umath as um
#from uncertainties import ufloat
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 16}

plt.rc('font', **font)
import imp
from astropy.io import fits as pyfits
import pandas as pd


# ## Iron Data Loading

# In[2]:


#data loading for DESI iron

ironrv = t1 = table.Table(pyfits.open('/raid/DESI/catalogs/iron/rvtab-hpxcoadd-all.fits')[1].data)
t1_fiber = table.Table(pyfits.open('/raid/DESI/catalogs/iron/rvtab-hpxcoadd-all.fits')[2].data)
t4 = table.Table(pyfits.open('/raid/DESI/catalogs/iron/rvtab-hpxcoadd-all.fits')[4].data)
t1_comb = table.hstack((t1,t1_fiber,t4))


# In[3]:


#isochrone loading with a age = 10 Gyr 
#Properties for the isochrone 
#MIX-LEN  Y      Z          Zeff        [Fe/H] [a/Fe]
# 1.9380  0.2459 5.4651E-04 5.4651E-04  -1.50   0.00 
iso_file = pd.read_csv('./draco_files/isochrone_10_1.csv')


# # Colorcut for better selection

# In[4]:


testiron= t1_comb


# In[5]:


#dust extinction correction
testiron['gmag'], testiron['rmag'], testiron['zmag'] = [22.5-2.5*np.log10(testiron['FLUX_'+_]) for _ in 'GRZ']

testiron['gmag0'] = testiron['gmag'] - testiron['EBV_2'] * 3.186
testiron['rmag0'] = testiron['rmag'] - testiron['EBV_2'] * 2.140
testiron['zmag0'] =testiron['zmag'] - testiron['EBV_2'] * 1.196
testiron['gmagerr'], testiron['rmagerr'], testiron['zmagerr'] = [2.5/np.log(10)*(np.sqrt(1./testiron['FLUX_IVAR_'+_])/testiron['FLUX_'+_]) for _ in 'GRZ']


# In[6]:


#error in the r mag
xdata = testiron['rmag']
ydata = np.log10(testiron['rmagerr'])
def betw(x,x1,x2): return (x>x1)&(x<x2)
xnew = xdata[betw(xdata, 15,24)&betw(ydata,-4,0)]
ynew = ydata[betw(xdata, 15,24)&betw(ydata,-4,0)]
#plotting stream regions
plt.plot(xnew, ynew,'.',alpha=0.01)

def log10_error_func(x, a, b):
    return a * x + b

from scipy.optimize import curve_fit
popt, pcov = curve_fit(log10_error_func, xnew, ynew)

xdata = np.linspace(15,24,100)
plt.plot(xdata, log10_error_func(xdata, *popt))
plt.xlabel('rmag')
plt.ylabel('log10(rmagerr)')


# In[7]:


#quality cut, exclude nans and RA/DEC cut
iqk,=np.where((testiron['TARGET_RA_1'] > 257) & (testiron['TARGET_RA_1'] < 263) & (testiron['TARGET_DEC_1'] > 55.9) & (testiron['TARGET_DEC_1'] < 59.9) & (testiron['RVS_WARN']==0) &(testiron['RR_SPECTYPE']!='QSO')&(testiron['VSINI']<50)
             
      &(~np.isnan(testiron["PMRA_ERROR"])) &(~np.isnan(testiron["PMDEC_ERROR"])) &(~np.isnan(testiron["PMRA_PMDEC_CORR"]))     )






# In[8]:


#making CMD diagram for the data
ra0 = 260.05972917
dec0 =  57.92121944
rad0 =1.6

stars = SkyCoord(ra=testiron['TARGET_RA_1'], dec=testiron['TARGET_DEC_1'], unit=u.deg)

# Calculate the angular separation between stars and the reference point
separations = stars.separation(SkyCoord(ra=ra0, dec=dec0, unit=u.deg))

testiron['dist1'] = separations


# In[9]:


ind = (testiron['dist1'][iqk] <0.3)
dm=19.53


# In[10]:


#making cut for the horizontal branch
dm_m92_harris = 14.59
m92ebv = 0.023
m92ag = m92ebv * 3.184
m92ar = m92ebv * 2.130
m92_hb_r = np.array([20.5, 20.8, 20.38, 20.2, 19.9, 19.8])
m92_hb_col = np.array([-0.25, -0.15, -0., 0.15, 0.25,0.33])
m92_hb_g = m92_hb_r + m92_hb_col
des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
des_m92_hb_g = des_m92_hb_g- dm_m92_harris
des_m92_hb_r = des_m92_hb_r  - dm_m92_harris


# In[11]:


dm_m92_harris = 14.59
m92ebv = 0.023
m92ag = m92ebv * 3.184
m92ar = m92ebv * 2.130
m92_hb_r = np.array([17.3, 15.8, 15.38, 15.1, 15.05, 15.0,14.95,14.9])
m92_hb_col = np.array([-0.39, -0.3, -0.2, -0.0, 0.1,0.2,0.3,0.4])
m92_hb_g = m92_hb_r + m92_hb_col
des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
des_m92_hb_g = des_m92_hb_g - m92ag - dm_m92_harris
des_m92_hb_r = des_m92_hb_r - m92ar - dm_m92_harris


# In[12]:


#isochrone on the sample with a radius =0.3 degree from the draco gal center 
plt.scatter(testiron['gmag0'][iqk][ind]-testiron['rmag0'][iqk][ind],testiron['rmag0'][iqk][ind],s=3)
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='orange',lw=2)

plt.ylim(21, 16)
plt.xlim(-0.4,1.8)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# In[13]:


# making actual selections/cuts -- CMD cut and color-color cut

# CMD cut for RGB

def cmd_selection(t, dm, dotter_g, dotter_r, gw=0.5):
    '''
    function for making the CMD cut 
    :param t datafile: data input
    :param dm float: distance modulus 
    :param dotter_g: isochrone g mag
    :param dotter_r: isochrone r mag
    :param gw: width in the CMD selection
    :return: color cut index 
    '''
    grw = np.sqrt(0.1**2 + (3*10**log10_error_func(dotter_r+dm, *popt))**2)
    gw = gw # RGB width in g
    rmin = 16
    rmax = 23
    grmin = -0.4
    grmax = 1.6
    magrange = (t['rmag'] > rmin) & (t['rmag'] < rmax) & (t['gmag0'] - t['rmag0'] < grmax) & (t['gmag0'] - t['rmag0'] > grmin)
    gr = t['gmag0'] - t['rmag0']
    grmax1 = np.interp(t['rmag0'], dotter_r[::-1] + dm, dotter_g[::-1] - dotter_r[::-1]+grw[::-1], left=np.nan, right=np.nan)
    grmax2 = np.interp(t['rmag0'], dotter_r[::-1] + dm + gw, dotter_g[::-1] - dotter_r[::-1]+grw[::-1], left=np.nan, right=np.nan)
    grmax3 = np.interp(t['rmag0'], dotter_r[::-1] + dm - gw, dotter_g[::-1] - dotter_r[::-1]+grw[::-1], left=np.nan, right=np.nan)
    grmax = np.max(np.array([grmax1, grmax2, grmax3]), axis=0)
    grmin1 = np.interp(t['rmag0'], dotter_r[::-1] + dm, dotter_g[::-1] - dotter_r[::-1]-grw[::-1], left=np.nan, right=np.nan)
    grmin2 = np.interp(t['rmag0'], dotter_r[::-1] + dm - gw, dotter_g[::-1] - dotter_r[::-1]-grw[::-1], left=np.nan, right=np.nan)
    grmin3 = np.interp(t['rmag0'], dotter_r[::-1] + dm + gw, dotter_g[::-1] - dotter_r[::-1]-grw[::-1], left=np.nan, right=np.nan)
    grmin = np.min(np.array([grmin1, grmin2, grmin3]), axis=0)
    colorsel = (gr < grmax) & (gr > grmin)
    colorrange = magrange & colorsel

    # CMD cut for BHB
    grw_bhb = 0.3 # BHB width in gr
    gw_bhb = 0.5  # BHB width in g
    grmin_bhb = -0.45
    grmax_bhb = 0.4
    magrange_bhb = (t['rmag'] > rmin) & (t['rmag'] < rmax) & (t['gmag0'] - t['rmag0'] < grmax_bhb) & (t['gmag0'] - t['rmag0'] > grmin_bhb)

    gr_bhb = np.interp(t['rmag0'], des_m92_hb_r[::-1] + dm , des_m92_hb_g[::-1] - des_m92_hb_r[::-1], left=np.nan, right=np.nan)
    rr_bhb = np.interp(t['gmag0'] - t['rmag0'], des_m92_hb_g - des_m92_hb_r, des_m92_hb_r + dm,left=np.nan, right=np.nan)
    del_color_cmd_bhb = t['gmag0'] - t['rmag0'] - gr_bhb
    del_g_cmd_bhb = t['rmag0'] - rr_bhb
    colorrange_bhb = magrange_bhb & ((abs(del_color_cmd_bhb) < grw_bhb) | (abs(del_g_cmd_bhb) < gw_bhb))

    colorrange = colorrange | colorrange_bhb
    
    return colorrange




# In[14]:


colorcut = cmd_selection(testiron[iqk], dm, iso_file['DECam_g'],iso_file['DECam_r'], gw=1.0)


# In[15]:


#colorcut for the sample 
#Investigating the data sample after colorcut 


plt.figure(figsize=(10,6))
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='orange',lw=2,label = 'Isochrone')
plt.scatter(testiron['gmag0'][iqk]-testiron['rmag0'][iqk],testiron['rmag0'][iqk],s=1,c='k',alpha=0.5,label ='Iron Data')

grw = np.sqrt(0.1**2 + (3*10**log10_error_func(iso_file['DECam_r']+dm, *popt))**2)



#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_file['DECam_r']+dm,'--r')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm,'--r')
plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_file['DECam_r']+dm+1,'--k',label='Colorcut Region')
plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm-1,'--k')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_file['DECam_r']+dm-1,'--k')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm+1,'--k')

plt.plot(des_m92_hb_g-des_m92_hb_r, des_m92_hb_r+dm, lw=2, color='orange')
plt.plot(des_m92_hb_g-des_m92_hb_r, des_m92_hb_r+dm-0.4,'--k')
plt.plot(des_m92_hb_g-des_m92_hb_r, des_m92_hb_r+dm+0.4,'--k')
#plt.plot(des_m92_hb_g-des_m92_hb_r-0.1, des_m92_hb_r+dm,'--r')
#plt.plot(des_m92_hb_g-des_m92_hb_r+0.1, des_m92_hb_r+dm,'--r')
plt.legend()
plt.ylim(21, 16)
plt.xlim(-0.3,1.5)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# # Input data profile (FeH, radial velocity, pmra, pmdec)

# In[16]:


iqk,=np.where((testiron['TARGET_RA_1'] > 257) & (testiron['TARGET_RA_1'] < 263) & (testiron['TARGET_DEC_1'] > 55.9) & (testiron['TARGET_DEC_1'] < 59.9) & (testiron['RVS_WARN']==0) &(testiron['RR_SPECTYPE']!='QSO')&(testiron['VSINI']<50)
             
      &(~np.isnan(testiron["PMRA_ERROR"])) &(~np.isnan(testiron["PMDEC_ERROR"])) &(~np.isnan(testiron["PMRA_PMDEC_CORR"])) )




# In[17]:


#radial velocity 
up = -150
low = -500
vtest = testiron["VRAD"][iqk][colorcut]
vcut  = (vtest > low) & (vtest  < up)
rv = testiron["VRAD"][iqk][colorcut][vcut]
rverr = testiron["VRAD_ERR"][iqk][colorcut][vcut]
# metallicity
feh = testiron["FEH"][iqk][colorcut][vcut]
feherr = testiron["FEH_ERR"][iqk][colorcut][vcut]
# proper motions
pmra = testiron["PMRA_3"][iqk][colorcut][vcut]
pmraerr = testiron["PMRA_ERROR"][iqk][colorcut][vcut]
pmdec = testiron["PMDEC_3"][iqk][colorcut][vcut]
pmdecerr = testiron["PMDEC_ERROR"][iqk][colorcut][vcut]


# In[18]:


#vrad,feh,pm distribution

fig, axes = plt.subplots(2,2,figsize=(9,9))
axes[0,0].hist(rv, bins='auto');
axes[0,0].set_xlabel("vhel")
axes[0,1].hist(feh, bins='auto');
axes[0,1].set_xlabel("[Fe/H]")
axes[1,0].hist(pmra, bins='auto');
axes[1,0].set_xlabel("pmra")
axes[1,1].hist(pmdec, bins='auto');
axes[1,1].set_xlabel("pmdec")


# In the given code, pmnorm is defined as 1/(np.pi * pmmax**2), where pmmax is the magnitude of the maximum proper motion vector. This expression is used to normalize the proper motion likelihood function for the entire data set.
# 
# In more detail, the proper motion likelihood function represents the probability of observing a particular proper motion vector for a star, given its position on the sky and any other relevant information. In this code, a uniform background distribution is assumed, which means that the likelihood of observing any particular proper motion vector is assumed to be constant across the entire sky.
# 
# To normalize the proper motion likelihood function, the maximum value of the proper motion vector magnitude is computed (pmmax), and the inverse of the product of pi and the square of pmmax is taken (1/(np.pi * pmmax**2)). This value (pmnorm) is then used to scale the proper motion likelihood function so that its integral over the entire sky is equal to one. This ensures that the probability of observing any proper motion vector is properly normalized, given the assumptions made in the analysis.

# We will model the smcnod data as a mixture of 2 gaussians. The parameters will be:
# 
# * pgal = fraction of stars in the galaxy
# * pmra = Heliocentric proper motion, RA of the galaxy in mas/yr
# * pmdec = Heliocentric proper motion, Dec of the galaxy in mas/yr
# * vhel = mean velocity of the galaxy in km/s
# * lsigv = log10 the velocity dispersion of the galaxy in km/s
# * feh = mean metallicity of the galaxy in dex
# * lsigfeh = log10 the metallicity dispersion of the galaxy in dex
# * vbg1, lsigvbg1, fehbg1, lsigfeh1 = same parameters for 1st background component

# In[19]:


def data_collect(datafile,ramin,ramax,decmin,decmax,fehmin,fehmax,vmin,vmax,galdis,iso_file,dm,gw):
    '''
    function for collecting all the data input for the likelihood (you can change ra,dec,feh,colorcut range in the above notebook in this function)
    :param datafile: data table
    :param ramin,ramax float: ra range
    :param decmin,decmax float: dec range
    :param fehmin,fehmax float: metallicity range
    :param vmin,vmax: radial velocity range 
    :param galdis: gal proper motion dispersion 
    :param iso_file: isochrone file
    :param dm: distance modulus
    :param gw: width for the colorcut
    :return: rv,rverr,feh,feherr and the proper motion data
    Note that a constant (0.025) term is added to the diagonal terms of the covariance for the proper motion data 
    to account for the gal dispersion  
    '''
    
    #ra,dec, feh and quality cut
    iqk,=np.where((datafile['TARGET_RA_1'] > ramin) & (datafile['TARGET_RA_1'] < ramax) & (datafile['TARGET_DEC_1'] > decmin) & (datafile['TARGET_DEC_1'] < decmax) & (datafile['RVS_WARN']==0) &(datafile['RR_SPECTYPE']!='QSO')&(datafile['VSINI']<50)
             
      &(~np.isnan(datafile["PMRA_ERROR"])) &(~np.isnan(datafile["PMDEC_ERROR"])) &(~np.isnan(datafile["PMRA_PMDEC_CORR"])) &(datafile["FEH"] >fehmin)&(datafile["FEH"] <fehmax))
    
    colorcut = cmd_selection(testiron[iqk], dm, iso_file['DECam_g'],iso_file['DECam_r'], gw=1.0)
    up = vmin
    low = vmax
    vtest = datafile["VRAD"][iqk][colorcut]
    vcut  = (vtest > low) & (vtest  < up)
    rv = datafile["VRAD"][iqk][colorcut][vcut]
    rverr = datafile["VRAD_ERR"][iqk][colorcut][vcut]
# metallicity
    feh = datafile["FEH"][iqk][colorcut][vcut]
    feherr = datafile["FEH_ERR"][iqk][colorcut][vcut]
# proper motions
    pmra = datafile["PMRA_3"][iqk][colorcut][vcut]
    pmraerr = datafile["PMRA_ERROR"][iqk][colorcut][vcut]
    pmdec = datafile["PMDEC_3"][iqk][colorcut][vcut]
    pmdecerr = datafile["PMDEC_ERROR"][iqk][colorcut][vcut]
    
    N = len(rv)
    datacut = datafile[iqk][colorcut][vcut]
# Create 2-D arrays for proper motion
    pms = np.zeros((N,2)) 
    pms[:,0] = testiron["PMRA_3"][iqk][colorcut][vcut]
    pms[:,1] = testiron["PMDEC_3"][iqk][colorcut][vcut]

# pms array is computed and assigned to the variable pmmax. 
# This is essentially finding the magnitude of the maximum proper motion vector.
    pmmax = np.max(np.sqrt(np.sum(pms**2, axis=1)))
# normalize the proper motion likelihood function for the entire data set
    pmnorm = 1/(np.pi * pmmax**2)
# Covariance Matrix for gal
    pmcovs = np.zeros((N,2,2))
    
    pmcovs[:,0,0] = testiron["PMRA_ERROR"][iqk][colorcut][vcut]**2+galdis**2
    pmcovs[:,1,1] = testiron["PMDEC_ERROR"][iqk][colorcut][vcut]**2+galdis**2
    pmcovs[:,0,1] = testiron["PMRA_ERROR"][iqk][colorcut][vcut]*testiron["PMDEC_ERROR"][iqk][colorcut][vcut]*testiron["PMRA_PMDEC_CORR"][iqk][colorcut][vcut]
    pmcovs[:,1,0] = testiron["PMRA_ERROR"][iqk][colorcut][vcut]*testiron["PMDEC_ERROR"][iqk][colorcut][vcut]*testiron["PMRA_PMDEC_CORR"][iqk][colorcut][vcut]


    return [rv,rverr,feh,feherr,pms,pmcovs],datacut


    


    


# In[20]:


datasum,datacut =data_collect(testiron,257,263,55.9,59.9,-3.9,0,-150,-400,0.025,iso_file,dm,1.0)


# # Likelihood function

# In[40]:


param_labels = ["pgal",
                "vhel","lsigv","feh","lsigfeh",
                "vbg1","lsigvbg1","fehbg1","lsigfeh1",
                "pmra","pmdec",
                "pmra1","pmdec1","lsigpmra1","lsigpmdec1"]
def full_like(theta):
    
    """ Likelihood and Prior """

    '''
    function for calculating the likelihood function 
    :param theta: parameters in the likelihood function 
    :return: final log likelihood
    Note that a constant (0.025) term is added to the diagonal terms of the covariance for the proper motion data 
    to account for the gal dispersion, so for the background pm dispersion, we need to first subtract the constant term,
    then add the parameter lsigpmra1 and lsigpmdec1 to the diagonal term for the background pm likelihood. 
    '''
    
    pgal, \
    vhel, lsigv, feh0, lsigfeh, \
    vbg1, lsigvbg1, fehbg1, lsigfeh1, \
    pmra_gal, pmdec_gal, pmra1, pmdec1, lsigpmra1, lsigpmdec1 = theta
    rv,rverr,feh,feherr,pms,pmcovs=datasum
    
    
    #data input 
    galdis=0.025
    N = len(rv)
    pm0s = np.zeros((N,2))
    pm0s[:,0] = pmra_gal
    pm0s[:,1] = pmdec_gal
    
    #pm mean for bg
    bgpm0s = np.zeros((N,2))
    bgpm0s[:,0] = pmra1
    bgpm0s[:,1] = pmdec1
    
    # Covariance Matrix for bg
    bgpmcovs = np.zeros((N,2,2))


    bgpmcovs[:,0,0] = pmcovs[:,0,0]+lsigpmra1**2-galdis**2
    bgpmcovs[:,1,1] =  pmcovs[:,1,1]+lsigpmdec1**2-galdis**2
    bgpmcovs[:,0,1] = pmcovs[:,0,1]
    bgpmcovs[:,1,0] = pmcovs[:,1,0]
    
    
    # The prior is just a bunch of hard cutoffs
    if (pgal > 1) or (pgal < 0) or \
        (lsigv > 3) or (lsigvbg1 > 3) or \
        (lsigv < -1) or (lsigvbg1 < -1) or \
        (lsigfeh > 1) or (lsigfeh1 > 1) or (lsigfeh1 > 1) or \
        (lsigfeh < -3) or (lsigfeh1 < -3) or (lsigfeh1 < -3) or \
        (vhel > 400) or (vhel < -400) or (vbg1 > 100) or (vbg1 < -300) or \
        (abs(pmra_gal) > 2) or (abs(pmdec_gal) > 2) or \
        (abs(pmra1) > 10) or (abs(pmdec1) > 10) or \
        (lsigpmra1 > 1.3) or (lsigpmra1 < -1) or \
        (lsigpmdec1 > 1.3) or (lsigpmdec1 < -1) :
        return -1e10
    
    # Compute log likelihood in rv
    lgal_vhel = stats.norm.logpdf(rv, loc=vhel, scale=np.sqrt(rverr**2 + (10**lsigv)**2))
    lbg1_vhel = stats.norm.logpdf(rv, loc=vbg1, scale=np.sqrt(rverr**2 + (10**lsigvbg1)**2))
    
    # Compute log likelihood in feh
    lgal_feh = stats.norm.logpdf(feh, loc=feh0, scale=np.sqrt(feherr**2 + (10**lsigfeh)**2))
    lbg1_feh = stats.norm.logpdf(feh, loc=fehbg1, scale=np.sqrt(feherr**2 + (10**lsigfeh1)**2))
    
    # Compute log likelihood in proper motions
    #for i in range(N):
        
    #using multivariat gaussian for the pm likelihood
    lgal_pm = [stats.multivariate_normal.logpdf(pms[i], mean=pm0s[i], cov=pmcovs[i]) for i in range(N)]
    lbg1_pm = [stats.multivariate_normal.logpdf(pms[i], mean=bgpm0s[i], cov=bgpmcovs[i]) for i in range(N)]
    #lbg1_pmdec = stats.norm.logpdf(pmdec, loc=pmdec1, scale=np.sqrt(pmdecerr**2 + (10**lsigpmdec1)**2))
    
    # Combine the components
    lgal = np.log(pgal)+lgal_vhel+lgal_pm+lgal_feh
    lbg1 = np.log(1-pgal)+lbg1_vhel+lbg1_pm+lbg1_feh
    ltot = np.logaddexp(lgal, lbg1)
    return ltot.sum()

def full_like_indi(theta,ii):
    
    """ Likelihood and Prior """

    '''
    function for calculating the posterior likelihood function for each star 
    :param theta: Best fitted parameters in the likelihood function 
    :param ii float: ith star
    :return: total posterior log likelihood for gal,background,  total posterior log likelihood and individual likelihood 
    for the vrad,pm, and Feh
    
    Note that a constant (0.025) term is added to the diagonal terms of the covariance for the proper motion data 
    to account for the gal dispersion, so for the background pm dispersion, we need to first subtract the constant term,
    then add the parameter lsigpmra1 and lsigpmdec1 to the diagonal term for the background pm likelihood. 
    '''
    
   
    pgal, \
    vhel, lsigv, feh0, lsigfeh, \
    vbg1, lsigvbg1, fehbg1, lsigfeh1, \
    pmra_gal, pmdec_gal, pmra1, pmdec1, lsigpmra1, lsigpmdec1 = theta
    rv,rverr,feh,feherr,pms,pmcovs=datasum
   
    galdis=0.025
    N = len(rv)
    pm0s = np.zeros((N,2))
    pm0s[:,0] = pmra_gal
    pm0s[:,1] = pmdec_gal
    
    
    #pm mean for bg
    bgpm0s = np.zeros((N,2))
    bgpm0s[:,0] = pmra1
    bgpm0s[:,1] = pmdec1
    
    # Covariance Matrix for bg
    bgpmcovs = np.zeros((N,2,2))


    bgpmcovs[:,0,0] = pmcovs[:,0,0]+lsigpmra1**2-galdis**2
    bgpmcovs[:,1,1] =  pmcovs[:,1,1]+lsigpmdec1**2-galdis**2
    bgpmcovs[:,0,1] = pmcovs[:,0,1]
    bgpmcovs[:,1,0] = pmcovs[:,1,0]
    
    
    # The prior is just a bunch of hard cutoffs
    if (pgal > 1) or (pgal < 0) or \
        (lsigv > 3) or (lsigvbg1 > 3) or \
        (lsigv < -1) or (lsigvbg1 < -1) or \
        (lsigfeh > 1) or (lsigfeh1 > 1) or (lsigfeh1 > 1) or \
        (lsigfeh < -3) or (lsigfeh1 < -3) or (lsigfeh1 < -3) or \
        (vhel > 400) or (vhel < -500) or (vbg1 > 100) or (vbg1 < -300) or \
        (abs(pmra_gal) > 2) or (abs(pmdec_gal) > 2) or \
        (abs(pmra1) > 10) or (abs(pmdec1) > 10) or \
        (lsigpmra1 > 1.3) or (lsigpmra1 < -1) or \
        (lsigpmdec1 > 1.3) or (lsigpmdec1 < -1) :
        return -1e10
    
    # Compute log likelihood in rv
    lgal_vhel = stats.norm.logpdf(rv[ii], loc=vhel, scale=np.sqrt(rverr[ii]**2 + (10**lsigv)**2))
    lbg1_vhel = stats.norm.logpdf(rv[ii], loc=vbg1, scale=np.sqrt(rverr[ii]**2 + (10**lsigvbg1)**2))
    
    # Compute log likelihood in feh
    lgal_feh = stats.norm.logpdf(feh[ii], loc=feh0, scale=np.sqrt(feherr[ii]**2 + (10**lsigfeh)**2))
    lbg1_feh = stats.norm.logpdf(feh[ii], loc=fehbg1, scale=np.sqrt(feherr[ii]**2 + (10**lsigfeh1)**2))
    
    # Compute log likelihood in proper motions
    #for i in range(N):
        
        #print (pms[i], "mean",pm0s[i], 'cov',pmcovs[i])
    lgal_pm = [stats.multivariate_normal.logpdf(pms[ii], mean=pm0s[ii], cov=pmcovs[ii]) ]
    lbg1_pm = [stats.multivariate_normal.logpdf(pms[ii], mean=bgpm0s[ii], cov=bgpmcovs[ii])]
    
    # Combine the components
    lgal = np.log(pgal)+lgal_vhel+lgal_pm+lgal_feh
    lbg1 = np.log(1-pgal)+lbg1_vhel+lbg1_pm+lbg1_feh
    
    ltot = np.logaddexp(lgal, lbg1)
    return  lgal,lbg1, ltot,[np.exp(lgal_vhel),np.exp(lgal_pm),np.exp(lgal_feh)]



def get_paramdict(theta):
    return OrderedDict(zip(param_labels, theta))


# In[41]:


def project_model(theta, p1min=-380, p1max=-150, p2min=-4, p2max=0.,key="vhel"):
    """ Turn parameters into p1 and p2 distributions """
    p1arr = np.linspace(p1min, p1max, 1000)
    p2arr = np.linspace(p2min, p2max, 1000)
    params = get_paramdict(theta)
    
    if key == 'vhel':
        p10 = params["pgal"]*stats.norm.pdf(p1arr, loc=params["vhel"], scale=10**params["lsigv"])
        p11 = (1-params["pgal"])*stats.norm.pdf(p1arr, loc=params["vbg1"], scale=10**params["lsigvbg1"])

        p20 = params["pgal"]*stats.norm.pdf(p2arr, loc=params["feh"], scale=10**params["lsigfeh"])
        p21 = (1-params["pgal"])*stats.norm.pdf(p2arr, loc=params["fehbg1"], scale=10**params["lsigfeh1"])
    else:
        p10 = params["pgal"]*stats.norm.pdf(p1arr, loc=params["pmra"], scale=10**0.025)
        p11 = (1-params["pgal"])*stats.norm.pdf(p1arr, loc=params["pmra1"], scale=10**params["lsigpmra1"])

        p20 = params["pgal"]*stats.norm.pdf(p2arr, loc=params["pmdec"], scale=10**0.025)
        p21 = (1-params["pgal"])*stats.norm.pdf(p2arr, loc=params["pmdec1"], scale=10**params["lsigpmdec1"])
        
    return p1arr, p10, p11, p2arr,p20,p21


# In[42]:


def plot_1d_distrs(theta,datasum,p1min=-380, p1max=-150, p2min=-4, p2max=0.,key="vhel"):
    '''
    function for plotting the likelihood distribution of two quantities p1 versus p2 for the gal/bg 
    :param theta: likelihood parameters (prior/posterior) 
    :param datasum: data table
    :param p1min,p1max float: p1 range
    :param p2min,p2max float: p2 range
    :param vmin,vmax: radial velocity range 
    :param key: key="vhel" for plotting vrad versus Feh / key="pmra" for plotting pmra versus pmdec 

    :return: plotting 

    '''
    colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    model_output = project_model(theta,p1min, p1max, p2min, p2max,key=key)
    fig, axes = plt.subplots(1,2,figsize=(10,5))
    if key == "vhel":
        ax = axes[0]
        ax.hist(datasum[0], density=True, color='grey', bins=200)
        xp, p0, p1 = model_output[0:3]
        ax.plot(xp, p0 + p1, 'k-', label="Total", lw=3)
        ax.plot(xp, p1, ':', color=colors[1], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="vhel (km/s)", ylabel="Prob. Density")
        ax.legend(fontsize='small')

        ax = axes[1]
        ax.hist(datasum[2], density=True, color='grey', bins='auto')
        xp, p0, p1 = model_output[3:6]
        ax.plot(xp, p0 + p1, 'k-', lw=3)
        ax.plot(xp, p1, ':', color=colors[1], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="[Fe/H] (dex)", ylabel="Prob. Density")
    else:
        ax = axes[0]
        ax.hist(datasum[-2][:,0], density=True, color='grey', bins=50)
        xp, p0, p1 = model_output[0:3]
        ax.plot(xp, p0 + p1, 'k-', label="Total", lw=3)
        ax.plot(xp, p1, ':', color=colors[1], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="PMRA", ylabel="Prob. Density")
        ax.legend(fontsize='small')

        ax = axes[1]
        ax.hist(datasum[-2][:,1], density=True, color='grey', bins='auto')
        xp, p0, p1 = model_output[3:6]
        ax.plot(xp, p0 + p1, 'k-', lw=3)
        ax.plot(xp, p1, ':', color=colors[1], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="PMDEC", ylabel="Prob. Density")
        
    return fig


# In[43]:


def plot_2d_distr(theta,datasum,key="vhel"):
    '''
    function for plotting the distribution of two quantities p1 versus p2 for the gal/bg 
    :param theta: likelihood parameters (prior/posterior) 
    :param datasum: data table
    :param key: key="vhel" for plotting vrad versus Feh / key="pmra" for plotting pmra versus pmdec 
    :return: plotting 

    '''
   
    fig, ax = plt.subplots(figsize=(10,5))
    if key == "vhel":
        ax.plot(datasum[2], datasum[0], 'k.')
        ax.set(xlabel="[Fe/H] (dex)", ylabel="vhel (km/s)", xlim=(-4,1), ylim=(-500,50))    
        params = get_paramdict(theta)
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ax.errorbar(params["feh"], params["vhel"],
               xerr=2*10**params["lsigfeh"], yerr=2*10**params["lsigv"],
               color=colors[0], marker='o', elinewidth=1, capsize=3, zorder=9999)
        ax.errorbar(params["fehbg1"], params["vbg1"],
               xerr=2*10**params["lsigfeh1"], yerr=2*10**params["lsigvbg1"],
               color=colors[1], marker='x', elinewidth=1, capsize=3, zorder=9999)
        ax.grid()
    else:
        ax.plot(datasum[-2][:,0], datasum[-2][:,1], 'k.')
        ax.set(xlabel="PMRA", ylabel="PMDEC", xlim=(-50,50), ylim=(-50,50))    
        params = get_paramdict(theta)
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ax.errorbar(params["pmra"], params["pmdec"],
               xerr=2*10**0.025, yerr=2*10**0.025,
               color=colors[0], marker='o', elinewidth=1, capsize=3, zorder=9999)
        ax.errorbar(params["pmra1"], params["pmdec1"],
              xerr=2*10**params["lsigpmra1"], yerr=2*10**params["lsigpmdec1"],
              color=colors[1], marker='x', elinewidth=1, capsize=3, zorder=9999)
        ax.grid()


# In[44]:


param_labels = ["pgal",
                "vhel","lsigv","feh","lsigfeh",
                "vbg1","lsigvbg1","fehbg1","lsigfeh1",
                "pmra","pmdec",
                "pmra1","pmdec1","lsigpmra1","lsigpmdec1"]


# # Optimize parameters
# 
# 

# In[45]:


## I found this guess by looking at the plot by eye and estimating. This part requires some futzing.
p0_guess = [0.47, 
            -290, 1.3, -1.5,-0.2,
            -200, 2.0, -0.8, -0.1,
            0.04, -1.18,
            0.07, -1, 1.2, 1.2]


# In[46]:


#vrad/Feh distribution

fig1 = plot_1d_distrs(p0_guess,datasum,p1min=-450, p1max=-150, p2min=-4, p2max=0.,key="vhel")
fig2 = plot_2d_distr(p0_guess,datasum,key="vhel")


# In[47]:


#pmra/pmdec distribution 
fig1 = plot_1d_distrs(p0_guess,datasum,p1min=-40, p1max=40, p2min=-20, p2max=20.,key="pmra")
fig2 = plot_2d_distr(p0_guess,datasum,key="pmra")


# In[48]:


#guess for the initial parameters
p0_guess


# In[49]:


optfunc = lambda theta: -full_like(theta)


# In[50]:


get_ipython().run_line_magic('timeit', 'optfunc(p0_guess)')


# In[51]:


optfunc(p0_guess)


# In[52]:


get_ipython().run_line_magic('time', 'res = optimize.minimize(optfunc, p0_guess, method="Nelder-Mead")')


# In[53]:


res.x


# In[54]:


optfunc(res.x)


# In[55]:


for label, p in zip(param_labels, res.x):
    print(f"{label}: {p:.3f}")


# In[56]:


fig1 = plot_1d_distrs(res.x,datasum,p1min=-400, p1max=-150, p2min=-4, p2max=0.,key="vhel")
fig2 = plot_2d_distr(res.x,datasum,key="vhel")


# ## Posterior Sampling
# The posterior is sampled using `emcee` with 64 walkers and 10,000 steps per chain.

# In[57]:


nw = 64
p0 = res['x']
nit = 2000
ep0 = np.zeros(len(p0_guess)) + 0.02
p0s = np.random.multivariate_normal(p0_guess, np.diag(ep0)**2, size=nw)
print(p0s)


# In[58]:


nparams = len(param_labels)
print(nparams)
nwalkers = 64
p0 = p0_guess
ep0 = np.zeros(len(p0)) + 0.02 # some arbitrary width that's pretty close; scale accordingly to your expectation of the uncertainty
p0s = np.random.multivariate_normal(p0, np.diag(ep0)**2, size=nwalkers)
## Check to see things are initialized ok
lkhds = [full_like(p0s[j]) for j in range(nwalkers)]
assert np.all(np.array(lkhds) > -9e9)


# In[ ]:


## Run emcee in parallel

from schwimmbad import MultiPool

nproc = 64 #use 32 cores
nit = 2000 

def get_rstate():
    return np.random.mtrand.RandomState(seed=np.random.randint(0,2**32-1))

with MultiPool(nproc) as pool:
    print("Running burnin with {} iterations".format(nit))
    start = time.time()
    es = emcee.EnsembleSampler(nw, len(p0_guess), full_like, pool=pool)
    PP = es.run_mcmc(p0s, nit, rstate0=get_rstate())
    print("Took {:.1f} seconds".format(time.time()-start))

    print(f"Now running the actual thing")
    es.reset()
    start = time.time()
    es.run_mcmc(PP.coords, nit, rstate0=get_rstate())
    print("Took {:.1f} seconds".format(time.time()-start))


# In[ ]:


outputs = es.flatchain


# ### Acceptance fraction
# Judging the convergence and performance of an algorithm is a non-trival problem. As a rule of thumb, the acceptance fraction should be between 0.2 and 0.5 (for example, Gelman, Roberts, & Gilks 1996).

# In[ ]:


# Another good test of whether or not the sampling went well is to 
# check the mean acceptance fraction of the ensemble
print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(es.acceptance_fraction)
    )
)


# In[ ]:


fig = corner.corner(outputs, labels=param_labels, quantiles=[0.16,0.50,0.84], show_titles=True,color='black', # add some colors
              **{'plot_datapoints': False, 'fill_contours': True})
#plt.savefig('SMCNOD_PM_Model_Cornerplot.png')


# In[ ]:


fig1 = corner.corner(outputs[:,1:3], labels=param_labels[1:3], quantiles=[0.16,0.50,0.84], show_titles=True,color='black', # add some colors
              **{'plot_datapoints': False, 'fill_contours': True})


# In[ ]:


fig2 = corner.corner(outputs[:,3:5], labels=param_labels[3:5], quantiles=[0.16,0.50,0.84], show_titles=True,color='black', # add some colors
              **{'plot_datapoints': False, 'fill_contours': True})


# In[ ]:


fig3 = corner.corner(outputs[:,9:11], labels=param_labels[9:11], quantiles=[0.16,0.50,0.84], show_titles=True,color='black', # add some colors
              **{'plot_datapoints': False, 'fill_contours': True})


# In[ ]:


def process_chain(chain, avg_error=True):
    pctl = np.percentile(chain, [16, 50, 84], axis=0)
    meds = pctl[1]
    ep = pctl[2]-pctl[1]
    em = pctl[0]-pctl[1]
    if avg_error: # just for simplicity, assuming no asymmetry
        err = (ep-em)/2
        return OrderedDict(zip(param_labels, meds)), OrderedDict(zip(param_labels, err))
    else:
        return OrderedDict(zip(param_labels, meds)), OrderedDict(zip(param_labels, ep)), OrderedDict(zip(param_labels, em))


# In[ ]:


meds, errs = process_chain(outputs)


# In[ ]:


for k,v in meds.items():
    print("{} {:.3f} {:.3f}".format(k, v, errs[k]))


# If things are well mixed, then you can just use the flat chain to concatenate all the walkers and steps.
# The results here may not be perfectly mixed, but it's not terrible.
# There are fancy ways to check things here involving autocorrelation times that Alex does not know about.
# To me this is the hard part of emcee: knowing when you're happy with the result, and setting things up so that it gets there as fast as possible. This is why I prefer dynesty, even though it's slower it has a motivated stopping condition.

# In[ ]:


chain = es.flatchain
chain.shape


# You can see the output of the fit as a corner plot. Basically you want everything to be nice and round, and if not that means you didn't initialize your walkers well enough or burn in for long enough.

# It's customary to summarize the data with percentiles, but you should check the corner plot diagonal to see if this is a good idea.

# In[ ]:


#plotting the posterior distribution for vral and FeH
fig1 = plot_1d_distrs(chain[1],datasum,p1min=-400, p1max=-150, p2min=-4, p2max=0.,key="vhel")
fig2 = plot_2d_distr(chain[1],datasum,key="vhel")


# In[ ]:


#plotting the posterior distribution for pmra and pmdec
fig1 = plot_1d_distrs(chain[1],datasum,p1min=-40, p1max=40, p2min=-40, p2max=40.,key="pmra")
fig2 = plot_2d_distr(chain[1],datasum,key="pmra")


# In[ ]:


chain_new = 10**(chain)
mean_vdisp = np.percentile(chain_new[:,2], 50)
std_vdisp = (np.percentile(chain_new[:,2], 84)-np.percentile(chain_new[:,2], 16))/2
mean_fehdisp = np.percentile(chain_new[:,4], 50)
std_fehdisp = (np.percentile(chain_new[:,4], 84)-np.percentile(chain_new[:,4], 16))/2
print("mean_vdisp: ",mean_vdisp, \
     "std_vdisp: ",std_vdisp)
print("mean_fehdisp: ",mean_fehdisp, \
     "std_fehdisp: ",std_fehdisp)


# In[ ]:


param_labels_1 = ["pgal =",
                "vhel =","lsigv =","feh =","lsigfeh =",
                "vbg1 =","lsigvbg1 =","fehbg1 =","lsigfeh1 =",
                "pmra =","pmdec =",
                "pmra1 =","pmdec1 =","lsigpmra1 =","lsigpmdec1 ="]

def process_chain_1(chain, avg_error=True):
    pctl = np.percentile(chain, [16, 50, 84], axis=0)
    median = pctl[1]
    ep = pctl[2]-pctl[1]
    em = pctl[0]-pctl[1]
    if avg_error:
        err = (ep-em)/2
        return OrderedDict(zip(param_labels_1, median)), OrderedDict(zip(param_labels_1, err))
    else:
        return OrderedDict(zip(param_labels_1, median)), OrderedDict(zip(param_labels_1, ep)), OrderedDict(zip(param_labels_1, em))

meds_1, errs_1 = process_chain_1(outputs)
pval  = []
for k,v in meds_1.items():
    pval.append(v)
    print("{} {:.3f}".format(k, v))


# In[ ]:


pval


# In[ ]:


#proability function for each stars to be member 
def prob(itot):
    probi=[]
    other = []
    for ii in range(itot):
       
        lgal,lbg1,ltot,_ = full_like_indi(pval,ii)
        print (ltot)
        probi.append(np.exp(lgal)/np.exp(ltot))
        other.append(_)
    return probi,other


# In[ ]:


testp,testindi = prob(len(datasum[0]))


# In[ ]:





# In[ ]:


#distribution for the membership probabilities
plt.hist(np.concatenate(testp),bins=15)
plt.xlabel('Probability')
plt.ylabel('Number of Stars')


# # Result Analysis
# 

# In[ ]:


#CMD diagram with probabiltiy 
plt.figure(figsize=(10,6))
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='red',lw=2)
plt.scatter(datacut['gmag0']-datacut['rmag0'],datacut['rmag0'],s=1,c=np.concatenate(testp),cmap='cool')
cbar=plt.colorbar(cmap='heatmap')
# Set the title for the colorbar
cbar.set_label('Probability')
plt.ylim(21, 16)
plt.xlim(-0.3,1.5)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# In[ ]:


#RA/DEC distribution with probability
plt.figure(figsize=(10,6))
plt.scatter(datacut['TARGET_RA_1'],datacut['TARGET_DEC_1'],s=2,c=np.concatenate(testp),cmap='cool')
cbar=plt.colorbar(cmap='heatmap')
# Set the title for the colorbar
cbar.set_label('Probability')
#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.ylabel('DEC')
plt.xlabel('RA')


# Comparing our results and Pace& Li 2022

# In[ ]:





# In[ ]:


#reading in results from Pace& Li 2022
df=Table.read('./draco_files/draco_1.dat', format='ascii', converters={'obsid': str})


# In[ ]:


ra2=df['ra'][df['mem_fixed_complete_ep']>0.000000001]
dec2=df['dec'][df['mem_fixed_complete_ep']>0.000000001]


# In[ ]:


#make a probability cut to exculde some bad data
probcut = np.concatenate(testp)>0.000000001


# In[ ]:


#cross match data 
ra1,dec1=datacut['TARGET_RA_1'],datacut['TARGET_DEC_1']
c = SkyCoord(ra=ra1*u.degree, dec=dec1*u.degree)
catalog = SkyCoord(ra=ra2*u.degree, dec=dec2*u.degree)


# In[ ]:


max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c.match_to_catalog_3d(catalog)
sep_constraint = d2d < max_sep
c_matches = c[sep_constraint]
catalog_matches = catalog[idx[sep_constraint]]


# In[ ]:


#difference between our probabilities and Pace&Li 2022 (using 'mem_fixed_complete_ep)
sub = np.concatenate(testp)[sep_constraint]-df['mem_fixed_complete_ep'][df['mem_fixed_complete_ep']>0.000000001][idx[sep_constraint]]

ind= sub>0.5


# In[ ]:


#plotting ra versus dec with Iron-Pace&Li 2022 prbabilities with the tidal radius of Draco

plt.figure(figsize=(10,6))
fig, ax = plt.subplots(figsize=(10,6))

# Set the aspect ratio of the plot to 'equal' to make the circle appear circular
ax.set_aspect('equal')

# Set the radius of the circle to be the half light radius
radius = 42/60
radius2 = 5*42/60
# Set the center coordinates of the circle
center = (ra0, dec0)

# Generate an array of angles from 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 100)

# Calculate the x and y coordinates of the points on the circumference of the circle
x = center[0] + radius * np.cos(theta)
y = center[1] + radius * np.sin(theta)
x2 = center[0] + radius2 * np.cos(theta)
y2 = center[1] + radius2 * np.sin(theta)

# Plot the circle
ax.plot(x, y,label = 'Tidal Radius',c='k')
ax.plot(x2, y2,label = '5 Times Tidal Radius',c='k')


#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(datacut['TARGET_RA_1'][sep_constraint],datacut['TARGET_DEC_1'][sep_constraint],s=6,c=sub,cmap='bwr')

cbar=plt.colorbar()
cbar.set_label('Iron-Pace&LI Probability')
plt.legend(loc=2)
#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.ylabel('DEC')
plt.xlabel('RA')


# In[ ]:


#Vrad versus FeH plot with Iron Probability

plt.figure(figsize=(10,6))

#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(datacut["VRAD"][sep_constraint],datacut['FEH'][sep_constraint],s=10,c=np.concatenate(testp)[sep_constraint],cmap='bwr')
cbar=plt.colorbar()
cbar.set_label('Iron - Pace&Li Probability')



#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.xlabel('VRAD')
plt.ylabel('FeH')


# In[ ]:


#Vrad versus FeH plot with Iron-Pace&Li 2022 Probability

plt.figure(figsize=(10,6))

#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(datacut["VRAD"][sep_constraint],datacut['FEH'][sep_constraint],s=10,c=sub,cmap='bwr')
cbar=plt.colorbar()
cbar.set_label('Iron - Pace&Li Probability')



#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.xlabel('VRAD')
plt.ylabel('FeH')


# In[ ]:


#pmra,pmdec with probability
plt.figure(figsize=(10,6))

#plt.scatter(df['pmra'][df['mem_fixed_complete_ep']>0.75],df['pmdec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(datacut["PMRA_3"][sep_constraint],datacut["PMDEC_3"][sep_constraint],s=10,c=np.concatenate(testp)[sep_constraint],cmap='bwr')
cbar= plt.colorbar()
cbar.set_label('Iron - Pace&Li Probability')

plt.ylim(-5, 5)

plt.xlim(-5,5)
plt.ylabel('PMDEC')
plt.xlabel('PMRA')


# In[ ]:


#pmra,pmdec with probability difference
plt.figure(figsize=(10,6))

#plt.scatter(df['pmra'][df['mem_fixed_complete_ep']>0.75],df['pmdec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(datacut["PMRA_3"][sep_constraint],datacut["PMDEC_3"][sep_constraint],s=10,c=sub,cmap='bwr')
cbar= plt.colorbar()
cbar.set_label('Iron - Pace&Li Probability')

plt.ylim(-5, 5)

plt.xlim(-5,5)
plt.ylabel('PMDEC')
plt.xlabel('PMRA')


# In[ ]:





# In[ ]:





# In[ ]:




