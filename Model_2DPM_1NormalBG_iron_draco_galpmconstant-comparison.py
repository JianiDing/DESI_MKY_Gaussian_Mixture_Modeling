#!/usr/bin/env python
# coding: utf-8

# # Iron Draco
# This notebook presents the mixture model of 3 gaussians built for Iron Draco data. The data is taken from the S5 Collaboration. With quality cut, we obtained 371 stars with good measurements to feed the model. The mixture model is built with 16 parameters, including radial velocity, metallicity and proper motion parameters of the smcnod and a set of parameters for the background components. We fit a Gaussian mixture model to this data using `emcee`.

# In[84]:


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
        'size'   : 10}

plt.rc('font', **font)

import imp
from astropy.io import fits as pyfits
import pandas as pd


# ## Iron Data Loading

# In[ ]:





# In[86]:


#data loading for DESI iron

ironrv = t1 = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[1].data)
t1_fiber = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[2].data)
t4 = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[4].data)

t1_comb = table.hstack((t1,t1_fiber,t4))




# In[87]:


len(set(t1_comb['REF_ID_2']))



# In[ ]:


#isochrone loading with a age = 10 Gyr 
#Properties for the isochrone 
#MIX-LEN  Y      Z          Zeff        [Fe/H] [a/Fe]
# 1.9380  0.2459 5.4651E-04 5.4651E-04  -1.50   0.00 
iso_file = pd.read_csv('./draco_files/isochrone_10_1.csv')


# In[ ]:





# # Colorcut for better selection

# In[ ]:


print('# before unique selection:', len(t1_comb))

# do a unique selection based on TARGET ID. Keep the first one for duplicates 
# (and first one has the smallest RV error)
t1_unique = table.unique(t1_comb, keys='TARGETID_1', keep='first')
print('# after unique selection:', len(t1_unique))


# In[ ]:





# In[ ]:


#taken from 
#https://docs.astropy.org/en/stable/generated/examples/coordinates/rv-to-gsr.html
def rv_to_gsr(c, v_sun=None):
    """Transform a barycentric radial velocity to the Galactic Standard of Rest
    (GSR).

    The input radial velocity must be passed in as a

    Parameters
    ----------
    c : `~astropy.coordinates.BaseCoordinateFrame` subclass instance
        The radial velocity, associated with a sky coordinates, to be
        transformed.
    v_sun : `~astropy.units.Quantity`, optional
        The 3D velocity of the solar system barycenter in the GSR frame.
        Defaults to the same solar motion as in the
        `~astropy.coordinates.Galactocentric` frame.

    Returns
    -------
    v_gsr : `~astropy.units.Quantity`
        The input radial velocity transformed to a GSR frame.

    """
    if v_sun is None:
        v_sun = coord.Galactocentric().galcen_v_sun.to_cartesian()

    gal = c.transform_to(coord.Galactic)
    cart_data = gal.data.to_cartesian()
    unit_vector = cart_data / cart_data.norm()

    v_proj = v_sun.dot(unit_vector)

    return c.radial_velocity + v_proj

import astropy.coordinates as coord
import astropy.units as u
coord.galactocentric_frame_defaults.set('latest')


icrs = coord.SkyCoord(ra=t1_unique['TARGET_RA_1']*u.deg, dec=t1_unique['TARGET_DEC_1']*u.deg,
                      radial_velocity=t1_unique['VRAD']*u.km/u.s, frame='icrs')
t1_unique['VGSR'] = rv_to_gsr(icrs)


# In[ ]:





# In[ ]:


testiron=t1_unique


# In[ ]:


rv_to_gsr(coord.SkyCoord(ra=t1_unique['TARGET_RA_1'][0]*u.deg, dec=t1_unique['TARGET_DEC_1'][0]*u.deg,radial_velocity=562.24*u.km/u.s, frame='icrs'))




# In[ ]:


testiron['VRAD']


# In[ ]:


testiron['VGSR']


# In[ ]:


testiron


# In[ ]:


#dust extinction correction
testiron['gmag'], testiron['rmag'], testiron['zmag'] = [22.5-2.5*np.log10(testiron['FLUX_'+_]) for _ in 'GRZ']

testiron['gmag0'] = testiron['gmag'] - testiron['EBV_2'] * 3.186
testiron['rmag0'] = testiron['rmag'] - testiron['EBV_2'] * 2.140
testiron['zmag0'] =testiron['zmag'] - testiron['EBV_2'] * 1.196
testiron['gmagerr'], testiron['rmagerr'], testiron['zmagerr'] = [2.5/np.log(10)*(np.sqrt(1./testiron['FLUX_IVAR_'+_])/testiron['FLUX_'+_]) for _ in 'GRZ']



# In[ ]:


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


# In[ ]:


#quality cut, exclude nans and RA/DEC cut
iqk,=np.where((testiron['TARGET_RA_1'] > 257) & (testiron['TARGET_RA_1'] < 263) & (testiron['TARGET_DEC_1'] > 55.9) & (testiron['TARGET_DEC_1'] < 59.9) & (testiron['RVS_WARN']==0) &(testiron['RR_SPECTYPE']!='QSO')&(testiron['VSINI']<50)
             
      &(~np.isnan(testiron["PMRA_ERROR"])) &(~np.isnan(testiron["PMDEC_ERROR"])) &(~np.isnan(testiron["PMRA_PMDEC_CORR"]))     )






# In[ ]:


#making CMD diagram for the data
ra0 = 260.0517
dec0 =  57.9153
rad0 =1.6

stars = SkyCoord(ra=testiron['TARGET_RA_1'], dec=testiron['TARGET_DEC_1'], unit=u.deg)

# Calculate the angular separation between stars and the reference point
separations = stars.separation(SkyCoord(ra=ra0, dec=dec0, unit=u.deg))

testiron['dist1'] = separations


# In[ ]:


ind = (testiron['dist1'][iqk] <0.3)
dm=19.53


# In[ ]:


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


# In[ ]:


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


# In[ ]:


#isochrone on the sample with a radius =0.3 degree from the draco gal center 
plt.scatter(testiron['gmag0'][iqk][ind]-testiron['rmag0'][iqk][ind],testiron['rmag0'][iqk][ind],s=3)
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='orange',lw=2)

plt.ylim(21, 16)
plt.xlim(-0.4,1.8)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# In[ ]:





# In[ ]:


def cmd_selection(t, dm, g_r, iso_r, cwmin=0.1, dobhb=True):
    # width in g-r
    grw = np.sqrt(0.1**2 + (3*10**log10_error_func(iso_r+dm, *popt))**2)

    gw = 0.3# min width (size) in r 
    # color selection box, in case we want something different from the mag cuts made earlier
    rmin = 16
    rmax = 23
    grmin = -0.5
    grmax = 1.5
    magrange = (t['rmag'] > rmin) & (t['rmag'] < rmax) & (t['gmag0'] - t['rmag0'] < grmax) & (t['gmag0'] - t['rmag0'] > grmin)
    gr = t['gmag0'] - t['rmag0']
    grmax1 = np.interp(t['rmag0'], iso_r[::-1] + dm, g_r[::-1]+grw[::-1], left=np.nan, right=np.nan)
    grmax2 = np.interp(t['rmag0'], iso_r[::-1] + dm + gw, g_r[::-1]+grw[::-1], left=np.nan, right=np.nan)
    grmax3 = np.interp(t['rmag0'], iso_r[::-1] + dm - gw, g_r[::-1]+grw[::-1], left=np.nan, right=np.nan)
    grmax = np.max(np.array([grmax1, grmax2, grmax3]), axis=0)
    grmin1 = np.interp(t['rmag0'], iso_r[::-1] + dm, g_r[::-1]-grw[::-1], left=np.nan, right=np.nan)
    grmin2 = np.interp(t['rmag0'], iso_r[::-1] + dm - gw, g_r[::-1]-grw[::-1], left=np.nan, right=np.nan)
    grmin3 = np.interp(t['rmag0'], iso_r[::-1] + dm + gw, g_r[::-1]-grw[::-1], left=np.nan, right=np.nan)
    grmin = np.min(np.array([grmin1, grmin2, grmin3]), axis=0)
    ismall,=np.where(grmax-grmin < cwmin)
    grmin[ismall] = np.interp(t['rmag0'][ismall],iso_r[::-1] + dm, g_r[::-1]-cwmin/2, left=np.nan, right=np.nan)
    grmax[ismall] = np.interp(t['rmag0'][ismall],iso_r[::-1] + dm, g_r[::-1]+cwmin/2, left=np.nan, right=np.nan)
    colorsel = (gr < grmax) & (gr > grmin)
    colorrange = magrange & colorsel

    
    if dobhb:
        grw_bhb = 1.0 # BHB width in gr
        gw_bhb = 1.0 # BHB width in g
        grmin_bhb = -0.6
        grmax_bhb = 0.6
        magrange_bhb = (t['rmag'] > rmin) & (t['rmag'] < rmax) & (t['gmag0'] - t['rmag0'] < grmax_bhb) & (t['gmag0'] - t['rmag0'] > grmin_bhb)

        gr_bhb = np.interp(t['rmag0'], des_m92_hb_r[::-1] + dm , des_m92_hb_g[::-1] - des_m92_hb_r[::-1], left=np.nan, right=np.nan)
        rr_bhb = np.interp(t['gmag0'] - t['rmag0'], des_m92_hb_g - des_m92_hb_r, des_m92_hb_r + dm,left=np.nan, right=np.nan)
        del_color_cmd_bhb = t['gmag0'] - t['rmag0'] - gr_bhb
        del_g_cmd_bhb = t['rmag0'] - rr_bhb
        colorrange_bhb = magrange_bhb & ((abs(del_color_cmd_bhb) < grw_bhb) | (abs(del_g_cmd_bhb) < gw_bhb))
        colorrange = colorrange | colorrange_bhb

    return  colorrange


# In[ ]:


from scipy import interpolate

xcut = (((iso_file['DECam_g']-iso_file['DECam_r']) < 1.8)& ((iso_file['DECam_g']-iso_file['DECam_r']) > -0.5))

ycut = (((iso_file['DECam_r']+dm) < 21)& ((iso_file['DECam_r']+dm) > 15.5))
fiso = interpolate.interp1d(((iso_file['DECam_g'].values)-(iso_file['DECam_r']).values)[xcut&ycut][-5:-1],((iso_file['DECam_r'].values)+dm)[xcut&ycut][-5:-1],kind='cubic',fill_value='extrapolate')



def extrapolate_poly(x, y, x_new):
    # Perform polynomial regression
    coeffs = np.polyfit(x, y, 1)

    # Extrapolate the polynomial
    y_new = np.polyval(coeffs, x_new)
    return y_new

isox = np.arange(1.18,1.5,0.1)
fiso=extrapolate_poly(((iso_file['DECam_g'].values)-(iso_file['DECam_r']).values)[xcut&ycut][-5:-1],((iso_file['DECam_r'].values)+dm)[xcut&ycut][-5:-1], isox)




# In[ ]:


iso_r = np.append(iso_file['DECam_r'].values[xcut&ycut],fiso-dm)
g_r = np.append(((iso_file['DECam_g'].values)-(iso_file['DECam_r']).values)[xcut&ycut],isox)


# In[ ]:






# In[ ]:


#colorcut for the sample 
#Investigating the data sample after colorcut 


plt.figure(figsize=(10,6))
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='orange',lw=2,label = 'Isochrone')
plt.scatter(testiron['gmag0'][iqk]-testiron['rmag0'][iqk],testiron['rmag0'][iqk],s=1,c='k',alpha=1,label ='Iron Data',cmap='bwr')
#plt.colorbar()
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
plt.legend(loc=2)
plt.ylim(21, 16)
plt.xlim(-0.3,1.5)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# In[ ]:


plt.figure(figsize=(10,6))
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='orange',lw=2,label = 'Isochrone')
plt.scatter(testiron['gmag0'][iqk]-testiron['rmag0'][iqk],testiron['rmag0'][iqk],s=1,c='k',alpha=1,label ='Iron Data',cmap='bwr')
#plt.colorbar()
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
plt.legend(loc=2)
plt.ylim(21, 16)
plt.xlim(-0.3,1.5)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# In[ ]:





# # Input data profile (FeH, radial velocity, pmra, pmdec)

# In[ ]:


iqk,=np.where((testiron['TARGET_RA_1'] > 257) & (testiron['TARGET_RA_1'] < 263) & (testiron['TARGET_DEC_1'] > 55.9) & (testiron['TARGET_DEC_1'] < 59.9) & (testiron['RVS_WARN']==0) &(testiron['RR_SPECTYPE']!='QSO')&(testiron['VSINI']<50)
             
      &(~np.isnan(testiron["PMRA_ERROR"])) &(~np.isnan(testiron["PMDEC_ERROR"])) &(~np.isnan(testiron["PMRA_PMDEC_CORR"])) )




# In[ ]:


print (len(testiron["VRAD"][iqk]))


# In[ ]:


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


# In[ ]:


plt.scatter(testiron["FEH"][iqk][colorcut],testiron["LOGG"][iqk][colorcut],s=2)
plt.xlabel('FEH')
plt.ylabel('LOGG')


# In[ ]:


#vrad,feh,pm distribution

fig, axes = plt.subplots(2,2,figsize=(9,9))
axes[0,0].hist(testiron["LOGG"], bins='auto');
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

# In[ ]:


plt.hist(testiron["VGSR"],bins=100)


# In[ ]:


def data_collect(datafile,ramin,ramax,decmin,decmax,fehmin,fehmax,vmin,vmax,logg,galdis,iso_file,dm,gw):
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
             
      &(~np.isnan(datafile["PMRA_ERROR"])) &(~np.isnan(datafile["PMDEC_ERROR"])) &(~np.isnan(datafile["PMRA_PMDEC_CORR"])) &(datafile["FEH"] >fehmin)&(datafile["FEH"] <fehmax) &(datafile["LOGG"] <logg) )
    
    colorcut = cmd_selection(testiron[iqk], dm, iso_file['DECam_g'],iso_file['DECam_r'], gw=gw)
    up = vmin
    low = vmax
    
    vtest = datafile["VGSR"][iqk][colorcut]
    vcut  = (vtest > low) & (vtest  < up)
    rv = datafile["VGSR"][iqk][colorcut][vcut]
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


    


    


# In[ ]:


#try 257/263 253/267

datasum,datacut =data_collect(testiron,253,267,55.9,59.9,-3.9,-0.5,900,-900,4,0.025,iso_file,dm,1.0)


# In[ ]:


len(datasum[0])


# In[ ]:


datacut


# In[ ]:


np.min(datacut['rmag0']),np.max(datacut['rmag0'])


# In[ ]:


plt.figure(figsize=(10,6))
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='orange',lw=2,label = 'Isochrone')
plt.scatter(datacut['gmag0']-datacut['rmag0'],datacut['rmag0'],s=1,c='k',alpha=1,label ='Iron Data',cmap='bwr')
#plt.colorbar()
grw = np.sqrt(0.1**2 + (3*10**log10_error_func(iso_file['DECam_r']+dm, *popt))**2)

iso1= np.array(iso_file['DECam_r']+dm+0.75)
iso2= np.array(iso_file['DECam_r']+dm-1.0)
colordiff = np.array(datacut['gmag0']-datacut['rmag0'])
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_file['DECam_r']+dm,'--r')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm,'--r')
cut1= ((iso2>np.min(datacut['rmag0']))&(iso2<np.min(des_m92_hb_r+dm-0.5)))
cut3 = ((iso2<np.max(datacut['rmag0']))&(iso2>np.min(des_m92_hb_r+dm+0.5)))
cut2= (iso1>np.min(datacut['rmag0']))&(iso1<np.max(datacut['rmag0']))
#cut4 = ((iso2<np.max(datacut['rmag0']))&(iso2>np.min(des_m92_hb_r+dm+0.5)))
plt.plot((iso_file['DECam_g']-iso_file['DECam_r']+grw)[cut2], iso1[cut2],'--k',label='Colorcut Region')
#plt.plot((iso_file['DECam_g']-iso_file['DECam_r']+grw)[cut3], iso1[cut3],'--k',label='Colorcut Region')
plt.plot((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut1], iso2[cut1] ,'--k')
plt.plot((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut3], iso2[cut3] ,'--k')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_cut1
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm+1,'--k')
plt.plot([np.min((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut3]),np.min((iso_file['DECam_g']-iso_file['DECam_r']+grw)[cut2])],[np.max(datacut['rmag0']),np.max(datacut['rmag0'])],'--',c='k')
#plt.plot(des_m92_hb_g-des_m92_hb_r, des_m92_hb_r+dm, lw=2, color='orange')
plt.plot(np.max(des_m92_hb_g-des_m92_hb_r),np.min(des_m92_hb_r+dm-0.5))
plt.plot((des_m92_hb_g-des_m92_hb_r)[des_m92_hb_g-des_m92_hb_r>-0.4], (des_m92_hb_r+dm-0.5)[des_m92_hb_g-des_m92_hb_r>-0.4],'--k')
plt.plot((des_m92_hb_g-des_m92_hb_r)[des_m92_hb_g-des_m92_hb_r>-0.4], (des_m92_hb_r+dm+0.5)[des_m92_hb_g-des_m92_hb_r>-0.4],'--k')
plt.plot([np.min((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut1]),np.max(des_m92_hb_g-des_m92_hb_r)],[np.max(iso2[cut1]),np.min(des_m92_hb_r+dm-0.5)],'--k')
#plt.a x= 0.3
#hor = 
plt.axhline(y=np.min(datacut['rmag0']),xmin = 0.63,xmax =0.98,c='k',linestyle='--')
plt.axvline(x=-0.33,ymin = 0.03,ymax = 0.25,c='k',linestyle='--')
#plt.plot(des_m92_hb_g-des_m92_hb_r-0.1, des_m92_hb_r+dm,'--r')
#plt.plot(des_m92_hb_g-des_m92_hb_r+0.1, des_m92_hb_r+dm,'--r')
plt.legend(loc=2)
plt.ylim(21.3, 17)
plt.xlim(-0.4,1.2)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')
plt.savefig('colorcut.pdf')


# In[ ]:


np.max((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut3])


# # Likelihood function

# In[ ]:


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


    bgpmcovs[:,0,0] = pmcovs[:,0,0]+(10**lsigpmra1)**2-galdis**2
    bgpmcovs[:,1,1] =  pmcovs[:,1,1]+(10**lsigpmdec1)**2-galdis**2
    bgpmcovs[:,0,1] = pmcovs[:,0,1]
    bgpmcovs[:,1,0] = pmcovs[:,1,0]
    
    
    # The prior is just a bunch of hard cutoffs
    if (pgal > 1) or (pgal < 0) or \
        (lsigv > 3) or (lsigvbg1 > 3) or \
        (lsigv < -1) or (lsigvbg1 < -1) or \
        (lsigfeh > 1) or (lsigfeh1 > 1) or (lsigfeh1 > 1) or \
        (lsigfeh < -3) or (lsigfeh1 < -3) or (lsigfeh1 < -3) or \
        (vhel > 600) or (vhel < -600) or (vbg1 > 500) or (vbg1 < -300) or \
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


    bgpmcovs[:,0,0] = pmcovs[:,0,0]+(10**lsigpmra1)**2-galdis**2
    bgpmcovs[:,1,1] =  pmcovs[:,1,1]+(10**lsigpmdec1)**2-galdis**2
    bgpmcovs[:,0,1] = pmcovs[:,0,1]
    bgpmcovs[:,1,0] = pmcovs[:,1,0]
    
    
    # The prior is just a bunch of hard cutoffs
    if (pgal > 1) or (pgal < 0) or \
        (lsigv > 3) or (lsigvbg1 > 3) or \
        (lsigv < -1) or (lsigvbg1 < -1) or \
        (lsigfeh > 1) or (lsigfeh1 > 1) or (lsigfeh1 > 1) or \
        (lsigfeh < -3) or (lsigfeh1 < -3) or (lsigfeh1 < -3) or \
        (vhel > 600) or (vhel < -600) or (vbg1 > 500) or (vbg1 < -300) or \
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


# In[ ]:


def project_model(theta, p1min=-600, p1max=350, p2min=-4, p2max=0.,key="vhel"):
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


# In[ ]:


def plot_1d_distrs(theta,datasum,p1min=-600, p1max=350, p2min=-4, p2max=0.,key="vhel"):
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
    fig, axes = plt.subplots(1,2,figsize=(18,8))
    if key == "vhel":
        ax = axes[0]
        ax.hist(datasum[0], density=True, color='grey', bins=100)
        xp, p0, p1 = model_output[0:3]
        ax.plot(xp, p0 + p1, 'k-', label="Total", lw=3)
        ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="Vgsr (km/s)", ylabel="Prob. Density")
        ax.legend(fontsize='small')

        ax = axes[1]
        ax.hist(datasum[2], density=True, color='grey', bins='auto')
        xp, p0, p1 = model_output[3:6]
        ax.plot(xp, p0 + p1, 'k-', lw=3)
        ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="[Fe/H] (dex)", ylabel="Prob. Density")
    else:
        ax = axes[0]
        ax.hist(datasum[-2][:,0], density=True, color='grey', bins=20)
        xp, p0, p1 = model_output[0:3]
        ax.plot(xp, p0 + p1, 'k-', label="Total", lw=3)
        ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel=r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$", ylabel="Prob. Density")
        ax.legend(fontsize='small')

        ax = axes[1]
        ax.hist(datasum[-2][:,1], density=True, color='grey', bins='auto')
        xp, p0, p1 = model_output[3:6]
        ax.plot(xp, p0 + p1, 'k-', lw=3)
        ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel=r"$\rm{\mu_{\delta} \ (mas/yr)}$", ylabel="Prob. Density")
    fig.savefig(str(key)+'distr1d.pdf')
    return fig


# In[ ]:


def plot_2d_distr(theta,datasum,key="vhel"):
    '''
    function for plotting the distribution of two quantities p1 versus p2 for the gal/bg 
    :param theta: likelihood parameters (prior/posterior) 
    :param datasum: data table
    :param key: key="vhel" for plotting vrad versus Feh / key="pmra" for plotting pmra versus pmdec 
    :return: plotting 

    '''
   
    fig, ax = plt.subplots(figsize=(18,8))
    if key == "vhel":
        ax.plot(datasum[2], datasum[0], 'k.',label='Sample')
        ax.set(xlabel="[Fe/H] (dex)", ylabel="Vgsr (km/s)", xlim=(-4,1), ylim=(-300,500))    
        params = get_paramdict(theta)
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ax.errorbar(params["feh"], params["vhel"],
               xerr=2*10**params["lsigfeh"], yerr=2*10**params["lsigv"],
               color=colors[0], marker='o', elinewidth=1, capsize=3, zorder=9999,label='Gal')
        ax.errorbar(params["fehbg1"], params["vbg1"],
               xerr=2*10**params["lsigfeh1"], yerr=2*10**params["lsigvbg1"],
               color=colors[2], marker='x', elinewidth=1, capsize=3, zorder=9999,label='Bg')
        ax.legend()
        ax.grid()
    else:
        ax.plot(datasum[-2][:,0], datasum[-2][:,1], 'k.',label='Sample')
        ax.set(xlabel=r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$", ylabel=r"$\rm{\mu_{\delta} \ (mas/yr)}$", xlim=(-5,5), ylim=(-5,5))    
        params = get_paramdict(theta)
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ax.errorbar(params["pmra"], params["pmdec"],
               xerr=2*10**0.025, yerr=2*10**0.025,
               color=colors[0], marker='o', elinewidth=1, capsize=3, zorder=9999,label='Gal')
        ax.errorbar(params["pmra1"], params["pmdec1"],
              xerr=2*10**params["lsigpmra1"], yerr=2*10**params["lsigpmdec1"],
              color=colors[2], marker='x', elinewidth=1, capsize=3, zorder=9999,label='Bg')
        ax.grid()
        ax.legend()
    fig.savefig(str(key)+'2ddistri.pdf')
    return fig


# In[ ]:


param_labels = ["pgal",
                "vhel","lsigv","feh","lsigfeh",
                "vbg1","lsigvbg1","fehbg1","lsigfeh1",
                "pmra","pmdec",
                "pmra1","pmdec1","lsigpmra1","lsigpmdec1"]


# In[ ]:





# walker2015 result 

# In[ ]:


Feh: -335.97883597883595, -0.747875354107649
-255.29100529100526, -0.747875354107649
-255.29100529100526, -3.501416430594901
-335.97883597883595, -3.501416430594901
Teff:
    -336.78756476683935, 4.110294117647059
-255.181347150259, 4.117647058823529
-335.49222797927456, 7.5808823529411775
-255.181347150259, 7.5808823529411775
logg:
-336.4269141531322, 0.24481327800829877
-255.2204176334106, 0.24481327800829877
-255.2204176334106, 3.7053941908713686
-336.4269141531322, 3.6970954356846466



# In[ ]:


dracohigh = pd.DataFrame()
dracohigh=pd.read_csv('draco_all.csv')


# In[ ]:


dracohigh['TARGET_RA_1']


# In[25]:


vloscut = [-255.23,-336.4]
loggcut = [0.24,3.7]
Teff = [4.11*1000,7.6*1000]
Feh = [-0.74,-3.5]


# In[26]:


Teff


# In[27]:


walker2015 = fits.open('draco1_walker2015.fits')


# In[28]:


walker2015[1].data


# In[29]:


cutvlos = (walker2015[1].data['vlos']<vloscut[0]) & (walker2015[1].data['vlos']>vloscut[1])
cutloggcut = (walker2015[1].data['logg']<loggcut[1]) & (walker2015[1].data['logg']>vloscut[0])
cutteff = (walker2015[1].data['Teff']<7580) & (walker2015[1].data['Teff']>loggcut[0])
cutFeh = (walker2015[1].data['__Fe_H_']<Feh[0]) & (walker2015[1].data['__Fe_H_']>Feh[1])
walkerselall = walker2015[1].data[cutvlos& cutloggcut& cutteff& cutFeh]


# In[30]:


print (len(walkerselall))


# In[45]:


walkerselected=pd.read_csv('walker2015colorhigh')


# In[46]:


walkerselected


# In[59]:


plt.figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(10,10))

# Set the aspect ratio of the plot to 'equal' to make the circle appear circular
#ax.set_aspect('equal')

# Set the radius of the circle to be the half light radius
r=10/60
e=0.31
theta0=89*0.0174533
b=r*np.sqrt(1-e)
a=r/np.sqrt(1-e)

r2=43/60
a2=r2/np.sqrt(1-e)
b2= r2*np.sqrt(1-e)
#radius = 42/60
#radius2 = 5*42/60
# Set the center coordinates of the circle
center = (ra0, dec0)

# Generate an array of angles from 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 100)

# Calculate the x and y coordinates of the points on the circumference of the circle
x =   a * np.cos(theta)
y = b* np.sin(theta)
#x =  center[0] + a * np.cos(theta)
#y = center[1]+b* np.sin(theta)
datacut = dracohigh
xd1=dracohigh['TARGET_RA_1']-center[0]
yd1=dracohigh['TARGET_DEC_1']-center[1]
x1= xd1*np.sin(theta0)- yd1*np.cos(theta0)

y1= xd1*np.cos(theta0)+ yd1*np.sin(theta0)
    

x2 =  a2 * np.cos(theta)
y2 = b2 * np.sin(theta)
#yt=y2*0.0174533
#xt=x2
#xd=np.cos(yt)*np.sin((xt-ra0)*0.0174533)/0.0174533
#yd=(np.sin(yt)*np.cos(dec0*0.0174533)-np.cos(yt)*np.sin(dec0*0.0174533)*np.cos((xt-ra0)*0.0174533))/0.0174533
# Plot the circle
#Rx= xd*np.cos(theta0)-yd*np.sin(theta0)
#Ry = xd*np.sin(theta0)+yd*np.cos(theta0)
#ax.plot(x, y,label = r'$r=r_{h}$',c='r')
ax.plot(x2, y2,label = 'Ellipse at the Tidal Radius',c='b')
ax.plot(x, y,label = 'Ellipse at the Half Light Radius',c='cyan')

proba=0.75

#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
#plt.scatter((x1+center[0])[np.concatenate(testp)>0.65],(y1+center[1])[np.concatenate(testp)>0.65],s=6,c='k',label= 'High Probability Members')
#plt.scatter((x1+center[0])[np.concatenate(testp)<0.6],(y1+center[1])[np.concatenate(testp)<0.6],s=6,c='grey',alpha=0.3,label = 'Low Probability Members')
x=np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
y=np.sin(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)
#Ryp = x*np.sin(theta0)+y*np.cos(theta0)

plt.scatter(x/0.0174533,y/0.0174533,s=30,label='High Probability Members')
proba= 0.75
x2=np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
y2=np.sin(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
#Rxp2= x2*np.cos(theta0)-y2*np.sin(theta0)
#Ryp2 = x2*np.sin(theta0)+y2*np.cos(theta0)
x3=np.cos(dataout['TARGET_DEC_1']*0.0174533)*np.sin((dataout['TARGET_RA_1']-ra0)*0.0174533)
y3=np.sin(dataout['TARGET_DEC_1']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(dataout['TARGET_DEC_1']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((dataout['TARGET_RA_1']-ra0)*0.0174533)
#Rxp2= x2*np.cos(theta0)-y2*np.sin(theta0)
#plt.scatter(x2/0.0174533,y2/0.0174533,s=6,alpha=0.3,c='grey',label='Low Probability Members')
#plt.scatter(x3/0.0174533,y3/0.0174533,s=200,c=(testp)[(testp)>probb][cut1],label='Eight Outskirts High Probability Members')
plt.plot([3.20833,0],[-1.20618556,0],'-',c='g')
plt.plot([-3.20833,0],[1.20618556,0],'--',c='g',label='Orbit from Qi et al. 2022')
#cbar=plt.colorbar()
plt.style.use('classic')
#cbar.set_label('Probability')


xq=np.cos(qi['DEC']*0.0174533)*np.sin((qi['RA']-ra0)*0.0174533)
yq=np.sin(qi['DEC']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(qi['DEC']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((qi['RA']-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

#plt.scatter(xq/0.0174533,yq/0.0174533,s=30,c='green',label='Qi2022 Members')


xw=np.cos(walkerselall['DEJ2000']*0.0174533)*np.sin((walkerselall['RAJ2000']-ra0)*0.0174533)
yw=np.sin(walkerselall['DEJ2000']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(walkerselall['DEJ2000']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((walkerselall['RAJ2000']-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

plt.scatter(xw/0.0174533,yw/0.0174533,s=30,c='red',label='Walker2015 Members')

xwa=np.cos(walker2015[1].data['DEJ2000']*0.0174533)*np.sin((walker2015[1].data['RAJ2000']-ra0)*0.0174533)
ywa=np.sin(walker2015[1].data['DEJ2000']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(walker2015[1].data['DEJ2000']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((walker2015[1].data['RAJ2000']-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

plt.scatter(xwa/0.0174533,ywa/0.0174533,s=30,c='grey',alpha=0.5,label='Walker2015 All',zorder = 0)

xw2=np.cos(walkerunique['col2']*0.0174533)*np.sin((walkerunique['col1']-ra0)*0.0174533)
yw2=np.sin(walkerunique['col2']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(walkerunique['col2']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((walkerunique['col1']-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)
xw2=xw2/0.0174533
yw2=yw2/0.0174533
cut1 = (xw2**2/a2**2+yw2**2/b2**2)>1

dataw = walkerunique[cut1]
plt.scatter(dataw['col1'],dataw['col2'],s=45,c='red')


xm=np.cos(walkerselected['dec'][idx[sep_constraint]]*0.0174533)*np.sin((walkerselected['ra'][idx[sep_constraint]]-ra0)*0.0174533)
ym=np.sin(walkerselected['dec'][idx[sep_constraint]]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(walkerselected['dec'][idx[sep_constraint]]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((walkerselected['ra'][idx[sep_constraint]]-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

#plt.scatter(xm/0.0174533,ym/0.0174533,s=60,c='orange',label='Matched Members',zorder=5)

xq=np.cos(qi['DEC'][idx2[sep_constraint2]]*0.0174533)*np.sin((qi['RA'][idx2[sep_constraint2]]-ra0)*0.0174533)
yq=np.sin(qi['DEC'][idx2[sep_constraint2]]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(qi['DEC'][idx2[sep_constraint2]]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((qi['RA'][idx2[sep_constraint2]]-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

#plt.scatter(xq/0.0174533,yq/0.0174533,s=60,c='purple',label='Qi Matched Members')

# Define the arrow properties
arrow_properties = dict(
    facecolor='orange',
    edgecolor='orange',
    linewidth=5,      # Adjust the arrow width
    head_width=0.05,  # Adjust the arrow head width
)

# Draw the arrow
#plt.arrow(ra0, dec0, 0.03/np.cos(dec0*0.0174533), -0.21, **arrow_properties)




# Draw your plot here...

# Turn off the grid
ax.grid(False)
plt.xlim(2,-2)
plt.ylim(-2,2)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.), ncol=3)
#plt.ylim(57, 59)
#plt.xlim(258,262)
plt.ylabel('DEC Projected [deg]')
plt.xlabel('RA Projected [deg]')


# In[50]:


c = SkyCoord(ra=dracohigh['TARGET_RA_1']*u.degree, dec=dracohigh['TARGET_DEC_1']*u.degree)
catalog = SkyCoord(ra=walkerselected['ra']*u.degree, dec=walkerselected['dec']*u.degree)

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c.match_to_catalog_3d(catalog)
sep_constraint = d2d < max_sep
c_matches = c[sep_constraint]
catalog_matches = catalog[idx[sep_constraint]]


# In[51]:


catalog[idx[sep_constraint]]


# In[52]:


testp=np.loadtxt('draco_prob')


# In[53]:


r2=43/60
e=0.31
a2=r2/np.sqrt(1-e)
b2= r2*np.sqrt(1-e)
probb=0.75
datacut = dracohigh
xc=np.cos(datacut['TARGET_DEC_1'][(testp)>probb]*0.0174533)*np.sin((datacut['TARGET_RA_1'][(testp)>probb]-ra0)*0.0174533)
yc=np.sin(datacut['TARGET_DEC_1'][(testp)>probb]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][(testp)>probb]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][(testp)>probb]-ra0)*0.0174533)
#Rxp2= x2*np.cos(theta0)-y2*np.sin(theta0)
x2=xc/0.0174533
y2=yc/0.0174533
cut1 = (x2**2/a2**2+y2**2/b2**2)>1

dataout = datacut[(testp)>probb][cut1]


# In[ ]:





# In[55]:


qira = [261.909753,259.142130,260.617774,256.354218,262.992811,261.600665,262.155953,258.784927,256.952365,260.717683]
qiid = [
1434492516786370176,
1433949770359264768,
1434014607185724544,
1433509931348820864,
1422046732355091584,
1420524699025187328,
1435623055258755840,
1437346161777223936,
1432365034801179648,
1437470136009868416]
qidec = [58.263091,58.498825,58.730676,57.552936,56.621840,56.016370,59.890716,60.251350,55.751991,60.847082]
qipmra=[0.047, 0.093, -0.084, 0.205, -0.093,-0.134, -0.057, -0.071,-0.149, -0.009]
qipmraerr = [0.127,0.187,0.186,0.184,0.141,0.116,0.168,0.202,0.182,0.221]
qipmdec = [-0.217, -0.206, -0.152,  -0.259, -0.311, -0.205,-0.066, -0.152, -0.159,  -0.169]
qipmdecerr = [0.138,0.172,0.197,0.281, 0.147,0.110 ,0.172, 0.227,0.229,0.251]
qi = pd.DataFrame()
qi['RA']=qira
qi['DEC']=qidec
qi['GaiaID']=qiid
qi['pmra'] = qipmra
qi['pmraerr'] = qipmraerr
qi['pmdec'] = qipmdec
qi['pmdecerr'] = qipmdecerr


# In[60]:


c = SkyCoord(ra=dracohigh['TARGET_RA_1']*u.degree, dec=dracohigh['TARGET_DEC_1']*u.degree)
catalog2 = SkyCoord(ra=qi['RA']*u.degree, dec=qi['DEC']*u.degree)

max_sep2 = 1.0 * u.arcsec
idx2, d2d2, d3d2 = c.match_to_catalog_3d(catalog2)
sep_constraint2 = d2d2 < max_sep2
c_matches2 = c[sep_constraint2]
catalog_matches2 = catalog2[idx2[sep_constraint2]]


# In[61]:


walkerunique = pd.read_csv('walker2015unique')


# In[62]:


walkerunique['col2']


# In[63]:


plt.figure(figsize=(10,10))
fig, ax = plt.subplots(figsize=(10,10))

# Set the aspect ratio of the plot to 'equal' to make the circle appear circular
#ax.set_aspect('equal')

# Set the radius of the circle to be the half light radius
r=10/60
e=0.31
theta0=89*0.0174533
b=r*np.sqrt(1-e)
a=r/np.sqrt(1-e)

r2=43/60
a2=r2/np.sqrt(1-e)
b2= r2*np.sqrt(1-e)
#radius = 42/60
#radius2 = 5*42/60
# Set the center coordinates of the circle
center = (ra0, dec0)

# Generate an array of angles from 0 to 2*pi
theta = np.linspace(0, 2 * np.pi, 100)

# Calculate the x and y coordinates of the points on the circumference of the circle
x =   a * np.cos(theta)
y = b* np.sin(theta)
#x =  center[0] + a * np.cos(theta)
#y = center[1]+b* np.sin(theta)
datacut = dracohigh
xd1=dracohigh['TARGET_RA_1']-center[0]
yd1=dracohigh['TARGET_DEC_1']-center[1]
x1= xd1*np.sin(theta0)- yd1*np.cos(theta0)

y1= xd1*np.cos(theta0)+ yd1*np.sin(theta0)
    

x2 =  a2 * np.cos(theta)
y2 = b2 * np.sin(theta)
#yt=y2*0.0174533
#xt=x2
#xd=np.cos(yt)*np.sin((xt-ra0)*0.0174533)/0.0174533
#yd=(np.sin(yt)*np.cos(dec0*0.0174533)-np.cos(yt)*np.sin(dec0*0.0174533)*np.cos((xt-ra0)*0.0174533))/0.0174533
# Plot the circle
#Rx= xd*np.cos(theta0)-yd*np.sin(theta0)
#Ry = xd*np.sin(theta0)+yd*np.cos(theta0)
#ax.plot(x, y,label = r'$r=r_{h}$',c='r')
ax.plot(x2, y2,label = 'Ellipse at the Tidal Radius',c='b')
ax.plot(x, y,label = 'Ellipse at the Half Light Radius',c='cyan')

proba=0.75

#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
#plt.scatter((x1+center[0])[np.concatenate(testp)>0.65],(y1+center[1])[np.concatenate(testp)>0.65],s=6,c='k',label= 'High Probability Members')
#plt.scatter((x1+center[0])[np.concatenate(testp)<0.6],(y1+center[1])[np.concatenate(testp)<0.6],s=6,c='grey',alpha=0.3,label = 'Low Probability Members')
x=np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
y=np.sin(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)
#Ryp = x*np.sin(theta0)+y*np.cos(theta0)

plt.scatter(x/0.0174533,y/0.0174533,s=30,label='High Probability Members')
proba= 0.75
x2=np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
y2=np.sin(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
#Rxp2= x2*np.cos(theta0)-y2*np.sin(theta0)
#Ryp2 = x2*np.sin(theta0)+y2*np.cos(theta0)
x3=np.cos(dataout['TARGET_DEC_1']*0.0174533)*np.sin((dataout['TARGET_RA_1']-ra0)*0.0174533)
y3=np.sin(dataout['TARGET_DEC_1']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(dataout['TARGET_DEC_1']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((dataout['TARGET_RA_1']-ra0)*0.0174533)
#Rxp2= x2*np.cos(theta0)-y2*np.sin(theta0)
#plt.scatter(x2/0.0174533,y2/0.0174533,s=6,alpha=0.3,c='grey',label='Low Probability Members')
#plt.scatter(x3/0.0174533,y3/0.0174533,s=200,c=(testp)[(testp)>probb][cut1],label='Eight Outskirts High Probability Members')
plt.plot([3.20833,0],[-1.20618556,0],'-',c='g')
plt.plot([-3.20833,0],[1.20618556,0],'--',c='g',label='Orbit from Qi et al. 2022')
#cbar=plt.colorbar()
plt.style.use('classic')
#cbar.set_label('Probability')


xq=np.cos(qi['DEC']*0.0174533)*np.sin((qi['RA']-ra0)*0.0174533)
yq=np.sin(qi['DEC']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(qi['DEC']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((qi['RA']-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

plt.scatter(xq/0.0174533,yq/0.0174533,s=30,c='green',label='Qi2022 Members')


xw=np.cos(walkerselected['dec']*0.0174533)*np.sin((walkerselected['ra']-ra0)*0.0174533)
yw=np.sin(walkerselected['dec']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(walkerselected['dec']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((walkerselected['ra']-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

plt.scatter(xw/0.0174533,yw/0.0174533,s=30,c='red',label='Walker2015 Members')


xw2=np.cos(walkerunique['col2']*0.0174533)*np.sin((walkerunique['col1']-ra0)*0.0174533)
yw2=np.sin(walkerunique['col2']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(walkerunique['col2']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((walkerunique['col1']-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)
xw2=xw2/0.0174533
yw2=yw2/0.0174533
cut1 = (xw2**2/a2**2+yw2**2/b2**2)>1

dataw = walkerunique[cut1]
plt.scatter(dataw['col1'],dataw['col2'],s=45,c='red')


xm=np.cos(walkerselected['dec'][idx[sep_constraint]]*0.0174533)*np.sin((walkerselected['ra'][idx[sep_constraint]]-ra0)*0.0174533)
ym=np.sin(walkerselected['dec'][idx[sep_constraint]]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(walkerselected['dec'][idx[sep_constraint]]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((walkerselected['ra'][idx[sep_constraint]]-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

plt.scatter(xm/0.0174533,ym/0.0174533,s=60,c='orange',label='Matched Members')

xq=np.cos(qi['DEC'][idx2[sep_constraint2]]*0.0174533)*np.sin((qi['RA'][idx2[sep_constraint2]]-ra0)*0.0174533)
yq=np.sin(qi['DEC'][idx2[sep_constraint2]]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(qi['DEC'][idx2[sep_constraint2]]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((qi['RA'][idx2[sep_constraint2]]-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)

plt.scatter(xq/0.0174533,yq/0.0174533,s=60,c='purple',label='Qi Matched Members')

# Define the arrow properties
arrow_properties = dict(
    facecolor='orange',
    edgecolor='orange',
    linewidth=5,      # Adjust the arrow width
    head_width=0.05,  # Adjust the arrow head width
)

# Draw the arrow
#plt.arrow(ra0, dec0, 0.03/np.cos(dec0*0.0174533), -0.21, **arrow_properties)




# Draw your plot here...

# Turn off the grid
ax.grid(False)
plt.xlim(3,-3)
plt.ylim(-3,3)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.), ncol=3)
#plt.ylim(57, 59)
#plt.xlim(258,262)
plt.ylabel('DEC Projected [deg]')
plt.xlabel('RA Projected [deg]')
#plt.savefig('spatial_gsr_matched.pdf')


# In[64]:


qi


# In[65]:


dataw['col1']


# In[61]:


walkerselall


# In[67]:


c = SkyCoord(ra=dracohigh['TARGET_RA_1']*u.degree, dec=dracohigh['TARGET_DEC_1']*u.degree)
catalog = SkyCoord(ra=dataw['col1']*u.degree, dec=dataw['col2']*u.degree)

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c.match_to_catalog_3d(catalog)
sep_constraint = d2d < max_sep
c_matches = c[sep_constraint]
catalog_matches = catalog[idx[sep_constraint]]


# In[68]:


dataw['col1'].values[1]


# In[69]:


[idx[sep_constraint]]


# In[70]:


dracohigh


# In[71]:


dracohigh['PMRA_3'][testp>0.75]


# In[72]:


dataout['PMRA_ERROR']


# In[73]:


dataout['FEH']


# In[74]:


plt.scatter(dataout['VGSR'],dataout['FEH'],c='red',label = 'Draco Members',s=10,zorder=0)
plt.errorbar(dataout['VGSR'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
#plt.plot(walkerunique['VRAD'],walkerunique['FEH'],'.',c='b',label = 'All Members')
#plt.plot(dataw['VRAD'],dataw['FEH'],'o',c='grey',label = 'Outskirts Members')
#plt.scatter(dataw['VRAD'].values[idx[sep_constraint]],dataw['FEH'].values[idx[sep_constraint]],s=10,c='g',label = 'Matched Outskirts Members',zorder=4)
#plt.errorbar(dataw['VRAD'],dataw['FEH'],xerr=dataw['VRAD_ERR'],yerr= dataw['FEH_ERR'],marker='o',linestyle = ' ',c='grey',label = 'Outskirts Members')
#plt.errorbar(dataw['VRAD'].values[idx[sep_constraint]],dataw['FEH'].values[idx[sep_constraint]],xerr=dataw['VRAD_ERR'].values[idx[sep_constraint]],yerr= dataw['FEH_ERR'].values[idx[sep_constraint]],linestyle = ' ',c='green')
plt.xlabel('VRAD')
plt.ylabel('FEH')
plt.legend(loc=4)


# In[ ]:





# In[72]:


c = SkyCoord(ra=walkerunique['col1']*u.degree, dec=walkerunique['col2']*u.degree)
catalog = SkyCoord(ra=dataw['col1']*u.degree, dec=dataw['col2']*u.degree)

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c.match_to_catalog_3d(catalog)
sep_constraint = d2d < max_sep
c_matches = c[sep_constraint]
catalog_matches = catalog[idx[sep_constraint]]


# In[73]:


dataw


# In[74]:


catalog_matches


# In[75]:


walkerunique[sep_constraint]


# In[75]:


dataw


# In[77]:


walkerunique[sep_constraint]


# In[76]:


ct = SkyCoord(ra=walkerselall['RAJ2000']*u.degree, dec=walkerselall['DEJ2000']*u.degree)
catalogt = SkyCoord(ra=dataw['col1']*u.degree, dec=dataw['col2']*u.degree)

max_sept = 1.0 * u.arcsec
idxt, d2dt, d3d = ct.match_to_catalog_3d(catalog)
sep_constraintt = d2dt < max_sept
c_matchest = ct[sep_constraintt]
catalog_matchest = catalogt[idxt[sep_constraintt]]


# In[83]:


# Create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(15, 13))  # Adjust figsize as needed

# Plot data in each panel
#axs[0, 0].scatter(dataout['VRAD'],dataout['FEH'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0, linewidth=0.0)
#axs[0, 0].scatter(dataout['VGSR'],dataout['FEH'],c='red',label = 'Draco Outskirts Members',s=10,zorder=0)
axs[0, 1].scatter(dracohigh['VRAD'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='red',label = 'DESI High Probability Members',s=10,zorder=1,linewidth=0.0)
axs[0, 1].scatter(walkerselall['vlos'],walkerselall['__Fe_H_'],c='b',label = 'Walker2015 Members',s=10,zorder=0,alpha = 0.3,linewidth=0.0)

#axs[0, 0].errorbar(dataout['VRAD'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
axs[0, 1].set_xlabel('VRAD (km/s)',fontsize=20)
axs[0, 1].set_ylabel('[Fe/H]',fontsize=20)

axs[0, 1].legend(loc=4,fontsize=12)


#axs[0, 0].scatter(walkerselall['vlos'],walkerselall['__Fe_H_'],c='b',label = 'Walker2015 Members',s=10,zorder=0,alpha = 0.5)
axs[0, 0].scatter(dracohigh['VRAD'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='red',label = 'DESI High Probability Members',s=10,zorder=0,linewidth=0.0)

axs[0, 0].scatter(dataout['VRAD'],dataout['FEH'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0,linewidth=0.0)
axs[0, 0].errorbar(dataout['VRAD'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
#axs[0, 1].scatter(walkerselall[sep_constraintt]['vlos'],walkerselall[sep_constraintt]['__Fe_H_'],c='blue',label = 'Walker2015 Outskirts Members',s=10,zorder=0)
#axs[0, 1].errorbar(walkerselall[sep_constraintt]['vlos'],walkerselall[sep_constraintt]['__Fe_H_'],xerr=walkerselall[sep_constraintt]['e__Fe_H_'],yerr= walkerselall[sep_constraintt]['e__Fe_H_'],linestyle = ' ',c='blue')

#axs[0, 1].scatter(dracohigh['VRAD'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='grey',label = 'DESI High Probability Members',s=10,zorder=0)
#axs[0, 1].errorbar(dataout['VGSR'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
#axs[0, 1].scatter(dracohigh['VGSR'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='grey',label = 'Draco High Probability Members',s=10,zorder=0)
axs[0, 0].set_xlabel('VRAD (km/s)',fontsize=20)
axs[0, 0].set_ylabel('[Fe/H]',fontsize=20)
axs[0, 0].legend(loc=2,fontsize=12)
#axs[1, 0].plot(data3)
#axs[1, 0].set_title('Panel 3')
axs[1, 1].scatter(dracohigh['PMRA_3'][testp>0.75],dracohigh['PMDEC_3'][testp>0.75],c='red', label = 'DESI Draco Members',s=10,zorder=5,linewidth=0.0,alpha = 0.3)
axs[1, 1].plot(walkerunique['pmra'],walkerunique['pmdec'],'.',c='b',label = 'Walker2015 All Members',alpha = 0.8,linewidth=0.0)
axs[1, 1].scatter(dataout['PMRA_3'],dataout['PMDEC_3'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0,linewidth=0.0)
axs[1, 1].errorbar(dataw['pmra'],dataw['pmdec'],xerr=dataw['pmra_error'],yerr= dataw['pmra_error'],marker='o',linestyle = ' ',c='grey',label = 'Walker2015 Outskirts Members')
axs[1, 1].errorbar(dataout['PMRA_3'],dataout['PMDEC_3'],xerr=dataout['PMRA_ERROR'],yerr= dataout['PMDEC_ERROR'],linestyle = ' ',c='red')
axs[1, 1].errorbar([],[],xerr=[],yerr= [],linestyle = ' ',c='green',label = 'Qi2022 Outskirts Members')

axs[1, 1].set_xlabel(r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$",fontsize=20)
axs[1, 1].set_ylabel(r"$\rm{\mu_{\delta} \ (mas/yr)}$",fontsize=20)
ax_inset = axs[1, 1].inset_axes([0.40, 0.05, 0.55, 0.45])  # [x, y, width, height]

# Generate data for the inset plot
ax_inset.scatter(dracohigh['PMRA_3'][testp>0.75],dracohigh['PMDEC_3'][testp>0.75],c='red', label = 'DESI Draco Members',s=10,zorder=5,linewidth=0.0,alpha = 0.3)
#axs[1, 0].plot(walkerunique['pmra'],walkerunique['pmdec'],'.',c='b',label = 'Walker2015 All Members',alpha = 0.8,linewidth=0.0)
ax_inset.scatter(dataout['PMRA_3'],dataout['PMDEC_3'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0,linewidth=0.0)
#axs[1, 0].er
ax_inset.scatter(qi['pmra'],qi['pmdec'],c='green',s=10,zorder=0,linewidth=0.0)
ax_inset.errorbar(qi['pmra'],qi['pmdec'],xerr=qi['pmraerr'],yerr= qi['pmraerr'],marker='o',linestyle = ' ',c='green',label = 'Qi2022 Beyond King Radius Members',zorder = 5)

ax_inset.errorbar(dataout['PMRA_3'],dataout['PMDEC_3'],xerr=dataout['PMRA_ERROR'],yerr= dataout['PMDEC_ERROR'],linestyle = ' ',c='red')
#legend_inset = ax_inset.legend(loc='upper left', title='Inset Legend')
#axs[1, 0].add_artist(legend_inset)
# Add decorations to the inset plot
#ax_inset.set_title('Inset Plot')
#ax_inset.set_xlabel('X')
#ax_inset.set_ylabel('Y')
axs[1, 1].plot(dataw['pmra'],dataw['pmdec'],'o',c='grey',markeredgewidth=0.0)
axs[1, 1].legend(loc=2,fontsize=10)
axs[1, 0].scatter(dracohigh['PMRA_3'][testp>0.75],dracohigh['PMDEC_3'][testp>0.75],c='red',alpha=0.3, label = 'DESI Draco Members',s=10,zorder=5, linewidth=0)
axs[1, 0].scatter(dataout['PMRA_3'],dataout['PMDEC_3'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0,linewidth=0.0)
axs[1, 0].errorbar(dataout['PMRA_3'],dataout['PMDEC_3'],xerr=dataout['PMRA_ERROR'],yerr= dataout['PMDEC_ERROR'],linestyle = ' ',c='red')


#axs[1, 1].scatter(dataw['pmra'].values[idx[sep_constraint]],dataw['pmdec'].values[idx[sep_constraint]],s=10,c='g',label = 'Matched Outskirts Members',zorder=4)
#axs[1, 1].errorbar(dataw['pmra'],dataw['pmdec'],xerr=dataw['pmra_error'],yerr= dataw['pmra_error'],marker='o',linestyle = ' ',c='grey',label = 'Walker2015 Outskirts Members')
#axs[1, 1].set_title('Panel 4')
axs[1, 0].set_xlabel(r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$",fontsize=20)
axs[1, 0].set_ylabel(r"$\rm{\mu_{\delta} \ (mas/yr)}$",fontsize=20)
# Adjust layout to avoid overlapping titles
plt.tight_layout()
axs[1, 0].legend(loc=2,fontsize=12)
# Show the plot
plt.savefig('Four_com.pdf')
plt.show()


# In[91]:


# Create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust figsize as needed

# Plot data in each panel
#axs[0, 0].scatter(dataout['VRAD'],dataout['FEH'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0, linewidth=0.0)
#axs[0, 0].scatter(dataout['VGSR'],dataout['FEH'],c='red',label = 'Draco Outskirts Members',s=10,zorder=0)
axs[0, 0].scatter(dracohigh['VRAD'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='red',label = 'DESI High Probability Members',s=10,zorder=1,linewidth=0.0)
axs[0, 0].scatter(walkerselall['vlos'],walkerselall['__Fe_H_'],c='b',label = 'Walker2015 Members',s=10,zorder=0,alpha = 0.3,linewidth=0.0)

#axs[0, 0].errorbar(dataout['VRAD'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
axs[0, 0].set_xlabel('VRAD (km/s)')
axs[0, 0].set_ylabel('FEH')

axs[0, 0].legend(loc=4,fontsize=10)


#axs[0, 0].scatter(walkerselall['vlos'],walkerselall['__Fe_H_'],c='b',label = 'Walker2015 Members',s=10,zorder=0,alpha = 0.5)
axs[0, 1].scatter(dracohigh['VRAD'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='red',label = 'DESI High Probability Members',s=10,zorder=0,linewidth=0.0)

axs[0, 1].scatter(dataout['VRAD'],dataout['FEH'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0,linewidth=0.0)
axs[0, 1].errorbar(dataout['VRAD'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
#axs[0, 1].scatter(walkerselall[sep_constraintt]['vlos'],walkerselall[sep_constraintt]['__Fe_H_'],c='blue',label = 'Walker2015 Outskirts Members',s=10,zorder=0)
#axs[0, 1].errorbar(walkerselall[sep_constraintt]['vlos'],walkerselall[sep_constraintt]['__Fe_H_'],xerr=walkerselall[sep_constraintt]['e__Fe_H_'],yerr= walkerselall[sep_constraintt]['e__Fe_H_'],linestyle = ' ',c='blue')

#axs[0, 1].scatter(dracohigh['VRAD'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='grey',label = 'DESI High Probability Members',s=10,zorder=0)
#axs[0, 1].errorbar(dataout['VGSR'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
#axs[0, 1].scatter(dracohigh['VGSR'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='grey',label = 'Draco High Probability Members',s=10,zorder=0)
axs[0, 1].set_xlabel('VRAD (km/s)')
axs[0, 1].set_ylabel('FEH')
axs[0, 1].legend(loc=2,fontsize=8)
#axs[1, 0].plot(data3)
#axs[1, 0].set_title('Panel 3')
axs[1, 0].scatter(dracohigh['PMRA_3'][testp>0.75],dracohigh['PMDEC_3'][testp>0.75],c='red', label = 'DESI Draco Members',s=10,zorder=5,linewidth=0.0,alpha = 0.3)
#axs[1, 0].plot(walkerunique['pmra'],walkerunique['pmdec'],'.',c='b',label = 'Walker2015 All Members',alpha = 0.8,linewidth=0.0)
axs[1, 0].scatter(dataout['PMRA_3'],dataout['PMDEC_3'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0,linewidth=0.0)
#axs[1, 0].errorbar(dataw['pmra'],dataw['pmdec'],xerr=dataw['pmra_error'],yerr= dataw['pmra_error'],marker='o',linestyle = ' ',c='grey',label = 'Walker2015 Outskirts Members')
axs[1, 0].scatter(qi['pmra'],qi['pmdec'],c='green',s=10,zorder=0,linewidth=0.0)
axs[1, 0].errorbar(qi['pmra'],qi['pmdec'],xerr=qi['pmraerr'],yerr= qi['pmraerr'],marker='o',linestyle = ' ',c='green',label = 'Qi2022 Beyond King Radius Members',zorder = 3)

axs[1, 0].errorbar(dataout['PMRA_3'],dataout['PMDEC_3'],xerr=dataout['PMRA_ERROR'],yerr= dataout['PMDEC_ERROR'],linestyle = ' ',c='red')
axs[1, 0].set_xlabel(r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$")
axs[1, 0].set_ylabel(r"$\rm{\mu_{\delta} \ (mas/yr)}$")
#axs[1, 0].plot(dataw['pmra'],dataw['pmdec'],'o',c='grey',label = 'Walker2015 Outskirts Members',markeredgewidth=0.0)
axs[1, 0].legend(loc=4,fontsize=8)
axs[1, 1].scatter(dracohigh['PMRA_3'][testp>0.75],dracohigh['PMDEC_3'][testp>0.75],c='red',alpha=0.3, label = 'DESI Draco Members',s=10,zorder=5, linewidth=0)
axs[1, 1].scatter(dataout['PMRA_3'],dataout['PMDEC_3'],c='red',label = 'DESI Beyond King Radius Members',s=10,zorder=0,linewidth=0.0)
axs[1, 1].errorbar(dataout['PMRA_3'],dataout['PMDEC_3'],xerr=dataout['PMRA_ERROR'],yerr= dataout['PMDEC_ERROR'],linestyle = ' ',c='red')


#axs[1, 1].scatter(dataw['pmra'].values[idx[sep_constraint]],dataw['pmdec'].values[idx[sep_constraint]],s=10,c='g',label = 'Matched Outskirts Members',zorder=4)
#axs[1, 1].errorbar(dataw['pmra'],dataw['pmdec'],xerr=dataw['pmra_error'],yerr= dataw['pmra_error'],marker='o',linestyle = ' ',c='grey',label = 'Walker2015 Outskirts Members')
#axs[1, 1].set_title('Panel 4')
axs[1, 1].set_xlabel(r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$")
axs[1, 1].set_ylabel(r"$\rm{\mu_{\delta} \ (mas/yr)}$")
# Adjust layout to avoid overlapping titles
plt.tight_layout()
axs[1, 1].legend(loc=2,fontsize=8)
# Show the plot
plt.show()


# In[90]:


# Create a 2x2 subplot grid
fig, axs = plt.subplots(2, 2, figsize=(10, 8))  # Adjust figsize as needed

# Plot data in each panel
#axs[0, 0].scatter(dataout['VGSR'],dataout['FEH'],c='red',label = 'Draco Outskirts Members',s=10,zorder=0)
#axs[0, 0].errorbar(dataout['VRAD'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
axs[0, 0].scatter(dracohigh['VRAD'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='red',label = 'DESI High Probability Members',s=10,zorder=0)
axs[0, 0].set_xlabel('VRAD (km/s)')
axs[0, 0].set_ylabel('FEH')
axs[0, 0].scatter(walkerselall['vlos'],walkerselall['__Fe_H_'],c='b',label = 'Walker2015 Members',s=10,zorder=0,alpha = 0.3)

axs[0, 0].legend(loc=4,fontsize=10)


axs[0, 0].scatter(walkerselall['vlos'],walkerselall['__Fe_H_'],c='b',label = 'Walker2015 Members',s=10,zorder=0,alpha = 0.5)
axs[0, 1].scatter(dataout['VRAD'],dataout['FEH'],c='red',label = 'DESI Outskirts Members',s=10,zorder=0)
axs[0, 1].errorbar(dataout['VRAD'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
axs[0, 1].scatter(walkerselall[sep_constraintt]['vlos'],walkerselall[sep_constraintt]['__Fe_H_'],c='blue',label = 'Walker2015 Outskirts Members',s=10,zorder=0)
axs[0, 1].errorbar(walkerselall[sep_constraintt]['vlos'],walkerselall[sep_constraintt]['__Fe_H_'],xerr=walkerselall[sep_constraintt]['e__Fe_H_'],yerr= walkerselall[sep_constraintt]['e__Fe_H_'],linestyle = ' ',c='blue')

axs[0, 1].scatter(dracohigh['VRAD'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='grey',label = 'DESI High Probability Members',s=10,zorder=0)
#axs[0, 1].errorbar(dataout['VGSR'],dataout['FEH'],xerr=dataout['VRAD_ERR'],yerr= dataout['FEH_ERR'],linestyle = ' ',c='red')
#axs[0, 1].scatter(dracohigh['VGSR'][(testp)>proba],dracohigh['FEH'][(testp)>proba],c='grey',label = 'Draco High Probability Members',s=10,zorder=0)
axs[0, 1].set_xlabel('VRAD (km/s)')
axs[0, 1].set_ylabel('FEH')
axs[0, 1].legend(loc=2,fontsize=8)
#axs[1, 0].plot(data3)
#axs[1, 0].set_title('Panel 3')
axs[1, 0].scatter(dracohigh['PMRA_3'][testp>0.75],dracohigh['PMDEC_3'][testp>0.75],c='red',alpha=0.8, label = 'DESI Draco Members',s=10,zorder=5)
axs[1, 0].plot(walkerunique['pmra'],walkerunique['pmdec'],'.',c='b',label = 'Walker2015 All Members',alpha = 0.5)
axs[1, 0].set_xlabel(r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$")
axs[1, 0].set_ylabel(r"$\rm{\mu_{\delta} \ (mas/yr)}$")
axs[1, 0].legend(loc=2,fontsize=8)
axs[1, 1].scatter(dracohigh['PMRA_3'][testp>0.75],dracohigh['PMDEC_3'][testp>0.75],c='red',alpha=0.3, label = 'DESI Draco Members',s=10,zorder=5)
axs[1, 1].scatter(dataout['PMRA_3'],dataout['PMDEC_3'],c='red',label = 'DESI Outskirts Draco Members',s=10,zorder=0)
axs[1, 1].errorbar(dataout['PMRA_3'],dataout['PMDEC_3'],xerr=dataout['PMRA_ERROR'],yerr= dataout['PMDEC_ERROR'],linestyle = ' ',c='red')

axs[1, 1].plot(dataw['pmra'],dataw['pmdec'],'o',c='grey',label = 'Walker2015 Outskirts Members')
#axs[1, 1].scatter(dataw['pmra'].values[idx[sep_constraint]],dataw['pmdec'].values[idx[sep_constraint]],s=10,c='g',label = 'Matched Outskirts Members',zorder=4)
axs[1, 1].errorbar(dataw['pmra'],dataw['pmdec'],xerr=dataw['pmra_error'],yerr= dataw['pmra_error'],marker='o',linestyle = ' ',c='grey',label = 'Walker2015 Outskirts Members')
#axs[1, 1].set_title('Panel 4')
axs[1, 1].set_xlabel(r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$")
axs[1, 1].set_ylabel(r"$\rm{\mu_{\delta} \ (mas/yr)}$")
# Adjust layout to avoid overlapping titles
plt.tight_layout()
axs[1, 1].legend(loc=4,fontsize=8)
# Show the plot
plt.show()


# In[ ]:


dataw['pmra'].values[idx[sep_constraint]]


# In[ ]:


walkerunique


# In[ ]:


w1 = SkyCoord(ra=walkerunique['col1']*u.degree, dec=walkerunique['col2']*u.degree)
catalogw = SkyCoord(ra=walkerselall['RAJ2000']*u.degree, dec=walkerselall['DEJ2000']*u.degree)

max_sep = 5.0 * u.arcsec
idxw, d2d, d3d = c.match_to_catalog_3d(catalog)
sep_constraintw = d2d < max_sep

len(walkerselall[idxw[sep_constraintw]])


# In[ ]:


plt.scatter(dracohigh['PMRA_3'][testp>0.75],dracohigh['PMDEC_3'][testp>0.75],c='red',alpha=0.3, label = 'DESI Draco Members',s=10,zorder=5)
plt.scatter(dataout['PMRA_3'],dataout['PMDEC_3'],c='red',label = 'DESI Outskirts Draco Members',s=10,zorder=0)
plt.errorbar(dataout['PMRA_3'],dataout['PMDEC_3'],xerr=dataout['PMRA_ERROR'],yerr= dataout['PMDEC_ERROR'],linestyle = ' ',c='red')
plt.plot(walkerunique['pmra'],walkerunique['pmdec'],'.',c='b',label = 'Walker2015 All Members')
plt.plot(dataw['pmra'],dataw['pmdec'],'o',c='grey',label = 'Walker2015 Outskirts Members')
plt.scatter(dataw['pmra'].values[idx[sep_constraint]],dataw['pmdec'].values[idx[sep_constraint]],s=10,c='g',label = 'Matched Outskirts Members',zorder=4)
plt.errorbar(dataw['pmra'],dataw['pmdec'],xerr=dataw['pmra_error'],yerr= dataw['pmra_error'],marker='o',linestyle = ' ',c='grey',label = 'Walker2015 Outskirts Members')
plt.errorbar(dataw['pmra'].values[idx[sep_constraint]],dataw['pmdec'].values[idx[sep_constraint]],xerr=dataw['pmra_error'].values[idx[sep_constraint]],yerr= dataw['pmra_error'].values[idx[sep_constraint]],linestyle = ' ',c='green')
plt.xlabel('pmra')
plt.ylabel('pmdec')
plt.legend(loc=4)


# In[ ]:


plt.plot(walkerunique['pmra_error'],walkerunique['pmdec_error'],'.',c='b',label = 'All Members')
plt.plot(dataw['pmra_error'],dataw['pmdec_error'],'o',c='r',label = 'Outskirts Members')

plt.xlabel('pmra_err')
plt.ylabel('pmdec_err')
plt.legend(loc=2)


# In[ ]:


c = SkyCoord(ra=testiron['TARGET_RA_1']*u.degree, dec=testiron['TARGET_DEC_1']*u.degree)
catalog = SkyCoord(ra=walkerselected['ra']*u.degree, dec=walkerselected['dec']*u.degree)

max_sep = 1.0 * u.arcsec
idx, d2d, d3d = c.match_to_catalog_3d(catalog)
sep_constraint = d2d < max_sep
c_matches = c[sep_constraint]
catalog_matches = catalog[idx[sep_constraint]]


# In[ ]:


len(testiron['TARGET_RA_1'])


# In[ ]:


c3 = SkyCoord(ra=dracohigh['TARGET_RA_1'][testp>0.5]*u.degree, dec=dracohigh['TARGET_DEC_1'][testp>0.5]*u.degree)
catalog3 = SkyCoord(ra=walkerselected['ra'][idx[sep_constraint]]*u.degree, dec=walkerselected['dec'][idx[sep_constraint]]*u.degree)

max_sep3 = 1.0 * u.arcsec
idx3, d2d3, d3d3 = c3.match_to_catalog_3d(catalog3)
sep_constraint3 = d2d3 < max_sep3
c_matches3 = c3[sep_constraint3]
catalog_matches3 = catalog3[idx3[sep_constraint3]]


# In[ ]:


nonmatch = catalog3[idx3[sep_constraint3]]


# In[ ]:


len(nonmatch)


# In[ ]:


dataw['psfmag_i'].values[idx2[sep_constraint2]]


# In[ ]:


dataw['psfmag_i'].values


# In[ ]:


c = SkyCoord(ra=testiron['TARGET_RA_1']*u.degree, dec=testiron['TARGET_DEC_1']*u.degree)
catalog2 = SkyCoord(ra=dataw['col1']*u.degree, dec=dataw['col2']*u.degree)

max_sep2 = 2.0 * u.arcsec
idx2, d2d2, d3d2 = c.match_to_catalog_3d(catalog2)
sep_constraint2 = d2d2 < max_sep2
c_matches2 = c[sep_constraint2]
catalog_matches2 = catalog2[idx2[sep_constraint2]]


# In[ ]:


catalog_matches2


# In[ ]:


c = SkyCoord(ra=testiron['TARGET_RA_1']*u.degree, dec=testiron['TARGET_DEC_1']*u.degree)
catalog2 = SkyCoord(ra=qi['RA']*u.degree, dec=qi['DEC']*u.degree)

max_sep2 = 1.0 * u.arcsec
idx2, d2d2, d3d2 = c.match_to_catalog_3d(catalog2)
sep_constraint2 = d2d2 < max_sep2
c_matches2 = c[sep_constraint2]
catalog_matches2 = catalog2[idx2[sep_constraint2]]


# In[ ]:


len(catalog_matches2)


# In[ ]:


#CMD diagram with probabiltiy 
plt.style.use('classic')
plt.figure(figsize=(10,8),facecolor='white')
plt.grid(color='white')
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='grey',lw=2,label='Isochrone')
#norm = colors.Normalize(vmin=0, vmax=1) 
#combined_colors = np.append(np.concatenate(testp), [0.99815984, 0.55174294, 0.99967203, 0.99981849, 0.92997065, 0.96598971])

probb=0.75
#plt.scatter(dataout['gmag0']-dataout['rmag0'],dataout['rmag0'],s=150,c=np.concatenate(testp)[np.isin(datacut['REF_ID_1'],dataout['REF_ID_1'])],marker='^',label='Outskirts High Probability Members')
#plt.scatter(datacut['gmag0']-datacut['rmag0'],datacut['rmag0'],s=10,c=(testp),cmap='bwr',label='EDR Sample')
plt.scatter(datacut['gmag0']-datacut['rmag0'],datacut['rmag0'],s=10,c=(testp),cmap='bwr',label='EDR Sample')

#cbar=plt.colorbar()
#plt.scatter(dataout['gmag0']-dataout['rmag0'],dataout['rmag0'],s=150,c=(testp)[(testp)>probb][cut1],marker='^',cmap='viridis',label='Outskirts High Probability Members')
#cbar=plt.colorbar()
plt.scatter(walkerselected['psfmag_g']-walkerselected['psfmag_r'],walkerselected['psfmag_r'],s=50,marker='^',label='Walker2015 Probability Members',c ='grey',alpha=0.5)
#cbar=plt.colorbar()
plt.scatter(testiron['gmag0'][sep_constraint]-testiron['rmag0'][sep_constraint],testiron['rmag0'][sep_constraint],s=50,marker='^',label='Matched Walker2015 Probability Members',c ='green')
#cbar=plt.colorbar()
plt.scatter(testiron['gmag0'][sep_constraint2]-testiron['rmag0'][sep_constraint2],testiron['rmag0'][sep_constraint2],s=50,marker='s',label='Qi Outskirts High Probability Members')
#cbar=plt.colorbar()
plt.grid(False)
# Customize the background color
plt.rcParams['figure.facecolor'] = 'white'

# Customize the tick labels font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set the title for the colorbar
#cbar.set_label('Probability')
plt.ylim(21, 16)
plt.xlim(-0.4,1.5)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')
plt.legend(loc=2,fontsize="11")
#plt.savefig('prob_gsr.pdf')


# In[ ]:


#CMD diagram with probabiltiy 
plt.style.use('classic')
plt.figure(figsize=(10,8),facecolor='white')
plt.grid(color='white')
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='grey',lw=2,label='Isochrone')
#norm = colors.Normalize(vmin=0, vmax=1) 
#combined_colors = np.append(np.concatenate(testp), [0.99815984, 0.55174294, 0.99967203, 0.99981849, 0.92997065, 0.96598971])

probb=0.75
#plt.scatter(dataout['gmag0']-dataout['rmag0'],dataout['rmag0'],s=150,c=np.concatenate(testp)[np.isin(datacut['REF_ID_1'],dataout['REF_ID_1'])],marker='^',label='Outskirts High Probability Members')
plt.scatter(datacut['gmag0']-datacut['rmag0'],datacut['rmag0'],s=10,c=(testp),cmap='bwr',label='EDR Sample')

cbar=plt.colorbar()
plt.scatter(dataout['gmag0']-dataout['rmag0'],dataout['rmag0'],s=150,c=(testp)[(testp)>probb][cut1],marker='^',cmap='viridis',label='Outskirts High Probability Members')
cbar=plt.colorbar()
plt.grid(False)
# Customize the background color
plt.rcParams['figure.facecolor'] = 'white'

# Customize the tick labels font size
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set the title for the colorbar
cbar.set_label('Probability')
plt.ylim(21, 16)
plt.xlim(-0.4,1.5)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# In[ ]:





# # Exponential fit

# In[170]:


def ellipse_bin_pro(xdata,ydata,vel,theta0=89*0.0174533,r0=10/60,e=0.31,number=2,ra0=ra0,dec0=dec0,prob=1,proba=0.):
    '''
    function for using ellipse bins to measure the number of stars in each bin
    :param xdata,ydata: ra and dec data
    :param theta0: position angle
    :param r0: half light radius
    :param e float: ellipticity 
    :param number: number of ellipse to be used
    :param ra0,dec0: galaxies center
    :param proba:counting probability > proba stars  

    :return: ellipses plotting and binning data (tot,prob,ra,dec,dis,probave)

    '''
    plt.figure(figsize=(10,6))
    fig, ax = plt.subplots(figsize=(10,6))
    err = []
    r=r0
    e=e
    theta0=theta0
    tot= [] 

    ra=[]
    dec=[]
    dis=[]
    probave=[]
    logg = []
    distot = []
    dis2=[]
    den=[]
    veldis =[]
    #rtot = [0.2,0.5,1.0,1.5,2,2.5,3.,3.5,4,4,5.5,6.5,8.0]
    rtot = [0.00001,0.25,0.5,0.75,1.0,1.25,1.5,2,3.,4.5,6,7,8.0,10,12]
    for ii in range(len(rtot)-1):
        
    
    
        
        b=r*rtot[ii]*np.sqrt(1-e)
        a=r*rtot[ii]/np.sqrt(1-e)
        b2=r*rtot[ii+1]*np.sqrt(1-e)
        a2=r*rtot[ii+1]/np.sqrt(1-e)
#radius = 42/60
#radius2 = 5*42/60
# Set the center coordinates of the circle
        center = (ra0, dec0)

# Generate an array of angles from 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, 100)

# Calculate the x and y coordinates of the points on the circumference of the circle
        xt =   a * np.cos(theta)
        yt =  b* np.sin(theta)
        
        #xd=np.cos(yt)*np.sin((xt-ra0)*0.0174533)/0.0174533
        #yd=(np.sin(yt)*np.cos(dec0*0.0174533)-np.cos(yt)*np.sin(dec0*0.0174533)*np.cos((xt-ra0)*0.0174533))/0.0174533
        #xd1=xdata-center[0]
        #yd1=ydata-center[1]
        #xdl=xdata-center[0]
        #ydl=ydata-center[1]
        #x1=xdl
        #y1=ydl
        #x1= xd1*np.sin(theta0)- yd1*np.cos(theta0)

        #y1= xd1*np.cos(theta0)+ yd1*np.sin(theta0)
        #x1= xd1*np.cos(theta0)- yd1*np.sin(theta0)

        #y1= xd1*np.sin(theta0)+ yd1*np.cos(theta0)
        x1=np.cos(ydata*0.0174533)*np.sin((xdata-ra0)*0.0174533)/0.0174533
        y1=(np.sin(ydata*0.0174533)*np.cos(dec0*0.0174533)-np.cos(ydata*0.0174533)*np.sin(dec0*0.0174533)*np.cos((xdata-ra0)*0.0174533))/0.0174533
        cut1 = (x1**2/a**2+y1**2/b**2)>1
        cut2 = (x1**2/a2**2+y1**2/b2**2)<1
       
        tot.append(len(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba]))
        err.append(np.sqrt(len(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba]))/(np.pi*a2*b2-(np.pi*a*b)))
        den.append(len(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba])/(np.pi*a2*b2-(np.pi*a*b)))
       # gco = np.histogram(vel[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],bins=8)
       # popt, covariance = curve_fit(gauss, gco[1][1:], gco[0],p0=[10,-290,2])

        #popt, covariance = curve_fit(gauss, gco[1][1:], gco[0],p0=[10,-290,2])
        #fit_y = gauss(gco[1][1:],*popt)
        #print (popt, np.sqrt(np.diagonal(covariance)))
        #plt.plot(gco[1][1:], gco[0], 'o', label='data')
        #plt.plot(gco[1][1:], fit_y, '-', label='fit')
        #plt.legend()
        #plt.show()
        #logg.append(testiron['LOGG'][np.isin(testiron['VRAD'],datacut['VRAD'])][[(cut1)&(cut2)]>proba])

        #dis.append(2/3*((r*rtot[ii+1])**3-(r*rtot[ii])**3)/((r*rtot[ii+1])**2-(r*rtot[ii])**2))
        dis.append(np.sqrt(((r*rtot[ii+1])**2+(r*rtot[ii])**2)/2))
        #dis.append(np.sqrt(((r*rtot[ii+1])**2-(r*rtot[ii])**2)/2))
        #prob.append(sub[(cut1)&(cut2)])
        print (len(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba]))
        
        ra.append(xdata[(cut1)&(cut2)])
        dec.append(ydata[(cut1)&(cut2)])
        
        veldis.append(np.std(vel[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba]))
        #veldis.append([popt[2],np.sqrt(np.diagonal(covariance))[2]])
        
        probave.append(prob)
        dis2.append(r*rtot[ii])
        #print (vel[(cut1)&(cut2)])
        
        
        
        

# Plot the circle
        #ax.plot(xt, yt,label = str(ii)+' r$_{h}$',c='k')
        plt.scatter(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],y1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],s=6)
        #cbar=plt.colorbar()
        #cbar.set_label('Iron-Pace&LI Probability')
        #plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
#plt.ylim(21, 16)
        #plt.xlim(258,262)
        #plt.ylabel('DEC')
        #plt.xlabel('RA')
        #cbar=plt.colorbar()
   # cbar.set_label('Iron Probability')   
    return tot,ra,dec,dis,probave,np.sqrt(x1**2+y1**2),den,err,dis2,veldis




# Set the aspect ratio of the plot to 'equal' to make the circle appear circular
#ax.set_aspect('equal')

# Set the radius of the circle to be the half light radiu


def ellipse_bin_pro_noprob(xdata,ydata,theta0=89*0.0174533,r0=1/60,e=0.31,number=2,ra0=ra0,dec0=dec0):
    '''
    function for using ellipse bins to measure the number of stars in each bin
    :param xdata,ydata: ra and dec data
    :param theta0: position angle
    :param r0: half light radius
    :param e float: ellipticity 
    :param number: number of ellipse to be used
    :param ra0,dec0: galaxies center
    :param proba:counting probability > proba stars  

    :return: ellipses plotting and binning data (tot,prob,ra,dec,dis,probave)

    '''
    plt.figure(figsize=(10,6))
    fig, ax = plt.subplots(figsize=(10,6))
    dis2=[]
    err = []
    r=r0
    e=e
    theta0=theta0
    tot= [] 

    ra=[]
    dec=[]
    dis=[]
    probave=[]
    logg = []
    distot = []
    den=[]
    veldis=[]
    #rtot = [0.2,0.5,1.0,1.5,2,2.5,3.,3.5,4,4,5.5,6.5,8.0]
    rtot = [0.00001,0.25,0.5,0.75,1.0,1.25,1.5,2,3.,4.5,6,7,8.0,10,12]
    for ii in range(len(rtot)-1):
        
        plt.figure(figsize=(10,6))
        fig, ax = plt.subplots(figsize=(10,6))
    
        
        b=r*rtot[ii]*np.sqrt(1-e)
        a=r*rtot[ii]/np.sqrt(1-e)
        b2=r*rtot[ii+1]*np.sqrt(1-e)
        a2=r*rtot[ii+1]/np.sqrt(1-e)
#radius = 42/60
#radius2 = 5*42/60
# Set the center coordinates of the circle
        center = (ra0, dec0)

# Generate an array of angles from 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, 100)

# Calculate the x and y coordinates of the points on the circumference of the circle
        xt =   a * np.cos(theta)
        yt =  b* np.sin(theta)
        
        #xd=np.cos(yt)*np.sin((xt-ra0)*0.0174533)/0.0174533
        #yd=(np.sin(yt)*np.cos(dec0*0.0174533)-np.cos(yt)*np.sin(dec0*0.0174533)*np.cos((xt-ra0)*0.0174533))/0.0174533
        #xd1=xdata-center[0]
        #yd1=ydata-center[1]
        #xdl=xdata-center[0]
        #ydl=ydata-center[1]
        #x1=xdl
        #y1=ydl
        #x1= xd1*np.sin(theta0)- yd1*np.cos(theta0)

        #y1= xd1*np.cos(theta0)+ yd1*np.sin(theta0)
        #x1= xd1*np.cos(theta0)- yd1*np.sin(theta0)

        #y1= xd1*np.sin(theta0)+ yd1*np.cos(theta0)
        x1=np.cos(ydata*0.0174533)*np.sin((xdata-ra0)*0.0174533)/0.0174533
        y1=(np.sin(ydata*0.0174533)*np.cos(dec0*0.0174533)-np.cos(ydata*0.0174533)*np.sin(dec0*0.0174533)*np.cos((xdata-ra0)*0.0174533))/0.0174533
        cut1 = (x1**2/a**2+y1**2/b**2)>1
        cut2 = (x1**2/a2**2+y1**2/b2**2)<1
       
        tot.append(len(x1[(cut1)&(cut2)]))
        err.append(np.sqrt(len(x1[(cut1)&(cut2)]))/(np.pi*a2*b2-(np.pi*a*b)))
        den.append(len(x1[(cut1)&(cut2)])/(np.pi*a2*b2-(np.pi*a*b)))
        #logg.append(testiron['LOGG'][np.isin(testiron['VRAD'],datacut['VRAD'])][[(cut1)&(cut2)]>proba])
        
       
        #dis.append(2/3*((r*rtot[ii+1])**3-(r*rtot[ii])**3)/((r*rtot[ii+1])**2-(r*rtot[ii])**2))
        dis.append(np.sqrt(((r*rtot[ii+1])**2+(r*rtot[ii])**2)/2))
        dis2.append(r*rtot[ii])
        #prob.append(sub[(cut1)&(cut2)])
        #print (len(xd1[(cut1)&(cut2)]))
        ra.append(xdata[(cut1)&(cut2)])
        dec.append(ydata[(cut1)&(cut2)])
        #probave.append(prob)
        
        
        
        

# Plot the circle
        ax.plot(xt, yt,label = str(ii)+' r$_{h}$',c='k')
        plt.scatter(x1[(cut1)&(cut2)],y1[(cut1)&(cut2)],s=6)
        #cbar=plt.colorbar()
        #cbar.set_label('Iron-Pace&LI Probability')
        #plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
#plt.ylim(21, 16)
        #plt.xlim(258,262)
        plt.ylabel('DEC')
        plt.xlabel('RA')
        #cbar=plt.colorbar()
   # cbar.set_label('Iron Probability')   
    return tot,ra,dec,dis,probave,np.sqrt(xd1**2+yd1**2),den,err,dis2




# Set the aspect ratio of the plot to 'equal' to make the circle appear circular
#ax.set_aspect('equal')

# Set the radius of the circle to be the half light radiu








# In[171]:


tot,ra,dec,r,probave,distot,dend,draerr,dis2,vral=ellipse_bin_pro(dracohigh['TARGET_RA_1'],dracohigh['TARGET_DEC_1'],dracohigh['VRAD'],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0,prob=(testp),proba=0.5)





# In[172]:


#Ting's file for completeness correction dr2
plt.style.use('classic')
targtab=Table.read('oc_gc_dg_targets.txt',format='ascii')
targra=targtab['col1']
targdec=targtab['col2']
targcoord=SkyCoord(targra*u.degree,targdec*u.degree,frame='icrs')
ra0 = 260.0517
dec0 =  57.9153
draco=SkyCoord(ra0*u.degree, dec0*u.degree,frame='icrs')
targsep=targcoord.separation(draco)
itargd,=np.where(targsep.value<2.)
irad,=np.where((testiron['TARGET_RA_1'] > 257) & (testiron['TARGET_RA_1'] < 263) & (testiron['TARGET_DEC_1'] > 55.9) & (testiron['TARGET_DEC_1'] < 59.9))
plt.plot(testiron['TARGET_RA_1'][irad],testiron['TARGET_DEC_1'][irad],'bo',ms=1)
plt.plot(targra[itargd],targdec[itargd],'ro',ms=1)


# In[173]:


mask =(1<<10)+(1<<38)
targiqk = np.where(((testiron['SV1_SCND_TARGET'][iqk] & mask) > 0))[0]


# In[174]:


allspectot,allspecxvec,allspecyvec,allspecrad,allspecprobave,allspecdistot,allspecden,allerr,dis2=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][irad],testiron['TARGET_DEC_1'][irad],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)
spectot,specxvec,specyvec,specrad,specprobave,specdistot,specden,specerr,dis2=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][iqk][targiqk],testiron['TARGET_DEC_1'][iqk][targiqk],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)

targtot,targxvec,targyvec,targrad,targprobave,targdistot,targden,tarerr,dis2=ellipse_bin_pro_noprob(targra[itargd],targdec[itargd],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)
 


# In[175]:


colorcut1 = cmd_selection(testiron[irad], dm, g_r, iso_r, cwmin=0.75, dobhb=True)


# In[176]:


len(colorcut1)


# In[177]:


colorcut2 = cmd_selection(testiron[iqk][targiqk], dm, g_r, iso_r, cwmin=0.75, dobhb=True)


# In[178]:


allspectot,allspecxvec,allspecyvec,allspecrad,allspecprobave,allspecdistot,allspecden,allerr,allr=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][irad][colorcut1],testiron['TARGET_DEC_1'][irad][colorcut1],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)
                                                                                                      

spectot,specxvec,specyvec,specrad,specprobave,specdistot,specden,err,specr=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][iqk][targiqk][colorcut2],testiron['TARGET_DEC_1'][iqk][targiqk][colorcut2],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)



# In[179]:


#plummer model comparison
theta0=89*0.0174533
#print (np.cos(90*0.0174533))
r0=10/60
e=0.31
a=r0/np.sqrt(1-e)
a=8.97/60
proba=0.
#x=np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba])*np.sin(datacut['TARGET_RA_1'][np.concatenate(testp)>proba]-ra0)
#y=np.sin(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba])*np.cos(dec0)-np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba])*np.sin(dec0)*np.cos(datacut['TARGET_RA_1'][np.concatenate(testp)>proba]-ra0)
x=np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)
y=np.sin(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][(testp)>proba]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][(testp)>proba]-ra0)*0.0174533)

x=x/0.0174533
y=y/0.0174533
Rx= x*np.cos(theta0)-y*np.sin(theta0)
Ry = x*np.sin(theta0)+y*np.cos(theta0)

Rx=Rx
Ry = Ry
R2 = Rx**2+Ry**2/(1-e)**2
Rx=np.arange(0,5,0.01)
Ry=np.arange(0,5,0.01)
R2 = Rx**2+Ry**2/(1-e)**2
den=1/(np.pi*a**2*(1-e))*(1+R2/a**2)**(-2)
#tratio=np.asarray(targden)/np.asarray(specden)
#print(tratio)
alltratio=np.asarray(targden)/np.asarray(allspecden)
print(alltratio)


# In[180]:


plt.plot(np.asarray(r)*60,dend*alltratio,'bo')
#plt.plot(np.asarray(r)*60,dend*tratio,'go')
plt.plot(np.asarray(r)*60,dend,'ro')
plt.plot(np.sqrt(R2)*60,den*560,'-')
plt.xlim(1,120)
plt.ylim(0.25,15000)
plt.xlabel('R Arcmin')

plt.yscale('log')
plt.yscale('log')
plt.xscale('log')


# In[181]:


#Ting's new file for completeness correction dr2
plt.style.use('classic')
targtab=Table.read('MWS_cluster_galaxy_deep.txt',format='ascii')
targra=targtab['col1']
targdec=targtab['col2']
targcoord=SkyCoord(targra*u.degree,targdec*u.degree,frame='icrs')
ra0 = 260.0517
dec0 =  57.9153
draco=SkyCoord(ra0*u.degree, dec0*u.degree,frame='icrs')
targsep=targcoord.separation(draco)
itargd,=np.where(targsep.value<2.)
irad,=np.where((testiron['TARGET_RA_1'] > 257) & (testiron['TARGET_RA_1'] < 263) & (testiron['TARGET_DEC_1'] > 55.9) & (testiron['TARGET_DEC_1'] < 59.9))
plt.plot(testiron['TARGET_RA_1'][irad],testiron['TARGET_DEC_1'][irad],'bo',ms=1)
plt.plot(targra[itargd],targdec[itargd],'ro',ms=1)


# In[182]:


allspectot,allspecxvec,allspecyvec,allspecrad,allspecprobave,allspecdistot,allspecden,allerr,dis2=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][irad],testiron['TARGET_DEC_1'][irad],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)
spectot,specxvec,specyvec,specrad,specprobave,specdistot,specden,specerr,dis2=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][iqk][targiqk],testiron['TARGET_DEC_1'][iqk][targiqk],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)

targtot,targxvec,targyvec,targrad,targprobave,targdistot,targden,tarerr,dis2=ellipse_bin_pro_noprob(targra[itargd],targdec[itargd],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)
 


# In[183]:


alltrationew=np.asarray(targden)/np.asarray(allspecden)
print(alltrationew)


# In[184]:


plt.plot(np.asarray(r)*60,dend*alltrationew,'bo')
#plt.plot(np.asarray(r)*60,dend*tratio,'go')
plt.plot(np.asarray(r)*60,dend,'ro')
plt.plot(np.sqrt(R2)*60,den*560,'-')
plt.xlim(1,120)
plt.ylim(0.25,15000)
plt.xlabel('R Arcmin')

plt.yscale('log')
plt.yscale('log')
plt.xscale('log')


# In[185]:


allerr/np.asarray(allspecden)

tarerr/np.asarray(targden)

draerr/np.array(dend)

draerr

errtot = dend*alltratio*np.sqrt((draerr/np.array(dend))**2+(allerr/np.asarray(allspecden))**2+(tarerr/np.asarray(targden))**2)


# In[272]:


plt.plot(np.asarray(r)*60,dend*alltrationew,'bo',label = 'Corrected Surface Density')
#plt.plot(xd,0.5* np.exp(np.array((xd-50)*(-0.17)))+10000)
plt.plot(xdfit,a_fit* np.exp(np.array((xdfit)*(b_fit)))+c_fit,c='purple')
#plt.plot(xd, y_fit, 'r--', label='Exponential Fitted Curve')
#plt.plot(np.asarray(r)*60,dend*tratio,'go')
plt.plot(np.asarray(r)*60,dend,'ro',label = 'Surface Density')
plt.plot(np.sqrt(R2)*60,den*560,'k-',label = 'Plummer Model')
plt.plot(np.asarray(r)*60,dend*alltratio,'go','Old Correction')
plt.errorbar(np.array(r)*60,dend*alltrationew, yerr=errtot, capsize=3, ls='none', color='black', 
            elinewidth=1)
plt.errorbar(np.array(r)*60,dend, yerr=draerr, capsize=3, ls='none', color='black', 
            elinewidth=1)
plt.axvline(x=43,linestyle='--',label='Tidal Radius')
plt.xlim(1,120)
plt.ylim(0.4,30000)
plt.xlabel('R arcmin')
plt.ylabel(r'Star Density [arcmin$^{-2}$]')
plt.yscale('log')
plt.yscale('log')
plt.xscale('log')
plt.legend(loc=3)
#plt.savefig('plummer_gsr.pdf')


# In[187]:


a_fit, b_fit, c_fit,_


# In[232]:


exponential_decay(18.88, a_fit, b_fit, c_fit)


# In[231]:


xdfit = np.arange(0,18.89,0.01)
xdfit[-1]


# In[268]:


# Exponential decay function
def exponential_decay(x, a, b, c):
    return a * np.exp(b* (x)) +c
num=9
#num2 = 
xd = np.asarray(r)*60
yd = dend*alltratio
xdfit = np.arange(0,22.84,0.01)

# Fit exponential decay curve
params, _ = curve_fit(exponential_decay,xd[:num], yd[:num],p0=[4,-0.17,-658])

# Extract fitted parameters
a_fit, b_fit, c_fit = params

# Generate fitted curve
y_fit = exponential_decay(xdfit, a_fit, b_fit, c_fit)

plt.scatter(xd, yd ,label='Corrected SD New')
#plt.plot(x, y_true, 'k-', label='True Curve')
plt.plot(xdfit, y_fit, 'r--', label='Exponential Fitted Curve')
plt.legend(loc=3)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exponential Decay Curve Fitting')
plt.yscale('log')
plt.xscale('log')
plt.ylim(-10,25000)
plt.xlim(1,120)
plt.show()


# In[190]:


y_fit


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




