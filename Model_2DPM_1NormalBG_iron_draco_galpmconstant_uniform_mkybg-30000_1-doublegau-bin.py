#!/usr/bin/env python
# coding: utf-8

# # Iron Draco
# This notebook presents the mixture model of 3 gaussians built for Iron Draco data. The data is taken from the S5 Collaboration. With quality cut, we obtained 371 stars with good measurements to feed the model. The mixture model is built with 16 parameters, including radial velocity, metallicity and proper motion parameters of the smcnod and a set of parameters for the background components. We fit a Gaussian mixture model to this data using `emcee`.

# In[4]:


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
from scipy import interpolate
from scipy import stats
#import uncertainties.umath as um
#from uncertainties import ufloat
font = {'family' : 'normal',
        'weight' : 'bold',
        'size'   : 22}

plt.rc('font', **font)

import imp
from astropy.io import fits as pyfits
import pandas as pd
import scipy.optimize as optim


# ## Iron Data Loading

# In[ ]:





# In[3]:


#data loading for DESI iron

ironrv = t1 = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[1].data)
t1_fiber = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[2].data)
t4 = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[4].data)

t1_comb = table.hstack((t1,t1_fiber,t4))


# In[5]:


len(set(t1_comb['REF_ID_2']))


# In[6]:


#isochrone loading with a age = 10 Gyr 
#Properties for the isochrone 
#MIX-LEN  Y      Z          Zeff        [Fe/H] [a/Fe]
# 1.9380  0.2459 5.4651E-04 5.4651E-04  -1.50   0.00 
iso_file = pd.read_csv('./draco_files/isochrone_10_1.csv')


# In[ ]:





# # Colorcut for better selection

# In[7]:


print('# before unique selection:', len(t1_comb))

# do a unique selection based on TARGET ID. Keep the first one for duplicates 
# (and first one has the smallest RV error)
t1_unique = table.unique(t1_comb, keys='TARGETID_1', keep='first')
print('# after unique selection:', len(t1_unique))


# In[ ]:





# In[8]:


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





# In[9]:


testiron=t1_unique


# In[10]:


rv_to_gsr(coord.SkyCoord(ra=t1_unique['TARGET_RA_1'][0]*u.deg, dec=t1_unique['TARGET_DEC_1'][0]*u.deg,radial_velocity=562.24*u.km/u.s, frame='icrs'))




# In[11]:


testiron['VRAD']


# In[12]:


testiron


# In[13]:


#dust extinction correction
testiron['gmag'], testiron['rmag'], testiron['zmag'] = [22.5-2.5*np.log10(testiron['FLUX_'+_]) for _ in 'GRZ']

testiron['gmag0'] = testiron['gmag'] - testiron['EBV_2'] * 3.186
testiron['rmag0'] = testiron['rmag'] - testiron['EBV_2'] * 2.140
testiron['zmag0'] =testiron['zmag'] - testiron['EBV_2'] * 1.196
testiron['gmagerr'], testiron['rmagerr'], testiron['zmagerr'] = [2.5/np.log(10)*(np.sqrt(1./testiron['FLUX_IVAR_'+_])/testiron['FLUX_'+_]) for _ in 'GRZ']



# In[14]:


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


# In[15]:


#quality cut, exclude nans and RA/DEC cut
iqk,=np.where((testiron['TARGET_RA_1'] > 257) & (testiron['TARGET_RA_1'] < 263) & (testiron['TARGET_DEC_1'] > 55.9) & (testiron['TARGET_DEC_1'] < 59.9) & (testiron['RVS_WARN']==0) &(testiron['RR_SPECTYPE']!='QSO')&(testiron['VSINI']<50)
             
      &(~np.isnan(testiron["PMRA_ERROR"])) &(~np.isnan(testiron["PMDEC_ERROR"])) &(~np.isnan(testiron["PMRA_PMDEC_CORR"])) &(testiron["FEH"] >-3.8)&(testiron["FEH"] <-0.3) &(testiron["LOGG"] <4))    






# In[16]:


#making CMD diagram for the data
ra0 = 260.0517
dec0 =  57.9153
rad0 =1.6

stars = SkyCoord(ra=testiron['TARGET_RA_1'], dec=testiron['TARGET_DEC_1'], unit=u.deg)

# Calculate the angular separation between stars and the reference point
separations = stars.separation(SkyCoord(ra=ra0, dec=dec0, unit=u.deg))

testiron['dist1'] = separations


# In[17]:


ind = (testiron['dist1'][iqk] <0.3)
dm=19.53


# In[18]:


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


# In[19]:


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





# In[20]:


xcut = (((iso_file['DECam_g']-iso_file['DECam_r']) < 1.8)& ((iso_file['DECam_g']-iso_file['DECam_r']) > -0.5))

ycut = (((iso_file['DECam_r']+dm) < 21)& ((iso_file['DECam_r']+dm) > 15.5))
fiso = interpolate.interp1d(((iso_file['DECam_g'].values)-(iso_file['DECam_r']).values)[xcut&ycut][-5:-1],((iso_file['DECam_r'].values)+dm)[xcut&ycut][-5:-1],kind='cubic',fill_value='extrapolate')



def extrapolate_poly(x, y, x_new):
    # Perform polynomial regression
    coeffs = np.polyfit(x, y, 1)

    # Extrapolate the polynomial
    y_new = np.polyval(coeffs, x_new)
    return y_new



# In[21]:


((iso_file['DECam_g'].values)-(iso_file['DECam_r']).values)[xcut&ycut]


# In[22]:


isox = np.arange(1.18,1.5,0.1)


# In[ ]:





# In[23]:


fiso=extrapolate_poly(((iso_file['DECam_g'].values)-(iso_file['DECam_r']).values)[xcut&ycut][-5:-1],((iso_file['DECam_r'].values)+dm)[xcut&ycut][-5:-1], isox)




# In[24]:


fiso


# In[25]:


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
    rmax = 21
    grmin = -0.5
    grmax = 1.3
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
    grw_bhb = 0.5 # BHB width in gr
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




# In[26]:


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


# In[27]:


iso_r = np.append(iso_file['DECam_r'].values[xcut&ycut],fiso-dm)
g_r = np.append(((iso_file['DECam_g'].values)-(iso_file['DECam_r']).values)[xcut&ycut],isox)


# In[ ]:





# In[28]:


colorcut = cmd_selection(testiron[iqk], dm, g_r,iso_r, cwmin=0.78, dobhb=True)


# In[ ]:





# In[29]:


len(testiron['gmag0'][iqk][colorcut])


# In[30]:


#colorcut for the sample 
#Investigating the data sample after colorcut 


plt.figure(figsize=(10,10))
plt.plot(g_r,iso_r+dm,c='orange',lw=2,label = 'Isochrone')
#plt.scatter(testiron['gmag0'][iqk]-testiron['rmag0'][iqk],testiron['rmag0'][iqk],s=1,c='k',alpha=1,label ='Iron Data',cmap='bwr')
#plt.colorbar()
plt.scatter(testiron['gmag0'][iqk][colorcut]-testiron['rmag0'][iqk][colorcut],testiron['rmag0'][iqk][colorcut],s=3,c='k',alpha=1,label ='Iron Data',cmap='bwr')
#plt.colorbar()
grw = np.sqrt(0.1**2 + (3*10**log10_error_func(iso_file['DECam_r']+dm, *popt))**2)



#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_file['DECam_r']+dm,'--r')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm,'--r')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_file['DECam_r']+dm+0.3,'--k',label='Colorcut Region')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm-0.3,'--k')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_file['DECam_r']+dm-1,'--k')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm+1,'--k')

plt.plot(des_m92_hb_g-des_m92_hb_r, des_m92_hb_r+dm, lw=2, color='orange')
#plt.plot(des_m92_hb_g-des_m92_hb_r, des_m92_hb_r+dm-0.4,'--k')
#plt.plot(des_m92_hb_g-des_m92_hb_r, des_m92_hb_r+dm+0.4,'--k')
#plt.plot(des_m92_hb_g-des_m92_hb_r-0.1, des_m92_hb_r+dm,'--r')
#plt.plot(des_m92_hb_g-des_m92_hb_r+0.1, des_m92_hb_r+dm,'--r')
plt.legend(loc=2)
plt.ylim(21, 15.5)
plt.xlim(-0.3,1.5)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# In[31]:


plt.scatter(testiron['gmag0'][iqk]-testiron['rmag0'][iqk],testiron['rmag0'][iqk],s=3)
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='orange',lw=2)

plt.ylim(21, 15.5)
plt.xlim(-0.4,1.8)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')


# In[32]:


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

# In[33]:


print (len(testiron["VRAD"][iqk]))


# In[34]:


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


# In[35]:


plt.scatter(testiron["FEH"][iqk][colorcut],testiron["LOGG"][iqk][colorcut],s=2)
plt.xlabel('FEH')
plt.ylabel('LOGG')


# In[36]:


plt.hist(feherr)


# In[37]:


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

# In[38]:


def data_collect(datafile,ramin,ramax,decmin,decmax,fehmin,fehmax,vmin,vmax,logg,galdis,dm,cwmin,g_r,iso_r):
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
    
    colorcut = cmd_selection(testiron[iqk], dm, g_r, iso_r, cwmin=cwmin, dobhb=True)
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


    return [rv,rverr,feh,feherr,pms,pmcovs],datacut,pmnorm


    


    


# In[237]:


#try 257/263 253/267

datasum,datacut,pmnorm =data_collect(testiron,253,267,55.9,59.9,-3.5,-0.5,50,-250,4,0.025,dm,0.75,g_r=g_r,iso_r=iso_r)




# In[238]:


pmnorm


# In[239]:


np.min(datacut['rmag0']),np.max(datacut['rmag0'])


# In[42]:


plt.figure(figsize=(10,10))
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
#plt.plot((iso_file['DECam_g']-iso_file['DECam_r']+grw)[cut2], iso1[cut2],'--k',label='Colorcut Region')
#plt.plot((iso_file['DECam_g']-iso_file['DECam_r']+grw)[cut3], iso1[cut3],'--k',label='Colorcut Region')
#plt.plot((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut1], iso2[cut1] ,'--k')
#plt.plot((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut3], iso2[cut3] ,'--k')
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']+grw, iso_cut1
#plt.plot(iso_file['DECam_g']-iso_file['DECam_r']-grw, iso_file['DECam_r']+dm+1,'--k')
#plt.plot([np.min((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut3]),np.min((iso_file['DECam_g']-iso_file['DECam_r']+grw)[cut2])],[np.max(datacut['rmag0']),np.max(datacut['rmag0'])],'--',c='k')
#plt.plot(des_m92_hb_g-des_m92_hb_r, des_m92_hb_r+dm, lw=2, color='orange')
plt.plot(np.max(des_m92_hb_g-des_m92_hb_r),np.min(des_m92_hb_r+dm-0.5))
plt.plot((des_m92_hb_g-des_m92_hb_r)[des_m92_hb_g-des_m92_hb_r>-0.4], (des_m92_hb_r+dm-0.5)[des_m92_hb_g-des_m92_hb_r>-0.4],'--k')
plt.plot((des_m92_hb_g-des_m92_hb_r)[des_m92_hb_g-des_m92_hb_r>-0.4], (des_m92_hb_r+dm+0.5)[des_m92_hb_g-des_m92_hb_r>-0.4],'--k')
plt.plot([np.min((iso_file['DECam_g']-iso_file['DECam_r']-grw)[cut1]),np.max(des_m92_hb_g-des_m92_hb_r)],[np.max(iso2[cut1]),np.min(des_m92_hb_r+dm-0.5)],'--k')
#plt.a x= 0.3
#hor = 
#plt.axhline(y=np.min(datacut['rmag0']),xmin = 0.63,xmax =0.98,c='k',linestyle='--')
#plt.axvline(x=-0.33,ymin = 0.03,ymax = 0.25,c='k',linestyle='--')
#plt.plot(des_m92_hb_g-des_m92_hb_r-0.1, des_m92_hb_r+dm,'--r')
#plt.plot(des_m92_hb_g-des_m92_hb_r+0.1, des_m92_hb_r+dm,'--r')
plt.legend(loc=2)
plt.ylim(21, 16)
plt.xlim(-0.4,1.5)
plt.ylabel('rmag0')
plt.xlabel('gmag0-rmag0')
#plt.savefig('colorcut.pdf')


# In[44]:


def two_gnfunc(dataarr, gnvals,params):
    '''
    function for model of the two gaussians
    :param dataarr: data array 
    :param params: parameter for the two gaussian distributions (center (center for gaussian narrow gaussian)
    sigma (sigma for gaussian narrow gaussian), center2 = (center2 for gaussian narrow gaussian),sigma2 = (sigma2 for gaussian narrow gaussian) 
    amp1 (fraction of area under narrow gaussian/area under broad gaussian)) 
    :return: normalizied two gaussian function
    '''
    
    center = params[0]
    sigma = params[1]
    center2 = params[2]
    sigma2 = params[3]
    Amp1 = params[4]
    
    #print(center, sigma, center2, sigma2, Amp1)
    
    gnvals = Amp1 * np.exp(-(dataarr-center)**2/(2*sigma*sigma))/sigma/np.sqrt(2*np.pi)\
            + (1-Amp1) * np.exp(-(dataarr-center2)**2/(2*sigma2*sigma2))/sigma2/np.sqrt(2*np.pi)
    return gnvals


# In[45]:


def twogau_like( xvals,gnvals,params):
    #likelihood used in fitting (normalizied)
    modelvals = two_gnfunc(xvals, gnvals,params)
   
    mlikelihood = - np.sum(np.log(modelvals)) 
    
    #print(mlikelihood)
    return (mlikelihood)



def fitting_reunbin(x,y):

   

    #guess6= [x_0, fwhm,c2,w2, h2] bounds=[(-10,10),(0,10),(-10,10),(0,10),(1e-5,1-1e-5)]
   
   # global xdata,ydata
   # xdata=x
    #ydata=y
    #LL = -np.sum(stats.norm.logpdf(ydata, loc=yPred, scale=sd) )
    #print (x)
    #bds1=((-100,100),(0,100),(-100,100),(100,1000),(1e-5,1-1e-5))
    #optim = minimize(twogau_like, guess6,args=(x),method = 'TNC',bounds=bds1,  options={'maxfun':100000,'disp':True})
    res2 = optim.minimize(twogau_like,[0,1,0,2,0.3],args=(x,y),method = 'SLSQP',
                    bounds=[(-0.1,0.1),(0,0.3),(-0.1,0.1),(0.5,5),(1e-5,1-1e-5)],
                    options={'maxfun':100000,'disp':True})
    #np.sum(((y-two_gaussians(x, *optim1))**2)/(poisson.std(50,loc=0)**2))/(bins-len(guess6))
    
    #chisq1 = chisquare(y,two_gaussians(x, *optim1))[0]
    
    
 
    #if plot == True:
  #  plt.scatter(x,y, c='pink', label='measurement', marker='.', edgecolors=None)
  #  plt.plot(x, ypred, c='b', label='fit of Two Gaussians')
   # plt.title("Two Gaussian Fitting")
   # plt.ylabel("Number of pairs")
   # plt.xlabel("Velocity Difference")
   # plt.legend(loc='upper left')
    #plt.savefig('./halo02_results/halo02_2_2fit'+str(k)+'12grav2.5largebin.png')
    #plt.show()
        #plt.scatter(x,y, c='pink', label='measurement', marker='.', edgecolors=None)
       # plt.plot(x, (gaussian(x, p2[0], p2[1], p2[2], p2[3])), c='b', label='fit of 1 Gaussians')
        #plt.title("One gaussian fitting")
       # plt.xlabel("Velocity Difference")
        #plt.legend(loc='upper left')
        #plt.savefig('./halo02_results/halo02_2_1fit'+str(k)+'12grav2.5largebin.png')
        #plt.show()
    
    #return 
    #np.sum(np.absolute((one_gaussian(x, *optim2) - y)**2/one_gaussian(x, *optim2)))
    #if np.absolute(aic1)-np.absolute(aic2) < 0:
        #return LL1,p1
    #else:
        #return LL2,p2
         
    #if np.absolute(chisq1) < np.absolute(chisq2):
        
        #interen = integrate.quad(lambda x: Lorentz1D_mo_ra(x, *optim1), -np.absolute((optim1[2]/2*3))+optim1[1],np.absolute(optim1[2]/2*3)+optim1[1])[0]
        #intereb = simps(y, dx=np.absolute(x[0]-x[1]))
        

        
    return res2.x


# In[ ]:





# In[46]:


dfm = pd.read_csv('metallicity_dist.csv')


# In[ ]:


dfm.keys()


# In[ ]:





# In[ ]:


plt.plot(dfm['x'],dfm['  y'],'.')


# In[ ]:


popt, pcov = curve_fit(two_gnfunc, dfm['x'],dfm['  y'],p0=[0,0,1,2,0.8])


# In[ ]:


popt


# In[ ]:


parag = fitting_reunbin(dfm['x'],dfm['  y'])


# In[47]:


parag=[0,0.5,0,1.5,0.9]


# In[48]:


plt.plot(dfm['x'],two_gnfunc(dfm['x'],dfm['  y'],parag),'.',c='b')
plt.plot(dfm['x'],dfm['  y'],'.',c='r')
plt.xlabel('x')
plt.ylabel('Number')


# In[ ]:


len(feh)


# In[ ]:


bins=16
binn = stats.binned_statistic(np.sort(feh), np.sort(feh), 'mean', bins=bins)


# In[ ]:


binn


# In[ ]:


fehbin=[]
for ii in range(1,bins+1):
    
    fehbin.append(np.repeat(binn[0][ii-1],len(np.array(binn[2])[binn[2]==ii])))


# In[ ]:


plt.hist(np.concatenate(fehbin),bins=16)


# In[ ]:


len(np.array(binn[2])[binn[2]==2])


# In[216]:


def log_prior(x,loc,scale):
        return stats.norm.logpdf(x, loc=loc, scale=scale)


# In[241]:


rv,rverr,feh,feherr,pms,pmcovs=datasum


# In[242]:


feh


# In[59]:


import scipy.integrate as integrate


# In[60]:


def gaum(x,sigma,mean):
    return (1/(10**sigma*np.sqrt(2*np.pi))) * np.exp(-(x- mean)**2/(2*(10**sigma)**2))


# # Likelihood function

# In[61]:


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
    
    
    
    
    feherr=np.sqrt(feherr**2+0.15**2)
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
    
    pm0sp = np.zeros((N,2))
    pm0sp[:,0] = -0.04
    pm0sp[:,1] = -0.19
    
    #pm mean for bg
    bgpm0sp = np.zeros((N,2))
    bgpm0sp[:,0] = -2.1
    bgpm0sp[:,1] = -1.1
    
    
    # Covariance Matrix for bg
    bgpmcovs = np.zeros((N,2,2))


    bgpmcovs[:,0,0] = pmcovs[:,0,0]+(10**lsigpmra1)**2-galdis**2
    bgpmcovs[:,1,1] =  pmcovs[:,1,1]+(10**lsigpmdec1)**2-galdis**2
    bgpmcovs[:,0,1] = pmcovs[:,0,1]
    bgpmcovs[:,1,0] = pmcovs[:,1,0]
    
    
    # The prior is just a bunch of hard cutoffs
    if (pgal > 1) or (pgal < 0) or \
        (lsigv > 3) or (lsigvbg1 > 3) or \
        (lsigv < -1) or (lsigvbg1 < -1)  or \
        (lsigfeh > 1) or (lsigfeh1 > 1) or (lsigfeh1 > 1) or  (feh0 > 0) or (feh0 < -5) or \
        (lsigfeh < -3) or (lsigfeh1 < -3) or (lsigfeh1 < -3) or \
        (vhel > 600) or (vhel < -600) or (vbg1 > 500) or (vbg1 < -300) or \
        (pmra_gal <-4) or ((pmdec_gal) > 2) or ((pmdec_gal) < -4) or \
        (pmra_gal > 2) or \
        (pmra1 > 2) or (pmra1 < -4) or\
        (pmdec1 > 2) or (pmdec1 < -4) or\
        (lsigpmra1 > 1.3) or (lsigpmra1 < -1) or \
        (lsigpmdec1 > 1.3) or (lsigpmdec1 < -1) :
        return -1e10
    
    # Compute log likelihood in rv
    lgal_vhel = stats.norm.logpdf(rv, loc=vhel, scale=np.sqrt(rverr**2 + (10**lsigv)**2))
    lbg1_vhel = stats.norm.logpdf(rv, loc=vbg1, scale=np.sqrt(rverr**2 + (10**lsigvbg1)**2))
    
    # Compute log likelihood in feh
    #feh covolved with double gaussian 
    #

    #fehbin = stats.binned_statistic(feh, , 'mean', bins=18)[0]
    fehbin=[]
    bins=11
    binn2 = stats.binned_statistic(np.sort(feh), np.sort(feh), 'mean', bins=bins)
    loglike = []
    loglike2 = []
    mt = []
    mt2 = []
    for ii in range(1,bins+1):
        #binned feh values
        fehbin.append(len(np.array(binn2[2])[binn2[2]==ii]))
     
        
       #m number of stars predict by the model belongs to Draco/ m2 belongs to the MKY background
       #total number of stars in the sample
        m = N*pgal*integrate.quad(gaum, binn2[1][ii-1],binn2[1][ii], args=(lsigfeh,feh0))[0]
        m2 = N*(1-pgal)*integrate.quad(gaum, binn2[1][ii-1],binn2[1][ii], args=(lsigfeh1,fehbg1))[0]
        #m2 = N*(1-pgal)*(1/(10**lsigfeh1*np.sqrt(2*np.pi))) * np.exp(-(binn2[0][ii-1]- fehbg1)**2/(2*(10**lsigfeh1)**2))*(3.9-0.3)/bins
        
        mt.append(m)
        mt2.append(m2)
    
        n1 = len(np.array(binn2[2])[binn2[2]==ii])
        n2=0
        #n2 is n!
        for k in range(1,len(np.array(binn2[2])[binn2[2]==ii])):
            n2=np.log(n1)+n2
            n1=n1-1
      
        loglike.append(len(np.array(binn2[2])[binn2[2]==ii])*np.log(m+m2)-n2-m-m2)
        
       # fehbin.append(np.repeat(binn[0][ii-1],len(np.array(binn[2])[binn[2]==ii])))
    lgal_feh = np.sum(loglike)
    #stats.norm.logpdf(np.concatenate(fehbin), loc=feh0, scale=np.sqrt( (10**lsigfeh)**2))
    #lgal_feh = a*stats.norm.logpdf(feh, loc=feh0, scale=np.sqrt((sigma1)**2 + (10**lsigfeh)**2))+(1-a)*stats.norm.logpdf(feh, loc=feh0, scale=np.sqrt(sigma2**2 + (10**lsigfeh)**2))
    #(1-a)*stats.norm.logpdf(feh, loc=feh0, scale=np.sqrt(sigma2^2**2 + (10**lsigsigma^feh)**2))
    #lbg1_feh = a*stats.norm.logpdf(feh, loc=fehbg1, scale=np.sqrt((sigma1)**2 + (10**lsigfeh1)**2))+(1-a)*stats.norm.logpdf(feh, loc=fehbg1, scale=np.sqrt(sigma2**2 + (10**lsigfeh1)**2))
    #lbg1_feh = np.sum(loglike2)
    #stats.norm.logpdf(np.concatenate(fehbin), loc=fehbg1,scale=np.sqrt( (10**lsigfeh1)**2))
    #stats.norm.logpdf(feh, loc=fehbg1, scale=np.sqrt((feherr)**2 + (10**lsigfeh1)**2))
    
    # Compute log likelihood in proper motions
    #for i in range(N):
        
    #using multivariat gaussian for the pm likelihood
    lgal_pm = [stats.multivariate_normal.logpdf(pms[i], mean=pm0s[i], cov=pmcovs[i]) for i in range(N)]
    lbg1_pm = [stats.multivariate_normal.logpdf(pms[i], mean=bgpm0s[i], cov=bgpmcovs[i]) for i in range(N)]
    
    
    # Combine the components
  
    lgal = np.log(pgal)+lgal_vhel+np.array(lgal_pm)+np.log(pmnorm)
    lbg1 = np.log(1-pgal)+lbg1_vhel+np.log(pmnorm)+lbg1_pm
    ltot = np.logaddexp(lgal, lbg1)
    ltot2 = np.logaddexp(ltot,lgal_feh)
    
    return ltot2.sum()

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
    #print (len(rv),len(feh))
    feherr=np.sqrt(feherr**2+0.15**2)
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
        (abs(pmra1) > 2) or (abs(pmdec1) > 2) or \
        (lsigpmra1 > 1.3) or (lsigpmra1 < -1) or \
        (lsigpmdec1 > 1.3) or (lsigpmdec1 < -1) :
        return -1e10
    
    # Compute log likelihood in rv
    lgal_vhel = stats.norm.logpdf(rv[ii], loc=vhel, scale=np.sqrt(rverr[ii]**2 + (10**lsigv)**2))
    lbg1_vhel = stats.norm.logpdf(rv[ii], loc=vbg1, scale=np.sqrt(rverr[ii]**2 + (10**lsigvbg1)**2))
    
    # Compute log likelihood in feh
    lgal_feh = stats.norm.logpdf(feh[ii], loc=feh0, scale=np.sqrt((10**lsigfeh)**2))
    lbg1_feh = stats.norm.logpdf(feh[ii], loc=fehbg1, scale=np.sqrt((10**lsigfeh1)**2))
    
    # Compute log likelihood in proper motions
    #for i in range(N):
        
        #print (pms[i], "mean",pm0s[i], 'cov',pmcovs[i])
    lgal_pm = [stats.multivariate_normal.logpdf(pms[ii], mean=pm0s[ii], cov=pmcovs[ii])]
    lbg1_pm = [stats.multivariate_normal.logpdf(pms[ii], mean=bgpm0s[ii], cov=bgpmcovs[ii])]
    
    # Combine the components
    
    lgal = np.log(pgal)+lgal_vhel+lgal_pm+np.log(pmnorm)+lgal_feh
    lbg1 = np.log(1-pgal)+lbg1_vhel+lbg1_pm+np.log(pmnorm)+lbg1_feh
    
    ltot = np.logaddexp(lgal, lbg1)
    return  lgal,lbg1, ltot,[np.exp(lgal_vhel),np.exp(lgal_pm),np.exp(lgal_feh)]







def get_paramdict(theta):
    return OrderedDict(zip(param_labels, theta))


# In[62]:


def project_model(theta, p1min=-600, p1max=350, p2min=-3.9, p2max=0.,key="vhel"):
    """ Turn parameters into p1 and p2 distributions """
    p1arr = np.linspace(p1min, p1max, 1000)
    p2arr = np.linspace(p2min, p2max, 1000)
    p3arr = np.linspace(-4,2,1000)
    params = get_paramdict(theta)
    
    if key == 'vhel':
        p10 = params["pgal"]*stats.norm.pdf(p1arr, loc=params["vhel"], scale=10**params["lsigv"])
        p11 = (1-params["pgal"])*stats.norm.pdf(p1arr, loc=params["vbg1"], scale=10**params["lsigvbg1"])

        p20 = params["pgal"]*stats.norm.pdf(p2arr, loc=params["feh"], scale=10**params["lsigfeh"])
        p21 = (1-params["pgal"])*stats.norm.pdf(p2arr, loc=params["fehbg1"], scale=10**params["lsigfeh1"])
        #p20 = 0.548*stats.norm.pdf(p2arr, loc=-2.197, scale=10**(-0.399))
        #p21 = (1-0.548)*stats.norm.pdf(p2arr, loc=-1.478, scale=10**(-0.440))
    else:
        p10 = params["pgal"]*stats.norm.pdf(p3arr, loc=params["pmra"], scale=0.025)
        p11 = (1-params["pgal"])*stats.norm.pdf(p3arr, loc=params["pmra1"], scale=10**params["lsigpmra1"])

        p20 = params["pgal"]*stats.norm.pdf(p3arr, loc=params["pmdec"], scale=0.025)
        p21 = (1-params["pgal"])*stats.norm.pdf(p3arr, loc=params["pmdec1"], scale=10**params["lsigpmdec1"])
        
    return p1arr, p10, p11, p2arr,p20,p21,p3arr
    


# In[100]:


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
        ax.hist(datasum[2], density=True, color='grey', bins=11)
        xp, p0, p1 = model_output[3:6]
        ax.plot(xp, p0 + p1, 'k-', lw=3)
        ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="[Fe/H] (dex)", ylabel="Prob. Density")
    else:
        ax = axes[0]
        ax.hist(datasum[-2][:,0], density=True, color='grey', bins=100)
        xp, p0, p1 = model_output[0:3]
        xp=model_output[-1]
        ax.plot(xp, p0 + p1, 'k-', label="Total", lw=3)
        ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel=r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$", ylabel="Prob. Density",xlim=(-5,5), ylim=(0,1))
        ax.legend(fontsize='small')

        ax = axes[1]
        ax.hist(datasum[-2][:,1], density=True, color='grey', bins='auto')
        xp, p0, p1 = model_output[3:6]
        xp=model_output[-1]
        ax.plot(xp, p0 + p1, 'k-', lw=3)
        ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel=r"$\rm{\mu_{\delta} \ (mas/yr)}$", ylabel="Prob. Density",xlim=(-5,5), ylim=(0,1))
    fig.savefig(str(key)+'distr1d.pdf')
    return fig


# In[101]:


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
               xerr=2*0.025, yerr=2*0.025,
               color=colors[0], marker='o', elinewidth=1, capsize=3, zorder=9999,label='Gal')
        ax.errorbar(params["pmra1"], params["pmdec1"],
              xerr=2*10**params["lsigpmra1"], yerr=2*10**params["lsigpmdec1"],
              color=colors[2], marker='x', elinewidth=1, capsize=3, zorder=9999,label='Bg')
        ax.grid()
        ax.legend()
    fig.savefig(str(key)+'2ddistri.pdf')
    return fig


# In[102]:


param_labels = ["pgal",
                "vhel","lsigv","feh","lsigfeh",
                "vbg1","lsigvbg1","fehbg1","lsigfeh1",
                "pmra","pmdec",
                "pmra1","pmdec1","lsigpmra1","lsigpmdec1"]


# # Optimize parameters
# 
# 

# In[103]:


## I found this guess by looking at the plot by eye and estimating. This part requires some futzing.
p0_guess = [0.6, 
            -84.974, 1.12, -2.196,-0.484,
           -49.579, 1.857, -1.479, -0.549,
            0.027, -0.184,
            -2.163, -0.999, 0.562, 0.455]


# In[104]:


#vrad/Feh distribution

fig1 = plot_1d_distrs(p0_guess,datasum,p1min=-300, p1max=300, p2min=-4, p2max=0.,key="vhel")
fig2 = plot_2d_distr(p0_guess,datasum,key="vhel")


# In[105]:


#pmra/pmdec distribution 
fig1 = plot_1d_distrs(p0_guess,datasum,p1min=-5, p1max=5, p2min=-5, p2max=5.,key="pmra")
fig2 = plot_2d_distr(p0_guess,datasum,key="pmra")


# In[106]:


#guess for the initial parameters
p0_guess


# In[107]:


optfunc = lambda theta: -full_like(theta)


# In[108]:


np.log(5*4*3*2)


# In[109]:


get_ipython().run_line_magic('timeit', 'optfunc(p0_guess)')


# In[110]:


optfunc(p0_guess)


# In[111]:


get_ipython().run_line_magic('time', 'res = optimize.minimize(optfunc, p0_guess, method="Nelder-Mead")')


# In[112]:


res.x


# In[76]:


optfunc(res.x)


# In[77]:


for label, p in zip(param_labels, res.x):
    print(f"{label}: {p:.3f}")


# In[78]:


fig1 = plot_1d_distrs(res.x,datasum,p1min=-300, p1max=300, p2min=-4, p2max=0.,key="vhel")
fig2 = plot_2d_distr(res.x,datasum,key="vhel")


# In[79]:


#plotting the posterior distribution for pmra and pmdec
fig1 = plot_1d_distrs(res.x,datasum,p1min=-5, p1max=5, p2min=-5, p2max=5,key="pmra")
fig2 = plot_2d_distr(res.x,datasum,key="pmra")


# ## Posterior Sampling
# The posterior is sampled using `emcee` with 64 walkers and 10,000 steps per chain.

# In[80]:


nw = 128
p0 = res['x']
nit = 5000
ep0 = np.zeros(len(p0_guess)) + 0.02
#p0s = np.random.multivariate_normal(p0_guess, np.diag(ep0)**2, size=nw)
#print(p0s)


# In[81]:


p0_guess


# In[82]:


np.random.uniform(low=-200, high=20, size=1)


# In[83]:


nparams = len(param_labels)
print(nparams)
nwalkers = 128
p0 = p0_guess
ep0 = np.zeros(len(p0)) + 0.02 # some arbitrary width that's pretty close; scale accordingly to your expectation of the uncertainty
p0s = np.random.multivariate_normal(p0, np.diag(ep0)**2, size=nwalkers)

## Check to see things are initialized ok
lkhds = [full_like(p0s[j]) for j in range(nwalkers)]
assert np.all(np.array(lkhds) > -9e9)


# In[84]:


## Run emcee in parallel

from schwimmbad import MultiPool

nproc = 64 #use 64 cores
nit = 3000

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


# In[85]:


outputs = es.flatchain


# In[86]:


plt.style.use('default')


# ### Acceptance fraction
# Judging the convergence and performance of an algorithm is a non-trival problem. As a rule of thumb, the acceptance fraction should be between 0.2 and 0.5 (for example, Gelman, Roberts, & Gilks 1996).

# In[126]:


# Another good test of whether or not the sampling went well is to 
# check the mean acceptance fraction of the ensemble
print(
    "Mean acceptance fraction: {0:.3f}".format(
        np.mean(es.acceptance_fraction)
    )
)


# In[127]:


fig = corner.corner(outputs, labels=param_labels, quantiles=[0.16,0.50,0.84], show_titles=True,color='black', # add some colors
              **{'plot_datapoints': False, 'fill_contours': True})

#plt.savefig('SMCNOD_PM_Model_Cornerplot.png')


# In[114]:


fig1 = corner.corner(outputs[:,1:3], labels=param_labels[1:3], quantiles=[0.16,0.50,0.84], show_titles=True,color='black', # add some colors
              **{'plot_datapoints': False, 'fill_contours': True})


# In[115]:


fig2 = corner.corner(outputs[:,3:5], labels=param_labels[3:5], quantiles=[0.16,0.50,0.84], show_titles=True,color='black', # add some colors
              **{'plot_datapoints': False, 'fill_contours': True})


# In[116]:


fig3 = corner.corner(outputs[:,9:11], labels=param_labels[9:11], quantiles=[0.16,0.50,0.84], show_titles=True,color='black', # add some colors
              **{'plot_datapoints': False, 'fill_contours': True})


# In[117]:


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
    
    
    


# In[118]:


meds, errs = process_chain(outputs)


# In[119]:


for k,v in meds.items():
    print("{} {:.3f} {:.3f}".format(k, v, errs[k]))


# In[120]:


get_paramdict(outputs[1])


# If things are well mixed, then you can just use the flat chain to concatenate all the walkers and steps.
# The results here may not be perfectly mixed, but it's not terrible.
# There are fancy ways to check things here involving autocorrelation times that Alex does not know about.
# To me this is the hard part of emcee: knowing when you're happy with the result, and setting things up so that it gets there as fast as possible. This is why I prefer dynesty, even though it's slower it has a motivated stopping condition.

# In[121]:


chain = es.flatchain
chain.shape


# You can see the output of the fit as a corner plot. Basically you want everything to be nice and round, and if not that means you didn't initialize your walkers well enough or burn in for long enough.

# It's customary to summarize the data with percentiles, but you should check the corner plot diagonal to see if this is a good idea.

# In[129]:


#plotting the posterior distribution for vral and FeH
fig1 = plot_1d_distrs(chain[1],datasum,p1min=-300, p1max=300, p2min=-3.5, p2max=-0.5,key="vh")
fig2 = plot_2d_distr(chain[1],datasum,key="vh")


# In[128]:


#plotting the posterior distribution for vral and FeH
fig1 = plot_1d_distrs(chain[1],datasum,p1min=-300, p1max=300, p2min=-3.5, p2max=-0.5,key="vhel")
fig2 = plot_2d_distr(chain[1],datasum,key="vhel")


# In[99]:


plt.hist(fehsample,density=True)


# In[130]:


plt.hist(outputs[:,3],density=False,bins=30)


# In[131]:


10**chain[:,4]


# In[235]:


datasum[2]


# In[180]:


fehsample = []
fehmsam = []
for ii in range(0,2000):
    xarr= np.linspace(-4, 0, 1000)
    fehm = np.random.choice(chain[:,3],1,replace = False)
    fehstd = np.random.choice(chain[:,4],1,replace = False)
    fehmm = np.random.choice(chain[:,7],1,replace = False)
    fehmstd = np.random.choice(chain[:,8],1,replace = False)
    
    fehsample.append(np.random.normal(loc=fehm, scale=10**fehstd,size=int(417*0.556)))
    fehmsam.append(np.random.normal(loc=fehmm, scale=10**fehmstd,size=int(417*(1-0.556))))
    
#print (np.percentile(chain_new[:,2], 50))
#
#plt.hist(datasum[2],density=False,alpha=0.5,bins=30)


# In[181]:


len(fehsample)


# In[182]:


np.mean(fehmsam,axis=1)


# In[183]:


feht= np.concatenate([np.mean(fehmsam,axis=0),np.mean(fehsample,axis=0)])


# In[184]:


plt.hist(fehmsam[:100],density=False,bins=15)
plt.hist(fehsample[:100],density=False,bins=15)
#plt.hist(feht,density=False,bins=15)
plt.hist(datasum[2],density=False,alpha=0.6,bins=15,label='real')
plt.xlabel('FeH')
plt.legend()


# In[185]:


datasum


# In[189]:


print (len(datasum[2][(datasum[2]< -2.0)& (datasum[2]> -3.5)]))


# In[190]:


leng = []
lengm = [] 
for ii in range(0,2000):
    leng.append(len(fehsample[ii][(fehsample[ii]< -2.0) & (fehsample[ii]> -3.5)]))
    lengm.append(len(fehmsam[ii][(fehmsam[ii]< -2.0)& (fehmsam[ii]> -3.5) ]))

plt.hist(lengm,density=False,alpha=0.5,bins=15,label='real')


# In[191]:


np.mean(leng)+np.mean(lengm),np.std(lengm),np.std(leng)


# In[192]:


#plotting the posterior distribution for pmra and pmdec
fig1 = plot_1d_distrs(chain[1],datasum,p1min=-5, p1max=5, p2min=-5, p2max=5,key="pmra")
fig2 = plot_2d_distr(chain[1],datasum,key="pmra")


# In[144]:


chain_new = 10**(chain)
mean_vdisp = np.percentile(chain_new[:,2], 50)
std_vdisp = (np.percentile(chain_new[:,2], 84)-np.percentile(chain_new[:,2], 16))/2
mean_fehdisp = np.percentile(chain_new[:,4], 50)
std_fehdisp = (np.percentile(chain_new[:,4], 84)-np.percentile(chain_new[:,4], 16))/2
print("mean_vdisp: ",mean_vdisp, \
     "std_vdisp: ",std_vdisp)
print("mean_fehdisp: ",mean_fehdisp, \
     "std_fehdisp: ",std_fehdisp)


# In[365]:


len(chain_new[:,3])


# In[202]:


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


# In[203]:


pval


# In[204]:


#proability function for each stars to be member 
def prob(itot):
    probi=[]
    other = []
    ltotf=[]
    lgalf=[]
    for ii in range(itot):
       
        lgal,lbg1,ltot,_ = full_like_indi(pval,ii)
        print (ltot)
        probi.append(np.exp(lgal)/np.exp(ltot))
        other.append(_)
        ltotf.append(ltot)
        lgalf.append(lgal)
    return probi,other,ltotf,lgalf


# In[205]:


testp,testindi,ltot,lgal = prob(len(datasum[0]))


# In[206]:


[np.exp(lgal_vhel),np.exp(lgal_pm),np.exp(lgal_feh)]


# In[207]:


np.array(ltot)[np.concatenate(testp)>probb][cut1],np.array(lgal)[np.concatenate(testp)>probb][cut1]


# In[208]:


np.array(ltot)[np.concatenate(testp)>probb][cut1]


# In[209]:


#distribution for the membership probabilities
plt.style.use('classic')
plt.hist(np.concatenate(testp),bins=15)
plt.grid(False)
plt.xlabel('Probability')
plt.ylabel('Number of Stars')
#plt.savefig('probdis.pdf')


# # Result Analysis
# 

# In[210]:


plt.style.available


# In[211]:


plt.hist(np.concatenate(testp))


# In[ ]:





# In[212]:


np.savetxt('draco_prob_binlike_30000',np.concatenate(testp), fmt='%.18e', delimiter=' ', newline='\n')
dracohigh= pd.DataFrame()
dracohigh= datacut
#dracohigh.write('draco_all_feherror0.15_30000.csv')


# In[ ]:





# In[213]:


dwarf_df= pd.DataFrame()
dwarf_df = datacut


# In[214]:


dwarf_df['VGSR']


# In[215]:


ra0 = ra0
dec0 = dec0
dist = 75.8
#dist = 100
#pmra0 = 0.045
pmra0 = 0.025
pmdec0 = -0.21
vlos0 = -291.12 #dwarftable[dwarftable['key']=='ursa_minor_1']['vlos_systemic'][0]
prob =0
c = 4.74047
vel_pmra0, vel_pmdec0 = pmra0 * c * dist, pmdec0 * c * dist
a = np.pi/180.
ca = np.cos(ra0*a)
cd = np.cos(dec0*a)
sa = np.sin(ra0*a)
sd = np.sin(dec0*a)
vx = vlos0 * cd * sa + vel_pmra0* cd * ca -  vel_pmdec0 * sd*sa
vy = -vlos0 * cd * ca + vel_pmdec0 * sd * ca +  vel_pmra0 * cd*sa
vz = vlos0 *sd + vel_pmdec0 *cd
dwarf_df['vlos_correct'] = np.zeros(len(dwarf_df), dtype=float)
deltax=dwarf_df['TARGET_RA_1']
deltay=dwarf_df['TARGET_DEC_1']
bx = np.cos(deltay*a)*np.sin(deltax*a)
by = -np.cos(deltay*a)*np.cos(deltax*a)
bz = np.sin(dwarf_df['TARGET_DEC_1']*a)
dwarf_df['delta_vlos_correct'] = bx*vx + vy*by + bz*vz - vlos0 
dwarf_df['vlos_correct'] = dwarf_df['delta_vlos_correct'] + dwarf_df['VGSR']

plt.scatter(dwarf_df['TARGET_RA_1'], dwarf_df['TARGET_DEC_1'], c=dwarf_df['delta_vlos_correct'], marker='o')
plt.xlabel('RA')
plt.ylabel('DEC')
plt.colorbar()


# In[216]:


#gaussian fitting 
import scipy.optimize as optim





# In[217]:


def gauss(x,  A, x0, sigma):
    return  A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))




# In[218]:


def mcmc_rv(nproc,nit,p0_guess,full_like,args=[1,1]):
    nw = 64
    p0 = res['x']
    nit = 2000
    ep0 = np.zeros(len(p0_guess)) + 0.02
    p0s = np.random.multivariate_normal(p0_guess, np.diag(ep0)**2, size=nw)
    
    
    nparams = len(param_labels)
   
    nwalkers = 64
    p0 = p0_guess
    ep0 = np.zeros(len(p0)) + 0.02 # some arbitrary width that's pretty close; scale accordingly to your expectation of the uncertainty
    p0s = np.random.multivariate_normal(p0, np.diag(ep0)**2, size=nwalkers)
## Check to see things are initialized ok
    #lkhds = [full_like(p0s[j]) for j in range(nwalkers)]
    #assert np.all(np.array(lkhds) > -9e9)
    
    from schwimmbad import MultiPool

    nproc = nproc #use 32 cores
    nit = nit



    with MultiPool(nproc) as pool:
        print("Running burnin with {} iterations".format(nit))
        start = time.time()
        es = emcee.EnsembleSampler(nw, len(p0_guess), full_like, args=args,pool=pool)
        PP = es.run_mcmc(p0s, nit, rstate0=get_rstate())
        print("Took {:.1f} seconds".format(time.time()-start))

        print(f"Now running the actual thing")
        es.reset()
        start = time.time()
        es.run_mcmc(PP.coords, nit, rstate0=get_rstate())
        print("Took {:.1f} seconds".format(time.time()-start))

    
    outputs = es.flatchain
    return outputs
    


# In[219]:


def process_chain_rv(chain, avg_error=True):
    pctl = np.percentile(chain, [16, 50, 84], axis=0)
    meds = pctl[1]
    ep = pctl[2]-pctl[1]
    em = pctl[0]-pctl[1]
    if avg_error: # just for simplicity, assuming no asymmetry
        err = (ep-em)/2
        return meds,err
    else:
        return meds,ep,em
    


# In[220]:


def like_rv(thetarv,vdata,verr):
    c,w=thetarv
   
    x=vdata
    rverr=verr
    likelog = stats.norm.logpdf(x, loc=c, scale=np.sqrt((rverr)**2 + (w)**2))
    return  likelog.sum()


# In[221]:


def ellipse_bin_rv(xdata,ydata,vel,theta0=89*0.0174533,r0=10/60,e=0.31,number=2,ra0=ra0,dec0=dec0,prob=1,proba=0.):
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
    rtot = [0.00001,0.5,0.75,1.1,1.5,1.8,2.3,3.0,5.0]
    

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


        #dis.append(2/3*((r*rtot[ii+1])**3-(r*rtot[ii])**3)/((r*rtot[ii+1])**2-(r*rtot[ii])**2))
        dis.append(np.sqrt(((r*rtot[ii+1])**2+(r*rtot[ii])**2)/2))
        #dis.append(np.sqrt(((r*rtot[ii+1])**2-(r*rtot[ii])**2)/2))
        #prob.append(sub[(cut1)&(cut2)])
        print (len(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba]))
        vdata=vel['vlos_correct'][(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba]
        verr = vel['VRAD_ERR'][(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba]
 
     
        guess_rv = [-94,11]
        #optfuncrv = lambda thetarv: like_rv(thetarv,vdata,verr)
        #timeit optfuncrv(guess_rv)
        test = mcmc_rv(60,2000,guess_rv,like_rv,args=[vdata,verr])
        plt.hist(vdata,bins=5)
        plt.show()
        meds, errs = process_chain_rv(test)
        ra.append(xdata[(cut1)&(cut2)])
        dec.append(ydata[(cut1)&(cut2)])
        
        veldis.append([meds[1],errs[1]])
        #veldis.append([popt[2],np.sqrt(np.diagonal(covariance))[2]])
        
      
        #print (vel[(cut1)&(cut2)])
        
        
        
        

# Plot the circle
        #ax.plot(xt, yt,label = str(ii)+' r$_{h}$',c='k')
        #plt.scatter(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],y1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],s=6)
        #cbar=plt.colorbar()
        #cbar.set_label('Iron-Pace&LI Probability')
        #plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
#plt.ylim(21, 16)
        #plt.xlim(258,262)
        #plt.ylabel('DEC')
        #plt.xlabel('RA')
        #cbar=plt.colorbar()
   # cbar.set_label('Iron Probability')   
    return tot,ra,dec,dis,den,err,dis2,veldis




# Set the aspect ratio of the plot to 'equal' to make the circle appear circular
#ax.set_aspect('equal')

# Set the radius of the circle to be the half light radiu






# In[222]:


tot,ra,dec,dis,den,err,dis2,veldis2=ellipse_bin_rv(dwarf_df['TARGET_RA_1'],dwarf_df['TARGET_DEC_1'],dwarf_df,theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0,prob=np.concatenate(testp),proba=0.75)


         
    


# In[ ]:


plt.hist(dwarf_df['VRAD'])


# In[303]:


#plt.plot(np.array(r[:5])*60,vral)
vralcor = veldis2
#vral = veldis2
plt.scatter(np.array(dis)*np.pi/180*75.8*1000,np.array(vralcor)[:,0])
plt.errorbar(np.array(dis)*np.pi/180*75.8*1000,np.array(vralcor)[:,0],yerr= np.array(vralcor)[:,1],linestyle = 'none',label='Corrected Velocity Dispersion')
plt.ylim(0,20)
plt.xlabel('R (pc)')
plt.ylabel('Velocity Dispersion (km/s)')
plt.legend()
#plt.savefig('velocitydisp_vcorvgsrrverr_0.5_8bins.pdf')


# In[223]:


r2=43/60
e=0.31
a2=r2/np.sqrt(1-e)
b2= r2*np.sqrt(1-e)
probb=0.75
xc=np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>probb]*0.0174533)*np.sin((datacut['TARGET_RA_1'][np.concatenate(testp)>probb]-ra0)*0.0174533)
yc=np.sin(datacut['TARGET_DEC_1'][np.concatenate(testp)>probb]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>probb]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][np.concatenate(testp)>probb]-ra0)*0.0174533)
#Rxp2= x2*np.cos(theta0)-y2*np.sin(theta0)
x2=xc/0.0174533
y2=yc/0.0174533
cut1 = (x2**2/a2**2+y2**2/b2**2)>1

dataout = datacut[np.concatenate(testp)>probb][cut1]


# In[224]:


np.concatenate(testp)[np.concatenate(testp)>probb][cut1]



# In[225]:


dataout['REF_ID_2']


# In[226]:


#plotting ra versus dec with Iron-Pace&Li 2022 prbabilities with the tidal radius of Draco

plt.figure(figsize=(15,10))
fig, ax = plt.subplots(figsize=(15,10))

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

xd1=datacut['TARGET_RA_1']-center[0]
yd1=datacut['TARGET_DEC_1']-center[1]
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
x=np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.sin((datacut['TARGET_RA_1'][np.concatenate(testp)>proba]-ra0)*0.0174533)
y=np.sin(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][np.concatenate(testp)>proba]-ra0)*0.0174533)
#Rxp= x*np.cos(theta0)-y*np.sin(theta0)
#Ryp = x*np.sin(theta0)+y*np.cos(theta0)

plt.scatter(x/0.0174533,y/0.0174533,s=6,label='High Probability Members')
proba= 0.0
x2=np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.sin((datacut['TARGET_RA_1'][np.concatenate(testp)>proba]-ra0)*0.0174533)
y2=np.sin(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][np.concatenate(testp)>proba]-ra0)*0.0174533)
#Rxp2= x2*np.cos(theta0)-y2*np.sin(theta0)
#Ryp2 = x2*np.sin(theta0)+y2*np.cos(theta0)
x3=np.cos(dataout['TARGET_DEC_1']*0.0174533)*np.sin((dataout['TARGET_RA_1']-ra0)*0.0174533)
y3=np.sin(dataout['TARGET_DEC_1']*0.0174533)*np.cos(dec0*0.0174533)-np.cos(dataout['TARGET_DEC_1']*0.0174533)*np.sin(dec0*0.0174533)*np.cos((dataout['TARGET_RA_1']-ra0)*0.0174533)
#Rxp2= x2*np.cos(theta0)-y2*np.sin(theta0)
plt.scatter(x2/0.0174533,y2/0.0174533,s=6,alpha=0.3,c='grey',label='Low Probability Members')
plt.scatter(x3/0.0174533,y3/0.0174533,s=200,c=np.concatenate(testp)[np.concatenate(testp)>probb][cut1],label='Eight Outskirts High Probability Members')
plt.plot([3.20833,0],[-1.20618556,0],'-',c='g')
plt.plot([-3.20833,0],[1.20618556,0],'--',c='g',label='Orbit from Qi et al. 2022')
cbar=plt.colorbar()
plt.style.use('classic')
#cbar.set_label('Probability')





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
#plt.savefig('spatial_gsr_test.pdf')


# In[230]:


dataout['VGSR'],dataout['VRAD_ERROR'],dataout['FEH'],dataout['FEH_ERR'],dataout['PMRA_3'],dataout['PMDEC_3'],dataout['PMRA_ERROR']


# In[168]:


dataout['gmag0']-dataout['rmag0'],np.concatenate(testp)[np.concatenate(testp)>probb][cut1]


# In[393]:


def plot_1d_distrs_high(theta,datasum,p1min=-600, p1max=350, p2min=-4, p2max=0.,key="vhel"):
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
    #colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    model_output = project_model(theta,p1min, p1max, p2min, p2max,key=key)
    fig, axes = plt.subplots(1,2,figsize=(18,8))
    if key == "vhel":
        ax = axes[0]
        ax.hist(datasum[0], density=False, color='grey', bins=100,label='All Sample')
        ax.hist(datasum[0][np.concatenate(testp)>probb], density=False, color='red', bins=100,label = 'High Probability Sample')
        #xp, p0, p1 = model_output[0:3]
        #ax.plot(xp, p0 + p1, 'k-', label="Total", lw=3)
        #ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        #ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="Vgsr (km/s)", ylabel="Prob. Density")
        ax.legend(fontsize='small')

        ax = axes[1]
        ax.hist(datasum[2], density=False, color='grey', bins='auto',label = 'All Sample')
        ax.hist(datasum[2][np.concatenate(testp)>probb], density=False, color='red', bins='auto',alpha=0.3,label = 'High Probability Sample')
        #xp, p0, p1 = model_output[3:6]
        #ax.plot(xp, p0 + p1, 'k-', lw=3)
        #ax.plot(xp, p1, ':', color=colors[2], label="bg1", lw=3)
        #ax.plot(xp, p0, ':', color=colors[0], label="gal", lw=3)
        ax.set(xlabel="[Fe/H] (dex)", ylabel="Prob. Density")
        plt.show()
   
    return fig


# In[394]:


plot_1d_distrs_high(theta,datasum,p1min=-600, p1max=350, p2min=-4, p2max=0.,key="vhel")


# In[315]:


dataout['REF_ID_2']


# In[ ]:


data


# In[395]:


#CMD diagram with probabiltiy 
plt.style.use('classic')
plt.figure(figsize=(10,8),facecolor='white')
plt.grid(color='white')
plt.plot(iso_file['DECam_g']-iso_file['DECam_r'],iso_file['DECam_r']+dm,c='grey',lw=2,label='Isochrone')
#norm = colors.Normalize(vmin=0, vmax=1) 
#combined_colors = np.append(np.concatenate(testp), [0.99815984, 0.55174294, 0.99967203, 0.99981849, 0.92997065, 0.96598971])


#plt.scatter(dataout['gmag0']-dataout['rmag0'],dataout['rmag0'],s=150,c=np.concatenate(testp)[np.isin(datacut['REF_ID_1'],dataout['REF_ID_1'])],marker='^',label='Outskirts High Probability Members')
plt.scatter(datacut['gmag0']-datacut['rmag0'],datacut['rmag0'],s=10,c=np.concatenate(testp),cmap='bwr',label='EDR Sample')

cbar=plt.colorbar()
plt.scatter(dataout['gmag0']-dataout['rmag0'],dataout['rmag0'],s=150,c=np.concatenate(testp)[np.concatenate(testp)>probb][cut1],marker='^',cmap='viridis',label='Outskirts High Probability Members')
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
#plt.legend(loc=2)
#plt.savefig('prob_gsr.pdf')


# In[ ]:





# In[150]:


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
    rtot = [0.00001,1.0,2,3.,4.5,6,7,8.0,10,12]
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
        #plt.scatter(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],y1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],s=6)
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






# In[399]:


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
    rtot = [0.00001,1.0,2,3.,4.5,6,7,8.0,10,12]
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
        probave.append(prob)
        
        
        
        

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






# In[152]:


tot,ra,dec,r,probave,distot,dend,draerr,dis2,vral=ellipse_bin_pro(datacut['TARGET_RA_1'],datacut['TARGET_DEC_1'],datacut['VRAD'],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0,prob=np.concatenate(testp),proba=0.5)





# In[119]:


vral = [[13.86,2.53],[11.20,0.61],[14.73, 1.75],[11.55, 3.324],[0,0]]
vralcor = [[13.59,2.62],[11.04,0.42],[14.62, 1.73],[10.66, 2.15],[0,0]]


# In[125]:


#Ting's file for completeness correction
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


# In[126]:


len(testiron['TARGET_RA_1'][irad])


# In[127]:


xdata=targra[itargd]
ydata=targdec[itargd]


# In[128]:


x1=np.cos(ydata*0.0174533)*np.sin((xdata-ra0)*0.0174533)/0.0174533
y1=(np.sin(ydata*0.0174533)*np.cos(dec0*0.0174533)-np.cos(ydata*0.0174533)*np.sin(dec0*0.0174533)*np.cos((xdata-ra0)*0.0174533))/0.0174533
cut1 = (x1**2/a**2+y1**2/b**2)>1


# In[129]:


plt.plot(x1,y1,'.')
plt.show()


# In[130]:


allspectot,allspecxvec,allspecyvec,allspecrad,allspecprobave,allspecdistot,allspecden,allerr,dis2=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][irad],testiron['TARGET_DEC_1'][irad],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)
                                                                                                      


# In[ ]:





# In[131]:


mask =(1<<10)+(1<<38)
targiqk = np.where(((testiron['SV1_SCND_TARGET'][iqk] & mask) > 0))[0]


# In[132]:


spectot,specxvec,specyvec,specrad,specprobave,specdistot,specden,specerr,dis2=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][iqk][targiqk],testiron['TARGET_DEC_1'][iqk][targiqk],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)



# In[133]:


targtot,targxvec,targyvec,targrad,targprobave,targdistot,targden,tarerr,dis2=ellipse_bin_pro_noprob(targra[itargd],targdec[itargd],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)
 


# In[134]:


plt.plot(targrad,targden,'bo')
plt.plot(specrad,specden,'ro')
plt.plot(allspecrad,allspecden,'go')


# In[135]:


plt.plot(np.asarray(targrad)*60,np.asarray(targden)/np.asarray(specden),'bo')
plt.plot(np.asarray(targrad)*60,np.asarray(targden)/np.asarray(allspecden),'ro')


# In[136]:


tratio=np.asarray(targden)/np.asarray(specden)
print(tratio)
alltratio=np.asarray(targden)/np.asarray(allspecden)
print(alltratio)


# In[137]:


r


# In[138]:


plt.plot(np.asarray(r)*60,dend*tratio,'bo')
plt.plot(np.asarray(r)*60,dend,'ro')
plt.plot(np.sqrt(R2)*60,den*250,'.')
plt.yscale('log')
plt.yscale('log')
plt.xscale('log')


# # Adding the colorcut 

# In[139]:


colorcut1 = cmd_selection(testiron[irad], dm, g_r, iso_r, cwmin=0.6, dobhb=True)


# In[140]:


colorcut2 = cmd_selection(testiron[iqk][targiqk], dm, g_r, iso_r, cwmin=0.6, dobhb=True)


# In[141]:


allspectot,allspecxvec,allspecyvec,allspecrad,allspecprobave,allspecdistot,allspecden,allerr,allr=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][irad][colorcut1],testiron['TARGET_DEC_1'][irad][colorcut1],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)
                                                                                                      

spectot,specxvec,specyvec,specrad,specprobave,specdistot,specden,err,specr=ellipse_bin_pro_noprob(testiron['TARGET_RA_1'][iqk][targiqk][colorcut2],testiron['TARGET_DEC_1'][iqk][targiqk][colorcut2],theta0=89*0.0174533,r0=10/60,e=0.31,number=10,ra0=ra0,dec0=dec0)





# In[142]:


plt.plot(targrad,targden,'bo')
plt.plot(specrad,specden,'ro')
plt.plot(allspecrad,allspecden,'go')


# In[398]:


plt.plot(targrad,targtot,'bo')
plt.plot(specrad,spectot,'ro')
plt.plot(allspecrad,allspectot,'go')


# In[144]:


spectot,allspectot


# In[145]:


plt.plot(np.asarray(targrad)*60,np.asarray(targden)/np.asarray(specden),'bo')
plt.plot(np.asarray(targrad)*60,np.asarray(targden)/np.asarray(allspecden),'ro')


# In[396]:


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
x=np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.sin((datacut['TARGET_RA_1'][np.concatenate(testp)>proba]-ra0)*0.0174533)
y=np.sin(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.cos(dec0*0.0174533)-np.cos(datacut['TARGET_DEC_1'][np.concatenate(testp)>proba]*0.0174533)*np.sin(dec0*0.0174533)*np.cos((datacut['TARGET_RA_1'][np.concatenate(testp)>proba]-ra0)*0.0174533)

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


# In[397]:


#tratio=np.asarray(targden)/np.asarray(specden)
#print(tratio)
alltratio=np.asarray(targden)/np.asarray(allspecden)
print(alltratio)


# In[148]:


dis2[0]=a * b / np.sqrt(a**2 + b**2)


# In[149]:


r


# In[150]:


r


# In[151]:


dis2


# In[152]:


plt.plot(np.asarray(dis2)*60,dend*alltratio,'bo')
#plt.plot(np.asarray(dis2)*60,dend*tratio,'go')
plt.plot(np.asarray(dis2)*60,dend,'ro')
plt.plot(np.sqrt(R2)*60,den*250,'.')
plt.yscale('log')
plt.yscale('log')
plt.xscale('log')


# In[153]:


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


# In[ ]:


allerr/np.asarray(allspecden)

tarerr/np.asarray(targden)

draerr/np.array(dend)

draerr

errtot = dend*alltratio*np.sqrt((draerr/np.array(dend))**2+(allerr/np.asarray(allspecden))**2+(tarerr/np.asarray(targden))**2)


# In[ ]:


errtot


# In[ ]:


draerr/np.array(dend)


# In[154]:


dend


# In[155]:


draerr,errtot


# In[156]:


plt.plot(np.asarray(r)*60,dend*alltratio,'bo',label = 'Corrected Surface Density')
#plt.plot(np.asarray(r)*60,dend*tratio,'go')
plt.plot(np.asarray(r)*60,dend,'ro',label = 'Surface Density')
plt.plot(np.sqrt(R2)*60,den*560,'k-',label = 'Plummer Model')
plt.errorbar(np.array(r)*60,dend*alltratio, yerr=errtot, capsize=3, ls='none', color='black', 
            elinewidth=1)
plt.errorbar(np.array(r)*60,dend, yerr=draerr, capsize=3, ls='none', color='black', 
            elinewidth=1)
plt.axvline(x=43,linestyle='--',label='Tidal Radius')
plt.xlim(1,120)
plt.ylim(0.5,15000)
plt.xlabel('R arcmin')
plt.ylabel(r'Star Density [arcmin$^{-2}$]')
plt.yscale('log')
plt.yscale('log')
plt.xscale('log')
plt.legend(loc=3)
plt.savefig('plummer_gsr.pdf')


# In[ ]:





# In[ ]:





# In[157]:





# In[151]:


R2,den


# In[152]:


np.array(den)[np.absolute(np.array(np.sqrt(R2))-0.5)<0.005]
np.array(dend)/np.array(dend)[np.absolute(np.array(r)-0.5)<0.01]
np.array(r)*60


# In[ ]:


1/(np.pi*a**2*(1-e))*(1+0**2/a**2)**(-2)


# In[ ]:


np.min(np.sqrt(R2)*60)


# In[ ]:


plt.plot(np.sqrt(R2)*60,den,'.')

plt.yscale('log')
plt.xscale('log')


# In[ ]:


np.sort(np.sqrt(R2))


# In[ ]:


np.array(den)[np.absolute(np.array(np.sqrt(R2))-0.5)<0.006]


# In[ ]:


plt.plot(np.sort(np.sqrt(R2)),np.array(np.array(den)/np.array(den)[np.absolute(np.array(np.sqrt(R2))-0.48)<0.002])[np.argsort(np.sqrt(R2))],'-')
plt.plot(np.array(r),np.array(dend)/np.array(dend)[np.absolute(np.array(r)-0.48)<0.04],'.')
plt.errorbar(np.array(r),np.array(dend)/np.array(dend)[np.absolute(np.array(r)-0.48)<0.04], yerr=err/np.array(dend)[np.absolute(np.array(r)-0.48)<0.04], capsize=5, ls='none', color='black', 
            elinewidth=2)

plt.yscale('log')

plt.xscale('log')
plt.ylim(0,150)
#plt.xlim(1,500)
plt.xlabel('R arcmin')
plt.ylabel(r'Number Density [stars arcmin$^{-2}$]')
plt.savefig('density.pdf')


# In[ ]:


simp3


# In[ ]:


den


# In[ ]:


ratio1=np.array(dend)/simp1
ratio2=np.array(dend2)/simp2
ratio3=den/simp3


# In[ ]:


lim = 0.0


# In[ ]:


plt.figure(figsize=(12,8))
plt.plot(np.sqrt(R2),den/den[np.absolute(np.sqrt(R2)-0.3)<0.01],'.',label = 'Plummer Model')
#plt.plot(np.array(r2),np.array(dend2)/dend2[np.absolute(np.array(r2)-0.3)<0.03],'.',label = 'Elliptical bins')
plt.plot(np.array(r),np.array(dend)/np.array(dend)[np.absolute(np.array(r)-0.3)<0.04],'.',label = 'Elliptical bins Iron')
plt.xlabel('R (degree)')
plt.legend()


# In[ ]:


datacut["LOGG"][np.isin(datacut['VRAD'],testiron['VRAD'])]


# In[ ]:


plt.figure(figsize=(10,6))
proba=0.2
#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(np.array(distot)[np.concatenate(testp)>proba],testiron["LOGG"][np.isin(testiron['VRAD'],datacut['VRAD'])][np.concatenate(testp)>proba],s=10,c=np.concatenate(testp)[np.concatenate(testp)>proba])
cbar=plt.colorbar()
cbar.set_label('Iron Probability')



#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.ylabel('LOGG')
plt.xlabel('Distance')


# In[ ]:


plt.figure(figsize=(10,6))
proba=0.2
#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(np.array(distot)[np.concatenate(testp)>proba],datacut["VRAD"][np.concatenate(testp)>proba],s=10,c=np.concatenate(testp)[np.concatenate(testp)>proba])
cbar=plt.colorbar()
cbar.set_label('Iron Probability')



#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.ylabel('VRAD')
plt.xlabel('Distance')


# In[ ]:


#logg versus rv
plt.figure(figsize=(10,6))
proba=0.2
#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(datacut["VRAD"][np.concatenate(testp)>proba],testiron["LOGG"][np.isin(testiron['VRAD'],datacut['VRAD'])][np.concatenate(testp)>proba],s=10,c=np.concatenate(testp)[np.concatenate(testp)>proba])
cbar=plt.colorbar()
cbar.set_label('Iron Probability')



#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.ylabel('VRAD')
plt.xlabel('Distance')


# In[ ]:


#Vrad versus FeH plot with Iron-Pace&Li 2022 Probability

plt.figure(figsize=(10,6))
proba=0.2
#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(datacut["FEH"][np.concatenate(testp)>proba],testiron["LOGG"][np.isin(testiron['VRAD'],datacut['VRAD'])][np.concatenate(testp)>proba],s=10,c=np.concatenate(testp)[np.concatenate(testp)>proba])
cbar=plt.colorbar()
cbar.set_label('Iron Probability')



#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.xlabel('FEH')
plt.ylabel('LOGG')


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


#Vrad versus FeH plot with Iron-Pace&Li 2022 Probability

plt.figure(figsize=(10,6))
proba=0.2
#plt.scatter(df['ra'][df['mem_fixed_complete_ep']>0.75],df['dec'][df['mem_fixed_complete_ep']>0.75],s=1)
plt.scatter(datacut["PMRA_3"][np.concatenate(testp)>proba],datacut['PMDEC_3'][np.concatenate(testp)>proba],s=10,c=np.concatenate(testp)[np.concatenate(testp)>proba])
cbar=plt.colorbar()
cbar.set_label('Iron Probability')



#plt.ylim(21, 16)
#plt.xlim(-0.3,1.5)
plt.xlabel('PMRA')
plt.ylabel('PMDEC')
plt.ylim(-5, 5)

plt.xlim(-5,5)


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





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




