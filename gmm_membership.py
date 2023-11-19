# Imports



import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import SkyCoord
import matplotlib.pyplot as plt
from scipy import optimize, stats
from scipy.optimize import curve_fit
import corner
from collections import OrderedDict
import time
from astropy import table
import scipy.integrate as integrate
from astropy.coordinates import SkyCoord
from scipy.signal import find_peaks
from scipy import interpolate
from scipy import stats
from schwimmbad import MultiPool
from astropy.io import fits as pyfits
import pandas as pd
import scipy.optimize as optim


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
    print (v_proj)
    return c.radial_velocity + v_proj



def extrapolate_poly(x, y, x_new):
    # Perform polynomial regression
    coeffs = np.polyfit(x, y, 1)

    # Extrapolate the polynomial
    y_new = np.polyval(coeffs, x_new)
    return y_new

def gaum(x,sigma,mean):
    return (1/(10**sigma*np.sqrt(2*np.pi))) * np.exp(-(x- mean)**2/(2*(10**sigma)**2))


def betw(x, x1, x2): return (x > x1) & (x < x2)

def log10_error_func(x, a, b):
    return a * x + b


def cmd_selection(t, dm, g_r, iso_r, cwmin=0.1, dobhb=True):
    # width in g-r
    grw = np.sqrt(0.1 ** 2 + (3 * 10 ** log10_error_func(iso_r + dm, *popt)) ** 2)

    gw = 0.3  # min width (size) in r
    # color selection box, in case we want something different from the mag cuts made earlier
    rmin = 16
    rmax = 23
    grmin = -0.5
    grmax = 1.5
    magrange = (t['rmag'] > rmin) & (t['rmag'] < rmax) & (t['gmag0'] - t['rmag0'] < grmax) & (
                t['gmag0'] - t['rmag0'] > grmin)
    gr = t['gmag0'] - t['rmag0']
    grmax1 = np.interp(t['rmag0'], iso_r[::-1] + dm, g_r[::-1] + grw[::-1], left=np.nan, right=np.nan)
    grmax2 = np.interp(t['rmag0'], iso_r[::-1] + dm + gw, g_r[::-1] + grw[::-1], left=np.nan, right=np.nan)
    grmax3 = np.interp(t['rmag0'], iso_r[::-1] + dm - gw, g_r[::-1] + grw[::-1], left=np.nan, right=np.nan)
    grmax = np.max(np.array([grmax1, grmax2, grmax3]), axis=0)
    grmin1 = np.interp(t['rmag0'], iso_r[::-1] + dm, g_r[::-1] - grw[::-1], left=np.nan, right=np.nan)
    grmin2 = np.interp(t['rmag0'], iso_r[::-1] + dm - gw, g_r[::-1] - grw[::-1], left=np.nan, right=np.nan)
    grmin3 = np.interp(t['rmag0'], iso_r[::-1] + dm + gw, g_r[::-1] - grw[::-1], left=np.nan, right=np.nan)
    grmin = np.min(np.array([grmin1, grmin2, grmin3]), axis=0)
    ismall, = np.where(grmax - grmin < cwmin)
    grmin[ismall] = np.interp(t['rmag0'][ismall], iso_r[::-1] + dm, g_r[::-1] - cwmin / 2, left=np.nan, right=np.nan)
    grmax[ismall] = np.interp(t['rmag0'][ismall], iso_r[::-1] + dm, g_r[::-1] + cwmin / 2, left=np.nan, right=np.nan)
    colorsel = (gr < grmax) & (gr > grmin)
    colorrange = magrange & colorsel

    # making cut for the horizontal branch
    dm_m92_harris = 14.59
    m92ebv = 0.023
    m92ag = m92ebv * 3.184
    m92ar = m92ebv * 2.130
    m92_hb_r = np.array([20.5, 20.8, 20.38, 20.2, 19.9, 19.8])
    m92_hb_col = np.array([-0.25, -0.15, -0., 0.15, 0.25, 0.33])
    m92_hb_g = m92_hb_r + m92_hb_col
    des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
    des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
    des_m92_hb_g = des_m92_hb_g - dm_m92_harris
    des_m92_hb_r = des_m92_hb_r - dm_m92_harris

    dm_m92_harris = 14.59
    m92ebv = 0.023
    m92ag = m92ebv * 3.184
    m92ar = m92ebv * 2.130
    m92_hb_r = np.array([17.3, 15.8, 15.38, 15.1, 15.05, 15.0, 14.95, 14.9])
    m92_hb_col = np.array([-0.39, -0.3, -0.2, -0.0, 0.1, 0.2, 0.3, 0.4])
    m92_hb_g = m92_hb_r + m92_hb_col
    des_m92_hb_g = m92_hb_g - 0.104 * (m92_hb_g - m92_hb_r) + 0.01
    des_m92_hb_r = m92_hb_r - 0.102 * (m92_hb_g - m92_hb_r) + 0.02
    des_m92_hb_g = des_m92_hb_g - m92ag - dm_m92_harris
    des_m92_hb_r = des_m92_hb_r - m92ar - dm_m92_harris



    if dobhb:
        grw_bhb = 1.0  # BHB width in gr
        gw_bhb = 1.0  # BHB width in g
        grmin_bhb = -0.6
        grmax_bhb = 0.6
        magrange_bhb = (t['rmag'] > rmin) & (t['rmag'] < rmax) & (t['gmag0'] - t['rmag0'] < grmax_bhb) & (
                    t['gmag0'] - t['rmag0'] > grmin_bhb)

        gr_bhb = np.interp(t['rmag0'], des_m92_hb_r[::-1] + dm, des_m92_hb_g[::-1] - des_m92_hb_r[::-1], left=np.nan,
                           right=np.nan)
        rr_bhb = np.interp(t['gmag0'] - t['rmag0'], des_m92_hb_g - des_m92_hb_r, des_m92_hb_r + dm, left=np.nan,
                           right=np.nan)
        del_color_cmd_bhb = t['gmag0'] - t['rmag0'] - gr_bhb
        del_g_cmd_bhb = t['rmag0'] - rr_bhb
        colorrange_bhb = magrange_bhb & ((abs(del_color_cmd_bhb) < grw_bhb) | (abs(del_g_cmd_bhb) < gw_bhb))
        colorrange = colorrange | colorrange_bhb

    return colorrange


def data_collect(datafile, ramin, ramax, decmin, decmax, fehmin, fehmax, vmin, vmax, logg, galdis, dm, cwmin, g_r,
                 iso_r):
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

    # ra,dec, feh and quality cut
    iqk, = np.where(
        (datafile['TARGET_RA_1'] > ramin) & (datafile['TARGET_RA_1'] < ramax) & (datafile['TARGET_DEC_1'] > decmin) & (
                    datafile['TARGET_DEC_1'] < decmax) & (datafile['RVS_WARN'] == 0) & (
                    datafile['RR_SPECTYPE'] != 'QSO') & (datafile['VSINI'] < 50)

        & (~np.isnan(datafile["PMRA_ERROR"])) & (~np.isnan(datafile["PMDEC_ERROR"])) & (
            ~np.isnan(datafile["PMRA_PMDEC_CORR"])) & (datafile["FEH"] > fehmin) & (datafile["FEH"] < fehmax) & (
                    datafile["LOGG"] < logg))

    colorcut = cmd_selection(testiron[iqk], dm, g_r, iso_r, cwmin=cwmin, dobhb=True)
    up = vmin
    low = vmax

    vtest = datafile["VGSR"][iqk][colorcut]
    vcut = (vtest > low) & (vtest < up)
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
    pms = np.zeros((N, 2))
    pms[:, 0] = datafile["PMRA_3"][iqk][colorcut][vcut]
    pms[:, 1] = datafile["PMDEC_3"][iqk][colorcut][vcut]

    # pms array is computed and assigned to the variable pmmax.
    # This is essentially finding the magnitude of the maximum proper motion vector.
    pmmax = np.max(np.sqrt(np.sum(pms ** 2, axis=1)))
    # normalize the proper motion likelihood function for the entire data set
    pmnorm = 1 / (np.pi * pmmax ** 2)
    # Covariance Matrix for gal
    pmcovs = np.zeros((N, 2, 2))

    pmcovs[:, 0, 0] = datafile["PMRA_ERROR"][iqk][colorcut][vcut] ** 2 + galdis ** 2
    pmcovs[:, 1, 1] = datafile["PMDEC_ERROR"][iqk][colorcut][vcut] ** 2 + galdis ** 2
    pmcovs[:, 0, 1] = datafile["PMRA_ERROR"][iqk][colorcut][vcut] * datafile["PMDEC_ERROR"][iqk][colorcut][vcut] * \
                      datafile["PMRA_PMDEC_CORR"][iqk][colorcut][vcut]
    pmcovs[:, 1, 0] = datafile["PMRA_ERROR"][iqk][colorcut][vcut] * datafile["PMDEC_ERROR"][iqk][colorcut][vcut] * \
                      datafile["PMRA_PMDEC_CORR"][iqk][colorcut][vcut]

    return [rv, rverr, feh, feherr, pms, pmcovs], datacut, pmnorm


def two_gnfunc(dataarr, gnvals, params):
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

    # print(center, sigma, center2, sigma2, Amp1)

    gnvals = Amp1 * np.exp(-(dataarr - center) ** 2 / (2 * sigma * sigma)) / sigma / np.sqrt(2 * np.pi) \
             + (1 - Amp1) * np.exp(-(dataarr - center2) ** 2 / (2 * sigma2 * sigma2)) / sigma2 / np.sqrt(2 * np.pi)
    return gnvals


def twogau_like(xvals, gnvals, params):
    # likelihood used in fitting (normalizied)
    modelvals = two_gnfunc(xvals, gnvals, params)

    mlikelihood = - np.sum(np.log(modelvals))

    # print(mlikelihood)
    return (mlikelihood)


def fitting_reunbin(x, y):
    # guess6= [x_0, fwhm,c2,w2, h2] bounds=[(-10,10),(0,10),(-10,10),(0,10),(1e-5,1-1e-5)]

    # global xdata,ydata
    # xdata=x
    # ydata=y
    # LL = -np.sum(stats.norm.logpdf(ydata, loc=yPred, scale=sd) )
    # print (x)
    # bds1=((-100,100),(0,100),(-100,100),(100,1000),(1e-5,1-1e-5))
    # optim = minimize(twogau_like, guess6,args=(x),method = 'TNC',bounds=bds1,  options={'maxfun':100000,'disp':True})
    res2 = optim.minimize(twogau_like, [0, 1, 0, 2, 0.3], args=(x, y), method='SLSQP',
                          bounds=[(-0.1, 0.1), (0, 0.3), (-0.1, 0.1), (0.5, 5), (1e-5, 1 - 1e-5)],
                          options={'maxfun': 100000, 'disp': True})
    # np.sum(((y-two_gaussians(x, *optim1))**2)/(poisson.std(50,loc=0)**2))/(bins-len(guess6))

    # chisq1 = chisquare(y,two_gaussians(x, *optim1))[0]

    # if plot == True:
    #  plt.scatter(x,y, c='pink', label='measurement', marker='.', edgecolors=None)
    #  plt.plot(x, ypred, c='b', label='fit of Two Gaussians')
    # plt.title("Two Gaussian Fitting")
    # plt.ylabel("Number of pairs")
    # plt.xlabel("Velocity Difference")
    # plt.legend(loc='upper left')
    # plt.savefig('./halo02_results/halo02_2_2fit'+str(k)+'12grav2.5largebin.png')
    # plt.show()
    # plt.scatter(x,y, c='pink', label='measurement', marker='.', edgecolors=None)
    # plt.plot(x, (gaussian(x, p2[0], p2[1], p2[2], p2[3])), c='b', label='fit of 1 Gaussians')
    # plt.title("One gaussian fitting")
    # plt.xlabel("Velocity Difference")
    # plt.legend(loc='upper left')
    # plt.savefig('./halo02_results/halo02_2_1fit'+str(k)+'12grav2.5largebin.png')
    # plt.show()

    # return
    # np.sum(np.absolute((one_gaussian(x, *optim2) - y)**2/one_gaussian(x, *optim2)))
    # if np.absolute(aic1)-np.absolute(aic2) < 0:
    # return LL1,p1
    # else:
    # return LL2,p2

    # if np.absolute(chisq1) < np.absolute(chisq2):

    # interen = integrate.quad(lambda x: Lorentz1D_mo_ra(x, *optim1), -np.absolute((optim1[2]/2*3))+optim1[1],np.absolute(optim1[2]/2*3)+optim1[1])[0]
    # intereb = simps(y, dx=np.absolute(x[0]-x[1]))

    return res2.x

def log_prior(x, loc, scale):
    return stats.norm.logpdf(x, loc=loc, scale=scale)



param_labels = ["pgal",
                "vhel", "lsigv", "feh", "lsigfeh",
                "vbg1", "lsigvbg1", "fehbg1", "lsigfeh1",
                "pmra", "pmdec",
                "pmra1", "pmdec1", "lsigpmra1", "lsigpmdec1"]




def get_paramdict(theta):
    return OrderedDict(zip(param_labels, theta))


def project_model(theta, p1min=-600, p1max=350, p2min=-3.9, p2max=0., key="vhel"):
    """ Turn parameters into p1 and p2 distributions """
    p1arr = np.linspace(p1min, p1max, 1000)
    p2arr = np.linspace(p2min, p2max, 1000)
    p3arr = np.linspace(-4, 2, 1000)
    params = get_paramdict(theta)

    if key == 'vhel':
        p10 = params["pgal"] * stats.norm.pdf(p1arr, loc=params["vhel"], scale=10 ** params["lsigv"])
        p11 = (1 - params["pgal"]) * stats.norm.pdf(p1arr, loc=params["vbg1"], scale=10 ** params["lsigvbg1"])

        p20 = params["pgal"] * stats.norm.pdf(p2arr, loc=params["feh"], scale=10 ** params["lsigfeh"])
        p21 = (1 - params["pgal"]) * stats.norm.pdf(p2arr, loc=params["fehbg1"], scale=10 ** params["lsigfeh1"])
        # p20 = 0.548*stats.norm.pdf(p2arr, loc=-2.197, scale=10**(-0.399))
        # p21 = (1-0.548)*stats.norm.pdf(p2arr, loc=-1.478, scale=10**(-0.440))
    else:
        p10 = params["pgal"] * stats.norm.pdf(p3arr, loc=params["pmra"], scale=0.025)
        p11 = (1 - params["pgal"]) * stats.norm.pdf(p3arr, loc=params["pmra1"], scale=10 ** params["lsigpmra1"])

        p20 = params["pgal"] * stats.norm.pdf(p3arr, loc=params["pmdec"], scale=0.025)
        p21 = (1 - params["pgal"]) * stats.norm.pdf(p3arr, loc=params["pmdec1"], scale=10 ** params["lsigpmdec1"])

    return p1arr, p10, p11, p2arr, p20, p21, p3arr

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
        ax.hist(datasum[2], density=True, color='grey', bins=16)
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


def plot_2d_distr(theta, datasum, key="vhel"):
    '''
    function for plotting the distribution of two quantities p1 versus p2 for the gal/bg
    :param theta: likelihood parameters (prior/posterior)
    :param datasum: data table
    :param key: key="vhel" for plotting vrad versus Feh / key="pmra" for plotting pmra versus pmdec
    :return: plotting

    '''

    fig, ax = plt.subplots(figsize=(18, 8))
    if key == "vhel":
        ax.plot(datasum[2], datasum[0], 'k.', label='Sample')
        ax.set(xlabel="[Fe/H] (dex)", ylabel="Vgsr (km/s)", xlim=(-4, 1), ylim=(-300, 500))
        params = get_paramdict(theta)
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ax.errorbar(params["feh"], params["vhel"],
                    xerr=2 * 10 ** params["lsigfeh"], yerr=2 * 10 ** params["lsigv"],
                    color=colors[0], marker='o', elinewidth=1, capsize=3, zorder=9999, label='Gal')
        ax.errorbar(params["fehbg1"], params["vbg1"],
                    xerr=2 * 10 ** params["lsigfeh1"], yerr=2 * 10 ** params["lsigvbg1"],
                    color=colors[2], marker='x', elinewidth=1, capsize=3, zorder=9999, label='Bg')
        ax.legend()
        ax.grid()
    else:
        ax.plot(datasum[-2][:, 0], datasum[-2][:, 1], 'k.', label='Sample')
        ax.set(xlabel=r"$\rm{\mu_{\alpha}\cos(\delta) \ (mas/yr)}$", ylabel=r"$\rm{\mu_{\delta} \ (mas/yr)}$",
               xlim=(-5, 5), ylim=(-5, 5))
        params = get_paramdict(theta)
        colors = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        ax.errorbar(params["pmra"], params["pmdec"],
                    xerr=2 * 0.025, yerr=2 * 0.025,
                    color=colors[0], marker='o', elinewidth=1, capsize=3, zorder=9999, label='Gal')
        ax.errorbar(params["pmra1"], params["pmdec1"],
                    xerr=2 * 10 ** params["lsigpmra1"], yerr=2 * 10 ** params["lsigpmdec1"],
                    color=colors[2], marker='x', elinewidth=1, capsize=3, zorder=9999, label='Bg')
        ax.grid()
        ax.legend()
    fig.savefig(str(key) + '2ddistri.pdf')
    return fig

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

    pgal, vhel, lsigv, feh0, lsigfeh, \
    vbg1, lsigvbg1, fehbg1, lsigfeh1, \
    pmra_gal, pmdec_gal, pmra1, pmdec1, lsigpmra1, lsigpmdec1 = theta
    rv, rverr, feh, feherr, pms, pmcovs = datasum

    feherr = np.sqrt(feherr ** 2 + 0.15 ** 2)
    # data input
    galdis = 0.025
    N = len(rv)
    pm0s = np.zeros((N, 2))
    pm0s[:, 0] = pmra_gal
    pm0s[:, 1] = pmdec_gal

         # pm mean for bg
    bgpm0s = np.zeros((N, 2))
    bgpm0s[:, 0] = pmra1
    bgpm0s[:, 1] = pmdec1

    pm0sp = np.zeros((N, 2))
    pm0sp[:, 0] = -0.04
    pm0sp[:, 1] = -0.19

    # pm mean for bg
    bgpm0sp = np.zeros((N, 2))
    bgpm0sp[:, 0] = -2.1
    bgpm0sp[:, 1] = -1.1

    # Covariance Matrix for bg
    bgpmcovs = np.zeros((N, 2, 2))

    bgpmcovs[:, 0, 0] = pmcovs[:, 0, 0] + (10 ** lsigpmra1) ** 2 - galdis ** 2
    bgpmcovs[:, 1, 1] = pmcovs[:, 1, 1] + (10 ** lsigpmdec1) ** 2 - galdis ** 2
    bgpmcovs[:, 0, 1] = pmcovs[:, 0, 1]
    bgpmcovs[:, 1, 0] = pmcovs[:, 1, 0]

    # The prior is just a bunch of hard cutoffs
    if (pgal > 1) or (pgal < 0) or \
            (lsigv > 3) or (lsigvbg1 > 3) or \
            (lsigv < -1) or (lsigvbg1 < -1) or \
            (lsigfeh > 1) or (lsigfeh1 > 1) or (lsigfeh1 > 1) or (feh0 > 0) or (feh0 < -5) or \
            (lsigfeh < -3) or (lsigfeh1 < -3) or (lsigfeh1 < -3) or \
            (vhel > 600) or (vhel < -600) or (vbg1 > 500) or (vbg1 < -300) or \
            (pmra_gal < -4) or ((pmdec_gal) > 2) or ((pmdec_gal) < -4) or \
            (pmra_gal > 2) or \
            (pmra1 > 2) or (pmra1 < -4) or \
            (pmdec1 > 2) or (pmdec1 < -4) or \
            (lsigpmra1 > 1.3) or (lsigpmra1 < -1) or \
            (lsigpmdec1 > 1.3) or (lsigpmdec1 < -1):
        return -1e10

    # Compute log likelihood in rv
    lgal_vhel = stats.norm.logpdf(rv, loc=vhel, scale=np.sqrt(rverr ** 2 + (10 ** lsigv) ** 2))
    lbg1_vhel = stats.norm.logpdf(rv, loc=vbg1, scale=np.sqrt(rverr ** 2 + (10 ** lsigvbg1) ** 2))

    # Compute log likelihood in feh
    # feh covolved with double gaussian
    #

    # fehbin = stats.binned_statistic(feh, , 'mean', bins=18)[0]
    fehbin = []
    bins = 11
    binn2 = stats.binned_statistic(np.sort(feh), np.sort(feh), 'mean', bins=bins)
    loglike = []
    loglike2 = []
    mt = []
    mt2 = []
    for ii in range(1, bins + 1):
        # binned feh values
            fehbin.append(len(np.array(binn2[2])[binn2[2] == ii]))

        # m number of stars predict by the model belongs to Draco/ m2 belongs to the MKY background
        # total number of stars in the sample
            m = N * pgal * integrate.quad(gaum, binn2[1][ii - 1], binn2[1][ii], args=(lsigfeh, feh0))[0]
            m2 = N * (1 - pgal) * integrate.quad(gaum, binn2[1][ii - 1], binn2[1][ii], args=(lsigfeh1, fehbg1))[0]
        # m2 = N*(1-pgal)*(1/(10**lsigfeh1*np.sqrt(2*np.pi))) * np.exp(-(binn2[0][ii-1]- fehbg1)**2/(2*(10**lsigfeh1)**2))*(3.9-0.3)/bins

            mt.append(m)
            mt2.append(m2)

            n1 = len(np.array(binn2[2])[binn2[2] == ii])
            n2 = 0
        # n2 is n!
            for k in range(1, len(np.array(binn2[2])[binn2[2] == ii])):
                n2 = np.log(n1) + n2
                n1 = n1 - 1

            loglike.append(len(np.array(binn2[2])[binn2[2] == ii]) * np.log(m + m2) - n2 - m - m2)

    # fehbin.append(np.repeat(binn[0][ii-1],len(np.array(binn[2])[binn[2]==ii])))
    lgal_feh = np.sum(loglike)
    # stats.norm.logpdf(np.concatenate(fehbin), loc=feh0, scale=np.sqrt( (10**lsigfeh)**2))
    # lgal_feh = a*stats.norm.logpdf(feh, loc=feh0, scale=np.sqrt((sigma1)**2 + (10**lsigfeh)**2))+(1-a)*stats.norm.logpdf(feh, loc=feh0, scale=np.sqrt(sigma2**2 + (10**lsigfeh)**2))
    # (1-a)*stats.norm.logpdf(feh, loc=feh0, scale=np.sqrt(sigma2^2**2 + (10**lsigsigma^feh)**2))
    # lbg1_feh = a*stats.norm.logpdf(feh, loc=fehbg1, scale=np.sqrt((sigma1)**2 + (10**lsigfeh1)**2))+(1-a)*stats.norm.logpdf(feh, loc=fehbg1, scale=np.sqrt(sigma2**2 + (10**lsigfeh1)**2))
    # lbg1_feh = np.sum(loglike2)
    # stats.norm.logpdf(np.concatenate(fehbin), loc=fehbg1,scale=np.sqrt( (10**lsigfeh1)**2))
    # stats.norm.logpdf(feh, loc=fehbg1, scale=np.sqrt((feherr)**2 + (10**lsigfeh1)**2))

    # Compute log likelihood in proper motions
    # for i in range(N):

    # using multivariat gaussian for the pm likelihood
    lgal_pm = [stats.multivariate_normal.logpdf(pms[i], mean=pm0s[i], cov=pmcovs[i]) for i in range(N)]
    lbg1_pm = [stats.multivariate_normal.logpdf(pms[i], mean=bgpm0s[i], cov=bgpmcovs[i]) for i in range(N)]

    # Combine the components

    lgal = np.log(pgal) + lgal_vhel + np.array(lgal_pm) + np.log(pmnorm)
    lbg1 = np.log(1 - pgal) + lbg1_vhel + np.log(pmnorm) + lbg1_pm
    ltot = np.logaddexp(lgal, lbg1)
    ltot2 = np.logaddexp(ltot, lgal_feh)

    return ltot2.sum()


def full_like_indi(theta, ii):

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
    rv, rverr, feh, feherr, pms, pmcovs = datasum
# print (len(rv),len(feh))
    feherr = np.sqrt(feherr ** 2 + 0.15 ** 2)
    galdis = 0.025
    N = len(rv)
    pm0s = np.zeros((N, 2))
    pm0s[:, 0] = pmra_gal
    pm0s[:, 1] = pmdec_gal

# pm mean for bg
    bgpm0s = np.zeros((N, 2))
    bgpm0s[:, 0] = pmra1
    bgpm0s[:, 1] = pmdec1

# Covariance Matrix for bg
    bgpmcovs = np.zeros((N, 2, 2))

    bgpmcovs[:, 0, 0] = pmcovs[:, 0, 0] + (10 ** lsigpmra1) ** 2 - galdis ** 2
    bgpmcovs[:, 1, 1] = pmcovs[:, 1, 1] + (10 ** lsigpmdec1) ** 2 - galdis ** 2
    bgpmcovs[:, 0, 1] = pmcovs[:, 0, 1]
    bgpmcovs[:, 1, 0] = pmcovs[:, 1, 0]

    # Compute log likelihood in rv
    lgal_vhel = stats.norm.logpdf(rv[ii], loc=vhel, scale=np.sqrt(rverr[ii] ** 2 + (10 ** lsigv) ** 2))
    lbg1_vhel = stats.norm.logpdf(rv[ii], loc=vbg1, scale=np.sqrt(rverr[ii] ** 2 + (10 ** lsigvbg1) ** 2))

# Compute log likelihood in feh
    lgal_feh = stats.norm.logpdf(feh[ii], loc=feh0, scale=np.sqrt((10 ** lsigfeh) ** 2))
    lbg1_feh = stats.norm.logpdf(feh[ii], loc=fehbg1, scale=np.sqrt((10 ** lsigfeh1) ** 2))

# Compute log likelihood in proper motions
# for i in range(N):

# print (pms[i], "mean",pm0s[i], 'cov',pmcovs[i])
    lgal_pm = [stats.multivariate_normal.logpdf(pms[ii], mean=pm0s[ii], cov=pmcovs[ii])]
    lbg1_pm = [stats.multivariate_normal.logpdf(pms[ii], mean=bgpm0s[ii], cov=bgpmcovs[ii])]

# Combine the components

    lgal = np.log(pgal) + lgal_vhel + lgal_pm + np.log(pmnorm) + lgal_feh
    lbg1 = np.log(1 - pgal) + lbg1_vhel + lbg1_pm + np.log(pmnorm) + lbg1_feh

    ltot = np.logaddexp(lgal, lbg1)
    return lgal, lbg1, ltot, [np.exp(lgal_vhel), np.exp(lgal_pm), np.exp(lgal_feh)]


def main():
    #data loading for DESI iron

    ironrv = t1 = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[1].data)
    t1_fiber = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[2].data)
    t4 = table.Table(pyfits.open('/home/jupyter-jianiding/desi_jiani/desi/rvtab-hpxcoadd-all.fits')[4].data)

    t1_comb = table.hstack((t1,t1_fiber,t4))

#isochrone loading with a age = 10 Gyr
#Properties for the isochrone
#MIX-LEN  Y      Z          Zeff        [Fe/H] [a/Fe]
# 1.9380  0.2459 5.4651E-04 5.4651E-04  -1.50   0.00
    iso_file = pd.read_csv('./draco_files/isochrone_10_1.csv')

    print('# before unique selection:', len(t1_comb))

# do a unique selection based on TARGET ID. Keep the first one for duplicates
# (and first one has the smallest RV error)
    t1_unique = table.unique(t1_comb, keys='TARGETID_1', keep='first')
    print('# after unique selection:', len(t1_unique))

#taken from
#https://docs.astropy.org/en/stable/generated/examples/coordinates/rv-to-gsr.html

    coord.galactocentric_frame_defaults.set('latest')


    icrs = coord.SkyCoord(ra=t1_unique['TARGET_RA_1']*u.deg, dec=t1_unique['TARGET_DEC_1']*u.deg,
                      radial_velocity=t1_unique['VRAD']*u.km/u.s, frame='icrs')
    t1_unique['VGSR'] = rv_to_gsr(icrs)

    testiron = t1_unique

    # dust extinction correction
    testiron['gmag'], testiron['rmag'], testiron['zmag'] = [22.5 - 2.5 * np.log10(testiron['FLUX_' + _]) for _ in 'GRZ']

    testiron['gmag0'] = testiron['gmag'] - testiron['EBV_2'] * 3.186
    testiron['rmag0'] = testiron['rmag'] - testiron['EBV_2'] * 2.140
    testiron['zmag0'] = testiron['zmag'] - testiron['EBV_2'] * 1.196
    testiron['gmagerr'], testiron['rmagerr'], testiron['zmagerr'] = [
        2.5 / np.log(10) * (np.sqrt(1. / testiron['FLUX_IVAR_' + _]) / testiron['FLUX_' + _]) for _ in 'GRZ']

    # error in the r mag
    xdata = testiron['rmag']
    ydata = np.log10(testiron['rmagerr'])



    xnew = xdata[betw(xdata, 15, 24) & betw(ydata, -4, 0)]
    ynew = ydata[betw(xdata, 15, 24) & betw(ydata, -4, 0)]
    # plotting stream regions
    plt.plot(xnew, ynew, '.', alpha=0.01)



    popt, pcov = curve_fit(log10_error_func, xnew, ynew)

    xdata = np.linspace(15, 24, 100)
    plt.plot(xdata, log10_error_func(xdata, *popt))
    plt.xlabel('rmag')
    plt.ylabel('log10(rmagerr)')

    # center/distance module of Draco
    ra0 = 260.0517
    dec0 = 57.9153
    rad0 = 1.
    dm = 19.53


    xcut = (((iso_file['DECam_g'] - iso_file['DECam_r']) < 1.8) & ((iso_file['DECam_g'] - iso_file['DECam_r']) > -0.5))

    ycut = (((iso_file['DECam_r'] + dm) < 21) & ((iso_file['DECam_r'] + dm) > 15.5))
    fiso = interpolate.interp1d(((iso_file['DECam_g'].values) - (iso_file['DECam_r']).values)[xcut & ycut][-5:-1],
                                ((iso_file['DECam_r'].values) + dm)[xcut & ycut][-5:-1], kind='cubic',
                                fill_value='extrapolate')
    isox = np.arange(1.18, 1.5, 0.1)

    fiso = extrapolate_poly(((iso_file['DECam_g'].values) - (iso_file['DECam_r']).values)[xcut & ycut][-5:-1],
                            ((iso_file['DECam_r'].values) + dm)[xcut & ycut][-5:-1], isox)

    iso_r = np.append(iso_file['DECam_r'].values[xcut & ycut], fiso - dm)
    g_r = np.append(((iso_file['DECam_g'].values) - (iso_file['DECam_r']).values)[xcut & ycut], isox)

    datasum, datacut, pmnorm = data_collect(testiron,253, 267, 55.9, 59.9, -3.5, -0.5, 50, -250, 4, 0.025, dm, 0.75,
                                            g_r=g_r, iso_r=iso_r)






    param_labels = ["pgal",
                "vhel","lsigv","feh","lsigfeh",
                "vbg1","lsigvbg1","fehbg1","lsigfeh1",
                "pmra","pmdec",
                "pmra1","pmdec1","lsigpmra1","lsigpmdec1"]

## I found this guess by looking at the plot by eye and estimating. This part requires some futzing.
    p0_guess = [0.6,
            -300, 1.12, -2.196,-0.484,
           -200, 1.857, -1.479, -0.549,
            0.027, -0.184,
            -2.163, -0.999, 0.562, 0.455]

    optfunc = lambda theta: -full_like(theta)
    %timeit optfunc(p0_guess)
    optfunc(p0_guess)

    %time res = optimize.minimize(optfunc, p0_guess, method="Nelder-Mead")


    optfunc(res.x)

    for label, p in zip(param_labels, res.x):
        print(f"{label}: {p:.3f}")

    for label, p in zip(param_labels, res.x):
        print(f"{label}: {p:.3f}")

    nw = 128
    p0 = res['x']
    nit = 5000
    ep0 = np.zeros(len(p0_guess)) + 0.02
    # p0s = np.random.multivariate_normal(p0_guess, np.diag(ep0)**2, size=nw)
    # print(p0s)
    nparams = len(param_labels)
    print(nparams)
    nwalkers = 128
    p0 = p0_guess
    ep0 = np.zeros(
        len(p0)) + 0.02  # some arbitrary width that's pretty close; scale accordingly to your expectation of the uncertainty
    p0s = np.random.multivariate_normal(p0, np.diag(ep0) ** 2, size=nwalkers)

    ## Check to see things are initialized ok
    lkhds = [full_like(p0s[j]) for j in range(nwalkers)]
    assert np.all(np.array(lkhds) > -9e9)

    ## Run emcee in parallel



    nproc = 64  # use 64 cores
    nit = 3000

    def get_rstate():
        return np.random.mtrand.RandomState(seed=np.random.randint(0, 2 ** 32 - 1))

    with MultiPool(nproc) as pool:
        print("Running burnin with {} iterations".format(nit))
        start = time.time()
        es = emcee.EnsembleSampler(nw, len(p0_guess), full_like, pool=pool)
        PP = es.run_mcmc(p0s, nit, rstate0=get_rstate())
        print("Took {:.1f} seconds".format(time.time() - start))

        print(f"Now running the actual thing")
        es.reset()
        start = time.time()
        es.run_mcmc(PP.coords, nit, rstate0=get_rstate())
        print("Took {:.1f} seconds".format(time.time() - start))

    outputs = es.flatchain
    plt.style.use('default')

    # Another good test of whether or not the sampling went well is to
    # check the mean acceptance fraction of the ensemble
    print(
        "Mean acceptance fraction: {0:.3f}".format(
            np.mean(es.acceptance_fraction)
        )
    )

    fig = corner.corner(outputs, labels=param_labels, quantiles=[0.16, 0.50, 0.84], show_titles=True, color='black',
                        # add some colors
                        **{'plot_datapoints': False, 'fill_contours': True})

    def process_chain(chain, avg_error=True):
        pctl = np.percentile(chain, [16, 50, 84], axis=0)
        meds = pctl[1]
        ep = pctl[2] - pctl[1]
        em = pctl[0] - pctl[1]
        if avg_error:  # just for simplicity, assuming no asymmetry
            err = (ep - em) / 2
            return OrderedDict(zip(param_labels, meds)), OrderedDict(zip(param_labels, err))
        else:
            return OrderedDict(zip(param_labels, meds)), OrderedDict(zip(param_labels, ep)), OrderedDict(
                zip(param_labels, em))

        # plotting stream regions

    meds, errs = process_chain(outputs)

    for k, v in meds.items():
        print("{} {:.3f} {:.3f}".format(k, v, errs[k]))





if __name__ == "__main__":
    main()


    # plotting stream regions















