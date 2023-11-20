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

from astropy.coordinates import SkyCoord
from scipy.signal import find_peaks
from scipy import interpolate
from scipy import stats
from schwimmbad import MultiPool
from astropy.io import fits as pyfits
import pandas as pd




def genration_test(chain, datasum):
    fehsample = []
    fehmsam = []
    for ii in range(0, 2000):
        xarr = np.linspace(-4, 0, 1000)
        fehm = np.random.choice(chain[:, 3], 1, replace=False)
        fehstd = np.random.choice(chain[:, 4], 1, replace=False)
        fehmm = np.random.choice(chain[:, 7], 1, replace=False)
        fehmstd = np.random.choice(chain[:, 8], 1, replace=False)

        fehsample.append(np.random.normal(loc=fehm, scale=10 ** fehstd, size=int(417 * 0.556)))
        fehmsam.append(np.random.normal(loc=fehmm, scale=10 ** fehmstd, size=int(417 * (1 - 0.556))))

    feht = np.concatenate([np.mean(fehmsam, axis=0), np.mean(fehsample, axis=0)])

    plt.hist(fehmsam[:100], density=False, bins=12)
    plt.hist(fehsample[:100], density=False, bins=12)
    # plt.hist(feht,density=False,bins=15)
    plt.hist(datasum[2], density=False, alpha=0.6, bins=12, label='real')
    plt.xlabel('FeH')
    plt.legend()
    leng = []
    lengm = []
    for ii in range(0, 2000):
        leng.append(len(fehsample[ii][(fehsample[ii] < -2.0) & (fehsample[ii] > -3.5)]))
        lengm.append(len(fehmsam[ii][(fehmsam[ii] < -2.0) & (fehmsam[ii] > -3.5)]))

    plt.hist(lengm, density=False, alpha=0.5, bins=15, label='real')
    return lengm,feht
    # plotting the posterior distribution for pmra and pmdec



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


# proability function for each stars to be member
def prob(itot):
    probi = []
    other = []
    for ii in range(itot):
        lgal, lbg1, ltot, _ = full_like_indi(pval, ii)
        print(ltot)
        probi.append(np.exp(lgal) / np.exp(ltot))
        other.append(_)
    return probi, other
import scipy.optimize as optim


def prob_out(datacut):



    np.savetxt('draco_prob_binlike_30000_1116',np.concatenate(testp), fmt='%.18e', delimiter=' ', newline='\n')
    dracohigh= pd.DataFrame()
    dracohigh= datacut
    dracohigh.write('draco_all_binlike_30000_1116.csv')

def gauss(x,  A, x0, sigma):
    return  A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))


def mcmc_rv(nproc, nit, p0_guess, full_like, args=[1, 1]):
    nw = 64
    p0 = res['x']
    nit = 2000
    ep0 = np.zeros(len(p0_guess)) + 0.02
    p0s = np.random.multivariate_normal(p0_guess, np.diag(ep0) ** 2, size=nw)

    nparams = len(param_labels)

    nwalkers = 64
    p0 = p0_guess
    ep0 = np.zeros(
        len(p0)) + 0.02  # some arbitrary width that's pretty close; scale accordingly to your expectation of the uncertainty
    p0s = np.random.multivariate_normal(p0, np.diag(ep0) ** 2, size=nwalkers)
    ## Check to see things are initialized ok
    # lkhds = [full_like(p0s[j]) for j in range(nwalkers)]
    # assert np.all(np.array(lkhds) > -9e9)

    from schwimmbad import MultiPool

    nproc = nproc  # use 32 cores
    nit = nit

    with MultiPool(nproc) as pool:
        print("Running burnin with {} iterations".format(nit))
        start = time.time()
        es = emcee.EnsembleSampler(nw, len(p0_guess), full_like, args=args, pool=pool)
        PP = es.run_mcmc(p0s, nit, rstate0=get_rstate())
        print("Took {:.1f} seconds".format(time.time() - start))

        print(f"Now running the actual thing")
        es.reset()
        start = time.time()
        es.run_mcmc(PP.coords, nit, rstate0=get_rstate())
        print("Took {:.1f} seconds".format(time.time() - start))

    outputs = es.flatchain
    return outputs

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


def like_rv(thetarv, vdata, verr):
    c, w = thetarv

    x = vdata
    rverr = verr
    likelog = stats.norm.logpdf(x, loc=c, scale=np.sqrt((rverr) ** 2 + (w) ** 2))
    return likelog.sum()


def ellipse_bin_rv(xdata, ydata, vel, theta0=89 * 0.0174533, r0=10 / 60, e=0.31, number=2, ra0=ra0, dec0=dec0, prob=1,
                   proba=0.):
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
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    err = []
    r = r0
    e = e
    theta0 = theta0
    tot = []

    ra = []
    dec = []
    dis = []
    probave = []
    logg = []
    distot = []
    dis2 = []
    den = []
    veldis = []
    # rtot = [0.2,0.5,1.0,1.5,2,2.5,3.,3.5,4,4,5.5,6.5,8.0]
    rtot = [0.00001, 0.8, 1.25, 1.5, 1.8, 2.3, 3.0, 5.0]
    # rtot = [0.9,6.0]

    for ii in range(len(rtot) - 1):
        b = r * rtot[ii] * np.sqrt(1 - e)
        a = r * rtot[ii] / np.sqrt(1 - e)
        b2 = r * rtot[ii + 1] * np.sqrt(1 - e)
        a2 = r * rtot[ii + 1] / np.sqrt(1 - e)
        # radius = 42/60
        # radius2 = 5*42/60
        # Set the center coordinates of the circle
        center = (ra0, dec0)

        # Generate an array of angles from 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, 100)

        # Calculate the x and y coordinates of the points on the circumference of the circle
        xt = a * np.cos(theta)
        yt = b * np.sin(theta)

        # xd=np.cos(yt)*np.sin((xt-ra0)*0.0174533)/0.0174533
        # yd=(np.sin(yt)*np.cos(dec0*0.0174533)-np.cos(yt)*np.sin(dec0*0.0174533)*np.cos((xt-ra0)*0.0174533))/0.0174533
        # xd1=xdata-center[0]
        # yd1=ydata-center[1]
        # xdl=xdata-center[0]
        # ydl=ydata-center[1]
        # x1=xdl
        # y1=ydl
        # x1= xd1*np.sin(theta0)- yd1*np.cos(theta0)

        # y1= xd1*np.cos(theta0)+ yd1*np.sin(theta0)
        # x1= xd1*np.cos(theta0)- yd1*np.sin(theta0)

        # y1= xd1*np.sin(theta0)+ yd1*np.cos(theta0)

        x1 = np.cos(ydata * 0.0174533) * np.sin((xdata - ra0) * 0.0174533) / 0.0174533
        y1 = (np.sin(ydata * 0.0174533) * np.cos(dec0 * 0.0174533) - np.cos(ydata * 0.0174533) * np.sin(
            dec0 * 0.0174533) * np.cos((xdata - ra0) * 0.0174533)) / 0.0174533
        cut1 = (x1 ** 2 / a ** 2 + y1 ** 2 / b ** 2) > 1
        cut2 = (x1 ** 2 / a2 ** 2 + y1 ** 2 / b2 ** 2) < 1

        tot.append(len(x1[(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba]))
        err.append(
            np.sqrt(len(x1[(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba])) / (np.pi * a2 * b2 - (np.pi * a * b)))

        # dis.append(2/3*((r*rtot[ii+1])**3-(r*rtot[ii])**3)/((r*rtot[ii+1])**2-(r*rtot[ii])**2))
        dis.append(np.sqrt(((r * rtot[ii + 1]) ** 2 + (r * rtot[ii]) ** 2) / 2))
        # dis.append(np.sqrt(((r*rtot[ii+1])**2-(r*rtot[ii])**2)/2))
        # prob.append(sub[(cut1)&(cut2)])
        print(len(x1[(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba]))
        vdata = vel['vlos_correct'][(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba]
        verr = vel['VRAD_ERR'][(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba]

        guess_rv = [-291, 13]
        # optfuncrv = lambda thetarv: like_rv(thetarv,vdata,verr)
        # timeit optfuncrv(guess_rv)
        test = mcmc_rv(60, 2000, guess_rv, like_rv, args=[vdata, verr])
        plt.hist(vdata, bins=5)
        plt.show()
        meds, errs = process_chain_rv(test)
        ra.append(xdata[(cut1) & (cut2)])
        dec.append(ydata[(cut1) & (cut2)])

        veldis.append([meds[1], errs[1]])
        # veldis.append([popt[2],np.sqrt(np.diagonal(covariance))[2]])

        # print (vel[(cut1)&(cut2)])


    return tot, ra, dec, dis, den, err, dis2, veldis




def epllitic_v(datacut):
    # making CMD diagram for the data
    ra0 = 260.0517
    dec0 = 57.9153
    rad0 = 1.6

    dm = 19.53

    dwarf_df = pd.DataFrame()
    dwarf_df = datacut

    ra0 = ra0
    dec0 = dec0
    dist = 75.8
    # dist = 100
    # pmra0 = 0.045
    pmra0 = 0.03
    pmdec0 = -0.18
    vlos0 = -291.21  # dwarftable[dwarftable['key']=='ursa_minor_1']['vlos_systemic'][0]
    prob = 0
    c = 4.74047
    vel_pmra0, vel_pmdec0 = pmra0 * c * dist, pmdec0 * c * dist
    a = np.pi / 180.
    ca = np.cos(ra0 * a)
    cd = np.cos(dec0 * a)
    sa = np.sin(ra0 * a)
    sd = np.sin(dec0 * a)
    vx = vlos0 * cd * sa + vel_pmra0 * cd * ca - vel_pmdec0 * sd * sa
    vy = -vlos0 * cd * ca + vel_pmdec0 * sd * ca + vel_pmra0 * cd * sa
    vz = vlos0 * sd + vel_pmdec0 * cd
    dwarf_df['vlos_correct'] = np.zeros(len(dwarf_df), dtype=float)
    deltax = dwarf_df['TARGET_RA_1']
    deltay = dwarf_df['TARGET_DEC_1']
    bx = np.cos(deltay * a) * np.sin(deltax * a)
    by = -np.cos(deltay * a) * np.cos(deltax * a)
    bz = np.sin(dwarf_df['TARGET_DEC_1'] * a)
    dwarf_df['delta_vlos_correct'] = bx * vx + vy * by + bz * vz - vlos0
    dwarf_df['vlos_correct'] = dwarf_df['delta_vlos_correct'] + dwarf_df['VRAD']

    plt.scatter(dwarf_df['TARGET_RA_1'], dwarf_df['TARGET_DEC_1'], c=dwarf_df['delta_vlos_correct'], marker='o')
    plt.xlabel('RA')
    plt.ylabel('DEC')
    plt.colorbar()

    tot, ra, dec, dis, den, err, dis2, veldis2 = ellipse_bin_rv(dwarf_df['TARGET_RA_1'], dwarf_df['TARGET_DEC_1'],
                                                                dwarf_df, theta0=89 * 0.0174533, r0=10 / 60, e=0.31,
                                                                number=10, ra0=ra0, dec0=dec0,
                                                                prob=np.concatenate(testp), proba=0.9)

    # plt.plot(np.array(r[:5])*60,vral)
    vralcor = veldis2
    # vral = veldis2
    plt.scatter(np.array(dis) * np.pi / 180 * 75.8 * 1000, np.array(vralcor)[:, 0])
    plt.errorbar(np.array(dis) * np.pi / 180 * 75.8 * 1000, np.array(vralcor)[:, 0], yerr=np.array(vralcor)[:, 1],
                 linestyle='none', label='Corrected Velocity Dispersion')
    plt.ylim(0, 20)
    plt.xlabel('R (pc)')
    plt.ylabel('Velocity Dispersion (km/s)')
    plt.legend()
    plt.savefig('velocitydisp_1116.pdf')


def ellipse_bin_pro(xdata, ydata, vel, theta0=89 * 0.0174533, r0=10 / 60, e=0.31, number=2, ra0=ra0, dec0=dec0, prob=1,
                    proba=0.):
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
    plt.figure(figsize=(10, 6))
    fig, ax = plt.subplots(figsize=(10, 6))
    err = []
    r = r0
    e = e
    theta0 = theta0
    tot = []

    ra = []
    dec = []
    dis = []
    probave = []
    logg = []
    distot = []
    dis2 = []
    den = []
    veldis = []
    # rtot = [0.2,0.5,1.0,1.5,2,2.5,3.,3.5,4,4,5.5,6.5,8.0]
    rtot = [0.00001, 1.0, 2, 3., 4.5, 6, 7, 8.0, 10, 12]
    for ii in range(len(rtot) - 1):
        b = r * rtot[ii] * np.sqrt(1 - e)
        a = r * rtot[ii] / np.sqrt(1 - e)
        b2 = r * rtot[ii + 1] * np.sqrt(1 - e)
        a2 = r * rtot[ii + 1] / np.sqrt(1 - e)
        # radius = 42/60
        # radius2 = 5*42/60
        # Set the center coordinates of the circle
        center = (ra0, dec0)

        # Generate an array of angles from 0 to 2*pi
        theta = np.linspace(0, 2 * np.pi, 100)

        # Calculate the x and y coordinates of the points on the circumference of the circle
        xt = a * np.cos(theta)
        yt = b * np.sin(theta)

        # xd=np.cos(yt)*np.sin((xt-ra0)*0.0174533)/0.0174533
        # yd=(np.sin(yt)*np.cos(dec0*0.0174533)-np.cos(yt)*np.sin(dec0*0.0174533)*np.cos((xt-ra0)*0.0174533))/0.0174533
        # xd1=xdata-center[0]
        # yd1=ydata-center[1]
        # xdl=xdata-center[0]
        # ydl=ydata-center[1]
        # x1=xdl
        # y1=ydl
        # x1= xd1*np.sin(theta0)- yd1*np.cos(theta0)

        # y1= xd1*np.cos(theta0)+ yd1*np.sin(theta0)
        # x1= xd1*np.cos(theta0)- yd1*np.sin(theta0)

        # y1= xd1*np.sin(theta0)+ yd1*np.cos(theta0)
        x1 = np.cos(ydata * 0.0174533) * np.sin((xdata - ra0) * 0.0174533) / 0.0174533
        y1 = (np.sin(ydata * 0.0174533) * np.cos(dec0 * 0.0174533) - np.cos(ydata * 0.0174533) * np.sin(
            dec0 * 0.0174533) * np.cos((xdata - ra0) * 0.0174533)) / 0.0174533
        cut1 = (x1 ** 2 / a ** 2 + y1 ** 2 / b ** 2) > 1
        cut2 = (x1 ** 2 / a2 ** 2 + y1 ** 2 / b2 ** 2) < 1

        tot.append(len(x1[(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba]))
        err.append(
            np.sqrt(len(x1[(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba])) / (np.pi * a2 * b2 - (np.pi * a * b)))
        den.append(len(x1[(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba]) / (np.pi * a2 * b2 - (np.pi * a * b)))
        # gco = np.histogram(vel[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],bins=8)
        # popt, covariance = curve_fit(gauss, gco[1][1:], gco[0],p0=[10,-290,2])

        # popt, covariance = curve_fit(gauss, gco[1][1:], gco[0],p0=[10,-290,2])
        # fit_y = gauss(gco[1][1:],*popt)
        # print (popt, np.sqrt(np.diagonal(covariance)))
        # plt.plot(gco[1][1:], gco[0], 'o', label='data')
        # plt.plot(gco[1][1:], fit_y, '-', label='fit')
        # plt.legend()
        # plt.show()
        # logg.append(testiron['LOGG'][np.isin(testiron['VRAD'],datacut['VRAD'])][[(cut1)&(cut2)]>proba])

        # dis.append(2/3*((r*rtot[ii+1])**3-(r*rtot[ii])**3)/((r*rtot[ii+1])**2-(r*rtot[ii])**2))
        dis.append(np.sqrt(((r * rtot[ii + 1]) ** 2 + (r * rtot[ii]) ** 2) / 2))
        # dis.append(np.sqrt(((r*rtot[ii+1])**2-(r*rtot[ii])**2)/2))
        # prob.append(sub[(cut1)&(cut2)])
        print(len(x1[(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba]))

        ra.append(xdata[(cut1) & (cut2)])
        dec.append(ydata[(cut1) & (cut2)])

        veldis.append(np.std(vel[(cut1) & (cut2)][prob[(cut1) & (cut2)] > proba]))
        # veldis.append([popt[2],np.sqrt(np.diagonal(covariance))[2]])

        probave.append(prob)
        dis2.append(r * rtot[ii])
        # print (vel[(cut1)&(cut2)])

    # Plot the circle
    # ax.plot(xt, yt,label = str(ii)+' r$_{h}$',c='k')
    # plt.scatter(x1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],y1[(cut1)&(cut2)][prob[(cut1)&(cut2)]>proba],s=6)
    # cbar=plt.colorbar()
    # cbar.set_label('Iron-Pace&LI Probability')
    # plt.legend(loc='upper left', bbox_to_anchor=(1, 0.5))
    # plt.ylim(21, 16)
    # plt.xlim(258,262)
    # plt.ylabel('DEC')
    # plt.xlabel('RA')
    # cbar=plt.colorbar()
    # cbar.set_label('Iron Probability')
    return tot, ra, dec, dis, probave, np.sqrt(x1 ** 2 + y1 ** 2), den, err, dis2, veldis

# Set the aspect ratio of the plot to 'equal' to make the circle appear circular
# ax.set_aspect('equal')



# Set the radius of the circle to be the half light radiu























