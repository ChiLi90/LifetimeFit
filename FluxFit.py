
import numpy as np
from Fitpost import ProcessMonthly
from scipy.optimize import curve_fit
from scipy.stats.stats import pearsonr
import EMFit
from scipy.stats import t
import os
import argparse




parser=argparse.ArgumentParser()
parser.add_argument('--month',type=int)
parser.add_argument('--year',type=int)
parser.add_argument('--level')
parser.add_argument('--DoBeta')
parser.add_argument('--subdir')
parser.add_argument('--DoDiff')
args=parser.parse_args()
month=args.month
year=args.year
level=args.level
strDoBeta=args.DoBeta
subdir=args.subdir
strDoDiff=args.DoDiff

if (strDoDiff=="True"):
    DoDiff=True
else:
    DoDiff=False
    
if (strDoBeta=="True"):
    DoBeta=True
else:
    DoBeta=False
DoFit=True
Pdir='/global/home/users/chili/AvgOMISO2/TA/'
if level=='All':
    pfiledir = Pdir
else:
    pfiledir = Pdir +level+'/'
outdir = pfiledir+subdir+'/'

if not os.path.exists(outdir):
    try:
        os.makedirs(outdir)
    except:
        print("outdir exist: "+outdir)


stryear='{:10.0f}'.format(year).strip()
VarNames=["SO2","AOD"]
Fluxtypes=['latlon','latt']
strmonth = ("{:10.0f}".format(month + 100).strip())[1:3]
outtxt = outdir + stryear + strmonth + '.fit'

px0 = np.zeros(len(Fluxtypes))
psigma = np.zeros(len(Fluxtypes))
pt0 = np.zeros(len(Fluxtypes))

outF = open(outtxt, "w")
for iflx in np.arange(len(Fluxtypes)):
    Fluxtype = Fluxtypes[iflx]

    [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG, Aer, AerSample, AerFlux, \
     AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG, u, v] = \
        ProcessMonthly(year, month, pfiledir, DoBeta, Fluxtype, DoDiff=DoDiff)

    for ivar in np.arange(len(VarNames)):
        VarName=VarNames[ivar]
        if VarName=='SO2':
            xflux = SO2Flux
            xfluxErr = SO2FluxErr
            Tj = SO2Tj
            TjErr = SO2TjErr
        else:
            xflux = AerFlux
            xfluxErr = AerFluxErr
            Tj = AerTj
            TjErr = AerTjErr

        relflxErrsq = np.nansum(xfluxErr ** 2) / (np.nansum(xflux) ** 2)
        relTErrsq = np.nansum(TjErr ** 2) / (np.nansum(Tj) ** 2)
        relTotErrsq = relflxErrsq + relTErrsq

        pars = ['Sample', 'flux_stdsq', 'time_stdsq']
        outdata = np.zeros(len(pars))
        outdata[0] = np.array(xflux).size
        outdata[1] = relflxErrsq
        outdata[2] = relTErrsq
        outF.write('{:>11}'.format(VarName + ','))
        outF.write(','.join(list(map('{:>10}'.format, pars))))
        outF.write('\n')
        outF.write('{:>11}'.format('Value,'))
        outF.write(','.join(list(map('{:10.3e}'.format, outdata))))
        outF.write('\n')

        if DoFit == True:
            successG = False
            ntry=0
            while (successG == False) & (ntry<10):
                [outEMG, successG] = EMFit.EMGFit(Tj, xflux, 20, fixparam=[2],fixvalue=[0.])  # )  #, fixparam=[2], fixvalue=[0.]
                ntry=ntry+1

            if successG == True:
                fity = EMFit.EMG(Tj, outEMG['a'].value, outEMG['x0'].value, outEMG['xsc'].value, \
                                 outEMG['sigma'].value, outEMG['b'].value)

                fvinds = ((np.isnan(fity) == False) & (np.isnan(xflux) == False)).nonzero()
                rsqr = pearsonr(fity[fvinds], xflux[fvinds])[0] ** 2

                if ivar == 0:
                    px0[iflx] = outEMG['x0'].value
                    psigma[iflx] = outEMG['sigma'].value
                    pt0[iflx] = outEMG['xsc'].value

                pars = ['a', 'x0', 'xsc', 'sigma', 'b', 'r2']

                outdata = np.zeros([2, len(pars)])
                for ipar in np.arange(len(pars) - 1):
                    outdata[0, ipar] = outEMG[pars[ipar]].value
                    outdata[1, ipar] = outEMG[pars[ipar]].brute_step
                outdata[0, len(pars) - 1] = rsqr
                outdata[1, len(pars) - 1] = 0.
                outF.write('{:>11}'.format('EMG M' + '{:10.0f}'.format(iflx + 1).strip() + ','))
                outF.write(','.join(list(map('{:>10}'.format, pars))))
                outF.write('\n')

                outF.write('{:>11}'.format('avg,'))
                outF.write(','.join(list(map('{:10.3e}'.format, outdata[0, :]))))
                outF.write('\n')

                outF.write('{:>11}'.format('std,'))
                outF.write(','.join(list(map('{:10.3e}'.format, outdata[1, :]))))
                outF.write('\n')

            if ivar > 0:
                successA = False
                ntry = 0
                while (successA == False) & (ntry < 10):
                    [outEMA, successA] = EMFit.EMAFit(Tj, xflux, 20, fixparam=[3, 4, 5],
                                                      fixvalue=[px0[iflx], pt0[iflx], pt0[iflx]], sDom=True)
                    ntry=ntry+1#

                if successA == True:
                    fity = EMFit.EMA(Tj, outEMA['a'].value, outEMA['c'].value, outEMA['xa'].value, \
                                     outEMA['xc'].value, outEMA['xsca'].value, outEMA['xscc'].value,
                                     outEMA['sigmaa'].value,
                                     outEMA['sigmac'].value, outEMA['b'].value)

                    fvinds = ((np.isnan(fity) == False) & (np.isnan(xflux) == False)).nonzero()

                    rsqr = pearsonr(fity[fvinds], (xflux)[fvinds])[0] ** 2

                    pars = ['a', 'c', 'xa', 'xc', 'xsca', 'xscc', 'sigmaa', 'sigmac', 'b', 'r2']
                    outdata = np.zeros([2, len(pars)])
                    for ipar in np.arange(len(pars) - 1):
                        outdata[0, ipar] = outEMA[pars[ipar]].value
                        outdata[1, ipar] = outEMA[pars[ipar]].brute_step
                    outdata[0, len(pars) - 1] = rsqr
                    outdata[1, len(pars) - 1] = 0.

                    outF.write('{:>11}'.format('EMA M' + '{:10.0f}'.format(iflx + 1).strip() + ','))
                    outF.write(','.join(list(map('{:>10}'.format, pars))))
                    outF.write('\n')

                    outF.write('{:>11}'.format('avg,'))
                    outF.write(','.join(list(map('{:10.3e}'.format, outdata[0, :]))))
                    outF.write('\n')

                    outF.write('{:>11}'.format('std,'))
                    outF.write(','.join(list(map('{:10.3e}'.format, outdata[1, :]))))
                    outF.write('\n')

outF.close()




