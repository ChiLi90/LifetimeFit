import matplotlib.pyplot as plt
import numpy as np
from Fitpost import ProcessMonthly,readfitfile,CalcErr
from GCProcess  import readGCSO4lt
from EMFit import EMA
from scipy.stats.stats import pearsonr






Bierlelt=[35,50,48,46,57]
Bierleunc=[6,12,10,10,16]
Heights=[1.5]
plt.rcParams.update({'font.size': 8})
fits=['Upper','Upper7']
folders=['H1.5']
GCcolors=['orange']
smonth=5
nmonth=5
nfolder=len(folders)
maxpressure=725

fitFunc='EMA'
plt.autoscale(False)

GCtopdir='/Volumes/users/chili/GC/data6/chili/GCv12/run_'
outfile='/Users/chili/AvgOMISO2/SO4ltcompare.png'
if fitFunc=='EMA':
    parinds = [1, 2, 7]
else:
    parinds = [0, 1, 3]
PlotVar='Aer'
Engmonths=['May','Jun','Jul','Aug','Sep']

# for month in np.arange(nmonth)+smonth:
#     strmonth = ("{:10.0f}".format(month + 100).strip())[1:3]
#     fitdir = '/Users/chili/AvgOMISO2/TA/Coarse2/UV2/' + fits[0] + '/U/'
#     [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG, Aer, AerSample, AerFlux,
#      AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG1, u, v] = ProcessMonthly(2008, month, fitdir, DoBeta=True,
#                                                                            Fluxtype='latlon',
#                                                                            DoDiff=False)
#     lon2d = np.stack([xs] * len(ys), axis=0)
#     lat2d = np.stack([ys] * len(xs), axis=1)
#     GCSO4lt = readGCSO4lt(GCtopdir+ 'H1.5_ns/OutputDir/', '2008', strmonth, AerPLG1, lat2d, lon2d, minpressure=1100.,maxpressure=0.)
#     print(month,GCSO4lt)
#
# exit()
fig, axs = plt.subplots(figsize=(6.2,9.6))  #
axs.remove()
for month in np.arange(nmonth)+smonth:

    imonth=month-smonth
    ax=fig.add_axes([0.08, 0.8-0.17*imonth, 0.7, 0.15])
    ifit=0
    strmonth = ("{:10.0f}".format(month + 100).strip())[1:3]
    ax.set_xticks([])
    ax.set_xlim(0,7)
    ax.set_ylim(0, 100)
    ax.set_title(Engmonths[imonth])

    ibar=0
    for fit in fits:
        #find the fitted lifetime
        fitdir='/Users/chili/AvgOMISO2/TA/Coarse2/UV2/'+fit+'/U/'
        ifolder = 0

        [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG, Aer, AerSample, AerFlux,
         AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG1, u, v] = ProcessMonthly(2008, month, fitdir, DoBeta=True, Fluxtype='latlon',
                                                                              DoDiff=False)
        lon2d = np.stack([xs] * len(ys), axis=0)
        lat2d = np.stack([ys] * len(xs), axis=1)
        fitfile = fitdir + 'Mass/'+'2008' + strmonth + '.fit'

        [fitPars, fitVals, fitstds] = readfitfile(fitfile, fluxtype='latt', Var=PlotVar, fitFunc=fitFunc)
        # #95% confidence of each parameter, source, lifetime and sigma
        [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG, Aer, AerSample, AerFlux,
         AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG, u, v] = ProcessMonthly(2008, month, fitdir, DoBeta=True,
                                                                               Fluxtype='latt',
                                                                               DoDiff=False)
        parErrs =  CalcErr(fitVals[parinds], fitstds[parinds], AerFlux, AerFluxErr, AerTj, AerTjErr,fitFunc,taug=[fitVals[3],fitstds[3],SO2Flux,SO2FluxErr,SO2Tj,SO2TjErr])
        fity = EMA(AerTj, *fitVals)
        if (fitVals[2]>1.25*fitVals[7]) & (pearsonr(fity, AerFlux)[0] ** 2>0.8):
            ax.bar([0.5+ifit*0.5+ibar], fitVals[2], width=1 , yerr=parErrs[1], color='red')
            ax.text(0.75+ifit*0.5+ibar, fitVals[2] * 1.1, '{:10.0f}'.format(np.round(fitVals[2])).strip(), color='red',
                   horizontalalignment='center',size=10)
        ibar=ibar+1

        [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG, Aer, AerSample, AerFlux,
         AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG2, u, v] = ProcessMonthly(2008, month, fitdir, DoBeta=False,
                                                                              Fluxtype='latlon',
                                                                              DoDiff=False)
        fitfile = fitdir + 'OD/' + '2008' + strmonth + '.fit'

        [fitPars, fitVals, fitstds] = readfitfile(fitfile, fluxtype='latt', Var=PlotVar, fitFunc=fitFunc)
        # #95% confidence of each parameter, source, lifetime and sigma
        [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG, Aer, AerSample, AerFlux,
         AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG, u, v] = ProcessMonthly(2008, month, fitdir, DoBeta=False,
                                                                              Fluxtype='latt',
                                                                              DoDiff=False)
        parErrs =  CalcErr(fitVals[parinds], fitstds[parinds], AerFlux, AerFluxErr, AerTj, AerTjErr,fitFunc,taug=[fitVals[3],fitstds[3],SO2Flux,SO2FluxErr,SO2Tj,SO2TjErr])
        fity = EMA(AerTj, *fitVals)
        if (fitVals[2] > 1.25 * fitVals[7]) & (pearsonr(fity, AerFlux)[0] ** 2 > 0.8):
            ax.bar([0.5 + ifit * 0.5 + ibar], fitVals[2], width=1, yerr=parErrs[1], color='olive')
            ax.text(0.75 + ifit * 0.5 + ibar, fitVals[2] * 1.1, '{:10.0f}'.format(np.round(fitVals[2])).strip(), color='olive',
                    horizontalalignment='center',size=10)
        ibar = ibar + 1

        #find the simulated lifetime of SO2
        for folder in folders:
            if fit=='Upper':
                minpressure = 875. + 12.5

            if fit=='Upper7':
                minpressure = 825. + 12.5


            GCfolder=GCtopdir+ folder + '_ns/OutputDir/'
            GCSO4lt=readGCSO4lt(GCfolder,'2008',strmonth,AerPLG1,lat2d,lon2d,minpressure=minpressure,maxpressure=maxpressure)

            ax.bar([0.5+ifit*0.5+ibar], GCSO4lt, width=1, color=GCcolors[ifolder])
            ax.text(0.75 + ifit * 0.5 + ibar, GCSO4lt * 1.1, '{:10.0f}'.format(np.round(GCSO4lt)).strip(), color=GCcolors[ifolder],
                    horizontalalignment='center',size=10)
            ibar=ibar+1
            ifolder=ifolder+1

        ifit=ifit+1

        # ax.plot([1.25,1.25],[0,110],linestyle='--',color='black')
        ax.plot([3.25, 3.25], [0, 110], linestyle='--', color='black')
        ax.plot([6.75, 6.75], [0, 110], linestyle='--', color='black')
        if imonth == 1:
            # plt.text(0.85,0.92,'Beirle14',color='green',transform=ax.transAxes,ha='center')
            plt.text(0.65, 0.91, 'This work (sulfate mass)', color='red', transform=ax.transAxes, ha='center')
            plt.text(0.65, 0.81, 'This work (AOD)', color='olive', transform=ax.transAxes, ha='center')
            for ifolder in np.arange(nfolder):
                plt.text(0.65, 0.71 - ifolder * 0.1, 'GC_' + '{:10.1f}'.format(Heights[ifolder]).strip() + 'km',
                         color=GCcolors[ifolder], transform=ax.transAxes, ha='center')
        if imonth == 2:
            ax.set_ylabel('SO' + r"$_4$" + ' lifetime (hr)')
    if imonth == nmonth - 1:
        # plt.text(1.25/2., -40, '1.50-2.50 km',ha='center',va='center',color='black',rotation=90)
        plt.text(0.167, -0.1, '1.10-2.73 km', ha='center', va='center', color='black',transform=ax.transAxes)
        plt.text(0.6, -0.1, '1.58-2.73 km', ha='center', va='center', color='black',transform=ax.transAxes)


    #ax.set_aspect(0.06)
    ax.spines['right'].set_visible(False)

    ifolder=0
    pos1 = ax.get_position()  # get the original position
    xax=fig.add_axes([0.8, 0.8-0.17*imonth, 0.1, 0.15])

    xax.set_ylim(0, 260)
    xax.yaxis.tick_right()
    xax.set_xlim(7.,8.)
    xax.spines['left'].set_visible(False)
    # find the simulated lifetime of SO2
    for folder in folders:
        GCfolder = GCtopdir + folder + '_ns/OutputDir/'
        GCSO4lt = readGCSO4lt(GCfolder, '2008', strmonth, AerPLG1, lat2d, lon2d, minpressure=1100.,
                              maxpressure=875.+12.5)

        xax.bar([0.5+ifit*0.5+ibar], GCSO4lt, width=1, color=GCcolors[ifolder])
        xax.text(0.75 + ifit * 0.5 + ibar, GCSO4lt * 1.02, '{:10.0f}'.format(np.round(GCSO4lt)).strip(), color=GCcolors[ifolder],
                horizontalalignment='center',size=10)
        ifolder=ifolder+1
        ibar=ibar+1
    if imonth == 2:
        xax.yaxis.set_label_position("right")
        xax.set_ylabel('SO' + r"$_4$" + ' lifetime (hr)')
    if imonth==nmonth-1:
        plt.text(0.5, -0.1, '0-1.10 km', ha='center', va='center', color='black', transform=xax.transAxes)
    xax.set_xticks([])




plt.savefig(outfile,dpi=1200)

