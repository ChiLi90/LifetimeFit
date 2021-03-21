import matplotlib.pyplot as plt
import numpy as np
from Fitpost import ProcessMonthly,readfitfile,CalcErr
from GCProcess  import readGCSO2lt



Bierlelt=[35,50,48,46,57]
Bierleunc=[6,12,10,10,16]
Heights=[1.22,1.5,1.8,2.1]
plt.rcParams.update({'font.size': 6})
fits=['Upper','Upper7']
folders=['def','H1.5','H1.8','H2.1']
GCcolors=['grey','orange','blue','purple']
smonth=5
nmonth=5
nfolder=len(folders)
maxpressure=725

fitFunc='EMG'

GCtopdir='/Volumes/users/chili/GC/data6/chili/GCv12/run_'
outfile='/Users/chili/AvgOMISO2/SO2ltcompare.png'
if fitFunc=='EMA':
    parinds = [1, 2, 7]
else:
    parinds = [0, 1, 3]
PlotVar='SO2'
Engmonths=['May','Jun','Jul','Aug','Sep']

fig, axs = plt.subplots(nmonth, figsize=(7.2,6.4))  #
for month in np.arange(nmonth)+smonth:

    imonth=month-smonth
    ax=axs[imonth]
    ax.bar([0.5], Bierlelt[imonth],width=1,yerr=Bierleunc[imonth],color='green')
    ax.text(0.5, Bierlelt[imonth] * 1.25, '{:10.0f}'.format(np.round(Bierlelt[imonth])).strip(), color='green',
            horizontalalignment='center')

    ifit=0
    strmonth = ("{:10.0f}".format(month + 100).strip())[1:3]
    ax.set_xticks([])
    ax.set_xlim(0,16.5)
    ax.set_ylim(0, 110)
    ax.set_title(Engmonths[imonth])

    ibar=0
    for fit in fits:
        #find the fitted lifetime
        fitdir='/Users/chili/AvgOMISO2/TA/Coarse2/UV2/'+fit+'/U/'
        ifolder = 0
        [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG, Aer, AerSample, AerFlux,
         AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG, u, v] = ProcessMonthly(2008, month, fitdir, DoBeta=True, Fluxtype='latlon',
                                                                              DoDiff=False)
        lon2d = np.stack([xs] * len(ys), axis=0)
        lat2d = np.stack([ys] * len(xs), axis=1)
        fitfile = fitdir + 'Mass/'+'2008' + strmonth + '.fit'

        [fitPars, fitVals, fitstds] = readfitfile(fitfile, fluxtype='latt', Var=PlotVar, fitFunc=fitFunc)
        # #95% confidence of each parameter, source, lifetime and sigma
        [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG1, Aer, AerSample, AerFlux,
         AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG, u, v] = ProcessMonthly(2008, month, fitdir, DoBeta=True,
                                                                              Fluxtype='latt',
                                                                              DoDiff=False)
        parErrs = CalcErr(fitVals[parinds], fitstds[parinds], SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, fitFunc)
        ax.bar([2.+ifit*0.5+ibar], fitVals[1], width=1 , yerr=parErrs[1], color='red')
        ax.text(2.+ifit*0.5+ibar, fitVals[1]*1.25,'{:10.0f}'.format(np.round(fitVals[1])).strip(),color='red',horizontalalignment='center')
        ibar=ibar+1

        #find the simulated lifetime of SO2
        for folder in folders:
            if fit=='Upper':
                minpressure = 875. + 12.5

            if fit=='Upper7':
                minpressure = 825. + 12.5


            GCfolder=GCtopdir+ folder + '_ns/OutputDir/'
            GCSO2lt=readGCSO2lt(GCfolder,'2008',strmonth,SO2PLG,lat2d,lon2d,minpressure=minpressure,maxpressure=maxpressure)

            ax.bar([2.+ifit*0.5+ibar], GCSO2lt, width=1, color=GCcolors[ifolder])
            ax.text(2. + ifit * 0.5 + ibar, GCSO2lt* 1.05, '{:10.0f}'.format(np.round(GCSO2lt)).strip(), color=GCcolors[ifolder],horizontalalignment='center')
            ibar=ibar+1
            ifolder=ifolder+1

        ifit=ifit+1

    ifolder=0
    # find the simulated lifetime of SO2
    for folder in folders:
        GCfolder = GCtopdir + folder + '_ns/OutputDir/'
        GCSO2lt = readGCSO2lt(GCfolder, '2008', strmonth, SO2PLG, lat2d, lon2d, minpressure=1100.,
                              maxpressure=875.+12.5)
        ax.bar([2.+ifit*0.5+ibar], GCSO2lt, width=1, color=GCcolors[ifolder])
        ax.text(2. + ifit * 0.5 + ibar, GCSO2lt * 1.05, '{:10.0f}'.format(np.round(GCSO2lt)).strip(), color=GCcolors[ifolder],
                horizontalalignment='center')
        ifolder=ifolder+1
        ibar=ibar+1

    ax.plot([1.25,1.25],[0,110],linestyle='--',color='black')
    ax.plot([6.75, 6.75], [0, 110], linestyle='--',color='black')
    ax.plot([12.25, 12.25], [0, 110], linestyle='--',color='black')
    if imonth==0:
        plt.text(0.85,0.92,'Beirle14',color='green',transform=ax.transAxes,ha='center')
        plt.text(0.85, 0.83, 'This work', color='red', transform=ax.transAxes, ha='center')
        for ifolder in np.arange(nfolder):
            plt.text(0.85, 0.74-ifolder*0.09, 'GC_'+'{:10.1f}'.format(Heights[ifolder]).strip()+'km', color=GCcolors[ifolder], transform=ax.transAxes, ha='center')
    if imonth==2:
        ax.set_ylabel('SO'+r"$_2$"+' lifetime (hr)')
    if imonth==nmonth-1:
        plt.text(1.25/2., -40, '1.50-2.50 km',ha='center',va='center',color='black',rotation=90)
        plt.text((1.25+6.75)/2, -20, '1.10-2.73 km', ha='center',va='center',color='black',rotation=20)
        plt.text((6.75+12.25)/2, -20, '1.58-2.73 km', ha='center',va='center',color='black',rotation=20)
        plt.text((12.25 + 16.5) / 2, -20, '0-1.10 km', ha='center',va='center',color='black',rotation=20)

    ax.set_aspect(0.06)


plt.savefig(outfile,dpi=1200)

