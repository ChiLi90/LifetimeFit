import matplotlib.pyplot as plt
import numpy as np
from Fitpost import ProcessMonthly
from GCProcess  import readGCSO2loss






Bierlelt=[35,50,48,46,57]
Bierleunc=[6,12,10,10,16]
Heights=[1.22,1.5,1.8,2.1]
plt.rcParams.update({'font.size': 11})
folders=['def']
GCcolors=['red','orange','blue','purple']
smonth=8
nmonth=1
nfolder=len(folders)
maxpressure=725

fitFunc='EMG'

GCtopdir='/Volumes/users/chili/GC/data6/chili/GCv12/run_'
outfile='/Users/chili/AvgOMISO2/GCSO2loss8.png'

PlotVar='SO2'
Engmonths=['Jul']
nplot=3

fig, axs = plt.subplots(nfolder,nplot,figsize=(7.2,4.8))  #

for month in np.arange(nmonth)+smonth:

    imonth=month-smonth

    # find the fitted lifetime
    fitdir = '/Users/chili/AvgOMISO2/TA/Coarse2/UV2/Upper/U/'
    ifolder = 0
    [xs, ys, SO2, SO2Sample, SO2Flux, SO2FluxErr, SO2Tj, SO2TjErr, SO2mask, SO2PLG, Aer, AerSample, AerFlux,
     AerFluxErr, AerTj, AerTjErr, Aermask, AerPLG, u, v] = ProcessMonthly(2008, month, fitdir, DoBeta=True,
                                                                          Fluxtype='latlon',
                                                                          DoDiff=False)
    lon2d = np.stack([xs] * len(ys), axis=0)
    lat2d = np.stack([ys] * len(xs), axis=1)

    # find the simulated lifetime of SO2
    for folder in folders:

        if nfolder==1:
            ax = axs[imonth]
        else:
            ax=axs[ifolder,imonth]


        strmonth = ("{:10.0f}".format(month + 100).strip())[1:3]



        #plt.text(0.5,0.95,Engmonths[imonth],transform=ax.transAxes,ha='center')

        GCfolder = GCtopdir + folder + '_ns/OutputDir/'
        [GCalts, GCSO2, GCSO2dloss, GCSO2wloss, GCSO2loss,GCProfCld] = readGCSO2loss(GCfolder, '2008', strmonth, SO2PLG,
                                                                           lat2d, lon2d, minpressure=1100.,
                                                                           maxpressure=maxpressure)


        ax.set_ylim(0.3, 2.8)
        ax.tick_params(size=1.5)

        if (imonth>0):
            ax.set_yticks([])
        else:
            ax.set_ylabel('Altitude (km)')
            plt.text(0.5, 0.85, 'DryDep', color='brown', transform=ax.transAxes, ha='center')
            plt.text(0.5, 0.8, 'WetScav', color='blue', transform=ax.transAxes, ha='center')
            plt.text(0.5, 0.75, 'Chem', color='red', transform=ax.transAxes, ha='center')

        xd = np.append(np.append([0], GCSO2dloss), 0)
        yd = np.append(np.append(GCalts[0:1], GCalts), GCalts[-1])
        ax.fill(xd, yd, facecolor='brown')

        xw = np.append(GCSO2dloss, np.flip(GCSO2dloss+GCSO2wloss))
        yw = np.append(GCalts, np.flip(GCalts))
        ax.fill(xw, yw, facecolor='blue')

        xc = np.append(GCSO2dloss+GCSO2wloss, np.flip(GCSO2loss))
        yc = np.append(GCalts, np.flip(GCalts))
        ax.fill(xc, yc, facecolor='red')
        ax.set_xlim(left=0)
        ax.set_xlabel('SO' + r'$_2$' + ' loss rate (kg/km' + r'$^3$' + '/h)')
        ax.tick_params(size=1.5)

        zax=axs[1]
        zax.set_xlabel('SO' + r'$_2$' + ' (kg/km' + r'$^3$' + ')', color = 'orange')
        zax.plot(GCSO2, GCalts, color='orange')
        yax=zax.twiny()
        yax.plot(GCSO2/GCSO2loss, GCalts, color='blue')
        yax.set_xlabel('SO' + r'$_2$' + ' lifetime (h)',color='blue')
        zax.tick_params(size=1.5)
        yax.tick_params(size=1.5)

        zax=axs[2]
        zax.plot(GCProfCld*100., GCalts, color='grey')
        zax.tick_params(size=1.5)
        zax.set_xlabel('Cloud fraction (%)', color='grey')
        # if imonth==2:
        #
        #     plt.text(-0.6,1.08,'SO' + r'$_2$' + ' (kg/km' + r'$^3$' + ')',color='orange',transform=ax.transAxes)
        #     plt.text(0., 1.08, 'Cloud fraction (%)', color='grey',transform=ax.transAxes)
        #     plt.text(0.8,1.08,'SO' + r'$_2$' + ' lifetime/10 (h)',color='green',transform=ax.transAxes)



        ifolder = ifolder + 1



plt.savefig(outfile,dpi=1200)

