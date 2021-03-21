import matplotlib.pyplot as plt
import numpy as np
from Fitpost import ProcessMonthly
from GCProcess  import readGCSO4loss





Bierlelt=[35,50,48,46,57]
Bierleunc=[6,12,10,10,16]
Heights=[1.22,1.5,1.8,2.1]
plt.rcParams.update({'font.size': 11})
folders=['H1.5']
GCcolors=['red','orange','blue','purple']
smonth=7
nmonth=1
nfolder=len(folders)
maxpressure=725

fitFunc='EMG'

GCtopdir='/Volumes/users/chili/GC/data6/chili/GCv12/run_'
outfile='/Users/chili/AvgOMISO2/GCSO4loss7.png'

PlotVar='SO2'
Engmonths=['May','Jun','Jul','Aug','Sep']
nplot=3
fig, axs = plt.subplots(nfolder,nplot,figsize=(6.8,4.8))  #
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



        #plt.text(0.5,0.97,Engmonths[imonth],transform=ax.transAxes,ha='center')

        GCfolder = GCtopdir + folder + '_ns/OutputDir/'
        [GCalts, GCSO4, GCSO4dloss, GCSO4wloss, GCProfCld,GCProfPrecp] = readGCSO4loss(GCfolder, '2008', strmonth, AerPLG,
                                                                           lat2d, lon2d, minpressure=1100.,
                                                                           maxpressure=maxpressure)


        ax.set_ylim(0.2, 2.8)
        ax.tick_params(size=1.5)

        if (imonth>0):
            ax.set_yticks([])
        else:
            ax.set_ylabel('Altitude (km)')
            plt.text(0.7, 0.9, 'DryDep', color='brown', transform=ax.transAxes, ha='center')
            plt.text(0.7, 0.85, 'WetScav', color='blue', transform=ax.transAxes, ha='center')

        GCSO4wloss[0:np.max(np.array((GCSO4wloss<0.).nonzero()))+1]=0.


        xd = np.append(np.append([0], GCSO4dloss), 0)
        yd = np.append(np.append(GCalts[0:1], GCalts), GCalts[-1])
        ax.fill(xd, yd, facecolor='brown')

        xw = np.append(GCSO4dloss, np.flip(GCSO4wloss+GCSO4dloss))
        yw = np.append(GCalts, np.flip(GCalts))
        ax.fill(xw, yw, facecolor='blue')
        ax.tick_params(size=1.5)

        ax.set_xlim(left=0)
        ax.set_xlabel('SO' + r'$_4$' + ' loss rate (kg/km' + r'$^3$' + '/h)')


        zax=axs[1]
        zax.plot(GCSO4, GCalts, color='red')
        zax.set_xlabel('SO' + r'$_4$' + ' (kg/km' + r'$^3$' + ')',color='red')
        yax = zax.twiny()
        yax.plot(GCSO4/(GCSO4wloss+GCSO4dloss), GCalts, color='blue')
        yax.set_xlabel('SO' + r'$_4$' + ' lifetime (h)',color='blue')
        zax.tick_params(size=1.5)
        yax.tick_params(size=1.5)

        zax=axs[2]
        zax.plot(GCProfCld*100., GCalts, color='grey')
        zax.set_xlabel('Cloud fraction (%)', color='grey')
        yax = zax.twiny()
        yax.plot(GCProfPrecp*3600*24*365.25, GCalts, color='purple')
        yax.set_xlabel('Precip_prod_rate (kg/kg/yr)', color = 'purple')
        zax.tick_params(size=1.5)
        yax.tick_params(size=1.5)




        ifolder = ifolder + 1



plt.savefig(outfile,dpi=1200)

