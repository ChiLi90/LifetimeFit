from Satellites import readMonthlyfile
from Fitpost import ProcessMonthly, mapConc
import numpy as np
import matplotlib.pyplot as plt
import shapefile as shp
import matplotlib.cm as cm
import matplotlib.ticker as ticker



Pdir='/Users/chili/AvgOMISO2/TA/Coarse2/UV/'
USshape='/Users/chili/AvgOMISO2/cb_2019_us_state_20m/cb_2019_us_state_20m.shp'
level='Upper'
subdir='Mass'
year=2008
resolution=[0.25,0.25]
plotVal='Aer'
outfile='/Users/chili/AvgOMISO2/TA/Coarse2/AerFluxup.png'

plt.rcParams.update({'font.size': 5})
plt.rcParams['axes.linewidth'] = 0.2

DoBeta=True
DoDiff=False
Fluxtype='latlon'
if plotVal=='SO2':
    cmap = cm.get_cmap('YlOrBr')  #
else:
    cmap=cm.get_cmap('inferno_r')
cmap.set_bad('white')
dmap = cm.get_cmap('bwr')
dmap.set_bad('white')
if level == 'All':
    folder = Pdir +'U/'
else:
    folder = Pdir + level + '/U/'
outdir = folder + subdir + '/'
sf = shp.Reader(USshape)
bgminlat=-4
bgmaxlat=3
bgminlon=190
bgmaxlon=205

if DoBeta:
    cbartitles = ["OMI SO" + r"$_2$" + " (kg/km" + r"$^2$" + ")", 'Sulfate mass ' + "(kg/km" + r"$^2$" + ")"]
    enhtitles = ['SO' + r'$_2$' + ' Enhancement', 'Sulfate Enhancement']
    #enhtitles = [r"$\delta$"+"SO" + r"$_2$" + " (kg/km" + r"$^2$" + ")", r"$\delta$"+ "Aerosol mass\n" + "(kg/km" + r"$^2$" + ")"]
    ytitles = ["SO" + r"$_2$" + " flux (Gg/h)", "Aerosol mass flux (Gg/h)"]
else:
    cbartitles = ["OMI SO" + r"$_2$" + " (kg/km" + r"$^2$" + ")", 'MODIS AOD']
    enhtitles = ['SO' + r'$_2$' + ' Enhancement', 'AOD Enhancement']
    #enhtitles = [r"$\delta$" + "SO" + r"$_2$" + " (kg/km" + r"$^2$" + ")",r"$\delta$" + "AOD"]
    ytitles = ["SO" + r"$_2$" + " flux (Gg/h)", "AOD flux (km" + r"$^2$" + "/s)"]

xticks=[145,165,185,205]
yticks=[0,10,20,30]

nmonth=5
smonth=5
#[xs,ys,SO2,SO2Sample,SO2Flux,SO2FluxErr,SO2Tj,SO2TjErr,SO2mask,SO2PLG,Aer,AerSample,AerFlux,AerFluxErr,AerTj,AerTjErr,Aermask,AerPLG,u,v]=ProcessMonthly(year,month,folder,DoBeta,Fluxtype)


#plot the concentrations and enhancements
fig, axs = plt.subplots(nmonth, 2, )
maxconcs=[50,20] #50
maxenhs=[10,5]   #30
strmonths=['May','June','July','August','September']


# colors=['red','orange','green','blue','purple']
#
# GALalts=np.array([0.111,0.323,0.540,0.762,0.989,1.220,1.457,1.700,1.949,2.204,2.466,3.012])
# GALbh=GALalts-GALalts
# for ilev in np.arange(len(GALalts)):
#     if ilev==0:
#         GALbh[ilev]=2*GALalts[ilev]
#     else:
#         GALbh[ilev]=2*(GALalts[ilev]-np.sum(GALbh[0:ilev]))
# fig, axs = plt.subplots(1,3,)

for imonth in np.arange(nmonth):

    month=imonth+smonth
    #[xs, ys, SO2, SO2Sample, Aer, AerSample] = ProcessMonthly(2008, month,folder, DoBeta,Fluxtype,MapOnly=True,DoPLG=False)
    [xs, ys, SO2, SO2Sample, SO2mask,SO2PLG, Aer, AerSample,Aermask, AerPLG] = \
        ProcessMonthly(2008, month,folder,DoBeta,Fluxtype,MapOnly=True,DoDiff=False)
    if imonth==0:
        x2d = np.stack([xs]*len(ys),axis=0)
        y2d = np.stack([ys] * len(xs), axis=1)
        minlon = np.min(xs) - 0.125
        maxlon = np.max(xs) + 0.125
        minlat = np.min(ys) - 0.125
        maxlat = np.max(ys) + 0.125
        ymin = np.min(ys) / 0.25
        xmin = np.min(xs) / 0.25

    # [xs, ys, preSO2, preSO2Sample, preSO2mask, preSO2PLG, preAer, preAerSample, preAermask, preAerPLG] = \
    #     ProcessMonthly(2008, month, folder,DoBeta, Fluxtype,MapOnly=True,DoDiff=True)


    preSO2 = np.zeros(SO2.shape)
    preAer = np.zeros(Aer.shape)
    preSO2Sample = np.zeros(SO2Sample.shape, dtype=int)
    preAerSample = np.zeros(AerSample.shape, dtype=int)
    # map of 2005-2007 avg
    for year in np.arange(3) + 2005:
        [xs, ys, yrSO2, yrSO2Sample, yrAer, yrAerSample] = ProcessMonthly(year, month, folder, DoBeta, Fluxtype,
                                                                          MapOnly=True, DoQF=False,DoPLG=False,DoDiff=False)

        SO2inds = ((np.isnan(yrSO2) == False)).nonzero()
        preSO2[SO2inds] = preSO2[SO2inds] + yrSO2[SO2inds]
        preSO2Sample[SO2inds] = preSO2Sample[SO2inds] + 1
        Aerinds = (np.isnan(yrAer) == False).nonzero()
        preAer[Aerinds] = preAer[Aerinds] + yrAer[Aerinds]
        preAerSample[Aerinds] = preAerSample[Aerinds] + 1

    preSO2 = preSO2 / preSO2Sample
    preAer = preAer / preAerSample


    # SO2Enhance = SO2 / preSO2
    # AerEnhance = Aer / preAer
    SO2tEnhance = np.nansum(SO2[SO2mask > 0]) / np.nansum(preSO2[SO2mask > 0])
    AertEnhance = np.nansum(Aer[Aermask > 0]) / np.nansum(preAer[Aermask > 0])

    SO2=SO2*1.e6
    preSO2=preSO2*1.e6
    if DoBeta:
        Aer=Aer*1.e6
        preAer=preAer*1.e6

    if plotVal=='SO2':
        Conc=SO2
        Enh=SO2/preSO2
        tEnh=SO2tEnhance
        maxenh=maxenhs[0]
        maxconc=maxconcs[0]
        cbartitle=cbartitles[0]
        enhtitle=enhtitles[0]
        PLG=SO2PLG
        #prePLG=preSO2PLG

    else:
        Conc = Aer
        Enh = Aer/preAer
        tEnh = AertEnhance
        maxenh = maxenhs[1]
        maxconc = maxconcs[1]
        cbartitle = cbartitles[1]
        enhtitle = enhtitles[1]
        PLG=AerPLG
        #prePLG=preAerPLG


    ax = axs[imonth, 0]
    cs= mapConc(ax,Conc,xmin,ymin,resolution,cmap,sf,minval=0,maxval=maxconc)
    plt.text(-0.2,0.5, strmonths[imonth],rotation='vertical',verticalalignment='center',transform=ax.transAxes)

    if imonth==0:
        ax.set_title(cbartitle, rotation='horizontal', pad=0.03)
        #ax.plot([bgmaxlon,bgmaxlon,bgminlon,bgminlon,bgmaxlon],[bgmaxlat,bgminlat,bgminlat,bgmaxlat,bgmaxlat], linewidth=0.5, color='blue')
    if imonth == nmonth - 1:
        cax = plt.axes([0.35, 0.08, 0.14, 0.01])
        cbar = plt.colorbar(cs,  orientation='horizontal', cax=cax, shrink=0.1,ticks=[0,maxconc/2.,maxconc])
        if (plotVal=='Aer') & (DoBeta==False):
            cbar.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        cbar.ax.tick_params(length=2,width=0.3,pad=0.03)

    ax.plot(x2d[PLG[:, 0], PLG[:, 1]], y2d[PLG[:, 0], PLG[:, 1]], linewidth=0.5, color='black',
            linestyle='--')

    ax.tick_params(axis='both', which='major', pad=1)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticks)
    if imonth==nmonth-1:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

    ax = axs[imonth,1]
    cs = mapConc(ax, Enh, xmin, ymin, resolution, dmap,sf, minval=1./maxenh, maxval=maxenh,logscale=True)
    plt.text(0.01, 0.9, '{:10.1f}'.format(tEnh), transform=ax.transAxes)
    if imonth==0:
        ax.set_title(enhtitle, rotation='horizontal', pad=0.03)
    if imonth == nmonth - 1:
        cax = plt.axes([0.51, 0.08, 0.14, 0.01])
        cbar = plt.colorbar(cs,  orientation='horizontal', cax=cax, shrink=0.1,ticks=[1./maxenh,1,maxenh])  #
        cbar.ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:.1f}"))
        cbar.ax.tick_params(length=2,width=0.3,pad=0.03)
    # ax.plot(x2d[prePLG[:, 0], prePLG[:, 1]], y2d[prePLG[:, 0], prePLG[:, 1]], linewidth=0.8, color='black',
    #         linestyle='--')



    if imonth==nmonth-1:
        ax.tick_params(axis='both', which='major', pad=1)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks)

plt.minorticks_off()

plt.subplots_adjust(left=0.343,right=0.657,wspace=0.015, hspace=0.005)

plt.savefig(outfile,dpi=1200)