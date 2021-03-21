from Satellites import readMonthlyfile
from Fitpost import ProcessMonthly, mapConc, readfitfile,CalcErr
import numpy as np
import matplotlib.pyplot as plt
import shapefile as shp
import matplotlib.cm as cm
from EMFit import EMG, EMA
import matplotlib.ticker as ticker
from scipy.stats.stats import pearsonr



Pdir='/Users/chili/AvgOMISO2/TA/Coarse2/UV/'
USshape='/Users/chili/AvgOMISO2/cb_2019_us_state_20m/cb_2019_us_state_20m.shp'
level='Upper'
subdir='Mass'
DoDiff=False
year=2008
resolution=[0.25,0.25]
month=8
# plt.rcParams.update({'font.size': 5})
# plt.rcParams['axes.linewidth'] = 0.2
outpng='/Users/chili/AvgOMISO2/TA/Coarse2/AerFitsupUV.png'
DoBeta=True
Fluxtype='latt'
cmap = cm.get_cmap('jet')  #
cmap.set_bad('white')
dmap = cm.get_cmap('bwr')
dmap.set_bad('white')
if level == 'All':
    folder = Pdir +'U/'
else:
    folder = Pdir + level + '/U/'
outdir = folder + subdir + '/'
sf = shp.Reader(USshape)
Heights=np.array([110.88,323.38,540.34,761.97,988.5,1220.19,1457.3,1700.13,1948.99,2204.23,2466.23,3012.18,3450])/1000.

if DoBeta:
    ytitles = ["SO" + r"$_2$" + " flux (Gg/h)", "Aerosol mass flux (Gg/h)"]
else:
    ytitles = ["SO" + r"$_2$" + " flux (Gg/h)", "AOD flux (km" + r"$^2$" + "/s)"]

xtitle='Plume age (hr)'
PlotVar='Aer'
fitFunc='EMA'
nmonth=5
smonth=5
#[xs,ys,SO2,SO2Sample,SO2Flux,SO2FluxErr,SO2Tj,SO2TjErr,SO2mask,SO2PLG,Aer,AerSample,AerFlux,AerFluxErr,AerTj,AerTjErr,Aermask,AerPLG,u,v]=ProcessMonthly(year,month,folder,DoBeta,Fluxtype)
#plot the concentrations and enhancements
#plt.figure(figsize=(7.2,2.4))
fig, axs = plt.subplots(nmonth, figsize=(4.8,7.2))

maxconcs=[50,0.3]
maxenhs=[10,5]
strmonths=['May','June','July','August','September']
colors=['red','orange','green','blue','purple']

if PlotVar=='SO2':
    strPars=['E'+r'$_g$',r'$\tau$' + r'$_g$',r'$\sigma$' + r'$_t$']
    ytitle = ytitles[0]

    tmax=100
else:
    strPars = ['fE' + r'$_g$', r'$\tau$' + r'$_a$', r'$\sigma$' + r'$_t$']
    ytitle = ytitles[1]

    tmax=300

if fitFunc=='EMA':
    parinds = [1, 2, 7]
else:
    parinds = [0, 1, 3]
Parunits = ['Gg/h', 'hr', 'hr']
if DoBeta==False:
    Parunits[0]='km' + r'$^2$' + '/s'

for imonth in np.arange(nmonth):

    month=imonth+smonth

    #[xs, ys, SO2, SO2Sample, Aer, AerSample] = ProcessMonthly(2008, month,folder, DoBeta,Fluxtype,MapOnly=True,DoPLG=False)
    [xs,ys,SO2,SO2Sample,SO2Flux,SO2FluxErr,SO2Tj,SO2TjErr,SO2mask,SO2PLG,Aer,AerSample,AerFlux,AerFluxErr,AerTj,AerTjErr,Aermask,AerPLG,u,v] = ProcessMonthly(2008, month,folder,DoBeta,Fluxtype,DoDiff=DoDiff)
    stryear = '{:10.0f}'.format(year).strip()
    strmonth = ("{:10.0f}".format(month + 100).strip())[1:3]
    fitfile=outdir+stryear+strmonth+'.fit'
    #
    if PlotVar=='SO2':
        Flux=SO2Flux
        Tj=SO2Tj


        [fitPars, fitVals, fitstds] = readfitfile(fitfile, Fluxtype, Var=PlotVar, fitFunc=fitFunc)
        # #95% confidence of each parameter, source, lifetime and sigma
        parErrs = CalcErr(fitVals[parinds], fitstds[parinds], Flux, SO2FluxErr, Tj, SO2TjErr, fitFunc)


    else:
        Flux = AerFlux
        Tj = AerTj
        [fitPars, fitVals, fitstds] = readfitfile(fitfile, Fluxtype,Var=PlotVar,fitFunc=fitFunc)
        parErrs = CalcErr(fitVals[parinds], fitstds[parinds], Flux, AerFluxErr, Tj, AerTjErr,fitFunc,taug=[fitVals[3],fitstds[3],SO2Flux,SO2FluxErr,SO2Tj,SO2TjErr])
    #


    axs[imonth].scatter(Tj,Flux,edgecolors=colors[imonth],marker='o',s=8,facecolors='none',linewidths=0.5)
    axs[imonth].set_xlim([-20, tmax])
    if imonth==2:
        plt.text(-0.13, 0., ytitle, rotation='vertical',transform=axs[imonth].transAxes)
    #fitted line
    if fitFunc == 'EMG':
        fity = EMG(Tj, *fitVals)
    else:
        fity = EMA(Tj, *fitVals)
    axs[imonth].plot(Tj, fity, color=colors[imonth])
    plt.text(0.03, 0.8, strmonths[imonth], color=colors[imonth], transform=axs[imonth].transAxes)

    if imonth<nmonth-1:
        axs[imonth].set_xticklabels([])
    else:
        axs[imonth].set_xlabel(xtitle)

    outtext = ''

    for ipar in np.arange(len(strPars)):
        outtext = outtext + strPars[ipar] + '=' + '{:10.1f}'.format(fitVals[parinds[ipar]]).strip() + r'$\pm$' \
                  + '{:10.1f}'.format(parErrs[ipar]).strip() + ' ' + Parunits[ipar] + '\n'
    rsqr = pearsonr(fity, Flux)[0] ** 2
    outtext = outtext + 'r' + r'$^2$' + '=' + '{:10.2f}'.format(rsqr).strip()
    plt.text(0.96, 0.3, outtext, color=colors[imonth], transform=axs[imonth].transAxes, horizontalalignment='right')

    print(month,fitVals[parinds[1]],parErrs[1])


plt.savefig(outpng,dpi=1200)