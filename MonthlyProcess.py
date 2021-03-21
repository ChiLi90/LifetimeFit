from datetime import datetime
import numpy as np
from SpatialFunctions import getConcwind,GetCaliopFraction
from Satellites import ReadMOD04, ReadOMISO2, ReadMOD04F
import glob
import matplotlib.cm as cm
from netCDF4 import Dataset
import argparse




PBLSO2=False
parser=argparse.ArgumentParser()
parser.add_argument('--month',type=int)
parser.add_argument('--year',type=int)
parser.add_argument('--Var')
parser.add_argument('--level')
parser.add_argument('--DoBeta')
parser.add_argument('--DoCaliop')  #take the calipso levels and fraction

args=parser.parse_args()
month=args.month
VarName=args.Var
year=args.year
level=args.level
strDoBeta=args.DoBeta
strDoCaliop=args.DoCaliop

UVdir='/clusterfs/aiolos//chili/ERA5UV/'
CPdir='/clusterfs/aiolos//chili/ERA5/'
RHdir='/clusterfs/aiolos//chili/ERA5RH/'
Caliopdir='/clusterfs/aiolos//chili/CALIPSO/'
Satdir='/clusterfs/aiolos/chili/'
outtop='/global/home/users/chili/AvgOMISO2/TA/UV/'

res=[0.25,0.25]
nx=281
ny=161
rest=1.
nt=401  #-50 to 350

#lat [-5,35] lon[140E,150W]
xs=np.arange(nx)*res[1]+140.
ys=np.arange(ny)*res[0]-5.
ts=np.arange(nt)*rest-50.

xmin=np.int(np.min(xs)/res[1])
ymin=np.int(np.min(ys)/res[0])



tupperprs = np.flip([0.,725, 762.5, 787.5, 812.5, 837.5,862.5,887.5,912.5,937.5,962.5,987.5])
tlowerprs = np.flip([725.,762.5, 787.5, 812.5, 837.5, 862.5,887.5,912.5,937.5,962.5,987.5,1200.])

#pressure range from Beirle et al. 2014
if level=='Upper':
    outpath=outtop+'Upper/'
    pressures=[880,745]
    stlevel= 5
    edlevel= 11

if level=='All1':
    outpath=outtop+'All1/'
    pressures=[1200,745]
    stlevel = 0
    edlevel = 11

if level == 'All':
    outpath=outtop
    pressures=[1200,695]
    stlevel=0
    edlevel=12

if level == 'Surf':
    outpath=outtop+'Surf/'
    pressures=[1200,920]
    stlevel = 0
    edlevel = 4

if level == 'Upper4':
    outpath=outtop+'Upper4/'
    pressures=[905,745]
    stlevel=4
    edlevel=11

if level == 'Upper6':
    outpath=outtop+'Upper6/'
    pressures=[855,745]
    stlevel=6
    edlevel=11

if level == 'Upper7':
    outpath=outtop+'Upper7/'
    pressures=[830,745]
    stlevel=7
    edlevel=11

upperprs=tlowerprs[stlevel:edlevel]
lowerprs=tupperprs[stlevel:edlevel]

LonFormat="PO"


uOnly=True
outdir=outpath



if (strDoBeta=="True"):
    DoBeta=True
else:
    DoBeta=False

if (strDoCaliop=='True'):
    DoCaliop=True
else:
    DoCaliop=False

#Caliopdir="/clusterfs/aiolos//chili/CALIPSO/opendap.larc.nasa.gov/opendap/hyrax/CALIPSO/LID_L3_Tropospheric_APro_CloudFree-Standard-V4-20/"
AerFrac=0.63212
xc=204.7
yc=19.4

cmap = cm.rainbow
cmap.set_bad('white')
stryear='{:10.0f}'.format(year).strip()


for VarName in [VarName]:
    if VarName == "AOD":
        IOFunc = ReadMOD04
        Sensor = 'TA'
        if Sensor == 'TERRA':
            Datadir = Satdir+'MOD04_L2/'
        if Sensor == 'AQUA':
            Datadir = Satdir+'MYD04_L2/'
        if Sensor == 'TA':
            Datadir = Satdir+'M?D04_L2/'
        concvar = 'AOD'
        uncvar = 'AODunc'
        concunit='unitless'

    if VarName == 'FAOD':
        IOFunc = ReadMOD04F
        Sensor = 'TA'
        if Sensor == 'TERRA':
            Datadir = Satdir+'MOD04_L2/'
        if Sensor == 'AQUA':
            Datadir = Satdir+'MYD04_L2/'
        if Sensor == 'TA':
            Datadir = Satdir+'M?D04_L2/'
        concvar = 'FAOD'
        uncvar = 'FAODunc'
        concunit = 'unitless'

    if VarName == 'SO2':
        IOFunc = ReadOMISO2
        Datadir = Satdir+'OMISO2/'
        concvar = 'SO2'
        uncvar = 'SO2unc'
        concunit = 'DU'

    for month in [month]:
        # determine pressure range
        strmonth = ('{:10.0f}'.format(month + 100).strip())[1:3]
        # try:
        #     [lat, lon, outpmin, outpmax, outpmed] = CalPressRange(CalDir, year, month, ys, xs, Fraction=AerFrac)
        # except:
        #     [lat, lon, outpmin, outpmax, outpmed] = CalPressRange(CalDir, year, month-1, ys, xs, Fraction=AerFrac)
        #
        # pressures=[np.percentile(outpmin[outpmin>0.],5.),np.percentile(outpmax[outpmax>0.],95.)]
        # print(month,pressures)
        AODfrs=np.zeros([ny,nx,len(upperprs)])
        #Calculate AOD fraction for ERA5 each layer & each ERA5 gridbox for this month
        if DoCaliop==True:
            #monthly AODfrs in [ny,nx,nl] for the ERA5 horizontal and vertical grid box
            AODfrs=GetCaliopFraction(Caliopdir,year,month,ys,xs,upperpres=tupperprs,lowerpres=tlowerprs,stlevel=stlevel,edlevel=edlevel,outCaliop=outdir+stryear+strmonth+'.CAL.nc',scalex=3,scaley=5,Interp=True)

        totalConc = np.zeros([ny, nx])
        totalConcErr = np.zeros([ny, nx])
        totalgflx = np.zeros([ny, nx])
        totalgflxErr = np.zeros([ny, nx])
        totalSample = np.zeros([ny, nx], dtype=int)
        totalu = np.zeros([ny, nx])
        totaluErr = np.zeros([ny, nx])
        totalv = np.zeros([ny, nx])
        totalvErr = np.zeros([ny, nx])
        totalws = np.zeros([ny, nx])
        totalwsErr = np.zeros([ny, nx])
        totalT2d = np.zeros([ny, nx])
        totalT2dErr = np.zeros([ny, nx])
        totalRH =  np.zeros([ny,nx])
        totalRHErr = np.zeros([ny, nx])
        totalTCC = np.zeros([ny, nx])
        totalTP = np.zeros([ny, nx])

        if DoBeta==True:
            totalConcms = np.zeros([ny, nx])
            totalConcmsErr = np.zeros([ny, nx])
            totalgflxms = np.zeros([ny, nx])
            totalgflxmsErr = np.zeros([ny, nx])
            totalTConcms = np.zeros([ny, nt])
            totalTConcmsErr = np.zeros([ny, nt])
            totalgrdflxms = np.zeros([ny, nt])
            totalgrdflxmsErr = np.zeros([ny, nt])

        totalTConc = np.zeros([ny, nt])
        totalTConcErr = np.zeros([ny, nt])
        totalgrdflx =  np.zeros([ny, nt])
        totalgrdflxErr =  np.zeros([ny, nt])
        totalTSample=  np.zeros([ny, nt], dtype=int)
        totalTErr =  np.zeros([ny, nt])
        totalTws = np.zeros([ny,nt])
        totalTwsErr = np.zeros([ny, nt])

        # Do accumulation
        obno = 0

        for day in np.arange(31) + 1:
            try:
                date = datetime(year, month, day)
            except:
                continue

            strday = ('{:10.0f}'.format(day + 100).strip())[1:3]
            # 1. search the file
            if (VarName == "AOD") | (VarName=="FAOD"):
                doy = (date - datetime(year, 1, 1)).days + 1
                strdoy = ('{:10.0f}'.format(doy + 1000).strip())[1:4]
                MDataDir = Datadir + stryear + '/' + strdoy + '/'
                files = glob.glob(MDataDir + '*.hdf')

            if VarName == "SO2":
                files = glob.glob(Datadir + "OMI-Aura_L2-OMSO2_" + stryear + "m" + strmonth + strday + "*.he5")

            if (len(files) < 1):
                print("wrong number of files for date " + Datadir)
                continue

            # 2. Read data and regridding
            for file in files:

                if (VarName == "AOD") | (VarName=="FAOD"):
                    strtime = file.split('/')[-1].split('.')[2]
                    hour = np.int(np.round(np.int(strtime[0:2]) + (np.float(strtime[2:4]) + 2.5) / 60.))
                else:
                    hour = np.int(np.round(IOFunc(file, TimeOnly=True)))
                if (hour>24) | (hour<0):
                    continue
                # concentration, 1-sigma uncertainty, sample, u, u-unc, v, v-unc, w (surface), w-unc
                # usq and vsq are "totals" of several layers
                #[outAOD, outAODunc, outSample, outu, outusq, outv, outvsq, outws, nLayer] = \
                #if uOnly == True, only use the u component to calculate plume age...
                if DoBeta==False:
                    [rgrdConc, rgrdConcErr,rgrdflux,rgrdfluxErr, rgSample, u, uErr, v, vErr, ws, wsErr,RH, RHErr, tcc, \
                     tp,T2d,T2dErr,nLayer, TConc, TConcErr, Tgridflx, TgridflxErr,TjErr,TSample,Tws,TwsErr]\
                        =getConcwind(file,UVdir,RHdir,CPdir, xs, ys,ts, xc,yc, res, date, hour, IOFunc,\
                                     pressures=pressures, LonFormat=LonFormat,uOnly=uOnly,DoBeta=DoBeta,PBLSO2=PBLSO2)
                else:
                    [rgrdConc, rgrdConcErr, rgrdflux, rgrdfluxErr, rgSample, u, uErr, v, vErr, ws, wsErr, RH, RHErr,\
                     tcc, tp, T2d, T2dErr, nLayer, TConc, TConcErr, Tgridflx, TgridflxErr, TjErr, TSample, Tws, TwsErr, \
                     rgrdConcmass, rgrdConcmassErr,TConcmass, TConcmassErr, gridflxmass, gridflxmassErr, Tgridflxmass,\
                     TgridflxmassErr] = getConcwind(file, UVdir, RHdir, CPdir, xs, ys, ts, xc, yc, res, date, hour, \
                                                    IOFunc, pressures=pressures,LonFormat=LonFormat, uOnly=uOnly, \
                                                    DoBeta=DoBeta,DoCaliop=[DoCaliop,AODfrs],PBLSO2=PBLSO2)

                if np.array(rgrdConc).size <= 1:
                    continue
                inds = (rgSample > 0).nonzero()
                if np.array(inds).size < 1:
                    continue

                totalnLayer = nLayer
                # 3. accumulation
                totalConc[inds] = totalConc[inds] + rgrdConc[inds] * rgSample[inds]
                totalConcErr[inds] = totalConcErr[inds] + rgrdConcErr[inds] * rgSample[inds]
                totalgflx[inds]=totalgflx[inds]+rgrdflux[inds]*rgSample[inds]
                totalgflxErr[inds] = totalgflxErr[inds] + rgrdfluxErr[inds] * rgSample[inds]
                totalSample[inds] = totalSample[inds] + rgSample[inds]

                totalu[inds] = totalu[inds] + u[inds] * rgSample[inds]
                totalv[inds] = totalv[inds] + v[inds] * rgSample[inds]
                totalTCC[inds] = totalTCC[inds] + tcc[inds] * rgSample[inds]
                totalTP[inds] = totalTP[inds] + tp[inds] * rgSample[inds]
                totalws[inds] = totalws[inds] + ws[inds] * rgSample[inds]
                totalRH[inds]=totalRH[inds]+RH[inds]*rgSample[inds]
                totalRHErr[inds] = totalRHErr[inds] + RHErr[inds] * rgSample[inds]
                totaluErr[inds] = totaluErr[inds] + uErr[inds] * rgSample[inds]
                totalvErr[inds] = totalvErr[inds] + vErr[inds] * rgSample[inds]
                totalwsErr[inds] = totalwsErr[inds] + wsErr[inds] * rgSample[inds]
                totalT2d[inds] = totalT2d[inds] + T2d[inds] * rgSample[inds]
                totalT2dErr[inds] = totalT2dErr[inds] + T2dErr[inds] * rgSample[inds]

                if DoBeta == True:

                    totalConcms[inds] = totalConcms[inds] + rgrdConcmass[inds] * rgSample[inds]
                    totalConcmsErr[inds] = totalConcmsErr[inds] + rgrdConcmassErr[inds] * rgSample[inds]
                    totalgflxms[inds] = totalgflxms[inds] + gridflxmass[inds] * rgSample[inds]
                    totalgflxmsErr[inds] = totalgflxmsErr[inds] + gridflxmassErr[inds] * rgSample[inds]

                #4. accumulation of lat-t system
                inds=(TSample>0.).nonzero()
                if np.array(inds).size < 1:
                    continue

                totalTConc[inds]=totalTConc[inds]+TConc[inds]*TSample[inds]
                totalTConcErr[inds]=totalTConcErr[inds]+TConcErr[inds]*TSample[inds]

                totalgrdflx[inds]=totalgrdflx[inds]+Tgridflx[inds]*TSample[inds]
                totalgrdflxErr[inds] = totalgrdflxErr[inds] + TgridflxErr[inds] * TSample[inds]

                totalTSample[inds]=totalTSample[inds]+TSample[inds]

                totalTErr[inds]=totalTErr[inds]+TjErr[inds]*TSample[inds]

                totalTws[inds] = totalTws[inds] + Tws[inds] * TSample[inds]
                totalTwsErr[inds] = totalTwsErr[inds] + TwsErr[inds] * TSample[inds]

                if DoBeta == True:

                    totalTConcms[inds] = totalTConcms[inds] + TConcmass[inds] * TSample[inds]
                    totalTConcmsErr[inds] = totalTConcmsErr[inds] + TConcmassErr[inds] * TSample[inds]
                    totalgrdflxms[inds] = totalgrdflxms[inds] + Tgridflxmass[inds] * TSample[inds]
                    totalgrdflxmsErr[inds] = totalgrdflxmsErr[inds] + TgridflxmassErr[inds] * TSample[inds]


        # 4. average from total
        meanConc = totalConc / totalSample
        meanConcErr = np.sqrt(totalConcErr / (totalSample ** 2))
        meangflx=totalgflx/totalSample
        meangflxErr=np.sqrt(totalgflxErr/(totalSample**2))
        # stdAOD=CalcStd(totalAOD,totalAODsq,totalSample)

        meanu = totalu / totalSample
        meanv = totalv / totalSample
        meanws = totalws / totalSample
        meanuErr = np.sqrt(totaluErr / (totalSample**2))
        meanvErr =  np.sqrt(totalvErr / (totalSample**2))
        meanwsErr = np.sqrt(totalwsErr / (totalSample**2))

        meantcc=totalTCC/totalSample
        meantp=totalTP/totalSample

        meanRH=totalRH/totalSample
        meanRHErr=np.sqrt(totalRHErr/(totalSample**2))

        meanT2d=totalT2d/totalSample
        meanT2dErr=np.sqrt(totalT2dErr/(totalSample**2))


        #5. avrage / accumulation for lat-t system

        totalTConc = totalTConc/totalTSample
        totalTConcErr = np.sqrt(totalTConcErr/(totalTSample**2))
        totalTws = totalTws / totalTSample
        totalTwsErr = np.sqrt(totalTwsErr / (totalTSample ** 2))
        totalgrdflx = totalgrdflx / totalTSample
        totalgrdflxErr = np.sqrt(totalgrdflxErr / (totalTSample ** 2))
        totalTErr = np.sqrt(totalTErr/ (totalTSample ** 2))


        if DoBeta==True:
            totalConcms=totalConcms/totalSample
            totalgflxms=totalgflxms/totalSample
            totalTConcms=totalTConcms/totalTSample
            totalgrdflxms=totalgrdflxms/totalTSample

            totalConcmsErr = np.sqrt(totalConcmsErr / (totalSample**2))
            totalgflxmsErr = np.sqrt(totalgflxmsErr / (totalSample**2))
            totalTConcmsErr = np.sqrt(totalTConcmsErr / (totalTSample**2))
            totalgrdflxmsErr = np.sqrt(totalgrdflxmsErr / (totalTSample**2))

        outfile = outdir + stryear + strmonth + '.' + VarName + '.nc'
        dso = Dataset(outfile, mode='w', format='NETCDF4')
        dso.createDimension('x', nx)
        dso.createDimension('y', ny)
        dso.createDimension('t', nt)


        outdata = dso.createVariable('latitude', np.float32, 'y')
        outdata.units = 'degree'
        outdata[:] = ys
        outdata = dso.createVariable('longitude', np.float32, 'x')
        outdata.units = 'degree'
        outdata[:] = xs
        outdata = dso.createVariable('AdvectionTime', np.float32, 't')
        outdata.units = 'hour'
        outdata[:] = ts

        outdata = dso.createVariable('Total precipitation', np.float32, ('y', 'x'))
        outdata.units = 'mm'
        outdata[:] = meantp

        outdata = dso.createVariable('Total cloud cover', np.float32, ('y', 'x'))
        outdata.units = 'unitless'
        outdata[:] = meantcc

        outdata = dso.createVariable('Relative humidity', np.float32, ('y', 'x'))
        outdata.units = '%'
        outdata[:] = meanRH

        outdata = dso.createVariable('Relative humidity Unc', np.float32, ('y', 'x'))
        outdata.units = '%'
        outdata[:] = meanRHErr


        outdata = dso.createVariable(concvar, np.float32, ('y', 'x'))
        outdata.units = concunit
        outdata[:] = meanConc
        outdata = dso.createVariable(uncvar, np.float32, ('y', 'x'))
        outdata.units = concunit
        outdata[:] = meanConcErr
        outdata = dso.createVariable('Sample', np.int, ('y', 'x'))
        outdata.units = 'unitless'
        outdata[:] = totalSample
        # outdata = dso.createVariable('AODstd', np.float32, ('y', 'x'))
        # outdata.units = 'unitless'
        # outdata[:] = stdAOD
        outdata = dso.createVariable(concvar + '_flux', np.float32, ('y', 'x'))
        outdata.units = concunit + '*m2/s'
        outdata[:] = meangflx

        outdata = dso.createVariable(uncvar + '_flux', np.float32, ('y', 'x'))
        outdata.units = concunit + '*m2/s'
        outdata[:] = meangflxErr

        outdata = dso.createVariable('u', np.float32, ('y', 'x'))
        outdata.units = 'm/s'
        outdata[:] = meanu
        outdata = dso.createVariable('v', np.float32, ('y', 'x'))
        outdata.units = 'm/s'
        outdata[:] = meanv
        outdata = dso.createVariable('ws', np.float32, ('y', 'x'))
        outdata.units = 'm/s'
        outdata[:] = meanws
        outdata = dso.createVariable('uunc', np.float32, ('y', 'x'))
        outdata.units = 'm/s'
        outdata[:] = meanuErr
        outdata = dso.createVariable('vunc', np.float32, ('y', 'x'))
        outdata.units = 'm/s'
        outdata[:] = meanvErr
        outdata = dso.createVariable('wunc', np.float32, ('y', 'x'))
        outdata.units = 'm/s'
        outdata[:] = meanwsErr

        outdata = dso.createVariable('T2d', np.float32, ('y', 'x'))
        outdata.units = 'hour'
        outdata[:] = meanT2d
        outdata = dso.createVariable('T2dunc', np.float32, ('y', 'x'))
        outdata.units = 'hour'
        outdata[:] = meanT2dErr


        outdata = dso.createVariable(concvar+'_lt', np.float32, ('y', 't'))
        outdata.units = concunit
        outdata[:] = totalTConc
        outdata = dso.createVariable(uncvar+'_lt', np.float32, ('y', 't'))
        outdata.units = concunit
        outdata[:] = totalTConcErr
        outdata = dso.createVariable('ws_lt', np.float32, ('y', 't'))
        outdata.units = 'm/s'
        outdata[:] = totalTws
        outdata = dso.createVariable('wsunc_lt', np.float32, ('y', 't'))
        outdata.units = 'm/s'
        outdata[:] = totalTwsErr
        outdata = dso.createVariable('LTSample', np.int, ('y', 't'))
        outdata.units = 'unitless'
        outdata[:] = totalTSample
        outdata = dso.createVariable(concvar + '_flux_lt', np.float32, ('y', 't'))
        outdata.units = concunit+'*m2/s'
        outdata[:] = totalgrdflx
        outdata = dso.createVariable(uncvar + '_flux_lt', np.float32, ('y', 't'))
        outdata.units = concunit+'*m2/s'
        outdata[:] = totalgrdflxErr
        outdata = dso.createVariable('time_error', np.float32, ('y', 't'))
        outdata.units = 'hour'
        outdata[:] = totalTErr

        if DoBeta==True:
            outdata = dso.createVariable(concvar + '_mass_flux', np.float32, ('y', 'x'))
            outdata.units = 'kg/s'
            outdata[:] = totalgflxms
            outdata = dso.createVariable(uncvar + '_mass_flux', np.float32, ('y', 'x'))
            outdata.units = 'kg/s'
            outdata[:] = totalgflxmsErr
            outdata = dso.createVariable(concvar + '_mass_flux_lt', np.float32, ('y', 't'))
            outdata.units = 'kg/s'
            outdata[:] = totalgrdflxms
            outdata = dso.createVariable(uncvar + '_mass_flux_lt', np.float32, ('y', 't'))
            outdata.units = 'kg/s'
            outdata[:] = totalgrdflxmsErr

            outdata = dso.createVariable(concvar+'_mass', np.float32, ('y', 'x'))
            outdata.units = 'kg/m2'
            outdata[:] = totalConcms

            outdata = dso.createVariable(uncvar + '_mass', np.float32, ('y', 'x'))
            outdata.units = 'kg/m2'
            outdata[:] = totalConcmsErr

            outdata = dso.createVariable(concvar + '_mass_lt', np.float32, ('y', 't'))
            outdata.units = 'kg/m2'
            outdata[:] = totalTConcms

            outdata = dso.createVariable(uncvar + '_mass_lt', np.float32, ('y', 't'))
            outdata.units = 'kg/m2'
            outdata[:] = totalTConcmsErr

        dso.close()



        # outpng = outdir + stryear + strdoy + '.' + Sensor + '.png'
        # fig, axs = plt.subplots(nrows=2, ncols=2)
        # # plot the AOD and wind fields
        # outconc = meanAOD
        # cbartitle = 'AOD'
        # outconc[outconc <= 0] = np.nan
        # ax = axs[0, 0]
        # pos = [0.3 * 0 - 0.1, 0.285, 0.4, 0.77]
        # [maxval, minval] = mapfield(ax, pos, outconc, xmin, ymin, res[0], cbartitle, cmap)
        # # qv = ax.quiver(outx[0:nx] + 0.5 * samplewd, outy[0:ny] + 0.5 * samplewd, outu, outv,
        # #                scale=100)  # plotu, plotv
        # # ax.quiverkey(qv, X=0.2, Y=1.05, U=10, label='10 m/s', labelpos='E')
        #
        # # plot the AOD uncertainty
        # outconc = stdAOD
        # cbartitle = 'AOD std'
        # outconc[outconc <= 0] = np.nan
        # ax = axs[0, 1]
        # pos = [0.3 * 1 - 0.1, 0.285, 0.4, 0.77]
        # [maxval, minval] = mapfield(ax, pos, outconc, xmin, ymin, res[0], cbartitle, cmap)
        #
        # # plot the Smoke Height
        # ax = axs[1, 0]
        # cbartitle = 'u(m/s)'
        # pos = [0.3 * 0 - 0.2, 0.05, 0.4, 0.77]
        # [maxval, minval] = mapfield(ax, pos, meanu, xmin, ymin, res[0], cbartitle, cm.bwr, pn=True)
        #
        # cbartitle = 'stdu (m/s)'
        # ax = axs[1, 1]
        # pos = [0.3 * 1 - 0.1, 0.05, 0.4, 0.77]
        # [maxval, minval] = mapfield(ax, pos, stdu, xmin, ymin, res[0], cbartitle, cm.bwr, pn=True)
        #
        # plt.subplots_adjust(hspace=0.0001)
        # plt.savefig(outpng, dpi=600)
        # plt.close()
        # exit()


