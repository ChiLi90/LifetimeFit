
import numpy as np
from Satellites import readMonthlyfile
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from scipy.optimize import curve_fit
from scipy.ndimage import median_filter, measurements
import matplotlib.path as mpltPath
from alphashape import alphashape
from scipy import stats


def readfitfile(file,fluxtype,Var,fitFunc):

    if fluxtype=='latlon':
        SO2met = 'EMG M1'
        AODmet=fitFunc+' M1'
    else:
        SO2met = 'EMG M2'
        AODmet=fitFunc+' M2'

    inSO2=False
    inAOD=False
    f = open(file, "r")
    while True:
        # read line
        line = f.readline()

        if 'SO2' in line:
            inSO2=True

        if 'AOD' in line:
            inAOD=True
            inSO2=False

        if (SO2met in line) & (inSO2==True):
            parline=line
            varline=f.readline()
            stdline=f.readline()
            pars = np.array([str.strip(aa) for aa in parline.split(',')][1:])
            vars = np.array(varline.split(',')[1:]).astype(float)
            stds = np.array(stdline.split(',')[1:]).astype(float)
            if Var=='SO2':
                break
            else:
                stdtg = stds[pars == 'x0']

        if (AODmet in line) & (inAOD==True):
            parline = line
            varline = f.readline()
            stdline = f.readline()
            pars = np.array([str.strip(aa) for aa in parline.split(',')][1:])
            vars = np.array(varline.split(',')[1:]).astype(float)
            stds = np.array(stdline.split(',')[1:]).astype(float)
            stds[pars=='xc']=stdtg
            break

        if not line:
            break

    f.close()

    return [pars[:-1],vars[:-1],stds[:-1]]






def mapConc(ax, conc, xmin, ymin, samplewd, cmap, sf, minval, maxval, **kwargs):
    ny, nx = conc.shape
    outx = (xmin - 0.5 + np.arange(nx + 1)) * samplewd[1]
    outy = (ymin - 0.5 + np.arange(ny + 1)) * samplewd[0]

    ax.set_xlim((xmin - 0.5) * samplewd[1], (xmin + nx - 0.5) * samplewd[1], auto=False)
    ax.set_ylim((ymin - 0.5) * samplewd[0], (ymin + ny - 0.5) * samplewd[0], auto=False)
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    logscale=False

    if 'logscale' in kwargs:
        logscale=kwargs['logscale']
    if logscale==False:
        cs = ax.pcolormesh(outx, outy, conc, vmin=minval, vmax=maxval, cmap=cmap)
    else:
        cs = ax.pcolormesh(outx, outy, conc, vmin=minval, vmax=maxval, cmap=cmap,norm=colors.LogNorm())
    ax.tick_params(size=0)
    ax.set_aspect(1.25)

    for shape in sf.shapes():
        points=np.array(shape.points)
        points[:,0]=points[:,0]+360.

        npt,nd=points.shape
        xpoint=points[:,0]
        ypoint=points[:,1]
        pind=0
        while (pind<npt-1):

            xstart = xpoint[0]
            ystart = ypoint[0]

            absdis=(xpoint-xstart)**2+(ypoint-ystart)**2

            repind=np.array((absdis<=0).nonzero()).flatten()
            pind = pind + repind[1] + 1
            if repind.size<2:

                xpoint = xpoint[repind[1] + 1:]
                ypoint = ypoint[repind[1] + 1:]
                continue
            else:
                partpts = np.zeros([repind[1] + 1, 2])
                partpts[:, 0] = xpoint[0:repind[1] + 1]
                partpts[:, 1] = ypoint[0:repind[1] + 1]
                ap = plt.Polygon(partpts, fill=False, closed=True, edgecolor="green", linewidth=0.4)
                ax.add_patch(ap)

                xpoint = xpoint[repind[1] + 1:]
                ypoint = ypoint[repind[1] + 1:]




    if 'plotuv' in kwargs:
        u, v = kwargs['plotuv']
        resuv = kwargs['resuv']
        xinds = np.array(resuv * np.arange(nx / resuv)).astype(int)
        yinds = np.array(resuv * np.arange(ny / resuv)).astype(int)

        qv = ax.quiver(outx[xinds] + 0.5 * resuv * samplewd[1], outy[yinds] + 0.5 * resuv * samplewd[0],
                       u[yinds, :][:, xinds], \
                       v[yinds, :][:, xinds], scale=200)  # plotu, plotv
        ax.quiverkey(qv, X=0.07, Y=1.05, U=10, label='10 m/s', labelpos='E')
    return cs

def PlumeMask(mask):

    # # 2. Find the Spatially connected part after masking that is closest to the volcano...
    # connectionkernel = np.ones([3, 3], dtype=int)
    # labeled, ncomponents = measurements.label(mask, connectionkernel)
    #
    # unique, counts = np.unique(labeled[(labeled > 0)], return_counts=True)
    #
    # mask[labeled != unique[counts == np.max(counts)]] = 0.

    # 1. Apply the median filter
    mask = median_filter(mask, size=3)
    #mask = gaussian_filter(mask, sigma=3.)

    #mth=np.nanpercentile(mask[mask>0.], 33.3)
    mask[mask > 0.] = 1
    mask[mask <= 0.] = 0

    #2. Find the Spatially connected part after masking that is closest to the volcano...
    connectionkernel = np.ones([3, 3], dtype=int)
    labeled, ncomponents = measurements.label(mask, connectionkernel)

    unique, counts = np.unique(labeled[(labeled>0)], return_counts=True)

    mask[labeled!=unique[counts==np.max(counts)]]=0.

    # #3. Concave hull of the mask
    Innerinds = np.transpose((mask>0.).nonzero())
    Outinds = np.transpose(np.array((mask<=0.).nonzero()))

    PLG=alphashape(Innerinds,0.01)

    try:
        Hull=PLG.exterior.coords
    except:
        Hull=PLG[0].exterior.coords

    path = mpltPath.Path(Hull)
    inside = path.contains_points(Outinds)
    ovinds = Outinds[inside == True].transpose()
    mask[ovinds[0], ovinds[1]] = 1.

    return [np.array(list(Hull)).astype(int),mask]

def Gauss(x, a, x0, sigma):
    return a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def expo(x,a,b,c):
    return a * (x**b)+c

def fitexpo(x,data):

    popt, pcov = curve_fit(expo, x, data, p0=[np.mean(data),1,np.min(data)],maxfev=10000)
    #ysigma[idx] = popt[2]
    return [popt,pcov]

def CalcErr(Vals, stds, Flux, FluxErr, Tj, TjErr,fitFunc,**kwargs):

    outErrs=np.zeros(len(Vals))

    npar=len(Flux)
    tvalsq=(stats.t(df=(npar-4)).ppf(0.975))**2/npar
    relmeasErrsq=np.sum(FluxErr**2)/((np.sum(Flux))**2)+np.sum(TjErr**2)/((np.sum(Tj))**2)

    if fitFunc=='EMA':
        taug=kwargs['taug']
        x0val=taug[0]
        x0std=taug[1]
        SO2Flux=taug[2]
        SO2FluxErr=taug[3]
        SO2Tj = taug[4]
        SO2TjErr = taug[5]
        nSO2par = len(SO2Flux)
        tvalsqg = (stats.t(df=(nSO2par - 4)).ppf(0.975)) ** 2 / nSO2par

        reltaugErrsq = tvalsqg * (np.sum(SO2FluxErr**2)/((np.sum(SO2Flux))**2)+np.sum(SO2TjErr**2)/((np.sum(SO2Tj))**2) + (x0std / x0val) ** 2) + 0.01 + 0.01

        for ivar in np.arange(len(Vals)):
            outErrs[ivar]=np.absolute(Vals[ivar])*np.sqrt(tvalsq*(relmeasErrsq+(stds[ivar]/Vals[ivar])**2)+0.01+0.01+0.04+reltaugErrsq)

    else:

        for ival in np.arange(len(Vals)):
            outErrs[ival]=np.absolute(Vals[ival])*np.sqrt(tvalsq*(relmeasErrsq+(stds[ival]/Vals[ival])**2)+0.01+0.01)

    return outErrs


#Direct integrate flux from gridflux on lat-t coordinates:
#Calculate average rather than integrated flux to account for possible lack of grids on each "column"
def IntegrateFlux(gridflx,gridflxErr,T2dErr,ts,dim):
    exdim=1-dim
    xflux = np.nansum(gridflx, axis=exdim)

    xfluxErr = np.sqrt(np.nansum(gridflxErr ** 2, axis=exdim))
    TjErr = np.sqrt(np.nansum(T2dErr**2,axis=exdim))
    return [xflux/1.e6*3600., xfluxErr/1.e6*3600., ts, TjErr]

#Calculate flux from arrays of concentration, velocity and time, on lat-lon coordinates...
def CalcFlux(gridflx,gridflxErr,T2d,T2dErr,Conc,ConcErr,dim):
    #u in m/s, dx, dy in meters
    #Conc in kg/m2...(unitless for AOD) then flux in kg/s
    #dim is the dimension of flux direction...
    exdim=1-dim
    if dim==1:
        nex,nmj=gridflx.shape
    else:
        nmj,nex=gridflx.shape

    xflux = np.nansum(gridflx, axis=exdim)
    xfluxErr = np.sqrt(np.nansum(gridflxErr ** 2, axis=exdim))



    #Time weighted by gridflux and averaged latitudally
    Weights=Conc
    WeightsErr=ConcErr
    Cj = np.nansum(Weights, axis=exdim)
    Cj2d = np.stack([Cj]*nex,axis=exdim)

    TCj = np.nansum(T2d*Weights, axis=exdim)
    TCj2d = np.stack([TCj] * nex, axis=exdim)

    Tj=TCj/Cj

    TjErr=np.sqrt(np.nansum((Weights/Cj2d*T2dErr)**2,axis=exdim)+np.nansum((T2d*Cj2d-TCj2d)/((Cj2d)**2)*(WeightsErr**2),axis=exdim))

    return [xflux/1.e6*3600., xfluxErr/1.e6*3600., Tj, TjErr]



def fitGaussian(x,data):

    mean = np.sum(x*data) / np.sum(data)
    sigma = np.sqrt(np.sum(data * (x - mean) ** 2) / np.sum(data))
    popt, pcov = curve_fit(Gauss, x, data, p0=[np.max(data), mean, sigma],maxfev=10000)
    #ysigma[idx] = popt[2]

    return [popt,pcov]

def ApplyQF(Sample,mincount,Conc,ConcErr,gridflx,gridflxErr,T2d,T2dErr):
    ConcInvinds = ((Sample < mincount) ).nonzero()  #| (np.absolute(gridflxErr / gridflx) > 1.) | (np.absolute(T2dErr / T2d) > 1.)
    Sample[ConcInvinds] = 0
    gridflx[ConcInvinds] = np.nan
    gridflxErr[ConcInvinds] = np.nan
    Conc[ConcInvinds] = np.nan
    ConcErr[ConcInvinds] = np.nan

    return [gridflx, gridflxErr, Conc, ConcErr, Sample]


def ProcessMonthly(year,month,folder,DoBeta,Fluxtype,**kwargs):

    DoBgMasking = True
    DoBgMinus = True
    MapOnly=False
    DoQF=True
    DoPLG=True
    DoDiff=False

    if 'DoPLG' in kwargs:
        DoPLG=kwargs['DoPLG']

    if 'MapOnly' in kwargs:
        MapOnly=kwargs['MapOnly']

    if 'DoQF' in kwargs:
        DoQF=kwargs['DoQF']

    if 'DoDiff' in kwargs:
        DoDiff = kwargs['DoDiff']

    if Fluxtype=='latlon':
        xc = 204.7
        yc = 19.4
    else:
        xc=0.
        yc=19.4

    NAv = 6.02214e23
    MSO2 = 6.4066e-2  # kg/mol
    stryear = '{:10.0f}'.format(year).strip()
    firstfile = 0
    mincounts = [5, 15]
    VarNames = ["SO2", "AOD"]
    bgpercent=50.
    SO2coeff = 2.6867 * 1.e20 / NAv * MSO2  # from DU to kg/m2
    strdfyears = ['2005', '2006', '2007']
    cmap = cm.get_cmap('gist_earth_r')  #
    cmap.set_bad('white')
    strmonth = ("{:10.0f}".format(month + 100).strip())[1:3]

    for ivar in np.arange(len(VarNames)):

        mincount = mincounts[ivar]
        VarName = VarNames[ivar]
        file = folder + stryear + strmonth + '.' + VarName + '.nc'

        # lat,lon: spatial coordinates
        # ys, ts: regridded coordinates for tracking the plume
        # Conc, u, v: concentration and wind information at [lat,lon] space
        # TConc: concentration*Area at [lat,time] space
        # Tgridflx: longitudal flux (TConc/s) at [lat,time] space
        # T2d: estimated plume age (hours) at [lat,lon] space
        # Ts2dErr: estimated Error of age at [lat,time] space
        coordvars = np.array(['latitude', 'longitude', 'AdvectionTime'])
        if (DoBeta == True) & (ivar == 1):
            if Fluxtype=='latlon':
                Datavars = np.array([VarName + '_mass', VarName + 'unc_mass', VarName + '_mass_flux', VarName + 'unc_mass_flux'])
            else:
                Datavars = np.array([VarName + '_mass_lt', VarName + 'unc_mass_lt',VarName + '_mass_flux_lt',VarName + 'unc_mass_flux_lt'])

        else:
            if Fluxtype == 'latlon':
                Datavars = np.array([VarName, VarName + 'unc',  VarName + '_flux', VarName + 'unc_flux'])
            else:
                Datavars = np.array([VarName + '_lt', VarName + 'unc_lt', VarName + '_flux_lt', VarName + 'unc_flux_lt'])

        if Fluxtype=='latlon':
            attvars = np.array(['Sample', 'u', 'v', 'uunc', 'vunc', 'ws', 'wunc', 'T2d','T2dunc'])
        else:
            attvars = np.array(['LTSample', 'u', 'v', 'uunc', 'vunc', 'ws_lt', 'wsunc_lt', 'T2d','time_error'])

        curdata = readMonthlyfile(file, np.concatenate((coordvars, Datavars, attvars)))

        lat, lon, ts, Conc, ConcErr, gridflx, gridflxErr, Sample, u, v, uErr, vErr, ws, wsErr, T2d, T2dErr = curdata

        if firstfile == 0:
            # coordinate system
            if Fluxtype=='latlon':
                ys=lat
                xs=lon
            else:
                xs = ts
                ys = lat
            nx = len(xs)
            ny = len(ys)
            firstfile = 1

        if Fluxtype=='latt':
            T2d=np.stack([xs] * ny, axis=0)

        if DoQF == True:
            [gridflx, gridflxErr, Conc, ConcErr, Sample] = ApplyQF(Sample, mincount, Conc, ConcErr, gridflx, gridflxErr,
                                                                   T2d, T2dErr)

        if DoPLG:
            # background values and mask initialization...
            if (DoBeta == True) & (ivar == 1):
                if Fluxtype == 'latlon':
                    DiffVars = np.array([VarName + '_mass',VarName+'unc_mass',VarName+'_mass_flux',VarName+'unc_mass_flux'])
                else:
                    DiffVars = np.array([VarName + '_mass_lt',VarName+'unc_mass_lt',VarName+'_mass_flux_lt',VarName+'unc_mass_flux_lt'])
            else:
                if Fluxtype == 'latlon':
                    DiffVars = np.array([VarName, VarName + 'unc',VarName+'_flux',VarName+'unc_flux'])
                else:
                    DiffVars = np.array([VarName + '_lt', VarName + 'unc_lt',VarName+'_flux_lt',VarName+'unc_flux_lt'])

            bgthres = np.percentile(Conc[np.isnan(Conc) == False], bgpercent)
            # Determine background and plume based on difference vs. 2005-2007 averages
            if DoBgMinus == True:
                mask = np.zeros(Conc.shape) + 1
                mask[np.isnan(Conc)] = 0.
                mask[Conc < bgthres] = 0.

                preConc=np.zeros([ny,nx])
                preConcunc=np.zeros([ny,nx])
                preflx=np.zeros([ny,nx])
                preflxunc=np.zeros([ny,nx])
                preSample = np.zeros([ny, nx],dtype=int)
                for iyr in np.arange(len(strdfyears)):
                    yrfile = folder + strdfyears[iyr] + strmonth + '.' + VarName + '.nc'
                    yrdata = readMonthlyfile(yrfile, DiffVars)
                    yrConc = yrdata[0]

                    yrConcunc = yrdata[1]
                    yrflx=yrdata[2]
                    yrflxunc=yrdata[3]

                    #if DoDiff:
                    vinds=(np.isnan(yrConc)==False).nonzero()
                    preConc[vinds]=preConc[vinds]+yrConc[vinds]
                    preflx[vinds]=preflx[vinds]+yrflx[vinds]
                    preSample[vinds]=preSample[vinds]+1
                    preConcunc[vinds]=preConcunc[vinds]+yrConcunc[vinds]**2
                    preflxunc[vinds]=preflxunc[vinds]+yrflxunc[vinds]**2
                mask[Conc <= 1.25 * preConc/preSample] = 0
                if DoDiff:
                    Conc=Conc-preConc/preSample
                    ConcErr=ConcErr+np.sqrt(preConcunc/preSample)
                    gridflx=gridflx-preflx/preSample
                    gridflxErr=gridflxErr+np.sqrt(preflxunc/preSample)



            else:
                mask = np.zeros([ny, nx])
                mask[Conc > bgthres] = 1
            Sample[Sample < mincount] = 0
            mask[np.isnan(Conc)] = 0
            if DoBgMasking:
                [PLG, mask] = PlumeMask(mask)
                if (np.all(mask <= 0)):
                    continue

        if VarName == "SO2":
            Conc = Conc * SO2coeff
            ConcErr = ConcErr * SO2coeff
            gridflx = gridflx * SO2coeff
            gridflxErr = gridflxErr * SO2coeff
        if MapOnly == True:
            if ivar == 0:
                SO2 = Conc
                SO2Sample = Sample
                if DoPLG:
                    SO2mask = mask.copy()
                    SO2PLG = PLG.copy()
            else:
                Aer = Conc
                AerSample = Sample
                if DoPLG:
                    Aermask = mask.copy()
                    AerPLG = PLG.copy()
            continue


        Conc[mask <= 0.] = np.nan
        ConcErr[mask <= 0.] = np.nan
        Sample[mask <= 0.] = 0
        gridflx[mask <= 0.] = np.nan
        gridflxErr[mask <= 0.] = np.nan
        acCount = np.count_nonzero(mask, axis=0)
        acnonNan = np.isnan(gridflx) == False
        # fraction of useful grids in each longitude or time "column"
        acPercent = np.count_nonzero(acnonNan, axis=0) / acCount


        # method1: AOD weighted time for each longitude band (normalized to the same y intervals)
        if Fluxtype=='latlon':
            [xflux, xfluxErr, Tj, TjErr] = CalcFlux(gridflx, gridflxErr, T2d, T2dErr, Conc, ConcErr, 1)
            xflux = -1 * xflux
        else:
            [xflux, xfluxErr, Tj, TjErr] = IntegrateFlux(gridflx, gridflxErr, T2dErr, xs, 1)
            xflux = -1 * xflux

        if ((ivar == 1) & (DoBeta == False)):
            xflux = xflux / 3600.
            xfluxErr = xfluxErr / 3600.

        if Fluxtype=='latlon':
            Tjdiff=np.zeros(len(Tj))
            for itj in np.arange(len(Tj)-1):
                Tjdiff[itj]=Tj[itj]-np.max(Tj[itj+1:])
            vinds = ((np.absolute(Tj) >= 0.) & (xflux >= 0.) & (Tjdiff>0.) & (acPercent > 0.8)).nonzero()  # & (np.absolute(xfluxErr / xflux) < 1.) & (np.absolute(TjErr / Tj) < 0.5)
        else:
            vinds = ((np.absolute(Tj) >= 0.) & (xflux >= 0.) & (acPercent > 0.8)).nonzero()  # & (np.absolute(xfluxErr / xflux) < 1.) & (np.absolute(TjErr / Tj) < 0.5)
        if ivar==0:
            SO2=Conc.copy()
            SO2Sample=Sample.copy()
            SO2Flux=xflux[vinds]
            SO2FluxErr=xfluxErr[vinds]
            SO2Tj=Tj[vinds]
            SO2TjErr=TjErr[vinds]
            SO2mask=mask.copy()
            SO2PLG=PLG.copy()
        else:
            Aer=Conc.copy()
            AerSample=Sample.copy()
            AerFlux=xflux[vinds]
            AerFluxErr = xfluxErr[vinds]
            AerTj = Tj[vinds]
            AerTjErr = TjErr[vinds]
            Aermask=mask.copy()
            AerPLG=PLG.copy()

    if (MapOnly==True):
        if (DoPLG==False):
            return [xs,ys,SO2,SO2Sample,Aer,AerSample]
        else:
            return [xs,ys,SO2,SO2Sample,SO2mask,SO2PLG,Aer,AerSample,Aermask,AerPLG]
    return [xs,ys,SO2,SO2Sample,SO2Flux,SO2FluxErr,SO2Tj,SO2TjErr,SO2mask,SO2PLG,Aer,AerSample,AerFlux,AerFluxErr,AerTj,AerTjErr,Aermask,AerPLG,u,v]










