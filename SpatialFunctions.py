
import matplotlib
matplotlib.use('Agg')
import numpy as np
from scipy import ndimage as nd
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from scipy.special import erfc
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import griddata
from scipy.stats import binned_statistic
from Satellites import GetUV, GetRH, GetCP,CalAODFr
from datetime import datetime, timedelta
import matplotlib.path as mpltPath
from netCDF4 import Dataset

def CalcStd(total,totalsq,sample):
    mean=total/sample
    return np.sqrt((totalsq-sample*(mean**2))/(sample-1))

def GetCaliopFraction(Caliopdir,year,month,ys,xs,upperpres,lowerpres,stlevel,edlevel,**kwargs):

    ny=len(ys)
    nx=len(xs)
    nl=len(upperpres)
    AODfrs=np.zeros([ny,nx,nl])

    dy=np.absolute(ys[1]-ys[0])
    dx = np.absolute(xs[1] - xs[0])

    scalex=1
    scaley=1

    if 'scalex' in kwargs:
        scalex=kwargs['scalex']
        scaley=kwargs['scaley']

    Interp=False

    if 'Interp' in kwargs:
        Interp=kwargs['Interp']

    #use 2007 to represent 2005-2007
    if (year<2007):
        inyr=2007
    else:
        inyr=year

    for il in np.arange(nl):

        #average of Daytime and Cloud-Free observations

        lat,lon,AODfr,AOD =CalAODFr(Caliopdir, inyr, month, ys, xs, PressRange=[lowerpres[il], upperpres[il]])
        # lat, lon, AODfrN, NAOD = CalAODFr(Caliopdir, inyr, month, ys, xs, PressRange=[lowerpres[il], upperpres[il]],DayNight='N')
        #
        # AODfrD[(AODfrD<=0)|np.isnan(AODfrD)]=0.
        # AODfrN[(AODfrN <= 0) | np.isnan(AODfrN)] = 0.
        # DAOD[(DAOD<=0.)|(np.isnan(DAOD))]=0.
        # NAOD[(NAOD<=0.)|(np.isnan(NAOD))]=0.
        # AODfr=(AODfrD*DAOD+AODfrN*NAOD)/(DAOD+NAOD)
        # AODfr[(DAOD<=0.)&(NAOD<=0.)]=0.

        if il == 0:
            nlat = len(lat)
            nlon = len(lon)
            latdata=np.stack([lat] * nlon, axis=1)
            londata=np.stack([lon]*nlat,axis=0)


        if (scalex>1) | (scaley>1):
            nlat, nlon = AODfr.shape
            newx = np.int(np.floor(nlon / scalex))  # 15
            newy = np.int(np.floor(nlat / scaley))  # 21
            xstart=np.int(nlon - newx * scalex)
            ystart=np.int(nlat - newy * scaley)
            AODfrrb = rebin(AODfr[ystart:, xstart:], [newy, newx])
            AODfr[ystart:, xstart:] = np.zeros([newy * scaley, newx * scalex]) - 1
            indentx = np.int(np.floor(scalex / 2) + xstart)
            indenty = np.int(np.floor(scaley / 2) + ystart)
            for irow in np.arange(newy):
                AODfr[np.int(irow * scaley + indenty), np.arange(newx) * scalex + indentx] = AODfrrb[irow, :]

        AODfr[AODfr <= 0.] = np.nan

        samplefr=np.zeros([nlat,nlon])+1
        samplefr[np.isnan(AODfr)]=0
        [fillfr, AODSample] = RegridMDdata(AODfr, samplefr, latdata, londata, ys, xs, [dy, dx])
        fillfr[(AODSample==0)]=0.
        AODfrs[:,:,il]=fillfr.squeeze()

    AODfrs[(AODfrs<=0.)|(np.isnan(AODfrs))]=0.
    #fill grids with NAN with nearest neighbour:
    tFrs=np.count_nonzero(AODfrs,axis=2)

    for ix in np.arange(nx):
        for iy in np.arange(ny):
            if tFrs[iy,ix]<=(3*nl/4):   #less than 10 available layers
                AODfrs[iy,ix,:]=np.nan
            elif tFrs[iy,ix]<nl:
                parAOD=AODfrs[iy,ix,:]
                noninds=np.array(((np.isnan(parAOD))|(parAOD<=0.)).nonzero()).flatten()
                valinds =np.array( ((np.isnan(parAOD)==False) & (parAOD > 0.)).nonzero()).flatten()
                parAOD[noninds]=np.interp(noninds,valinds,parAOD[valinds])
                AODfrs[iy, ix, :]=parAOD

    for il in np.arange(nl):
        fillfr=AODfrs[:,:,il]
        if Interp==True:
            fillfr=fill(fillfr,None,interp=True)
        fillfr = fill(fillfr, None)
        AODfrs[:,:,il]=fillfr.squeeze()

    AODfrs=AODfrs/np.stack([np.nansum(AODfrs,axis=2)]*nl,axis=2)
    AODfrs=AODfrs[:,:,stlevel:edlevel]
    if 'outCaliop' in kwargs:
        outfile = kwargs['outCaliop']
        dso = Dataset(outfile, mode='w', format='NETCDF4')
        dso.createDimension('x', nx)
        dso.createDimension('y', ny)
        dso.createDimension('l', edlevel-stlevel)

        outdata = dso.createVariable('latitude', np.float32, 'y')
        outdata.units = 'degree'
        outdata[:] = ys
        outdata = dso.createVariable('longitude', np.float32, 'x')
        outdata.units = 'degree'
        outdata[:] = xs
        outdata = dso.createVariable('pressure', np.float32, 'l')
        outdata.units = 'hpa'
        outdata[:] = 0.5*(np.array(lowerpres[stlevel:edlevel])+np.array(upperpres[stlevel:edlevel]))

        outdata = dso.createVariable('AODFraction', np.float32, ('y','x','l'))
        outdata.units = 'unitless'
        outdata[:] = AODfrs

        dso.close()

    return AODfrs[:,:]

#Calculate mass extinction efficience (m2/kg) from known humidity for SO4, based on Table A1 of Latimar (2019)
def InterpBeta(cRH):

    RH=cRH.copy()
    RH[cRH>99.]=99.
    kappa=0.61
    row=1.7 #g/cm3
    RHs=[0,35,50,70,80,90,95,99]
    reffs=[0.101,0.101,0.118,0.136,0.152,0.188,0.235,0.398432]
    Qs=[0.603,0.603,0.656,0.742,0.847,1.116,1.5,2.57]

    outreff=np.zeros(RH.shape)
    outQ=np.zeros(RH.shape)

    outreff[RH<35]=reffs[0]
    outreff[(RH>=35)&(RH<40)]=reffs[0]+reffs[0]*(RH[(RH>=35)&(RH<40)]-35)/(40-35)*(((1+kappa*(40/(100-40)))**(1/3))-1)
    outreff[RH>=40]=reffs[0]*((1+kappa*(RH[RH>=40]/(100-RH[RH>=40.])))**(1/3))
    #Interpolate the radius and Q for the RH

    outQ=np.interp(outreff,reffs,Qs,left=np.nan,right=np.nan)

    Beta=0.75*((outreff/reffs[0])**2)*outQ/row/reffs[0]*1.e3  #m2/g to m2/kg

    return [outreff,Beta]

def convex_hull(points):
    """Computes the convex hull of a set of 2D points.

    Input: an iterable sequence of (x, y) pairs representing the points.
    Output: a list of vertices of the convex hull in counter-clockwise order,
      starting from the vertex with the lexicographically smallest coordinates.
    Implements Andrew's monotone chain algorithm. O(n log n) complexity.
    """

    # Sort the points lexicographically (tuples are compared lexicographically).
    # Remove duplicates to detect the case we have just one unique point.
    points = sorted(set(points))

    # Boring case: no points or a single point, possibly repeated multiple times.
    if len(points) <= 1:
        return points

    # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
    # Returns a positive value, if OAB makes a counter-clockwise turn,
    # negative for clockwise turn, and zero if the points are collinear.
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    # Build lower hull
    lower = []
    for p in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    # Build upper hull
    upper = []
    for p in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    # Concatenation of the lower and upper hulls gives the convex hull.
    # Last point of each list is omitted because it is repeated at the beginning of the other list.
    return lower[:-1] + upper[:-1]

def ConvertTsystem(Conc,ConcErr,Sample,u,uErr,v,vErr,ws,wsErr,ys,xs,ycind,xcind,newys,ts,**kwargs):

    nx=len(xs)
    ny=len(ys)

    vCalc=v.copy()
    vErrCalc=vErr.copy()
    wsqr=u**2+v**2
    uOnly=False
    if "uOnly" in kwargs:
        if kwargs["uOnly"]==True:
            uOnly=True
            wsqr=u**2

    DoBeta=False

    if "DoBeta" in kwargs:
        DoBeta=kwargs["DoBeta"]
    if DoBeta==True:
        Concmass=kwargs["Concmass"]
        ConcmassErr = kwargs["ConcmassErr"]

    reslt = [np.absolute(newys[1] - newys[0]), np.absolute(ts[1] - ts[0])]

    defval=-9999.
    # Calculate dx and dy
    [dy, dx] = Calckm(ys, xs)  # meters
    dmj = np.stack([dx] * nx, axis=1)
    dex = np.stack([dy] * ny, axis=0)
    lat2d = np.stack([ys] * nx, axis=1)
    lon2d = np.stack([xs] * ny, axis=0)

    #ts is in hours
    #time step is 20 minutes
    tstep=1200.
    #center trajectory
    T2d=np.zeros([ny,nx])+defval
    T2dErr = np.zeros([ny, nx]) + defval

    #It is more stable to use the latitudal line across the source as the "zero time"
    VMode=True

    #This is the perpendicular vector! (NOT A TYPO)
    uc=-1*v[ycind,xcind]
    vc=u[ycind,xcind]
    totalLines=np.max([ny-ycind-1,ycind-1])*2+1 #vertical pixels
    for iy in np.arange(totalLines):
        if iy==0:
            xc=xcind
            yc=ycind
        else:
            ymod = ((iy-1) % 2)
            yhalf = (iy+1-ymod)/2

            if VMode==True:
                dxy=[0,yhalf]
            else:
                #if np.absolute(uc) > np.absolute(vc):
                #    dxy = np.array([1., vc / uc]) * yhalf
                #else:
                dxy = np.array([uc / vc, 1.]) * yhalf

            if ymod == 0:
                xc=np.int(np.round(xcind+dxy[0]))
                yc = np.int(np.round(ycind + dxy[1]))
            else:
                xc = np.int(np.round(xcind - dxy[0]))
                yc = np.int(np.round(ycind - dxy[1]))
            #yc=np.int(ycind+((-1)**ymod)*((iy-1-ymod)/2+1))
        if (xc<0) | (xc>nx-1) | (yc<0) | (yc>ny-1):
            continue

        T2d,T2dErr = TrackPlumeCenter(dex, dmj, tstep, fill(vCalc, None), fill(u, None),fill(vErrCalc,None),\
                                      fill(uErr,None), fill(wsqr,None), yc, xc, ny, nx, T2d,T2dErr, defval,uOnly=uOnly)

    T2d[(np.absolute(T2d-defval)<1.e-4)]=np.nan
    T2dErr[(np.absolute(T2dErr - defval) < 1.e-4)] = np.nan
    try:
        fillT2d=fill(T2d,None,interp=True)
        fillT2dErr=fill(T2dErr,None,interp=True)
    except:
        if (DoBeta==False):
            return np.zeros(12)
        else:
            return np.zeros(18)
    #estimate the plume age over locations within the convex hull of available estimates 	
    Innerinds=np.transpose(np.array((np.isnan(T2d)==False).nonzero()))
    Outinds = np.transpose(np.array((np.isnan(T2d) == True).nonzero()))

    Hull=convex_hull([(Innerinds[i,0], Innerinds[i,1]) for i in range(len(Innerinds))])
    #Hull=concaveHull(Innerinds,50)

    path=mpltPath.Path(Hull)
    inside=path.contains_points(Outinds)
    ovinds=Outinds[inside==True].transpose()

    T2d[ovinds[0],ovinds[1]]=fillT2d[ovinds[0],ovinds[1]]
    T2dErr[ovinds[0],ovinds[1]] = fillT2dErr[ovinds[0],ovinds[1]]


    T2d=T2d/3600.  #hours
    T2dErr = T2dErr / 3600.  # hours

    #exclude locations that the time is out of the sampling range, or the error is too large due to accumulation
    T2d[(np.isnan(Conc)==True) | (T2d < np.min(ts) - 0.5 * reslt[1]) | (T2d > np.max(ts) + 0.5 * reslt[1]) | ((np.absolute(T2d)>0.)&(np.absolute(T2dErr/T2d)>1.)) ] = np.nan
    T2dErr[np.isnan(T2d)==True] = np.nan

    #exclude the concentration without a time estimates
    Conc[np.isnan(T2d)==True]=np.nan
    ConcErr[np.isnan(T2d)==True]=np.nan
    Sample[np.isnan(T2d)==True]=0


    # # select pixels that has negative u wind direction, small v/u ratio and continous vs. the center:
    # mask = np.zeros([ny, nx], dtype=int) + 1
    # # mask[(u >= 0)] = 0
    #
    # # if mask[ycind,xcind]<1:
    # #    return np.zeros(15)+0.
    #
    # # Only pixels having the same wind direction as the center longitude are used...
    # for iy in np.arange(ny):
    #     if u[iy, xcind] >= 0:
    #         mask[iy, u[iy, :] < 0] = 0
    #     else:
    #         mask[iy, u[iy, :] >= 0] = 0
    #     labeled, ncomponents = measurements.label(mask[iy, :])  # , connectionkernel)
    #     centerlabel = labeled[xcind]
    #     mask[iy, labeled != centerlabel] = 0
    #
    # ucp = fill(u.copy(),None)
    # Conccp = Conc.copy()
    # ConcErrcp = ConcErr.copy()
    # Samplecp = Sample.copy()
    # ucp[mask < 1] = np.nan
    # uErr[mask < 1] = np.nan
    # Conccp[mask < 1] = np.nan
    # ConcErrcp[mask < 1] = np.nan
    # Samplecp[mask < 1] = 0
    #
    #Calculate grid flux in the spatial    coordinates
    gridflx = Conc * fill(u,None) * dex  #Conc(DU or unitless)*m2/s
    gridflxErr = np.sqrt((ConcErr / Conc) ** 2 + fill(uErr / u,None) ** 2) * gridflx

    if DoBeta==True:
        gridflxmass = Concmass * fill(u, None) * dex  # Conc(DU or unitless)*m2/s
        gridflxmassErr = np.sqrt((ConcmassErr / Concmass) ** 2 + fill(uErr / u, None) ** 2) * gridflxmass


    #
    # deltaT = dmj / ucp / 3600.  # hours
    # deltaTErr = np.absolute(uErr / ucp * deltaT)
    #
    # # "T" dimension (2D)
    # T2d = np.zeros([ny, nx])
    # T2dErr = np.zeros([ny, nx])
    # for imj in np.arange(nx):
    #     if imj < xcind:
    #         T2d[:, imj] = np.nansum(deltaT[:, imj:xcind], axis=1) * -1
    #         T2dErr[:, imj] = np.sqrt(np.nansum(deltaTErr[:, imj:xcind] ** 2, axis=1))
    #     if imj > xcind:
    #         T2d[:, imj] = np.nansum(deltaT[:, xcind + 1:imj + 1], axis=1)
    #         T2dErr[:, imj] = np.sqrt(np.nansum(deltaTErr[:, xcind + 1:imj + 1] ** 2, axis=1))
    #     if imj == xcind:
    #         T2d[:, imj] = 0.
    #         T2dErr[:, imj] = deltaTErr[:, imj]
    #
    # T2d[np.isnan(ucp)]=np.nan
    # T2dErr[np.isnan(ucp)]=np.nan
    if DoBeta==False:
        Concdata = np.stack((Conc, gridflx, Sample * ((ConcErr) ** 2), Sample * (gridflxErr ** 2), \
                             Sample * (T2dErr ** 2),ws,Sample*(wsErr**2)),axis=2)
    else:
        Concdata = np.stack((Conc, gridflx, Sample * ((ConcErr) ** 2), Sample * (gridflxErr ** 2), \
                             Sample * (T2dErr ** 2), ws, Sample * (wsErr ** 2),Concmass,gridflxmass,\
                             Sample*(ConcmassErr**2),Sample*(gridflxmassErr**2)), axis=2)

    # regrid concentration (Error) and flux (Error) at each spatial grid to the "lat-t" grid...

    [rgrdConc, rgrdsample] = RegridMDdata(Concdata, Sample, lat2d, T2d, newys, ts, reslt)

    TConc = rgrdConc[:, :, 0]
    TConcErr = np.sqrt(rgrdConc[:, :, 2]/rgrdsample)
    Tgridflx = rgrdConc[:, :, 1]
    TgridflxErr = np.sqrt(rgrdConc[:, :, 3]/rgrdsample)
    TjErr = np.sqrt(rgrdConc[:, :, 4]/rgrdsample)

    Tws=rgrdConc[:, :, 5]
    TwsErr = np.sqrt(rgrdConc[:, :, 6]/rgrdsample)

    if DoBeta==True:
        TConcmass=rgrdConc[:,:,7]
        Tgridflxmass=rgrdConc[:,:,8]
        TConcmassErr=np.sqrt(rgrdConc[:,:,9]/rgrdsample)
        TgridflxmassErr=np.sqrt(rgrdConc[:,:,10]/rgrdsample)

    TSample = rgrdsample


    if DoBeta==False:
        return [TConc,gridflx,Tgridflx,TSample,TConcErr,gridflxErr,TgridflxErr,TjErr,T2d,T2dErr,Tws,TwsErr]
    else:
        return [TConc, gridflx, Tgridflx, TSample, TConcErr, gridflxErr, TgridflxErr, TjErr, T2d, T2dErr, Tws, TwsErr,\
                TConcmass, TConcmassErr, gridflxmass, gridflxmassErr, Tgridflxmass, TgridflxmassErr]

def Calckm(lat,lon,**kwargs):

    #dx is different for each "lat-row", while dy is the same for each "column"
    rearth=6371010   #meter

    if "res" in kwargs:
        res=kwargs["res"]
    else:
        res = [np.absolute(lat[1] - lat[0]), np.absolute(lon[1] - lon[0])]

    resy=res[0]
    resx=res[1]

    ny=len(lat)
    nx=len(lon)

    dy=np.zeros(nx)+rearth*resy*np.pi/180.
    dx=np.zeros(ny)

    for irow in np.arange(ny):
        dx[irow]=rearth*np.cos(lat[irow]*np.pi/180.)*resx*np.pi/180.

    return [dy,dx]

#determine the indexing based on closeness
def InnerIndexing (xs,refxs,**kwargs):

    if "res" in kwargs:
        keyres = kwargs["res"]
        reslow = keyres[0]
        resend = keyres[1]
    else:
        reslow = np.max([np.absolute(refxs[1]-refxs[0]),np.absolute(xs[1] - xs[0])])
        resend = np.max([np.absolute(refxs[-1]-refxs[-2]),np.absolute(xs[-1] - xs[-2])])

    xmin = np.argwhere(xs - np.min(refxs) >= -0.5 * reslow).flatten()[0]
    xmax = np.argwhere(xs - np.max(refxs) < 0.5 * resend).flatten()[-1]
    return [xmin,xmax]

def rebin(arr, new_shape,**kwargs):

    new_shape=np.array(new_shape).astype(int)

    DoAdd=False
    if 'Add' in kwargs:
        if kwargs['Add']==True:
            DoAdd=True

    if new_shape[0]<arr.shape[0]:
        shape = (new_shape[0], arr.shape[0] // new_shape[0],
                 new_shape[1], arr.shape[1] // new_shape[1])
        if DoAdd:
            return np.nansum(np.nansum(arr.reshape(shape), axis=-1), axis=1)
        else:
            return np.nanmean(np.nanmean(arr.reshape(shape), axis=-1), axis=1)
    else:
        shape = (arr.shape[0], new_shape[0] // arr.shape[0],
                 arr.shape[1], new_shape[1] // arr.shape[1])

        stackarr = np.stack([np.stack([arr] * shape[1], axis=1)]*shape[3],axis=3)
        return stackarr.reshape(new_shape)

def Findspikes(data,wd,nstd):

    nx,ny=data.shape
    xlowinds=np.zeros([nx,ny],dtype=bool)
    ylowinds = np.zeros([nx, ny], dtype=bool)

    datanbx=np.zeros([nx,ny,wd*2])
    datanby=np.zeros([nx,ny,wd*2])

    xlowinds[:]=False
    ylowinds[:]=False

    for i in np.arange(wd):
        datanbx[:-1*(i+1),:,2*i]=data[(i+1):,:] #"right"
        datanbx[(i+1):,:,2*i+1]=data[:-1*(i+1),:]  #"left"

        datanby[:,:-1 * (i + 1), 2 * i] = data[:,(i + 1):]  # "up"
        datanby[:,(i + 1):, 2 * i + 1] = data[:,:-1 * (i + 1)]  # "down"

    datanbx[datanbx<=0.]=np.nan
    datanby[datanby<=0.]=np.nan

    xlowinds[(data>=np.nanmean(datanbx,axis=2)*(1.+nstd)) & (data>0.) & (np.nanstd(datanbx,axis=2)>0.)]=True   #| (data >= (np.nanmean(datanbx, axis=2) + 2 * np.nanstd(datanbx, axis=2)))
    ylowinds[(data >= np.nanmean(datanby, axis=2)*(1+nstd)) & (data>0.) & (np.nanstd(datanby,axis=2)>0.)] = True   #| (data >= (np.nanmean(datanby, axis=2) + 2 * np.nanstd(datanby, axis=2)))

    return [xlowinds,ylowinds]

def FindLows(data,wd):

    nx,ny=data.shape
    xlowinds=np.zeros([nx,ny],dtype=bool)
    ylowinds = np.zeros([nx, ny], dtype=bool)

    datanbx=np.zeros([nx,ny,wd*2])
    datanby=np.zeros([nx,ny,wd*2])

    xlowinds[:]=False
    ylowinds[:]=False

    for i in np.arange(wd):
        datanbx[:-1*(i+1),:,2*i]=data[(i+1):,:] #"right"
        datanbx[(i+1):,:,2*i+1]=data[:-1*(i+1),:]  #"left"

        datanby[:,:-1 * (i + 1), 2 * i] = data[:,(i + 1):]  # "up"
        datanby[:,(i + 1):, 2 * i + 1] = data[:,:-1 * (i + 1)]  # "down"

    datanbx[datanbx<=0.]=np.nan
    datanby[datanby<=0.]=np.nan


    xlowinds[((data<=(np.nanmean(datanbx,axis=2)-2*np.nanstd(datanbx,axis=2))) ) & (data>0.) & (np.nanstd(datanbx,axis=2)>0.)]=True   #| (data >= (np.nanmean(datanbx, axis=2) + 2 * np.nanstd(datanbx, axis=2)))
    ylowinds[((data <= (np.nanmean(datanby, axis=2) - 2*np.nanstd(datanby, axis=2))) ) & (data>0.) & (np.nanstd(datanby,axis=2)>0.)] = True   #| (data >= (np.nanmean(datanby, axis=2) + 2 * np.nanstd(datanby, axis=2)))


    return [xlowinds,ylowinds]


#return 2d array quantiles (nth in percentile)
def quantile2d(x,y,Nbins,nth):

    def myperc(x,n=nth):
        return(np.percentile(x,n))
    t=binned_statistic(x,y,statistic=myperc,bins=Nbins)
    v=[]
    for i in range(len(t[0])): v.append((t[1][i+1]+t[1][i])/2.)
    v=np.array(v)
    ii = []
    for i in range(Nbins):
        ii = ii + np.argwhere(((t.binnumber == i) & (y < t.statistic[i]))).flatten().tolist()
    ii = np.array(ii, dtype=int)

    return(x[ii],y[ii])


#Do regridding of data and collocate with wind information based on pressure level range
def getConcwind(file,UVdir,RHdir,CPdir,xs,ys,ts,xcenter,ycenter,res,date,hour,IOFunc,**kwargs):

    #construct the regular grids
    #date in 00:00:00 of the day
    #Jan 30: we are now trying to directly calculate grid flux for each scene and estimate its uncertainty
    #April 5: we are now deriveing the cloud fraction, precipitation and relative humidity from the met field too
    #Correspondingly, we built a separate lat-time coordinate system as well
    #We now use the inherent lat-lon coordinate (0.25x0.25) of the ERA5 meteorology to reduce computation

    minx=np.min(xs)
    miny=np.min(ys)
    maxx=np.max(xs)
    maxy=np.max(ys)

    resy=res[0]
    resx=res[1]

    xcind = np.argmin(np.absolute(xs - xcenter))
    ycind = np.argmin(np.absolute(ys - ycenter))

    DoBeta=False
    DoCaliop=False
    PBLSO2=False
    if 'PBLSO2' in kwargs:
        PBLSO2=kwargs['PBLSO2']
    if "DoBeta" in kwargs:
        DoBeta=kwargs["DoBeta"]

    if "DoCaliop" in kwargs:
        keyCaliop=kwargs["DoCaliop"]
        DoCaliop=keyCaliop[0]

        if DoCaliop:
            AODfrs=keyCaliop[1]

    if DoBeta==False:
        noutval=26
    else:
        noutval=34

    #pressure level range
    pressures=kwargs['pressures']

    uOnly=False
    if "uOnly" in kwargs:
        uOnly=kwargs["uOnly"]

    LonFormat='PN'  #positive and negative (-180 to 180)
    if 'LonFormat' in kwargs:
        LonFormat=kwargs['LonFormat']

    [lat, lon] = IOFunc(file, GeoOnly=True)

    if np.array(lat).size < 2:
        return np.zeros(noutval)+0.

    if (LonFormat == 'PO'):
        lon[lon < 0] = lon[lon < 0.] + 360.
    if (LonFormat == 'PN'):
        lon[lon > 180.] = lon[lon > 180.] - 360.
    insideinds = (((lat - miny) > -0.5 * resy) & ((lat - maxy) < 0.5 * resy) & \
                  ((lon - minx) > -0.5 * resx) & ((lon - maxx) < 0.5 * resx)).nonzero()

    if np.array(insideinds).flatten().size < 1:
        return np.zeros(noutval)+0.
    
    if hour<=23:	    
        filedate=datetime(date.year,date.month,date.day,hour)
    else:
        newdate=date+timedelta(days=1)
        filedate=datetime(newdate.year,newdate.month,newdate.day,hour-24)


    [Conc, Concunc] = IOFunc(file, DataOnly=True, PBLSO2=PBLSO2)



    if np.array(Conc).size<2:
        return np.zeros(noutval)+0.

    if (np.all(Conc <= 0.)) | (np.all(np.isnan(Conc))):
        return np.zeros(noutval)+0.
    #WindInfo, latitude, longitude, (layer averaged) u, v, u (v) uncertainties (1-sigma), number of layers, (surface) w
    #Now xs and ys are carefully selected, we do not do regridding of met fields. The regridding could be done on regular grids after monthly average...
    #u,v,ws are averaged wind speeds...
    #usq,vsq,wsq are total squares from the n Layers...
    [metlat,metlon,u, v, usq, vsq, nLayer, ws, wsq] = \
        GetUV(UVdir,filedate, xs,ys,Press=pressures, LonFormat=LonFormat,NoFile=4)

    #RH, also from levels that are used
    if DoCaliop==False:
        [RH, RHsq] = GetRH(RHdir, filedate, xs, ys, Press=pressures, LonFormat=LonFormat, NoFile=4)
    else:
        [RH, RHsq,RHlayers] = GetRH(RHdir, filedate, xs, ys, Press=pressures, LonFormat=LonFormat, NoFile=4,DoCaliop=True)

    RHErr = np.sqrt((RHsq - nLayer * (RH ** 2)) / (nLayer - 1))
    RHrelsq = (RHErr / RH) ** 2

    #tcc,tp from other files
    [tcc,tp]=GetCP(CPdir,filedate,xs,ys,Press=pressures, LonFormat=LonFormat)



    if np.array(metlat).size<2:
        return np.zeros(noutval)+0.

    #do the regridding to lat-lon system
    Concsample = Conc - Conc
    Concsample[Conc > 0.] = 1
    Concdata=np.stack((Conc,Concsample*(Concunc**2)), axis=2)

    [rgConc, rgsample] = RegridMDdata(Concdata, Concsample, lat, lon, ys, xs, res)
    rgrdConc=rgConc[:,:,0]
    rgrdConcErr=np.sqrt(rgConc[:,:,1]/rgsample)

    if DoBeta == True:
        if DoCaliop==False:
            [Reff, Beta] = InterpBeta(RH)
            rgrdConcmass = rgrdConc / Beta
        else:
            rgrdConcmass=np.zeros(rgrdConc.shape)
            rhnz=(AODfrs.shape)[2]
            partConc=np.zeros(rgrdConc.shape)

            for iz in np.arange(rhnz):
                #AOD
                AODLayer=rgrdConc*AODfrs[:,:,iz]
                AODLayer[(AODLayer<0.)|np.isnan(AODLayer)]=0.
                partConc=partConc+AODLayer
                #mass
                [Reff,Beta]=InterpBeta(RHlayers[iz,:,:])
                massLayer=AODLayer / Beta
                massLayer[(Beta < 0.) | np.isnan(Beta)] = 0.
                rgrdConcmass=rgrdConcmass+massLayer
            rgrdConc=partConc
            rgrdConcmass[rgrdConcmass<=0.]=np.nan



        rgrdConc[np.isnan(rgrdConcmass)]=np.nan
        rgrdConcmassErr=rgrdConcmass*(np.sqrt((rgrdConcErr/rgrdConc)**2+RHrelsq))

    # we estimate the error of wind speed as std of n Layers
    uErr = np.sqrt((usq - nLayer * (u ** 2)) / (nLayer - 1))
    vErr = np.sqrt((vsq - nLayer * (v ** 2)) / (nLayer - 1))

    wsErr = np.sqrt(wsq-3*(ws**2))/2.

    #do the regridding to lat-time system for concentration and gridflux of the concentration
    if DoBeta==False:
        [TConc,gridflx,Tgridflx,TSample,TConcErr,gridflxErr,TgridflxErr,TjErr,T2d,T2dErr,Tws,TwsErr]=ConvertTsystem(rgrdConc,rgrdConcErr,rgsample,u,uErr,v,vErr,ws,wsErr,ys,xs,ycind,xcind,ys,ts,uOnly=uOnly,DoBeta=DoBeta)
    else:
        [TConc, gridflx, Tgridflx, TSample, TConcErr, gridflxErr, TgridflxErr, TjErr, T2d, T2dErr, Tws,
         TwsErr,TConcmass,TConcmassErr,gridflxmass,gridflxmassErr,Tgridflxmass,TgridflxmassErr] = ConvertTsystem(rgrdConc, rgrdConcErr, rgsample, u, uErr, v, vErr, ws, wsErr, ys, xs, ycind, xcind,
                                  ys, ts, uOnly=uOnly, DoBeta=DoBeta,Concmass=rgrdConcmass,ConcmassErr=rgrdConcmassErr)

    if np.array(TConc).size<2:
        return np.zeros(noutval)+0.

    #the pixels without an age indicate little chance that pixel has signals from the source emissions
    rgsample[np.isnan(T2d)==True]=0
    rgrdConc[np.isnan(T2d)==True]=np.nan

    # #The regridding of "regular grids" should be simpler...
    # uvdata = np.stack((u, v,usq,vsq, ws), axis=2)
    # [rgu,rgv,rgusq,rgvsq,rgws]=RebinMDdata(uvdata,metlat,metlon,ys,xs,res)#,interimres=[0.05,0.05])
    # adding new met fields, and mass column concentration converted from AOD based on RH
    if DoBeta==False:
        return [rgrdConc, rgsample*(rgrdConcErr**2), gridflx,rgsample*(gridflxErr**2),rgsample, u, rgsample*(uErr**2), \
                v, rgsample*(vErr**2), ws,rgsample*(wsErr**2),RH,rgsample*(RHErr**2),tcc,tp,T2d,rgsample*(T2dErr**2),\
                nLayer,TConc,TSample*(TConcErr**2),Tgridflx,TSample*(TgridflxErr**2),TSample*(TjErr**2),TSample,\
                Tws,TSample*(TwsErr**2)]
    else:
        return [rgrdConc, rgsample*(rgrdConcErr**2), gridflx,rgsample*(gridflxErr**2),rgsample, u, rgsample*(uErr**2), \
                v, rgsample*(vErr**2), ws,rgsample*(wsErr**2),RH,rgsample*(RHErr**2),tcc,tp,T2d,rgsample*(T2dErr**2),\
                nLayer,TConc,TSample*(TConcErr**2),Tgridflx,TSample*(TgridflxErr**2),TSample*(TjErr**2),TSample,\
                Tws,TSample*(TwsErr**2),rgrdConcmass,rgsample*(rgrdConcmassErr**2),TConcmass,TSample*(TConcmassErr**2),\
                gridflxmass,rgsample*(gridflxmassErr**2),Tgridflxmass,TSample*(TgridflxmassErr**2)]


#"regular grid" regridding
def RebinMDdata(data,latdata,londata,ys,xs,gridres,**kwargs):

    ny=(data.shape)[0]
    nx=(data.shape)[1]
    if len(data.shape)==3:
        nv=(data.shape)[2]
    else:
        nv=1

    oriresy=np.absolute(latdata[1] - latdata[0])
    oriresx=np.absolute(londata[1] - londata[0])

    resy = gridres[0]
    resx = gridres[1]

    if "interimres" in kwargs:
        #interimres is a resolution that bridges the two resolution from latdata... to resx rexy....,
        #after rebinning to interimres, the grids are finner than the final resolution
        itresy=kwargs["interimres"][0]
        itresx = kwargs["interimres"][1]
        rebinfactory = oriresy / itresy
        rebinfactorx = oriresx / itresx

        itmshape = [np.int(np.round(ny*rebinfactory)),np.int(np.round(nx*rebinfactorx))]
        refresx = np.max([itresx,resx])
        refresy = np.max([itresy,resy])

        rebinx = londata[0] + 0.5 * itresx - 0.5*oriresx + itresx * np.arange(itmshape[1])
        rebiny = latdata[0] + 0.5 * itresy - 0.5*oriresy + itresy * np.arange(itmshape[0])

        recoverfactory = itresy / resy
        recoverfactorx = itresx / resx
        #newshape=[np.int(np.round(itmshape[0]*recoverfactory)),np.int(np.round(itmshape[1]*recoverfactorx))]

    else:
        refresx=np.max([oriresx,resx])
        refresy=np.max([oriresy,resy])
        recoverfactory = oriresy / resy
        recoverfactorx = oriresx / resx
        rebinx = londata
        rebiny = latdata
        #newshape = [np.int(np.round(ny*rebinfactory)),np.int(np.round(nx*rebinfactorx))]

    recovershape = [len(ys), len(xs)]

    outdata = []

    for iv in np.arange(nv):
        if len(data.shape) == 3:
            subdata=data[:,:,iv]
        else:
            subdata=data

        if "interimres" in kwargs:
            subdata=rebin(subdata,itmshape)


        if iv == 0:


            [xmin, xmax] = InnerIndexing(rebinx, xs, res=[refresx,refresx])
            [ymin, ymax] = InnerIndexing(rebiny, ys, res=[refresy, refresy])

            # numeric error might cause -1 or -2 grids lacking, needs further examination.....
            if ymax - ymin + 1 < np.round(len(ys) / recoverfactory):
                if np.min(ys) - rebiny[ymin] < 0.5 * resy:
                    ymin = ymin - 1
                if np.max(ys)-rebiny[ymax]>=0.5*resy:
                    ymax = ymax + 1
            if xmax - xmin + 1 < np.round(len(xs) / recoverfactorx):
                if np.min(xs) - rebinx[xmin] < 0.5 * resx:
                    xmin = xmin - 1
                if np.max(xs) - rebinx[xmax] >= 0.5 * resx:
                    xmax = xmax + 1

        rebindata = rebin(subdata[ymin:ymax+1,xmin:xmax+1], recovershape)
        outdata.append(rebindata)

    return np.squeeze(np.array(outdata))

def FillAcrossWind(deltaT,u,v,x,y,nx,ny,defval):

    tanwd = v/ u

    if (tanwd >= -0.414) & (tanwd < -0.414):  # x+1 or x-1
        if (x-1>=0):
            if deltaT[y, x - 1] != defval:
                deltaT[y, x - 1] = deltaT[y, x]
        if (x+1<=nx-1):
            if deltaT[y, x + 1] != defval:
                deltaT[y, x + 1] = deltaT[y, x]

    elif (tanwd >= 0.414) & (tanwd < 2.414):  # x+1 and y+1

        if (y+1<=ny-1) & (x+1<=nx-1):
            if deltaT[y + 1, x + 1] != defval:
                deltaT[y + 1, x + 1] = deltaT[y, x]
        if (x-1>=0) & (y-1>=0):
            if deltaT[y - 1, x - 1] != defval:
                deltaT[y - 1, x - 1] = deltaT[y, x]

    elif (tanwd >= 2.414) | (tanwd < -2.414):  # y+1 or y-1
        if (y + 1 <= ny - 1):
            if deltaT[y + 1, x] != defval:
                deltaT[y + 1, x] = deltaT[y, x]
        if (y - 1 >= 0) :
            if deltaT[y - 1, x] != defval:
                deltaT[y - 1, x] = deltaT[y, x]


    else:  # x-1 and y+1
        if (y + 1 <= ny - 1) & (x - 1 >= 0):
            if deltaT[y + 1, x - 1] != defval:
                deltaT[y + 1, x - 1] = deltaT[y, x]
        if (y - 1 >= 0) & (x + 1 <= nx-1):
            if deltaT[y - 1, x + 1] != defval:
                deltaT[y - 1, x + 1] = deltaT[y, x]

    return deltaT

#regrid 3-dimensional data (with the first 2 dimensions indicating the 2-D spatial shape) into one reference grid
def RegridMDdata(data,sample,latdata,londata,ys,xs,gridres):

    #e.g. gridres=0.05 for ~5 km resolution
    #number of the 3rd dimension
    if len(data.shape)==3:
        nv=(data.shape)[2]
    else:
        nv=1

    resy=gridres[0]
    resx=gridres[1]

    nx=len(xs)
    ny=len(ys)
    minx=np.min(xs)
    miny=np.min(ys)

    dx = np.round((londata-minx)/resx)
    dy = np.round((latdata-miny)/resy)

    outdata = np.zeros([ny, nx, nv])
    outSample = np.zeros([ny, nx], dtype=int)

    dz = np.round(dx + dy * nx).astype(int)

    liminds = ((dx>=0) & (dx<=nx-1) & (dy>=0) & (dy<=ny-1)).nonzero()

    if np.array(liminds).size < 1:
        return [outdata, outSample]

    for iv in np.arange(nv):
        if len(data.shape) == 3:
            subdata=data[:,:,iv]
        else:
            subdata=data

        [tempdata, tempNo, templocs] = \
            SortAvg(subdata[liminds].flatten(),sample[liminds].flatten(), dz[liminds].flatten(), 0)

        Griddata = np.zeros(ny * nx)
        GridNo = np.zeros(ny * nx, dtype=int)

        if iv==0:
            GridNo[templocs] = tempNo
            outSample = GridNo.reshape([ny, nx])

        if len(tempdata) > 1:
            Griddata[templocs] = tempdata
            outdata[:,:,iv]=Griddata.reshape([ny,nx])

    return [outdata,outSample]

def GaussianPuff(Q,Kx,Ky,Kz,x,y,z,u,kdecay,dt):

    t=(np.arange(dt)+0.5)*10

    xshape=x.shape
    xsize=x.size

    ct=np.zeros([xsize,dt])
    t=np.stack([t]*xsize,axis=0)
    flx=np.stack([x.flatten()]*dt,axis=1)
    flz = np.stack([z.flatten()] * dt, axis=1)
    fly = np.stack([y.flatten()] * dt, axis=1)

    ct=Q*np.exp(-kdecay*t)/8./np.sqrt(Kx*Ky*Kz)/((np.pi*t)**(3./2.))\
       *np.exp(-1*((flx-u*t)**2)/4./Kx/t-(fly**2)/4./Ky/t-(flz**2)/4./Kz/t)



    return np.sum(ct*10,axis=1).reshape(xshape)

    #return Kx*x/(r**2)*(x+Kx/Ky*y+Kx/Kz*z)*(F/r+F**2+B/x+1/(r**2))-F*Kx/r-Kx*B/x-kdecay+u*F*x/r
def GaussianPuff2D(Q,Kx,Ky,x,y,u,kdecay,dt,deltat,steady):

    convergence=False

    while(convergence==False):
        t = (np.arange(dt) + 0.5) * deltat

        xshape = x.shape
        xsize = x.size

        t = np.stack([t] * xsize, axis=0)
        flx = np.stack([x.flatten()] * dt, axis=1)
        fly = np.stack([y.flatten()] * dt, axis=1)

        ct = Q * np.exp(-kdecay * t) / np.pi / 8. / np.sqrt(Kx * Ky) / t \
             * np.exp(-1 * ((flx - u * t) ** 2) / 4. / Kx / t - (fly ** 2) / 4. / Ky / t)

        intend=np.sum(ct * deltat, axis=1)
        intlast=np.sum(ct[:,0:dt-1] * deltat, axis=1)

        if steady==False:
            return np.sum(ct * deltat, axis=1).reshape(xshape)

        RMS=np.sqrt(np.mean((intend-intlast)**2))/np.mean(intlast)
        if RMS<0.00001:
            convergence=True

        dt=dt*2

    return np.sum(ct*deltat,axis=1).reshape(xshape)

#v1=Vd-W/2.
def GaussianDep (x,y,z,Ky,Kz,u,Vd,w,q):

    v1=Vd-w/2.

    sigmay=np.sqrt(Ky*2.*x/u)
    sigmaz=np.sqrt(Kz*2.*x/u)
    g1=np.exp(-0.5*(y/sigmay)**2)
    r=2*x/u/sigmaz

    eta=(z/sigmaz+v1*r)/np.sqrt(2.)

    g2=np.exp(-0.5*w*z*r/sigmaz-(w*r)**2/8)*np.exp(-0.5*(y/sigmaz)**2)\
       *(1.-np.sqrt(2.*np.pi)*v1*r*np.exp(eta**2)*erfc(eta))

    return q/u/2./np.pi/sigmay/sigmaz*g1*g2


#1D gaussian model along the wind at the plume center (cross-wind direction)
#Q: total emission from source
#x: along-wind downwind distance from source
#u: wind speed, m/s
#Kd: eddy diffusivity, m2/s
#k: decay constant, s-1
#return value F: column concentration at distance x, kg/m2
def Gaussiandisp1D (x, Q, u, Kd, k):

    F=x-x
    inds=(x<=2000000.).nonzero()
    F[inds]=Q/2/np.pi/Kd/np.sqrt(x[inds]**2+1**2)

    # inds = (x > 0.).nonzero()
    # F[inds] = Q*np.exp(-1*k*x[inds]/u)/2./np.sqrt(np.pi*Kd*x[inds]*u)
    return F

def GaussianFull (x,y,z,sigmax,sigmay,sigmaz,u,q,x0):

    ax=sigmax/u
    ay=sigmay/u
    az=sigmaz/u

    r=np.sqrt(x**2+ax/ay*(y**2)+ax/az*(z**2))


    decayterm = x-x+1
    if np.absolute(x0) > 1.e-20:
        decayterm[x>0]=np.exp(-1*x[x>0]/x0)


    return q/2/np.pi/np.sqrt(ay*az)/r*np.exp(-1*u/ax*(r-x))*decayterm


def Gaussian2DDecay(x,y,sigmax,sigmay,u,q,x0):

    fterm=sigmax**2/x/x0
    xterm=x/x0
    yterm=y**2/2./(sigmay**2)
    return q/np.sqrt(2.*np.pi)/u/sigmay/np.sqrt(1.+fterm)*np.exp(-1*yterm-xterm+fterm*(2.*xterm-yterm))


def GaussianDecay (x,y,z,Kx,Ky,Kz,u,q,kdecay):

    #q: kg/s  continous emission at source (0,0,0)
    #Kx, Ky, Kz: diffussivity (m2/s) at each dimension
    #u: wind speed (m/s)
    #x0: tau*u, e-folding distance, m
    # c : kg/m3 (concentration)
    #equivelent to gaussianfull if kdecay=0

    r=np.sqrt(x**2+Kx/Ky*(y**2)+Kx/Kz*(z**2))
    K=np.sqrt(Ky*Kz)


    if np.absolute(u)<1.e-10:
        F=q / 4. / np.pi / K / r*np.exp(-1/2./Kx*r*np.sqrt(4*np.pi*K*kdecay))
    elif np.absolute(kdecay)<1.e-10:
        F= q / 4. / np.pi / K / r * np.exp(-1 * u / 2. / Kx * (r - x))
    else:
        x0 = u / kdecay
        F=q / 4. / np.pi / K / r * np.exp(-1 * u / 2. / Kx * (r * np.sqrt(1. + 4. *np.pi* K / u / x0) - x))

    zinds=((np.absolute(x)<1.e-3)|(np.absolute(y)<1.e-3)|(np.absolute(z)<1.e-3)).nonzero()

    if np.array(zinds).flatten().size>0:

        F[zinds] = np.nan

        interpx = x[:, 0, 0]
        interpy = y[0, :, 0]
        interpz = z[0, 0, :]

        interpx = interpx[np.absolute(interpx) >= 1.e-3].flatten()
        interpy = interpy[np.absolute(interpy) >= 1.e-3].flatten()
        interpz = interpz[np.absolute(interpz) >= 1.e-3].flatten()

        inds = (np.isnan(F) == False).nonzero()
        my_interpolating_function = RegularGridInterpolator \
            ((interpx, interpy, interpz), F[inds].reshape([len(interpx), len(interpy), len(interpz)]),
             bounds_error=False)

        G = F.flatten()
        inds = (np.isnan(F) == True).nonzero()
        fillinds = np.array([x[inds], y[inds], z[inds]]).transpose()
        G[np.isnan(G) == True] = my_interpolating_function(fillinds)
        F = G.reshape(x.shape)

        F[np.isnan(F) == True] = 0.



    return F

def DoTimeStep(v,u,plumeys,plumexs,ys,xs,dy,dx,rest):

    ny,nx=v.shape

    svu = np.array([v[plumeys, plumexs], u[plumeys, plumexs]])
    tdxy = np.array([dy[plumeys, plumexs], dx[plumeys, plumexs]])
    [sdy, sdx] = [ys, xs] + svu / tdxy * rest
    intdxy = np.round(np.array([sdy, sdx])).astype(int)

    if (intdxy[0]<=ny-1) & (intdxy[0]>=0) & (intdxy[1]<=nx-1) & (intdxy[1]>=0) & ((intdxy[0] != plumeys) | (intdxy[1] != plumexs)):
        midvu=(svu + np.array([v[intdxy[0], intdxy[1]], u[intdxy[0], intdxy[1]]]))*0.5
        if (midvu[0]*svu[0]>0.) & (midvu[1]*svu[1]>0.):  #avoid opposite direction for two time steps...
            [ys, xs] = [ys, xs] + midvu / tdxy * rest
        else:
            [ys, xs] = [sdy, sdx]
    else:
        [ys, xs] = [sdy, sdx]
    return [ys,xs]

def Harversine(lat1,lat2,long1,long2,Rearth):

    dfi=np.absolute(lat1-lat2)/180.*np.pi
    dlamda=np.absolute(long1-long2)/180.*np.pi

    a = (np.sin(dfi/2))**2 + np.cos(lat1/180.*np.pi)*np.cos(lat2/180.*np.pi)*((np.sin(dlamda / 2))**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return c*Rearth


def detect_peaks(image,width):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    #neighborhood = generate_binary_structure(2,2)
    neighborhood = np.zeros([width,width],dtype=bool)
    neighborhood[:]=True
    #print(neighborhood)

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    return detected_peaks

def TrackPlumeCenter(dy,dx,rest,v,u,vErr,uErr,wsqr,ycind,xcind,ny,nx,deltaT,deltaTErr,defval,**kwargs):

    #dts in seconds

    #float
    xe=xcind
    xs=xcind
    ye=ycind
    ys=ycind

    reluvErrsq = (uErr / u) ** 2 + (vErr / v) ** 2
    vcpy=v.copy()
    wsqr=u**2+v**2
    uOnly=False
    if "uOnly" in kwargs:
        uOnly=kwargs["uOnly"]
        if uOnly==True:
            reluvErrsq=(uErr/u)**2
            v[:]=0.
            vErr[:]=0.

    #integer
    plumexs=xcind
    plumeys=ycind
    plumexe = xcind
    plumeye = ycind

    finishs = False  #negative tracking
    finishe = False  #positive tracking

    deltaTlc=np.zeros([ny,nx])+defval
    deltaTErrlc=np.zeros([ny,nx])+defval

    deltaTlc[ycind, xcind] = 0.
    deltaTErrlc[ycind,xcind] = 0.

    accumts=0.
    accumte=0.
    accumtsErr=0.
    accumteErr=0.

    #deltaT=FillAcrossWind(deltaT, u[ycind,xcind], v[ycind,xcind], xcind, ycind, nx, ny, defval)
    #start time steps
    while ((finishs == False) | (finishe == False)):

        # if reluvErrsq[plumeys, plumexs]>1.:  #the error of wind speed in the grid box is too large for a good estimates of time
        #     finishs=True
        # if reluvErrsq[plumeye, plumexe]>1.:
        #     finishe=True
	
	#Limit the tracking to only locations with dominant meridonal wind...
        if uOnly==True:
            if np.absolute(vcpy[plumeys,plumexs]/u[plumeys,plumexs])>0.5:
                finishs=True
            if np.absolute(vcpy[plumeye,plumexe]/u[plumeye,plumexe])>0.5:
                finishe=True
        
	# stop tracking if the wind speed is lower than 3 m/s
        if wsqr[plumeys,plumexs]<9.:
            finishs=True
        if wsqr[plumeye,plumexe]<9.:
            finishe=True


        if finishs==False:
            # one across-wind pixel at each side
            [ys, xs] = DoTimeStep(v, u, plumeys, plumexs, ys, xs, dy, dx, -1 * rest)

        if finishe==False:
            [ye, xe] = DoTimeStep(v, u, plumeye, plumexe, ye, xe, dy, dx, rest)

        if (xs < 0) | (xs > nx-1) | (ys < 0) | (ys > ny-1):
            finishs = True
        if (xe < 0) | (xe > nx-1) | (ye < 0) | (ye > ny-1):
            finishe = True

        if ((finishs == True) & (finishe == True)):
            continue

        outse=np.round(np.array([xs,xe,ys,ye])).astype(int)
        xsint=outse[0]
        xeint=outse[1]
        ysint=outse[2]
        yeint=outse[3]
        errSteps = rest * np.sqrt(reluvErrsq[plumeys, plumexs])
        errStepe = rest * np.sqrt(reluvErrsq[plumeye, plumexe])

        if finishs==False:
            if (xsint != plumexs) | (ysint != plumeys):
                if (np.absolute(deltaTlc[ysint, xsint]-defval)>1.e-4):
                    finishs = True
                else:
                    deltaTlc[ysint, xsint] = accumts - rest
                    #accumulate error terms of time estimates
                    deltaTErrlc[ysint, xsint] = accumtsErr + errSteps

                    plumexs = xsint
                    plumeys = ysint

            accumts=accumts-rest
            accumtsErr=accumtsErr+errSteps


        if finishe==False:

            if (xeint != plumexe) | (yeint != plumeye):
                if (np.absolute(deltaTlc[yeint, xeint] - defval)>1.e-4):
                    finishe = True
                else:
                    deltaTlc[yeint, xeint] = accumte + rest
                    deltaTErrlc[yeint, xeint] = accumteErr + errStepe
                    plumexe = xeint
                    plumeye = yeint
            accumte=accumte+rest
            accumteErr = accumteErr + errStepe

    updateinds=((np.absolute(deltaT-defval)<1.e-4)&(np.absolute(deltaTlc-defval)>1.e-4)).nonzero()
    deltaT[updateinds]=deltaTlc[updateinds]
    deltaTErr[updateinds] = deltaTErrlc[updateinds]
    #nearest neighbourhood interpolation
    return [deltaT,deltaTErr]




def TracePlume(xc,yc,xarray,yarray,ufield,vfield,oriAOD,existx,existy,res,minw):

    xs = xc
    xe = xc
    ys = yc
    ye = yc

    xmin=np.min(xarray)
    xmax=np.max(xarray)
    ymin=np.min(yarray)
    ymax=np.max(yarray)

    Plumex = [xc]
    Plumey = [yc]
    Plumez = [0.]
    PlumeAOD = oriAOD[Plumey - ymin, Plumex - xmin]

    CAOD=oriAOD[Plumey - ymin, Plumex - xmin]

    finishs = False
    finishe = False
    Repeatflag=False

    while ((finishs == False) | (finishe == False)):


        if (xs <= xmin) | (xs >= xmax) | (ys <= ymin) | (ys >= ymax):
            finishs = True

        if (xe <= xmin) | (xe >= xmax) | (ye <= ymin) | (ye >= ymax):
            finishe = True

        if ((finishs == True) & (finishe == True)):
            continue

        [xs, xe, ys, ye] = FindPlumeTrace(xarray, yarray, xs, xe, ys, ye, ufield, vfield, finishs, finishe,minw)

        if finishs == False:
            Plumez = np.append(-1*Harversine(ys*res,Plumey[0]*res,xs*res,Plumex[0]*res,Rearth=6371.)+Plumez[0],Plumez)
            Plumex = np.append(xs, Plumex)
            Plumey = np.append(ys, Plumey)

        if finishe == False:
            Plumez = np.append(Plumez, Harversine(ye*res,Plumey[-1]*res,xe*res,Plumex[-1]*res,Rearth=6371.)+Plumez[-1])
            Plumex = np.append(Plumex, xe)
            Plumey = np.append(Plumey, ye)

        PlumeAOD = oriAOD[np.round(Plumey).astype(int) - ymin, np.round(Plumex).astype(int) - xmin]
        # determine each side if finished based on AOD gradient

        if len(PlumeAOD) < 6:
            continue
        PlumeAOD[PlumeAOD == 0] = np.nan
        PlumeAOD[PlumeAOD >= 4.] = np.nan


        # if (Plumegrd[0] < 0) & (Plumegrd[1] < 0) & (Plumegrd[2] < 0):
        #     # if ((PlumeAOD[0]-PlumeAOD[1])>0) & (PlumeAOD[1]-PlumeAOD[2]<0):
        #     finishs = True
        # if (Plumegrd[-1] > 0) & (Plumegrd[-2] > 0) & (Plumegrd[-3] > 0):
        #     finishe = True
            # if ((PlumeAOD[-1]-PlumeAOD[-2])>0) & (PlumeAOD[-2]-PlumeAOD[-3]<0):

        if (Plumex[0]==Plumex[1]) & (Plumey[0]==Plumey[1]) &  (Plumex[0]==Plumex[2]) & (Plumey[0]==Plumey[2]):
            finishs=True

        if (Plumex[-1]==Plumex[-2]) & (Plumey[-1]==Plumey[-2]) &  (Plumex[-1]==Plumex[-3]) & (Plumey[-1]==Plumey[-3]):
            finishs=True


        if (PlumeAOD[2]>=CAOD):
            finishs = True
        if (PlumeAOD[2]-np.nanmean(PlumeAOD[3:6]) > np.nanstd(PlumeAOD[3:6])):
            finishs = True
        if (PlumeAOD[-3]>=CAOD):
            finishe=True
        if (PlumeAOD[-3]-np.nanmean(PlumeAOD[-6:-3]) > np.nanstd(PlumeAOD[-6:-3])):
            finishe = True

        if ((np.isnan(PlumeAOD[0])==True) & (np.isnan(PlumeAOD[1])==True) & (np.isnan(PlumeAOD[2])==True)):
            finishs=True

        if (np.isnan(PlumeAOD[-1])==True) & (np.isnan(PlumeAOD[-2])==True) & (np.isnan(PlumeAOD[-3])==True):
            finishe=True

        for ix in np.arange(len(Plumex)):
            inds = ((np.round(Plumex).astype(int) == np.round(Plumex[ix]).astype(int)) & \
                    (np.round(Plumey).astype(int) == np.round(Plumey[ix]).astype(int))).nonzero()
            if Plumex[inds].size > 1:
                finishs = True
                finishe = True

        for iloc in np.arange(np.array(existx).size):
            finishx=np.round(existx[iloc])
            finishy=np.round(existy[iloc])

            inds = ((np.round(Plumex) == finishx) & (np.round(Plumey) == finishy)).nonzero()
            if Plumex[inds].size > 0:
                finishs = True
                finishe = True
                Repeatflag=True
                break


    if (Repeatflag==True) | (len(Plumex)<6):
        return [0,0,0,0,0]


    npl = len(Plumex)
    Plumex = Plumex[3:npl - 3]
    Plumey = Plumey[3:npl - 3]
    Plumez = Plumez[3:npl - 3]
    PlumeAOD = PlumeAOD[3:npl - 3]
    # m/s
    Plumew = np.sqrt((ufield[np.round(Plumey).astype(int) - ymin, np.round(Plumex).astype(int) - xmin]) ** 2 \
                     + (vfield[np.round(Plumey).astype(int) - ymin, np.round(Plumex).astype(int) - xmin]) ** 2)



    return [Plumex,Plumey,Plumew,Plumez,PlumeAOD]

def fill(data, invalid=None,**kwargs):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid')
    by the value of the nearest valid data cell

    Input:
        data:    numpy array of any dimension
        invalid: a binary array of same shape as 'data'.
                 data value are replaced where invalid is True
                 If None (default), use: invalid  = np.isnan(data)

    Output:
        Return a filled array.
    """
    if invalid is None: invalid = np.isnan(data)
    if 'interp' in kwargs:
        if kwargs['interp']==True:
            x, y = np.indices(data.shape)
            interp = data.copy()
            interp[np.isnan(interp)] = griddata((x[invalid==False], y[invalid==False]),data[invalid==False],\
                                                (x[invalid==True], y[invalid==True]),method='linear')  # points to interpolate
            return interp
    else:
        ind = nd.distance_transform_edt(invalid,
                                        return_distances=False,
                                        return_indices=True)
        return data[tuple(ind)]




def SortAvg(data,sample,loc,mincount):

    #sort data first
    valind=(sample>0).nonzero()
    if np.array(valind).size <= 0:
        return [[0],[0],[0]]

    data=data[valind]
    sample=sample[valind]
    loc=loc[valind]

    sortinds=np.argsort(loc)
    sample = sample[sortinds]
    data=data[sortinds]*sample
    loc=loc[sortinds]

    #more efficient aggregation
    loc=np.round(loc).astype(int)
    datasum=np.bincount(loc, weights=data, minlength=0)
    samplesum=np.bincount(loc, weights=sample, minlength=0)


    datasum=datasum[samplesum>0]
    samplesum=samplesum[samplesum>0]
    uloc = np.unique(loc)

    inds=(samplesum>mincount).nonzero()
    if np.array(inds).size <= 0:
        return [[0],[0],[0]]

    return [(datasum/samplesum)[inds],np.array(samplesum[inds]).astype(int),uloc[inds]]

def divergence(f):
    num_dims = len(f.shape)
    # fgrd=np.gradient(f)
    return np.ufunc.reduce(np.add, [np.gradient(f,5,axis=i) for i in np.arange(num_dims)])


# def winddirection (u,v):
#
#     tanwd = v / u
#
#     if ((tanwd >= -0.414) & (tanwd < 0.414) & (u >= 0.)):
#         return [1,0]
#
#     if ((tanwd >= -0.414) & (tanwd < 0.414) & (u < 0.)):
#         return [-1,0]
#
#     if ((tanwd >= 0.414) & (tanwd < 2.414) & (u >= 0.)):
#         return [1,1]
#
#     if ((tanwd >= 0.414) & (tanwd < 2.414) & (u < 0.)):
#         return [-1,-1]
#
#     if (((tanwd >= 2.414) | (tanwd < -2.414)) & (v >= 0.)):
#         return [0,1]
#
#     if (((tanwd >= 2.414) | (tanwd < -2.414)) & (v < 0.)):
#         return [0,-1]
#
#     if ((tanwd >= -2.414) & (tanwd < -0.414) & (u >= 0.)):
#         return [1,-1]
#
#     if ((tanwd >= -2.414) & (tanwd < -0.414) & (u < 0.)):
#         return [-1,1]
#
#     # wdob[(tanwd >= -0.414) & (tanwd < 0.414) & (u10 >= 0.)] = 0
#     # wdob[(tanwd >= -0.414) & (tanwd < 0.414) & (u10 < 0.)] = 4
#     # wdob[(tanwd >= 0.414) & (tanwd < 2.414) & (u10 >= 0.)] = 1
#     # wdob[(tanwd >= 0.414) & (tanwd < 2.414) & (u10 < 0.)] = 5
#     # wdob[((tanwd >= 2.414) | (tanwd < -2.414)) & (v10 >= 0.)] = 2
#     # wdob[((tanwd >= 2.414) | (tanwd < -2.414)) & (v10 < 0.)] = 6
#     # wdob[(tanwd >= -2.414) & (tanwd < -0.414) & (u10 >= 0.)] = 7
#     # wdob[(tanwd >= -2.414) & (tanwd < -0.414) & (u10 < 0.)] = 3


def FindPlumeTrace(xarray,yarray,xs,xe,ys,ye,ufield,vfield,finishs,finishe,minw):

    xmin=np.min(xarray[:])
    ymin = np.min(yarray[:])

    xmax=np.max(xarray[:])
    ymax=np.max(yarray[:])

    oxs=-999
    oys=-999
    oxe=-999
    oye=-999

    if finishs==False:
        us=ufield[np.round(ys).astype(int)-ymin,np.round(xs).astype(int)-xmin]
        vs = vfield[np.round(ys).astype(int)-ymin,np.round(xs).astype(int)-xmin]
        #wds=winddirection(us,vs)

        if np.sqrt(us**2+vs**2)<minw:
            oxs=xs
        else:
            dz = np.max(np.absolute([us, vs]))

            oxs = xs - us / dz
            oys = ys - vs / dz



    if finishe == False:
        ue = ufield[np.round(ye).astype(int)-ymin,np.round(xe).astype(int)-xmin]
        ve = vfield[np.round(ye).astype(int)-ymin,np.round(xe).astype(int)-xmin]

        if np.sqrt(ue**2+ve**2)<minw:
            oys=ys
        else:
            dz = np.max(np.absolute([ue, ve]))
            oxe = xe + ue / dz
            oye = ye + ve / dz


    if oxs<xmin:
        oxs=xmin
    if oxs>xmax:
        oxs=xmax

    if oys<ymin:
        oys=ymin
    if oys>ymax:
        oys=ymax

    if oxe < xmin:
        oxe = xmin
    if oxe > xmax:
        oxe = xmax

    if oye < ymin:
        oye = ymin
    if oye > ymax:
        oye = ymax

    return [oxs, oxe, oys, oye]







