from pyhdf.SD import SD, SDC
from netCDF4 import Dataset
import h5py
import numpy as np
import glob
from os import path
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def readMonthlyfile(file,varlist,**kwargs):

    ds = Dataset(file, 'r')
    outdata=[]

    for var in varlist:

        data=ds[var][:]
        outdata.append(data)

    ds.close()

    return outdata

#return the nth location of a binary
def convToBin(data,n):

    #n from 0...
    indata=(data/(2**n)).astype(int)
    outdata= (indata % 2)

    return outdata

def mapfield(ax, conc, xmin, ymin, samplewd, cmap, **kwargs):

    ny, nx = conc.shape
    outx = (xmin - 0.5 + np.arange(nx + 1)) * samplewd[1]
    outy = (ymin - 0.5 + np.arange(ny + 1)) * samplewd[0]


    ax.set_xlim((xmin - 0.5) * samplewd[1], (xmin + nx - 0.5) * samplewd[1], auto=False)
    ax.set_ylim((ymin - 0.5) * samplewd[0], (ymin + ny - 0.5) * samplewd[0], auto=False)

    if 'pn' in kwargs:
        if kwargs['pn'] == True:
            maxval = np.nanpercentile(np.absolute(conc[np.isnan(conc) == False]), 95)
            minval = -1 * maxval
    else:
        maxval = np.nanpercentile(conc[np.isnan(conc) == False], 95)
        minval = np.nanpercentile(conc[np.isnan(conc) == False], 5)

    if "maxval" in kwargs:
        maxval = kwargs['maxval']
    if "minval" in kwargs:
        minval = kwargs['minval']



    cs = ax.pcolormesh(outx, outy, conc, vmin=minval, vmax=maxval, cmap=cmap)

    cbar = plt.colorbar(cs, orientation='horizontal', ax=ax, shrink=0.4)  # , aspect=50
    #cbar.ax.set_title(cbartitle, pad=0.01)
    cbar.ax.tick_params(size=0)

    #ax.set_aspect(1)
    ax.tick_params(size=0)

    if 'plotuv' in kwargs:
        u,v=kwargs['plotuv']
        resuv=kwargs['resuv']
        xinds=np.array(resuv*np.arange(nx/resuv)).astype(int)
        yinds =np.array(resuv * np.arange(ny / resuv)).astype(int)


        qv = ax.quiver(outx[xinds] + 0.5*resuv * samplewd[1], outy[yinds] + 0.5*resuv * samplewd[0], u[yinds, :][:, xinds], \
                       v[yinds, :][:, xinds],scale=200)  # plotu, plotv
        ax.quiverkey(qv, X=0.07, Y=1.05, U=10, label='10 m/s', labelpos='E')

    return [maxval, minval]

def FindExtMajority(AOD,pressure,alt,frac,**kwargs):

    spalt=alt[AOD>0.]
    spAOD=AOD[AOD>0.]
    sppres=pressure[AOD>0.]

    nz=len(spalt)

    zmax=np.argmax(spAOD).flatten()[0]
    tAOD=np.nansum(spAOD)

    for dz in np.arange(np.max([zmax,nz-zmax-1]))+1:
        minz=np.max([0,zmax - dz])
        maxz=np.min([nz,zmax + dz + 1])
        subAOD = np.nansum(spAOD[minz:maxz])
        if subAOD>=frac*tAOD:
            return [sppres[zmax],sppres[minz:maxz]]



def CalPressRange (filedir,year,month,ys,xs,**kwargs):

    outdir="/Users/chili/CALIPSO/"
    localvars=["ext", "extstd", "sample", "pressure", "pressurestd"]
    # for localvar in localvars:
    #     global localvar

    global ext, extstd, sample, pressure, pressurestd

    aerfrac=0.667

    if "Fraction" in kwargs:
        aerfrac=kwargs["Fraction"]

    LongType = "PN"
    if np.min(xs) < 0:
        LongType = "PN"

    if np.max(xs) > 180.:
        LongType = "PO"

    DataOnly=False

    if "DataOnly" in kwargs:
        if kwargs["DataOnly"] == True:
            lat=kwargs["lat"]
            lon=kwargs["lon"]
            alt=kwargs["alt"]
            DataOnly = True

    #daytime or nighttime flag
    DN="D"
    if "DayNight" in kwargs:
        DN=kwargs["DayNight"]

    expFileNo=1
    if DN=="DN":
        expFileNo=2
        DN="*"

    stryear='{:10.0f}'.format(year).strip()
    strmonth=('{:10.0f}'.format(month+100).strip())[1:3]
    file=glob.glob(filedir+stryear+"/*"+stryear+"-"+strmonth+DN+".hdf")

    if len(file) != expFileNo:
        print("Wrong Number of file detected: "+stryear+strmonth,len(file))
        return [False,False,False,False,False]

    if DataOnly==True:
        [ext, extstd, sample, pressure, pressurestd]=ReadCALIPSO(file[0],DataOnly=True)
    else:
        [lat, lon, alt, ext, extstd, sample, pressure, pressurestd] = ReadCALIPSO(file[0])

    #adjust data based on LongType
    if (LongType=="PN") & (np.max(lon)>180.):
        lon[lon>180.]=lon[lon>180.]-360.
    if (LongType=="PO") & (np.min(lon)<0.):
        lon[lon < 0.] = lon[lon < 0.] + 360.

    xsorts = np.argsort(lon)
    lon = np.take(lon, xsorts)
    for localvar in localvars:
        varlocal = globals()[localvar]
        globals()[localvar] = np.take(varlocal, xsorts, axis=1)
    #sort the data at each dimension
    # yinds=np.argsort(lat)
    # for localvar in localvars:
    #     np.take(locals()[localvar], yinds, axis=0)

    # zinds = np.argsort(alt)
    # for localvar in localvars:
    #     np.take(locals()[localvar], zinds, axis=2)

    resx=np.max([np.absolute(xs[1]-xs[0]),np.absolute(lon[1]-lon[0])])
    resy=np.max([np.absolute(ys[1]-ys[0]),np.absolute(lat[1]-lat[0])])
    xmin=np.argwhere((lon-np.min(xs)>=-0.5*resx)&(lon-np.min(xs)<0.5*resx)).flatten()[0]
    xmax=np.argwhere((lon-np.max(xs)>=-0.5*resx)&(lon-np.max(xs)<0.5*resx)).flatten()[0]
    ymin = np.argwhere((lat-np.min(ys)>=-0.5*resy)&(lat-np.min(ys)<0.5*resy)).flatten()[0]
    ymax = np.argwhere((lat-np.max(ys)>=-0.5*resy)&(lat-np.max(ys)<0.5*resy)).flatten()[0]

    lon = lon[xmin:xmax+1]
    lat = lat[ymin:ymax+1]
    for localvar in localvars:
        varlocal = globals()[localvar]
        globals()[localvar] = varlocal[ymin:ymax+1,xmin:xmax+1,:]


    nx=len(lon)
    ny=len(lat)
    # cmap = cm.rainbow
    # cmap.set_bad('white')
    # line_colors = cmap(np.linspace(0, 1, nx))
    # outpng = outdir + "Lat=20."+strmonth+".png"
    # ax=plt.subplot()


    outpmin = np.zeros([ny, nx])
    outpmax = np.zeros([ny, nx])
    outpmed = np.zeros([ny, nx])
    icolor=0
    nz=len(alt)

    # Calculate AOD
    dz = alt[1:nz] - alt[0:nz - 1]
    dz = np.append(alt[0] * 2, dz.flatten())



    for ix in np.arange(nx):
        for iy in np.arange(ny):

            AOD = ext[iy,ix,:] * dz
            AOD[ext[iy,ix,:] <= 0.] = 0.

            if np.all(AOD<=0.):
                continue
            presmax,press=FindExtMajority(AOD,pressure[iy,ix,:],alt,aerfrac)

            # thislat = lat[iy]
            # thislon = lon[ix]
            # if np.absolute(thislat-20.)<0.1:
            #     thisext=ext[iy,ix,:].flatten()
            #     vinds=(thisext>0.).nonzero()
            #     ax.plot(AOD/np.nansum(AOD),alt,color=line_colors[icolor],linewidth=0.5)
            #     ax.scatter(AOD/np.nansum(AOD),alt,color=line_colors[icolor], s=0.5)
            #     outtext="Lon="+"{:10.2f}".format(thislon).strip()
            #     plt.text(0.8,0.9-0.03*icolor,outtext,color=line_colors[icolor],transform=ax.transAxes)
            #     icolor=icolor+1



            outpmin[iy, ix] = np.min(press)
            outpmax[iy, ix] = np.max(press)
            outpmed[iy,ix]=presmax

    # #plt.xscale("log")
    # ax.set_ylim(0,3)
    # plt.savefig(outpng, dpi=600)
    # plt.close()

    return [lat,lon,outpmin,outpmax,outpmed]


def CalAODFr (filedir,year,month,ys,xs,**kwargs):

    localvars=["ext", "extstd", "sample", "pressure", "pressurestd"]
    # for localvar in localvars:
    #     global localvar

    global ext, extstd, sample, pressure, pressurestd

    ReadAOD=False
    if "ReadAOD" in kwargs:
        ReadAOD=kwargs["ReadAOD"]

    if "PressRange" in kwargs:
        minpress=np.min(kwargs["PressRange"])
        maxpress = np.max(kwargs["PressRange"])


    if "AltRange" in kwargs:
        minheight=np.min(kwargs["AltRange"])
        maxheight = np.max(kwargs["AltRange"])

    if "Fraction" in kwargs:
        aerfrac=kwargs["Fraction"]

    LongType = "PN"
    if np.min(xs) < 0:
        LongType = "PN"

    if np.max(xs) > 180.:
        LongType = "PO"

    DataOnly=False

    if "DataOnly" in kwargs:
        if kwargs["DataOnly"] == True:
            lat=kwargs["lat"]
            lon=kwargs["lon"]
            alt=kwargs["alt"]
            DataOnly = True

    #daytime or nighttime flag
    DN="D"
    if "DayNight" in kwargs:
        DN=kwargs["DayNight"]

    expFileNo=1
    if DN=="DN":
        expFileNo=2
        DN="*"

    stryear='{:10.0f}'.format(year).strip()
    strmonth=('{:10.0f}'.format(month+100).strip())[1:3]
    file=glob.glob(filedir+stryear+"/*"+stryear+"-"+strmonth+DN+".hdf")

    if len(file) != expFileNo:
        print("Wrong Number of file detected: "+stryear+strmonth,len(file))
        return [False,False,False,False]

    if DataOnly==True:
        [ext, extstd, sample, pressure, pressurestd]=ReadCALIPSO(file[0],DataOnly=True)
    else:
        [lat, lon, alt, ext, extstd, sample, pressure, pressurestd] = ReadCALIPSO(file[0])

    #adjust data based on LongType
    if (LongType=="PN") & (np.max(lon)>180.):
        lon[lon>180.]=lon[lon>180.]-360.
    if (LongType=="PO") & (np.min(lon)<0.):
        lon[lon < 0.] = lon[lon < 0.] + 360.

    xsorts = np.argsort(lon)
    lon = np.take(lon, xsorts)
    for localvar in localvars:
        varlocal = globals()[localvar]
        globals()[localvar] = np.take(varlocal, xsorts, axis=1)
    #sort the data at each dimension
    # yinds=np.argsort(lat)
    # for localvar in localvars:
    #     np.take(locals()[localvar], yinds, axis=0)

    # zinds = np.argsort(alt)
    # for localvar in localvars:
    #     np.take(locals()[localvar], zinds, axis=2)

    resx=np.max([np.absolute(xs[1]-xs[0]),np.absolute(lon[1]-lon[0])])
    resy=np.max([np.absolute(ys[1]-ys[0]),np.absolute(lat[1]-lat[0])])
    xmin=np.argwhere((lon-np.min(xs)>=-0.5*resx)&(lon-np.min(xs)<0.5*resx)).flatten()[0]
    xmax=np.argwhere((lon-np.max(xs)>=-0.5*resx)&(lon-np.max(xs)<0.5*resx)).flatten()[0]
    ymin = np.argwhere((lat-np.min(ys)>=-0.5*resy)&(lat-np.min(ys)<0.5*resy)).flatten()[0]
    ymax = np.argwhere((lat-np.max(ys)>=-0.5*resy)&(lat-np.max(ys)<0.5*resy)).flatten()[0]

    lon = lon[xmin:xmax+1]
    lat = lat[ymin:ymax+1]
    for localvar in localvars:
        varlocal = globals()[localvar]
        globals()[localvar] = varlocal[ymin:ymax+1,xmin:xmax+1,:]


    nx=len(lon)
    ny=len(lat)
    # cmap = cm.rainbow
    # cmap.set_bad('white')
    # line_colors = cmap(np.linspace(0, 1, nx))
    # outpng = outdir + "Lat=20."+strmonth+".png"
    # ax=plt.subplot()
    aerfrac = np.zeros([ny, nx])
    totalAOD = np.zeros([ny,nx])
    nz=len(alt)

    # Calculate AOD
    dz = alt[1:nz] - alt[0:nz - 1]
    dz = np.append(dz[0], dz.flatten())


    if ReadAOD==True:
        dz3d=np.stack([np.stack([dz]*nx,axis=0)]*ny,axis=0)
        AOD=ext*dz3d
        AOD[ext <= 0.] = 0.
        return [lat,lon,np.sum(AOD,axis=2)]


    for ix in np.arange(nx):
        for iy in np.arange(ny):

            AOD = ext[iy,ix,:] * dz

            AOD[ext[iy,ix,:] <= 0.] = 0.

            if np.all(AOD<=0.):
                continue


            localpressure=pressure[iy, ix, :]

            if "PressRange" in kwargs:

                vertinds=((localpressure>=minpress)&(localpressure<=maxpress)).nonzero()

            if "AltRange" in kwargs:

                vertinds = ((alt >= minheight) & (alt <= maxheight)).nonzero()
            totalAOD[iy, ix] = np.nansum(AOD)
            aerfrac[iy,ix]=np.nansum(AOD[vertinds])


    return [lat,lon,aerfrac,totalAOD]

def ReadCALIPSO (file,**kwargs):

    varsGeo=["Latitude_Midpoint","Longitude_Midpoint","Altitude_Midpoint"]
    varsData=["Extinction_Coefficient_532_Mean","Extinction_Coefficient_532_Standard_Deviation",\
              "Samples_Averaged","Pressure_Mean","Pressure_Standard_Deviation"]

    outdata = []

    GeoOnly=False
    DataOnly=False
    Complementary=False

    if "GeoOnly" in kwargs:
        if kwargs["GeoOnly"]==True:
            GeoOnly=True

    if "DataOnly" in kwargs:
        if kwargs["DataOnly"]==True:
            DataOnly=True

    if "Complementary" in kwargs:
        Complementary=True
        varsComplement=kwargs["Complementary"]

    ds = SD(file, SDC.READ)

    if DataOnly==False:
        for var in varsGeo:
            outdata.append(ds.select(var).get().flatten())


    if GeoOnly==True:
        ds.end()
        return outdata   #Following the dimensions of actual data


    #basic data for analysis
    for var in varsData:
        outdata.append(ds.select(var).get())

    if Complementary==True:
        for var in varsComplement:
            outdata.append(ds.select(var).get())

    ds.end()
    return outdata

def ReadOMISO2 (file,**kwargs):

    GeoOnly=False
    DataOnly=False
    TimeOnly=False

    try:
        f = h5py.File(file, 'r')
        ds = f["HDFEOS/SWATHS/OMI Total Column Amount SO2/"]  # OMI_Total_Column_Amount_SO2/Data_Fields/

        if "TimeOnly" in kwargs:
            if kwargs['TimeOnly'] == True:
                TimeOnly = True

        if 'DataOnly' in kwargs:
            if kwargs['DataOnly'] == True:
                DataOnly = True
        if 'GeoOnly' in kwargs:
            if kwargs['GeoOnly'] == True:
                GeoOnly = True

        if TimeOnly == True:
            return np.round(np.nanmean(ds["Geolocation Fields/SecondsInDay"]) / 3600.)
            f.close()

        Lat = ds["Geolocation Fields/Latitude"][:]
        Lon = ds["Geolocation Fields/Longitude"][:]

        if GeoOnly == True:
            f.close()
            return [Lat, Lon]

        # default AOD varaible
        SO2var = 'ColumnAmountSO2_TRL'
        # we ignore the QA flags in the file for now as suggested
        SO2QAvar = 'QualityFlags_TRL'
        if 'SO2var' in kwargs:
            SO2var = kwargs['SO2var']
            SO2QAvar = kwargs['SO2QAvar']

        # unit: DU
        SO2 = ds["Data Fields/" + SO2var][:]
        SZA = ds["Geolocation Fields/SolarZenithAngle"][:]
        CF = ds["Data Fields/RadiativeCloudFraction"][:]

        # the 8th bit of QA indicates ascending or descending mode...
        QA = convToBin(ds["Data Fields/" + SO2QAvar][:], 7)
        f.close()

        # null possible negative concentration (unphysical)
        neginds = (SO2 <= 0.).nonzero()
        if np.array(neginds).size > 0:
            SO2[neginds] = np.nan
        fltinds = ((SZA >= 70.) | (CF >= 0.3) | (QA > 0)).nonzero()
        if np.array(fltinds).size > 0:
            SO2[fltinds] = np.nan

        #rows affected by row anormaly...
        SO2[:,53:55]=np.nan
        SO2[:,37:43]=np.nan


        SO2unc = SO2 - SO2 + 0.3  # DU

        if DataOnly == True:
            return [SO2, SO2unc]

        return [Lat, Lon, SO2, SO2unc]
    except:
        print('File not openable!: '+ file)
        if TimeOnly==True:
            return [False]
        if (GeoOnly==True) | (DataOnly==True):
            return [False,False]
        return [False,False,False,False]

#Fine AOD reader
def ReadMOD04F (file,**kwargs):

    GeoOnly=False
    DataOnly=False

    if 'DataOnly' in kwargs:
        if kwargs['DataOnly'] == True:
            DataOnly = True
    if 'GeoOnly' in kwargs:
        if kwargs['GeoOnly'] == True:
            GeoOnly = True

    try:
        ds = SD(file, SDC.READ)

        Lat = ds.select('Latitude').get()
        Lon = ds.select('Longitude').get()

        if GeoOnly == True:
            ds.end()
            return [Lat, Lon]

        # default AOD varaible
        AODvar = 'Optical_Depth_Small_Best_Ocean'
        AODQAvar = 'AOD_550_Dark_Target_Deep_Blue_Combined' + '_QA_Flag'

        if 'AODvar' in kwargs:
            AODvar = kwargs['AODvar']
            AODQAvar = kwargs['AODQAvar']

        AODobj = ds.select(AODvar)
        AOD = (AODobj.get() * AODobj.attributes()['scale_factor'])[1,:,:]  #550 nm
        AODQA = ds.select(AODQAvar).get()

        # if AOD in more than 1 wavelength and we need to select one
        if 'AODind' in kwargs:
            AODind = kwargs['AODind']
            AOD = AOD[AODind, :, :]
            AODQA = AODQA[AODind, :, :]

        # null possible negative AODs (unphysical)
        neginds = (AOD <= 0.).nonzero()
        if np.array(neginds).size > 0:
            AOD[neginds] = np.nan

        # 1=land 0=water 2=coastal
        LSvar = 'Land_sea_Flag'
        LSMask = ds.select(LSvar).get()
        ds.end()
        # select high quality data (QA>=1 for ocean and QA=3 for land), not necessary for the merged SDS:
        if 'SelHighAQ' in kwargs:
            if kwargs['SelHighAQ'] == True:
                oceaninds = ((LSMask == 0) & (AODQA < 1)).nonzero()
                if np.array(oceaninds).size > 0:
                    AOD[oceaninds] = np.nan
                landinds = ((LSMask > 0) & (AOD < 3)).nonzero()
                if np.array(landinds).size > 0:
                    AOD[landinds] = np.nan

        # 1-sigma uncertainty of AOD retrievals according to Levy et al. (2015) and Sayer et al. (2014)
	#assume Fine AOD has 10% + 0.03 uncertainty
        AODunc = AOD - AOD
        oceaninds = ((LSMask == 0) & (AOD > 0.) & (AODQA >= 1)).nonzero()
        if np.array(oceaninds).size > 0:
            AODunc[oceaninds] = 0.1 * AOD[oceaninds] + 0.03
        landinds = ((LSMask > 0) & (AOD > 0.) & (AODQA >= 2)).nonzero()
        if np.array(landinds).size > 0:
            AOD[landinds]=np.nan
            AODunc[landinds] = np.nan

        if DataOnly == True:
            return [AOD, AODunc]

        return [Lat, Lon, AOD, AODunc]
    except:
        print('File not openable!: '+ file)
        if (GeoOnly==True) | (DataOnly==True):
            return [False,False]
        return [False,False,False,False]


def ReadMOD04 (file,**kwargs):

    GeoOnly=False
    DataOnly=False

    if 'DataOnly' in kwargs:
        if kwargs['DataOnly'] == True:
            DataOnly = True
    if 'GeoOnly' in kwargs:
        if kwargs['GeoOnly'] == True:
            GeoOnly = True

    try:
        ds = SD(file, SDC.READ)

        Lat = ds.select('Latitude').get()
        Lon = ds.select('Longitude').get()

        if GeoOnly == True:
            ds.end()
            return [Lat, Lon]

        # default AOD varaible
        AODvar = 'AOD_550_Dark_Target_Deep_Blue_Combined'
        AODQAvar = AODvar + '_QA_Flag'

        if 'AODvar' in kwargs:
            AODvar = kwargs['AODvar']
            AODQAvar = kwargs['AODQAvar']

        AODobj = ds.select(AODvar)
        AOD = AODobj.get() * AODobj.attributes()['scale_factor']
        AODQA = ds.select(AODQAvar).get()

        # if AOD in more than 1 wavelength and we need to select one
        if 'AODind' in kwargs:
            AODind = kwargs['AODind']
            AOD = AOD[AODind, :, :]
            AODQA = AODQA[AODind, :, :]

        # null possible negative AODs (unphysical)
        neginds = (AOD <= 0.).nonzero()
        if np.array(neginds).size > 0:
            AOD[neginds] = np.nan

        # 1=land 0=water 2=coastal
        LSvar = 'Land_sea_Flag'
        LSMask = ds.select(LSvar).get()
        ds.end()
        # select high quality data (QA>=1 for ocean and QA=3 for land), not necessary for the merged SDS:
        if 'SelHighAQ' in kwargs:
            if kwargs['SelHighAQ'] == True:
                oceaninds = ((LSMask == 0) & (AODQA < 1)).nonzero()
                if np.array(oceaninds).size > 0:
                    AOD[oceaninds] = np.nan
                landinds = ((LSMask > 0) & (AOD < 3)).nonzero()
                if np.array(landinds).size > 0:
                    AOD[landinds] = np.nan

        # 1-sigma uncertainty of AOD retrievals according to Levy et al. (2015) and Sayer et al. (2014)
        AODunc = AOD - AOD
        oceaninds = ((LSMask == 0) & (AOD > 0.) & (AODQA >= 1)).nonzero()
        if np.array(oceaninds).size > 0:
            AODunc[oceaninds] = 0.05 * AOD[oceaninds] + 0.03
        landinds = ((LSMask > 0) & (AOD > 0.) & (AODQA >= 2)).nonzero()
        if np.array(landinds).size > 0:
            AODunc[landinds] = 0.15 * AOD[landinds] + 0.05

        if DataOnly == True:
            return [AOD, AODunc]

        return [Lat, Lon, AOD, AODunc]
    except:
        print('File not openable!: '+ file)
        if (GeoOnly==True) | (DataOnly==True):
            return [False,False]
        return [False,False,False,False]

def GetRH(ECdir, date, Lons,Lats,Press,LonFormat,**kwargs):

    #Lons,Lats,Press are the spatial range in x,y,z coordinates
    # extract year month day UTC from obtime

    NoFile=1
    DoCaliop=False
    if 'DoCaliop' in kwargs:
        DoCaliop=kwargs['DoCaliop']

    if 'NoFile' in kwargs:
        NoFile=kwargs['NoFile']
    refday=date.day
    hour=np.round(date.hour+date.minute/60.)
    stryymm='{:10.0f}'.format(date.year).strip()+('{:10.0f}'.format(date.month+100).strip())[1:3]
    #the flag if modify longitude formats (-180-180 for "PN"  or  0-360 for "PO")
    ModifyLon=False
    for fileind in np.arange(NoFile):
        strind='{:10.0f}'.format(fileind).strip()
        RHfile = ECdir + stryymm + '.RH.'+strind+'.nc'

        if (path.exists(RHfile)==False):
            return [0,0]

        ds = Dataset(RHfile, 'r')

        if fileind==0:
            lat = ds["latitude"][:].flatten()
            lon = ds["longitude"][:].flatten()

            if (LonFormat == 'PN') & (np.max(lon) > 180.):
                ModifyLon=True
                lontemp = lon.copy()
                #convert from [0,359.75] to [-180,179.75]
                nx=len(lon)
                nxhf=np.int(nx/2)
                lon[0:nxhf]=lontemp[nxhf:nx]-360.
                lon[nxhf:nx] = lontemp[0:nxhf]
            if (LonFormat == 'PO') & (np.max(lon) < 300.):
                ModifyLon = True
                lontemp = lon.copy()
                #convert from [-180,179.75] to [0,359.75]
                nx = len(lon)
                nxhf = np.int(nx / 2)
                lon[0:nxhf] = lontemp[nxhf:nx]
                lon[nxhf:nx] = lontemp[0:nxhf]+360.
            reslat = np.max([np.absolute(Lats[1]-Lats[0]),np.absolute(lat[1] - lat[0])])
            reslon = np.max([np.absolute(Lons[1]-Lons[0]),np.absolute(lon[1] - lon[0])])


            #determine the geolocation fraction from met fields
            xinds=np.array(((lon - np.min(Lons)>=-0.5*reslon) & (lon-np.max(Lons)< 0.5*reslon)).nonzero()).flatten()
            yinds=np.array(((lat - np.min(Lats)>=-0.5*reslat) & (lat-np.max(Lats)< 0.5*reslat)).nonzero()).flatten()

            thour = (refday - 1) * 24. + hour
            thours = np.arange(len(ds['time'][:]))

            #determine the temporal slice
            tind = np.array(np.argmin(np.absolute(thours - thour)))
            rhf = ds["r"]


            #pressure should be flipped first to be continuous
            rhft = np.flip(rhf[tind, :, :, :],axis=0)  # lev,lat,lon
            pressure=np.flip(ds['level'])
        else:
            rhft=np.append(rhft,np.flip(rhf[tind, :, :, :],axis=0),axis=0)
            pressure = np.append(pressure,np.flip(ds['level']))

        ds.close()
        #dsv.close()
        resp = np.absolute(pressure[0] - pressure[1])

    pinds=np.array(((pressure - np.min(Press)>=-0.5 * resp) & (pressure - np.max(Press)<0.5*resp)).nonzero()).flatten()
    nl, ny, nx = rhft.shape
    nxhf = np.int(nx / 2)

    if ModifyLon == True:
        rhfttemp = rhft.copy()
        rhft[:, :, 0:nxhf] = rhfttemp[:, :, nxhf:nx]
        rhft[:, :, nxhf:nx] = rhfttemp[:, :, 0:nxhf]

    #the x,y range is conservative due to possible regridding issue (coarse-fine-coarse)
    pmin=np.min(pinds)
    pmax=np.max(pinds)+1
    ymin=np.min(yinds)
    ymax=np.max(yinds)+1
    xmin=np.min(xinds)
    xmax=np.max(xinds)+1


    if "extend" in kwargs:
        if kwargs["extend"]==True:
            ymin = ymin - 1
            ymax = ymax + 1
            xmin = xmin - 1
            xmax = xmax + 1
    #[metlat, metlon, u, v, usq, vsq, nLayer, ws]
    if DoCaliop==False:
        return [np.flip(np.mean(rhft[pmin:pmax,ymin:ymax,xmin:xmax],axis=0),axis=0),\
                np.flip(np.sum(rhft[pmin:pmax,ymin:ymax,xmin:xmax]**2,axis=0),axis=0)]
    else:
        return [np.flip(np.mean(rhft[pmin:pmax,ymin:ymax,xmin:xmax],axis=0),axis=0),\
                np.flip(np.sum(rhft[pmin:pmax,ymin:ymax,xmin:xmax]**2,axis=0),axis=0),\
                np.flip(rhft[pmin:pmax,ymin:ymax,xmin:xmax],axis=1)]

def GetCP(ECdir, date, Lons,Lats,Press,LonFormat,**kwargs):

    #Lons,Lats,Press are the spatial range in x,y,z coordinates
    # extract year month day UTC from obtime

    NoFile=1
    if 'NoFile' in kwargs:
        NoFile=kwargs['NoFile']
    refday=date.day
    hour=np.round(date.hour+date.minute/60.)
    stryymm='{:10.0f}'.format(date.year).strip()+('{:10.0f}'.format(date.month+100).strip())[1:3]
    #the flag if modify longitude formats (-180-180 for "PN"  or  0-360 for "PO")
    ModifyLon=False


    RHfile = ECdir + stryymm + '.nc'

    if (path.exists(RHfile)==False):
        return [0,0]
    ds = Dataset(RHfile, 'r')
    lat = ds["latitude"][:].flatten()
    lon = ds["longitude"][:].flatten()

    if (LonFormat == 'PN') & (np.max(lon) > 180.):
        ModifyLon=True
        lontemp = lon.copy()
        #convert from [0,359.75] to [-180,179.75]
        nx=len(lon)
        nxhf=np.int(nx/2)
        lon[0:nxhf]=lontemp[nxhf:nx]-360.
        lon[nxhf:nx] = lontemp[0:nxhf]
    if (LonFormat == 'PO') & (np.max(lon) < 300.):
        ModifyLon = True
        lontemp = lon.copy()
        #convert from [-180,179.75] to [0,359.75]
        nx = len(lon)
        nxhf = np.int(nx / 2)
        lon[0:nxhf] = lontemp[nxhf:nx]
        lon[nxhf:nx] = lontemp[0:nxhf]+360.
    reslat = np.max([np.absolute(Lats[1]-Lats[0]),np.absolute(lat[1] - lat[0])])
    reslon = np.max([np.absolute(Lons[1]-Lons[0]),np.absolute(lon[1] - lon[0])])


    #determine the geolocation fraction from met fields
    xinds=np.array(((lon - np.min(Lons)>=-0.5*reslon) & (lon-np.max(Lons)< 0.5*reslon)).nonzero()).flatten()
    yinds=np.array(((lat - np.min(Lats)>=-0.5*reslat) & (lat-np.max(Lats)< 0.5*reslat)).nonzero()).flatten()

    thour = (refday - 1) * 24. + hour
    thours = np.arange(len(ds['time'][:]))

    #determine the temporal slice
    tind = np.array(np.argmin(np.absolute(thours - thour)))
    tccf = ds["tcc"]
    tpf = ds["tp"]

    ymin = np.min(yinds)
    ymax = np.max(yinds) + 1
    xmin = np.min(xinds)
    xmax = np.max(xinds) + 1
    #pressure should be flipped first to be continuous
    tcc = tccf[tind,ymin:ymax, xmin:xmax] # lev,lat,lon
    tp = tpf[tind, ymin:ymax, xmin:xmax]

    ds.close()

    return [tcc,tp]

def GetUV(ECdir, date, Lons,Lats,Press,LonFormat,**kwargs):

    #Lons,Lats,Press are the spatial range in x,y,z coordinates
    # extract year month day UTC from obtime

    NoFile=1
    if 'NoFile' in kwargs:
        NoFile=kwargs['NoFile']
    refday=date.day
    hour=np.round(date.hour+date.minute/60.)
    stryymm='{:10.0f}'.format(date.year).strip()+('{:10.0f}'.format(date.month+100).strip())[1:3]
    #the flag if modify longitude formats (-180-180 for "PN"  or  0-360 for "PO")
    ModifyLon=False
    for fileind in np.arange(NoFile):
        strind='{:10.0f}'.format(fileind).strip()
        ECUfile = ECdir + stryymm + '.U.'+strind+'.nc'
        ECVfile = ECdir + stryymm + '.V.'+strind+'.nc'

        if (path.exists(ECUfile)==False) | (path.exists(ECUfile)==False):
            return [0,0,0,0,0,0,0,0,0]

        dsu = Dataset(ECUfile, 'r')
        dsv = Dataset(ECVfile, 'r')

        if fileind==0:
            lat = dsu["latitude"][:].flatten()
            lon = dsu["longitude"][:].flatten()


            if (LonFormat == 'PN') & (np.max(lon) > 180.):
                ModifyLon=True
                lontemp = lon.copy()
                #convert from [0,359.75] to [-180,179.75]
                nx=len(lon)
                nxhf=np.int(nx/2)
                lon[0:nxhf]=lontemp[nxhf:nx]-360.
                lon[nxhf:nx] = lontemp[0:nxhf]
            if (LonFormat == 'PO') & (np.max(lon) < 300.):
                ModifyLon = True
                lontemp = lon.copy()
                #convert from [-180,179.75] to [0,359.75]
                nx = len(lon)
                nxhf = np.int(nx / 2)
                lon[0:nxhf] = lontemp[nxhf:nx]
                lon[nxhf:nx] = lontemp[0:nxhf]+360.
            reslat = np.max([np.absolute(Lats[1]-Lats[0]),np.absolute(lat[1] - lat[0])])
            reslon = np.max([np.absolute(Lons[1]-Lons[0]),np.absolute(lon[1] - lon[0])])


            #determine the geolocation fraction from met fields
            xinds=np.array(((lon - np.min(Lons)>=-0.5*reslon) & (lon-np.max(Lons)< 0.5*reslon)).nonzero()).flatten()
            yinds=np.array(((lat - np.min(Lats)>=-0.5*reslat) & (lat-np.max(Lats)< 0.5*reslat)).nonzero()).flatten()

            thour = (refday - 1) * 24. + hour
            thours = np.arange(len(dsu['time'][:]))

            #determine the temporal slice
            tind = np.array(np.argmin(np.absolute(thours - thour)))
            usf = dsu["u"]
            vsf = dsv["v"]

            #pressure should be flipped first to be continuous
            usft = np.flip(usf[tind, :, :, :],axis=0)  # lev,lat,lon
            vsft = np.flip(vsf[tind, :, :, :],axis=0)
            pressure=np.flip(dsu['level'])
        else:
            usft=np.append(usft,np.flip(usf[tind, :, :, :],axis=0),axis=0)
            vsft = np.append(vsft, np.flip(vsf[tind, :, :, :],axis=0), axis=0)
            pressure = np.append(pressure,np.flip(dsu['level']))

        dsu.close()
        dsv.close()
        resp = np.absolute(pressure[0] - pressure[1])

    pinds=np.array(((pressure - np.min(Press)>=-0.5 * resp) & (pressure - np.max(Press)<0.5*resp)).nonzero()).flatten()


    nl, ny, nx = usft.shape
    nxhf = np.int(nx / 2)

    if ModifyLon == True:
        usfttemp = usft.copy()
        usft[:, :, 0:nxhf] = usfttemp[:, :, nxhf:nx]
        usft[:, :, nxhf:nx] = usfttemp[:, :, 0:nxhf]
        vsfttemp = vsft.copy()
        vsft[:, :, 0:nxhf] = vsfttemp[:, :, nxhf:nx]
        vsft[:, :, nxhf:nx] = vsfttemp[:, :, 0:nxhf]

    #the x,y range is conservative due to possible regridding issue (coarse-fine-coarse)
    pmin=np.min(pinds)
    pmax=np.max(pinds)+1
    ymin=np.min(yinds)
    ymax=np.max(yinds)+1
    xmin=np.min(xinds)
    xmax=np.max(xinds)+1

    if "extend" in kwargs:
        if kwargs["extend"]==True:
            ymin = ymin - 1
            ymax = ymax + 1
            xmin = xmin - 1
            xmax = xmax + 1

    #ws error from 3 layers
    ws=np.sqrt(np.flip(usft[0:3, ymin:ymax, xmin:xmax], axis=1) ** 2 + np.flip(vsft[0:3, ymin:ymax, xmin:xmax], axis=1) ** 2)

    #[metlat, metlon, u, v, usq, vsq, nLayer, ws]
    return [np.flip(lat[ymin:ymax]),lon[xmin:xmax],\
            np.flip(np.mean(usft[pmin:pmax,ymin:ymax,xmin:xmax],axis=0),axis=0),\
            np.flip(np.mean(vsft[pmin:pmax,ymin:ymax,xmin:xmax],axis=0),axis=0),\
            np.flip(np.sum(usft[pmin:pmax,ymin:ymax,xmin:xmax]**2,axis=0),axis=0),\
            np.flip(np.sum(vsft[pmin:pmax,ymin:ymax,xmin:xmax]**2,axis=0),axis=0),pmax-pmin,\
            np.mean(ws,axis=0),np.sum(ws**2,axis=0)]
