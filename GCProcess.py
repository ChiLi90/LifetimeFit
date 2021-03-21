from netCDF4 import Dataset
import numpy as np
import matplotlib.path as mpltPath
import matplotlib.pyplot as plt


def readGCSO4loss(topdir,stryear,strmonth,AerPLG,lat2d,lon2d,minpressure,maxpressure):

    lats = [-5., 35.]  # -5.,35.
    lons = [140., 210.]  # 140.,210
    prs = [minpressure-12.5, maxpressure+12.5]
    NAv = 6.022e23
    nested=False

    Aermsfile = topdir + 'GEOSChem.AerosolMass.' + stryear + strmonth + '01_0000z.nc4'
    ppbfile = topdir + 'GEOSChem.SpeciesConc.' + stryear + strmonth + '01_0000z.nc4'
    plfile = topdir + 'GEOSChem.ProdLoss.' + stryear + strmonth + '01_0000z.nc4'
    GCmetfile = topdir + 'GEOSChem.StateMet.' + stryear + strmonth + '01_0000z.nc4'
    DryDepfile = topdir + 'GEOSChem.DryDep.' + stryear + strmonth + '01_0000z.nc4'
    WetLossConvfile = topdir + 'GEOSChem.WetLossConv.' + stryear + strmonth + '01_0000z.nc4'
    WetLossLSfile = topdir + 'GEOSChem.WetLossLS.' + stryear + strmonth + '01_0000z.nc4'

    [GCbh, GCprs,PBLH,PrcpLS,PrcpConv,Cloud3D] = readGC(GCmetfile, ['Met_BXHEIGHT', 'Met_PMIDDRY','Met_PBLH','Met_DQRLSAN','Met_DQRCU','Met_CLDF'], lats, lons, prs, 0.25, 0.25, 25.,
                           DataOnly=True, HOnly=True, Nested=nested)

    # determine the number of layers used for column calculation
    minz = np.min(np.array((np.min(GCprs, axis=(1, 2)) <= (prs[0] + 12.5)).nonzero()))
    maxz = np.max(np.array((np.max(GCprs, axis=(1, 2)) > (prs[1] - 12.5)).nonzero()))

    #pressure at PBLH...

    GCPBLprs = Convertpres(PBLH+ Convertasl(GCprs[0,:,:])-0.5*GCbh[0,:,:] )  #+ Convertasl(GCprs[0,:,:])-0.5*GCbh[0,:,:]

    GCbh=GCbh[minz:maxz+1,:,:]
    GCprs=GCprs[minz:maxz+1,:,:]
    Cloud3D=Cloud3D[minz:maxz+1,:,:]
    Precp3D=(PrcpLS+PrcpConv)[minz:maxz+1,:,:]




    [SO4mass, SO2mass] = readGCSO2(Aermsfile, ppbfile, lats, lons, prs, 0.25, 0.25, 25., DataOnly=True, HOnly=True,
                                   zrange=[minz, maxz], Nested=nested)
    SO4mass = SO4mass * GCbh * 1.e-3
    SO2mass = SO2mass * GCbh * 1.e-3

    # SO2 chemical conversion rate (SO4 chemical formation rate)
    [gclat,gclon,SO4fmr] = readGC(plfile, ['Prod_SO4'], lats, lons, prs, 0.25, 0.25, 25., HOnly=True,zrange=[minz, maxz], Nested=nested)

    SO4fmr = SO4fmr[0] / NAv * 1.e6 * 96. * 1.e-3 * 1.e9 * GCbh * 1.e-3  # molecs/cm3/s to kg/km2/s
    aqSO4Vars = ['ProdSO4fromH2O2inCloud', 'ProdSO4fromHOBrInCloud', 'ProdSO4fromO2inCloudMetal',
                 'ProdSO4fromO3inCloud', 'ProdSO4fromO3inSeaSalt', 'ProdSO4fromO3s', 'ProdSO4fromSRO3',
                 'ProdSO4fromSRHObr', 'AREA']
    SO4fmrs = readGC(plfile, aqSO4Vars, lats, lons, prs, 0.25, 0.25, 25., DataOnly=True, HOnly=True,
                     zrange=[minz, maxz], Nested=nested)


    for ivar in np.arange(len(aqSO4Vars) - 1):
        if ivar == 0:
            aqSO4fmr = SO4fmrs[ivar] * 3
        else:
            aqSO4fmr = aqSO4fmr + SO4fmrs[ivar] * 3  # kg SO4/s
    Areas = SO4fmrs[8]  # meters
    Area3D = np.stack([Areas] * (maxz - minz + 1), axis=0)  # m2
    aqSO4fmr = aqSO4fmr / Area3D * 1.e6  # kg/km2/s

    # SO2 and SO4 loss rate from drydep and wet loss files  molec cm-2 s-1 to kg/m2/s
    [SO2drydep, SO4drydep] = readGC(DryDepfile, ['DryDep_SO2', 'DryDep_SO4'], lats, lons, prs, 0.25, 0.25, 25.,
                                    DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)




    # # kg/s
    [SO2WLConv, SO4WLConv] = readGC(WetLossConvfile, ['WetLossConv_SO2', 'WetLossConv_SO4'], lats, lons, prs, 0.25,
                                    0.25, 25., DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)
    [SO2WLLS, SO4WLLS] = readGC(WetLossLSfile, ['WetLossLS_SO2', 'WetLossLS_SO4'], lats, lons, prs, 0.25, 0.25, 25.,
                                DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)
    #
    GCmask = PLGslice(gclat, gclon,
                         np.transpose((lon2d[AerPLG[:, 0], AerPLG[:, 1]], lat2d[AerPLG[:, 0], AerPLG[:, 1]])))
    #
    # # kg/km2/s
    # SO2Wloss = (SO2WLConv + SO2WLLS) / Area3D * 1.e6
    # SO2dloss = SO2drydep / NAv * 64. * 1.e-3 * 1.e4 * 1.e6
    #
    #
    # # distribute the dry depsition into the layers below PBL:
    # SO2dloss = DistributeDrydep(GCprs,GCPBLprs,SO2dloss,SO2mass)
    #
    # SO2loss = SO2Wloss+ SO2dloss
    #
    # SO2loss = (SO4fmr + aqSO4fmr) / 96. * 64. + SO2loss

    SO4wloss = (SO4WLConv + SO4WLLS) / Area3D * 1.e6
    #
    SO4dloss = SO4drydep / NAv * 96. * 1.e-3 * 1.e4 * 1.e6
    SO4dloss = DistributeDrydep(GCprs,GCPBLprs,SO4dloss,SO4mass)
    SO4loss = SO4dloss+SO4wloss

    # SO2totmass = np.sum(SO2mass, axis=0).squeeze()
    # SO2rgmeanmass = np.mean(SO2totmass[SO2GCmask == True])
    # SO2gcbgmass = np.nanmean(
    #     SO2totmass[(gclat2d >= bgminlat) & (gclat2d < bgmaxlat) & (gclon2d >= bgminlon) & (gclon2d <= bgmaxlon)])


    nGClev, nGClat, nGClon = SO4mass.shape
    # SO2 and SO4 profile over SO2 box region, kg/km3
    Aer3DbMask = np.stack([GCmask] * nGClev, axis=0)
    Aer3DMask = np.zeros(Aer3DbMask.shape)
    Aer3DMask[Aer3DbMask == True] = 1
    Aer3DMask[Aer3DbMask == False] = np.nan



    # GCProfSO2 = np.nanmean(SO2mass * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    # #
    # GCProfSO2loss = np.nanmean(SO2loss * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSO2dloss = np.nanmean(SO2dloss * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSO2Wloss = np.nanmean(SO2Wloss * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    GCProfCld = np.nanmean(Cloud3D * Aer3DMask, axis=(1, 2))
    GCProfPrecp = np.nanmean(Precp3D * Aer3DMask, axis=(1, 2))

    # GCProfSO4fmr = np.nanmean(SO4fmr * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfaqSO4fmr = np.nanmean(aqSO4fmr * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    GCProfSO4mass = np.nanmean(SO4mass * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    GCProfSO4wloss = np.nanmean(SO4wloss * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    GCProfSO4dloss = np.nanmean(SO4dloss * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))

    GCAeralts = Convertasl(
        np.nansum(GCprs * SO4mass * Aer3DMask, axis=(1, 2)) / np.nansum(SO4mass * Aer3DMask, axis=(1, 2))) * 1.e-3
    # GCAeralts = Convertasl(np.nanmean(GCprs * Aer3DMask, axis=(1, 2))) * 1.e-3
    #
    # # GCProfSO2deploss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    # # GCProfSO2loss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    # # GCProfSO4loss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    #
    # GCProfSO4 = np.nanmean(AODdata[3] * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSAL = np.nanmean((AODdata[4] + AODdata[5]) * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # # convert to refrence pressures (AOD), km-1, over Aerosol box region
    # GCProf = np.nanmean(GCAOD3D * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfPrcp = np.nanmean((PrcpLS + PrcpConv) * Aer3DMask, axis=(1, 2))
    # GCProfCld = np.nanmean(Cloud3D * SO23DMask, axis=(1, 2))

    # determine regional vertical slicing
    # altitudes, concentration, dry loss, wet loss, total loss
    return [GCAeralts,GCProfSO4mass,GCProfSO4dloss*3600,GCProfSO4wloss*3600,GCProfCld,GCProfPrecp]

def readGCSO2loss(topdir,stryear,strmonth,SO2PLG,lat2d,lon2d,minpressure,maxpressure):

    lats = [-5., 35.]  # -5.,35.
    lons = [140., 210.]  # 140.,210
    prs = [minpressure-12.5, maxpressure+12.5]
    NAv = 6.022e23
    nested=False

    Aermsfile = topdir + 'GEOSChem.AerosolMass.' + stryear + strmonth + '01_0000z.nc4'
    ppbfile = topdir + 'GEOSChem.SpeciesConc.' + stryear + strmonth + '01_0000z.nc4'
    plfile = topdir + 'GEOSChem.ProdLoss.' + stryear + strmonth + '01_0000z.nc4'
    GCmetfile = topdir + 'GEOSChem.StateMet.' + stryear + strmonth + '01_0000z.nc4'
    DryDepfile = topdir + 'GEOSChem.DryDep.' + stryear + strmonth + '01_0000z.nc4'
    WetLossConvfile = topdir + 'GEOSChem.WetLossConv.' + stryear + strmonth + '01_0000z.nc4'
    WetLossLSfile = topdir + 'GEOSChem.WetLossLS.' + stryear + strmonth + '01_0000z.nc4'

    [GCbh, GCprs,PBLH,PrcpLS,PrcpConv,Cloud3D] = readGC(GCmetfile, ['Met_BXHEIGHT', 'Met_PMIDDRY','Met_PBLH','Met_DQRLSAN','Met_DQRCU','Met_CLDF'], lats, lons, prs, 0.25, 0.25, 25.,
                           DataOnly=True, HOnly=True, Nested=nested)

    # determine the number of layers used for column calculation
    minz = np.min(np.array((np.min(GCprs, axis=(1, 2)) <= (prs[0] + 12.5)).nonzero()))
    maxz = np.max(np.array((np.max(GCprs, axis=(1, 2)) > (prs[1] - 12.5)).nonzero()))

    #pressure at PBLH...

    GCPBLprs = Convertpres(PBLH+ Convertasl(GCprs[0,:,:])-0.5*GCbh[0,:,:] )  #

    GCbh=GCbh[minz:maxz+1,:,:]
    GCprs=GCprs[minz:maxz+1,:,:]
    Cloud3D=Cloud3D[minz:maxz+1,:,:]




    [SO4mass, SO2mass] = readGCSO2(Aermsfile, ppbfile, lats, lons, prs, 0.25, 0.25, 25., DataOnly=True, HOnly=True,
                                   zrange=[minz, maxz], Nested=nested)
    SO4mass = SO4mass * GCbh * 1.e-3
    SO2mass = SO2mass * GCbh * 1.e-3

    # SO2 chemical conversion rate (SO4 chemical formation rate)
    [gclat,gclon,SO4fmr] = readGC(plfile, ['Prod_SO4'], lats, lons, prs, 0.25, 0.25, 25., HOnly=True,zrange=[minz, maxz], Nested=nested)

    SO4fmr = SO4fmr[0] / NAv * 1.e6 * 96. * 1.e-3 * 1.e9 * GCbh * 1.e-3  # molecs/cm3/s to kg/km2/s
    aqSO4Vars = ['ProdSO4fromH2O2inCloud', 'ProdSO4fromHOBrInCloud', 'ProdSO4fromO2inCloudMetal',
                 'ProdSO4fromO3inCloud', 'ProdSO4fromO3inSeaSalt', 'ProdSO4fromO3s', 'ProdSO4fromSRO3',
                 'ProdSO4fromSRHObr', 'AREA']
    SO4fmrs = readGC(plfile, aqSO4Vars, lats, lons, prs, 0.25, 0.25, 25., DataOnly=True, HOnly=True,
                     zrange=[minz, maxz], Nested=nested)


    for ivar in np.arange(len(aqSO4Vars) - 1):
        if ivar == 0:
            aqSO4fmr = SO4fmrs[ivar] * 3
        else:
            aqSO4fmr = aqSO4fmr + SO4fmrs[ivar] * 3  # kg SO4/s
    Areas = SO4fmrs[8]  # meters
    Area3D = np.stack([Areas] * (maxz - minz + 1), axis=0)  # m2
    aqSO4fmr = aqSO4fmr / Area3D * 1.e6  # kg/km2/s

    # SO2 and SO4 loss rate from drydep and wet loss files  molec cm-2 s-1 to kg/m2/s
    [SO2drydep, SO4drydep] = readGC(DryDepfile, ['DryDep_SO2', 'DryDep_SO4'], lats, lons, prs, 0.25, 0.25, 25.,
                                    DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)




    # kg/s
    [SO2WLConv, SO4WLConv] = readGC(WetLossConvfile, ['WetLossConv_SO2', 'WetLossConv_SO4'], lats, lons, prs, 0.25,
                                    0.25, 25., DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)
    [SO2WLLS, SO4WLLS] = readGC(WetLossLSfile, ['WetLossLS_SO2', 'WetLossLS_SO4'], lats, lons, prs, 0.25, 0.25, 25.,
                                DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)

    SO2GCmask = PLGslice(gclat, gclon,
                         np.transpose((lon2d[SO2PLG[:, 0], SO2PLG[:, 1]], lat2d[SO2PLG[:, 0], SO2PLG[:, 1]])))

    # kg/km2/s
    SO2Wloss = (SO2WLConv + SO2WLLS) / Area3D * 1.e6
    SO2dloss = SO2drydep / NAv * 64. * 1.e-3 * 1.e4 * 1.e6


    # distribute the dry depsition into the layers below PBL:
    SO2dloss = DistributeDrydep(GCprs,GCPBLprs,SO2dloss,SO2mass)

    SO2loss = SO2Wloss+ SO2dloss

    SO2loss = (SO4fmr + aqSO4fmr) / 96. * 64. + SO2loss

    # SO4Wloss = (SO4WLConv + SO4WLLS) / Area3D * 1.e6
    #
    # SO4dloss = SO4drydep / NAv * 96. * 1.e-3 * 1.e4 * 1.e6
    # SO4dloss = DistributeDrydep(GCprs,GCPBLprs,SO4dloss,SO4mass)
    # SO4loss = SO4Wloss+SO4Wloss

    # SO2totmass = np.sum(SO2mass, axis=0).squeeze()
    # SO2rgmeanmass = np.mean(SO2totmass[SO2GCmask == True])
    # SO2gcbgmass = np.nanmean(
    #     SO2totmass[(gclat2d >= bgminlat) & (gclat2d < bgmaxlat) & (gclon2d >= bgminlon) & (gclon2d <= bgmaxlon)])


    nGClev, nGClat, nGClon = SO2mass.shape
    # SO2 and SO4 profile over SO2 box region, kg/km3
    SO23DbMask = np.stack([SO2GCmask] * nGClev, axis=0)
    SO23DMask = np.zeros(SO23DbMask.shape)
    SO23DMask[SO23DbMask == True] = 1
    SO23DMask[SO23DbMask == False] = np.nan


    GCProfSO2 = np.nanmean(SO2mass * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    #
    GCProfSO2loss = np.nanmean(SO2loss * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    GCProfSO2dloss = np.nanmean(SO2dloss * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    GCProfSO2Wloss = np.nanmean(SO2Wloss * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    GCProfCld = np.nanmean(Cloud3D * SO23DMask, axis=(1, 2))

    # GCProfSO4fmr = np.nanmean(SO4fmr * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfaqSO4fmr = np.nanmean(aqSO4fmr * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSO4mass = np.nanmean(SO4mass * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSO4loss = np.nanmean(SO4loss * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))

    GCSO2alts = Convertasl(
        np.nansum(GCprs * SO2mass * SO23DMask, axis=(1, 2)) / np.nansum(SO2mass * SO23DMask, axis=(1, 2))) * 1.e-3
    # GCAeralts = Convertasl(np.nanmean(GCprs * Aer3DMask, axis=(1, 2))) * 1.e-3
    #
    # # GCProfSO2deploss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    # # GCProfSO2loss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    # # GCProfSO4loss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    #
    # GCProfSO4 = np.nanmean(AODdata[3] * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSAL = np.nanmean((AODdata[4] + AODdata[5]) * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # # convert to refrence pressures (AOD), km-1, over Aerosol box region
    # GCProf = np.nanmean(GCAOD3D * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfPrcp = np.nanmean((PrcpLS + PrcpConv) * Aer3DMask, axis=(1, 2))
    # GCProfCld = np.nanmean(Cloud3D * SO23DMask, axis=(1, 2))

    # determine regional vertical slicing
    # altitudes, concentration, dry loss, wet loss, total loss
    return [GCSO2alts,GCProfSO2,GCProfSO2dloss*3600,GCProfSO2Wloss*3600,GCProfSO2loss*3600,GCProfCld]

def readGCSO4lt(topdir,stryear,strmonth,AerPLG,lat2d,lon2d,minpressure,maxpressure):

    lats = [-5., 35.]  # -5.,35.
    lons = [140., 210.]  # 140.,210
    prs = [minpressure-12.5, maxpressure+12.5]
    NAv = 6.022e23
    nested=False

    Aermsfile = topdir + 'GEOSChem.AerosolMass.' + stryear + strmonth + '01_0000z.nc4'
    ppbfile = topdir + 'GEOSChem.SpeciesConc.' + stryear + strmonth + '01_0000z.nc4'
    plfile = topdir + 'GEOSChem.ProdLoss.' + stryear + strmonth + '01_0000z.nc4'
    GCmetfile = topdir + 'GEOSChem.StateMet.' + stryear + strmonth + '01_0000z.nc4'
    DryDepfile = topdir + 'GEOSChem.DryDep.' + stryear + strmonth + '01_0000z.nc4'
    WetLossConvfile = topdir + 'GEOSChem.WetLossConv.' + stryear + strmonth + '01_0000z.nc4'
    WetLossLSfile = topdir + 'GEOSChem.WetLossLS.' + stryear + strmonth + '01_0000z.nc4'

    [GCbh, GCprs] = readGC(GCmetfile, ['Met_BXHEIGHT', 'Met_PMIDDRY'], lats, lons, prs, 0.25, 0.25, 25.,
                           DataOnly=True, HOnly=True, Nested=nested)

    # determine the number of layers used for column calculation
    minz = np.min(np.array((np.min(GCprs, axis=(1, 2)) <= (prs[0] + 12.5)).nonzero()))
    maxz = np.max(np.array((np.max(GCprs, axis=(1, 2)) > (prs[1] - 12.5)).nonzero()))

    GCbh=GCbh[minz:maxz+1,:,:]
    GCprs=GCprs[minz:maxz+1,:,:]

    [SO4mass, SO2mass] = readGCSO2(Aermsfile, ppbfile, lats, lons, prs, 0.25, 0.25, 25., DataOnly=True, HOnly=True,
                                   zrange=[minz, maxz], Nested=nested)
    SO4mass = SO4mass * GCbh * 1.e-3
    SO2mass = SO2mass * GCbh * 1.e-3

    # SO2 chemical conversion rate (SO4 chemical formation rate)
    [gclat,gclon,SO4fmr] = readGC(plfile, ['Prod_SO4'], lats, lons, prs, 0.25, 0.25, 25., HOnly=True,zrange=[minz, maxz], Nested=nested)

    SO4fmr = SO4fmr[0] / NAv * 1.e6 * 96. * 1.e-3 * 1.e9 * GCbh * 1.e-3  # molecs/cm3/s to kg/km2/s
    aqSO4Vars = ['ProdSO4fromH2O2inCloud', 'ProdSO4fromHOBrInCloud', 'ProdSO4fromO2inCloudMetal',
                 'ProdSO4fromO3inCloud', 'ProdSO4fromO3inSeaSalt', 'ProdSO4fromO3s', 'ProdSO4fromSRO3',
                 'ProdSO4fromSRHObr', 'AREA']
    SO4fmrs = readGC(plfile, aqSO4Vars, lats, lons, prs, 0.25, 0.25, 25., DataOnly=True, HOnly=True,
                     zrange=[minz, maxz], Nested=nested)


    for ivar in np.arange(len(aqSO4Vars) - 1):
        if ivar == 0:
            aqSO4fmr = SO4fmrs[ivar] * 3
        else:
            aqSO4fmr = aqSO4fmr + SO4fmrs[ivar] * 3  # kg SO4/s
    Areas = SO4fmrs[8]  # meters
    Area3D = np.stack([Areas] * (maxz - minz + 1), axis=0)  # m2
    aqSO4fmr = aqSO4fmr / Area3D * 1.e6  # kg/km2/s

    # SO2 and SO4 loss rate from drydep and wet loss files  molec cm-2 s-1 to kg/m2/s
    [SO2drydep, SO4drydep] = readGC(DryDepfile, ['DryDep_SO2', 'DryDep_SO4'], lats, lons, prs, 0.25, 0.25, 25.,
                                    DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)
    # kg/s
    [SO2WLConv, SO4WLConv] = readGC(WetLossConvfile, ['WetLossConv_SO2', 'WetLossConv_SO4'], lats, lons, prs, 0.25,
                                    0.25, 25., DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)
    [SO2WLLS, SO4WLLS] = readGC(WetLossLSfile, ['WetLossLS_SO2', 'WetLossLS_SO4'], lats, lons, prs, 0.25, 0.25, 25.,
                                DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)

    AerGCmask = PLGslice(gclat, gclon,
                         np.transpose((lon2d[AerPLG[:, 0], AerPLG[:, 1]], lat2d[AerPLG[:, 0], AerPLG[:, 1]])))


    SO4Wloss = (SO4WLConv + SO4WLLS) / Area3D * 1.e6
    SO4loss = SO4Wloss
    SO4dloss = SO4drydep / NAv * 96. * 1.e-3 * 1.e4 * 1.e6
    SO4loss[0, :, :] = SO4loss[0, :, :] + SO4dloss

    # SO2totmass = np.sum(SO2mass, axis=0).squeeze()
    # SO2rgmeanmass = np.mean(SO2totmass[SO2GCmask == True])
    # SO2gcbgmass = np.nanmean(
    #     SO2totmass[(gclat2d >= bgminlat) & (gclat2d < bgmaxlat) & (gclon2d >= bgminlon) & (gclon2d <= bgmaxlon)])


    nGClev, nGClat, nGClon = SO2mass.shape
    # SO2 and SO4 profile over SO2 box region, kg/km3
    Aer3DbMask = np.stack([AerGCmask] * nGClev, axis=0)
    Aer3DMask = np.zeros(Aer3DbMask.shape)
    Aer3DMask[Aer3DbMask == True] = 1
    Aer3DMask[Aer3DbMask == False] = np.nan


    # GCProfSO2 = np.nanmean(SO2mass * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    #
    # GCProfSO2loss = np.nanmean(SO2loss * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSO2deploss = np.nanmean(SO2deploss * SO23DMask / (GCbh * 1.e-3), axis=(1, 2))
    #
    # GCProfSO4fmr = np.nanmean(SO4fmr * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfaqSO4fmr = np.nanmean(aqSO4fmr * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSO4mass = np.nanmean(SO4mass * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSO4loss = np.nanmean(SO4loss * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    #
    # GCSO2alts = Convertasl(
    #     np.nansum(GCprs * SO2mass * SO23DMask, axis=(1, 2)) / np.nansum(SO2mass * SO23DMask, axis=(1, 2))) * 1.e-3
    # GCAeralts = Convertasl(np.nanmean(GCprs * Aer3DMask, axis=(1, 2))) * 1.e-3
    #
    # # GCProfSO2deploss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    # # GCProfSO2loss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    # # GCProfSO4loss[(GCProfSO4mass/GCProfSO4loss/3600>1000.)|(GCProfSO4loss<0.)]=np.nan
    #
    # GCProfSO4 = np.nanmean(AODdata[3] * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfSAL = np.nanmean((AODdata[4] + AODdata[5]) * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # # convert to refrence pressures (AOD), km-1, over Aerosol box region
    # GCProf = np.nanmean(GCAOD3D * Aer3DMask / (GCbh * 1.e-3), axis=(1, 2))
    # GCProfPrcp = np.nanmean((PrcpLS + PrcpConv) * Aer3DMask, axis=(1, 2))
    # GCProfCld = np.nanmean(Cloud3D * SO23DMask, axis=(1, 2))

    # determine regional vertical slicing
    rgGCprs = np.nanmean(GCprs * Aer3DMask, axis=(1, 2))
    vinds = np.array(((rgGCprs < minpressure)&(rgGCprs>=maxpressure)).nonzero()).flatten()


    rgSO4lifetime = np.nansum(SO4mass[vinds, :, :] * Aer3DMask[vinds, :, :]) / np.nansum(
        SO4loss[vinds, :, :] * Aer3DMask[vinds, :, :])
    return rgSO4lifetime/3600.

def readGCSO2lt(topdir,stryear,strmonth,SO2PLG,lat2d,lon2d,minpressure,maxpressure):

    lats = [-5., 35.]  # -5.,35.
    lons = [140., 210.]  # 140.,210
    prs = [minpressure-12.5, maxpressure+12.5]
    NAv = 6.022e23
    nested=False

    Aermsfile = topdir + 'GEOSChem.AerosolMass.' + stryear + strmonth + '01_0000z.nc4'
    ppbfile = topdir + 'GEOSChem.SpeciesConc.' + stryear + strmonth + '01_0000z.nc4'
    plfile = topdir + 'GEOSChem.ProdLoss.' + stryear + strmonth + '01_0000z.nc4'
    GCmetfile = topdir + 'GEOSChem.StateMet.' + stryear + strmonth + '01_0000z.nc4'
    DryDepfile = topdir + 'GEOSChem.DryDep.' + stryear + strmonth + '01_0000z.nc4'
    WetLossConvfile = topdir + 'GEOSChem.WetLossConv.' + stryear + strmonth + '01_0000z.nc4'
    WetLossLSfile = topdir + 'GEOSChem.WetLossLS.' + stryear + strmonth + '01_0000z.nc4'

    [GCbh, GCprs] = readGC(GCmetfile, ['Met_BXHEIGHT', 'Met_PMIDDRY'], lats, lons, prs, 0.25, 0.25, 25.,
                           DataOnly=True, HOnly=True, Nested=nested)

    # determine the number of layers used for column calculation
    minz = np.min(np.array((np.min(GCprs, axis=(1, 2)) <= (prs[0] + 12.5)).nonzero()))
    maxz = np.max(np.array((np.max(GCprs, axis=(1, 2)) > (prs[1] - 12.5)).nonzero()))

    GCbh=GCbh[minz:maxz+1,:,:]
    GCprs=GCprs[minz:maxz+1,:,:]

    [SO4mass, SO2mass] = readGCSO2(Aermsfile, ppbfile, lats, lons, prs, 0.25, 0.25, 25., DataOnly=True, HOnly=True,
                                   zrange=[minz, maxz], Nested=nested)
    SO4mass = SO4mass * GCbh * 1.e-3
    SO2mass = SO2mass * GCbh * 1.e-3

    # SO2 chemical conversion rate (SO4 chemical formation rate)
    [gclat,gclon,SO4fmr] = readGC(plfile, ['Prod_SO4'], lats, lons, prs, 0.25, 0.25, 25., HOnly=True,zrange=[minz, maxz], Nested=nested)

    SO4fmr = SO4fmr[0] / NAv * 1.e6 * 96. * 1.e-3 * 1.e9 * GCbh * 1.e-3  # molecs/cm3/s to kg/km2/s
    aqSO4Vars = ['ProdSO4fromH2O2inCloud', 'ProdSO4fromHOBrInCloud', 'ProdSO4fromO2inCloudMetal',
                 'ProdSO4fromO3inCloud', 'ProdSO4fromO3inSeaSalt', 'ProdSO4fromO3s', 'ProdSO4fromSRO3',
                 'ProdSO4fromSRHObr', 'AREA']
    SO4fmrs = readGC(plfile, aqSO4Vars, lats, lons, prs, 0.25, 0.25, 25., DataOnly=True, HOnly=True,
                     zrange=[minz, maxz], Nested=nested)


    for ivar in np.arange(len(aqSO4Vars) - 1):
        if ivar == 0:
            aqSO4fmr = SO4fmrs[ivar] * 3
        else:
            aqSO4fmr = aqSO4fmr + SO4fmrs[ivar] * 3  # kg SO4/s
    Areas = SO4fmrs[8]  # meters
    Area3D = np.stack([Areas] * (maxz - minz + 1), axis=0)  # m2
    aqSO4fmr = aqSO4fmr / Area3D * 1.e6  # kg/km2/s

    # SO2 and SO4 loss rate from drydep and wet loss files  molec cm-2 s-1 to kg/m2/s
    [SO2drydep, SO4drydep] = readGC(DryDepfile, ['DryDep_SO2', 'DryDep_SO4'], lats, lons, prs, 0.25, 0.25, 25.,
                                    DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)
    # kg/s
    [SO2WLConv, SO4WLConv] = readGC(WetLossConvfile, ['WetLossConv_SO2', 'WetLossConv_SO4'], lats, lons, prs, 0.25,
                                    0.25, 25., DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)
    [SO2WLLS, SO4WLLS] = readGC(WetLossLSfile, ['WetLossLS_SO2', 'WetLossLS_SO4'], lats, lons, prs, 0.25, 0.25, 25.,
                                DataOnly=True, HOnly=True, zrange=[minz, maxz], Nested=nested)

    SO2GCmask = PLGslice(gclat, gclon,
                         np.transpose((lon2d[SO2PLG[:, 0], SO2PLG[:, 1]], lat2d[SO2PLG[:, 0], SO2PLG[:, 1]])))

    # kg/km2/s
    SO2Wloss = (SO2WLConv + SO2WLLS) / Area3D * 1.e6
    SO2dloss = SO2drydep / NAv * 64. * 1.e-3 * 1.e4 * 1.e6
    SO2loss = SO2Wloss
    SO2loss[0, :, :] = SO2loss[0, :, :] + SO2dloss
    SO2deploss = SO2loss

    SO2loss = (SO4fmr + aqSO4fmr) / 96. * 64. + SO2loss



    nGClev, nGClat, nGClon = SO2mass.shape
    # SO2 and SO4 profile over SO2 box region, kg/km3
    SO23DbMask = np.stack([SO2GCmask] * nGClev, axis=0)
    SO23DMask = np.zeros(SO23DbMask.shape)
    SO23DMask[SO23DbMask == True] = 1
    SO23DMask[SO23DbMask == False] = np.nan



    # determine regional vertical slicing
    rgGCprs = np.nanmean(GCprs * SO23DMask, axis=(1, 2))
    vinds = np.array(((rgGCprs < minpressure)&(rgGCprs>=maxpressure)).nonzero()).flatten()


    rgSO2lifetime = np.nansum(SO2mass[vinds, :, :] * SO23DMask[vinds, :, :]) / np.nansum(
        SO2loss[vinds, :, :] * SO23DMask[vinds, :, :])
    return rgSO2lifetime/3600.

def PLGslice(gclat,gclon,PLG):

    nx=len(gclon)
    ny=len(gclat)
    #PLG, high resolution mask boundaries
    xres=np.absolute(gclon[1]-gclon[0])
    yres=np.absolute(gclat[1]-gclat[0])

    xl2d=np.stack([gclon-0.5*xres]*ny,axis=0).flatten()
    xr2d=np.stack([gclon+0.5*xres]*ny,axis=0).flatten()
    yu2d=np.stack([gclat+0.5*yres]*nx,axis=1).flatten()
    yl2d=np.stack([gclat-0.5*yres]*nx,axis=1).flatten()


    PLGpath=mpltPath.Path(PLG)

    mask =( (PLGpath.contains_points(np.transpose((xl2d,yu2d)))==True) | (PLGpath.contains_points(np.transpose((xl2d,yl2d)))==True) |   \
            (PLGpath.contains_points(np.transpose((xr2d,yu2d)))==True) | (PLGpath.contains_points(np.transpose((xr2d,yl2d)))==True) )

    mask=np.array(mask).reshape([ny,nx])

    return mask

def xyslice(gclat,gclon,lats,lons,dlat,dlon):
    xinds = np.array(
        (((gclon - np.min(lons)) >= -0.5 * dlon) & ((gclon - np.max(lons)) < 0.5 * dlon)).nonzero()).flatten()
    yinds = np.array(
        (((gclat - np.min(lats)) >= -0.5 * dlat) & ((gclat - np.max(lats)) < 0.5 * dlat)).nonzero()).flatten()

    return [yinds,xinds]


#hpa to meter
def Convertasl (pressures):

    p0=1013.25
    t0=15.
    asl=pressures-pressures
    asl=((p0/pressures)**(1./5.257)-1)*(t0+273.15)/0.0065

    return asl


#m to pressure (hpa)
def Convertpres (Hgt):

    p0=1013.25
    t0=15.
    pres=Hgt-Hgt
    pres=p0/((Hgt*0.0065/(t0+273.15)+1.)**5.257)

    return pres

def DistributeDrydep(prs,PBLprs,dloss,mass):

    nz,ny,nx=prs.shape

    PBLmask=((prs-np.stack([PBLprs]*nz,axis=0))>0.)

    PBLmask3D=np.zeros(np.array(PBLmask).shape)
    PBLmask3D[PBLmask==True]=1
    PBLmask3D[PBLmask!=True]=0
    lossWgts=mass*PBLmask3D/np.stack([np.nansum(mass*PBLmask3D,axis=0)]*nz,axis=0)

    dloss3D=np.stack([dloss]*nz,axis=0)*lossWgts

    dloss3D[np.isnan(dloss3D)]=0.
    return dloss3D

#data in (z,y,x)
def ConvertPressures (data,pres3D,strefprs,edrefprs,**kwargs):

    nz,ny,nx=data.shape
    nrefz=len(edrefprs)
    outdata=np.zeros([nrefz,ny,nx])

    DoSum=False
    if 'DoSum' in kwargs:
        DoSum=kwargs['DoSum']

    for ix in np.arange(nx):
        for iy in np.arange(ny):

            grddata=data[:,iy,ix]
            grdprs=pres3D[:,iy,ix]

            for iz in np.arange(nrefz):
                zinds=np.array(((grdprs>=edrefprs[iz])&(grdprs<strefprs[iz])).nonzero())
                if zinds.size<1:
                    continue

                if DoSum:

                    outdata[iz,iy,ix]=np.nansum(grddata[zinds])
                else:
                    outdata[iz,iy,ix]=np.nanmean(grddata[zinds])


    return outdata

def readGC(gcfile,varNames,lats,lons,prs,dlat,dlon,dpr,**kwargs):

    outdata=[]

    nested = False

    if 'Nested' in kwargs:
        nested = kwargs["Nested"]

    if nested==False:
        ds=Dataset(gcfile,'r')
        gclat = ds['lat'][:]
        gclon = ds['lon'][:]
    else:
        ds=[Dataset(gcfile[0],'r'),Dataset(gcfile[1],'r')]
        gclat = ds[0]['lat'][:]
        gclon = np.append(ds[0]['lon'][:],ds[1]['lon'][:])

    if np.max(lons)>180.:
        lontype='PO'
    else:
        lontype='PN'

    if (lontype=='PO')&(np.max(gclon)<=180.):
        gclon[gclon<0.]=gclon[gclon<0.]+360.

    HOnly = False
    if 'HOnly' in kwargs:
        HOnly = kwargs['HOnly']

    RefPrs=True
    if 'RefPrs' in kwargs:
        RefPrs=kwargs['RefPrs']

    #use simple pressures
    if HOnly==False:
        if RefPrs==True:
            #ETAedge(I,J,L)   = [ Pedge(I,J,L)   – Ptop ] / [ Psurface – Ptop ]
            psurf=1013.25
            ptop=0.01
            if nested==False:
                gcpr = ds['lev'][:] * (psurf - ptop) + ptop
            else:
                gcpr=ds[0]['lev'][:]*(psurf-ptop)+ptop
        else:
            HOnly=True

    #calculate x y and z slicing
    xinds=np.array((((gclon-np.min(lons))>=-0.5*dlon)&((gclon-np.max(lons))<0.5*dlon)).nonzero()).flatten()
    yinds=np.array((((gclat-np.min(lats))>=-0.5*dlat)&((gclat-np.max(lats))<0.5*dlat)).nonzero()).flatten()

    minx=np.min(xinds)
    maxx=np.max(xinds)

    miny=np.min(yinds)
    maxy=np.max(yinds)

    if HOnly==False:
        zinds=np.array((((gcpr-np.min(prs))>-0.5*dpr)&((gcpr-np.max(prs))<=0.5*dpr)).nonzero()).flatten()
    else:
        if 'zrange' in kwargs:
            zrange = kwargs['zrange']
            minz = zrange[0]
            maxz = zrange[1]
        else:
            minz = 0
            if nested==False:
                maxz = len(ds['lev'][:]) - 1
            else:
                maxz = len(ds[0]['lev'][:])-1

    for varName in varNames:
        if nested==False:
            varData=ds[varName][:].squeeze()

        else:
            varData = ds[0][varName][:].squeeze()
            varData1 = ds[1][varName][:].squeeze()


        if varData.ndim==4:
            if HOnly:
                if nested==False:
                    outdata.append(varData[0,minz:maxz+1,miny:maxy+1,xinds].squeeze())
                else:
                    outdata.append(np.append(varData,varData1,axis=3)[0, minz:maxz + 1, miny:maxy+1,minx:maxx+1].squeeze())
            else:
                if nested==False:
                    outdata.append(varData[0,minz:maxz+1,miny:maxy+1,xinds].squeeze())
                else:
                    outdata.append(np.append(varData,varData1,axis=3)[0, minz:maxz+1, miny:maxy+1,minx:maxx+1].squeeze())

        if varData.ndim==2:
            if HOnly:
                if nested==False:
                    outdata.append(varData[miny:maxy+1,xinds].squeeze())
                else:
                    outdata.append(np.append(varData,varData1,axis=1)[miny:maxy+1,minx:maxx+1].squeeze())
            else:
                if nested == False:
                    outdata.append(varData[miny:maxy+1,xinds].squeeze())
                else:
                    outdata.append(np.append(varData, varData1, axis=1)[miny:maxy+1,minx:maxx+1].squeeze())


        if varData.ndim==3:
            if HOnly:
                if nested == False:
                    outdata.append(varData[minz:maxz + 1, miny:maxy+1,xinds].squeeze())
                else:
                    outdata.append(np.append(varData, varData1, axis=2)[minz:maxz + 1,miny:maxy+1,minx:maxx+1].squeeze())
            else:
                if nested == False:
                    outdata.append(varData[minz:maxz + 1, miny:maxy+1,xinds].squeeze())
                else:
                    outdata.append(np.append(varData, varData1, axis=2)[minz:maxz + 1, miny:maxy+1,minx:maxx+1].squeeze())

    if nested==False:
        ds.close()
    else:
        ds[0].close()
        ds[1].close()

    DataOnly=False
    if 'DataOnly' in kwargs:
        DataOnly=kwargs['DataOnly']
    if DataOnly:
        return outdata
    else:
        if HOnly:
            return [gclat[yinds], gclon[xinds], outdata]
        else:
            return [gclat[yinds],gclon[xinds],gcpr[zinds],outdata]



def readGCSO2(Aermsfile,ppbfile,lats,lons,prs,dlat,dlon,dpr,**kwargs):
    nested = False

    if 'Nested' in kwargs:
        nested = kwargs["Nested"]

    if 'zrange' in kwargs:
        [gclat, gclon, SO4data] = readGC(Aermsfile, ['AerMassSO4'], lats, lons, prs, dlat, dlon, dpr, HOnly=True,zrange=kwargs['zrange'],Nested=nested)  # ug/m3
        ppbdata = readGC(ppbfile, ['SpeciesConc_SO4', 'SpeciesConc_SO2'], lats, lons, prs, dlat, dlon, dpr, DataOnly=True, HOnly=True,zrange=kwargs['zrange'],Nested=nested)  # ppb
    else:
        [gclat, gclon, SO4data] = readGC(Aermsfile, ['AerMassSO4'], lats, lons, prs, dlat, dlon, dpr,
                                         HOnly=True,Nested=nested)  # ug/m3
        ppbdata = readGC(ppbfile, ['SpeciesConc_SO4', 'SpeciesConc_SO2'], lats, lons, prs, dlat, dlon, dpr,
                         DataOnly=True, HOnly=True,Nested=nested)  # ppb


    SO4mass=SO4data[0]  #ug/m3
    SO4ppb=ppbdata[0]
    SO2ppb=ppbdata[1]
    SO2mass=SO4mass/96.*SO2ppb/SO4ppb*64. #ug/m3
    DataOnly = False
    if 'DataOnly' in kwargs:
        DataOnly = kwargs['DataOnly']
    if DataOnly:
        return [SO4mass,SO2mass]
    else:
        return [gclat,gclon,SO4mass,SO2mass]