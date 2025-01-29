import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import geopandas as gpd
import multiprocessing as mp
import numpy as np

def mprun(catchment_ID):
    
    model = 'grfr' #'noah' #

    print(f'Processing {model} - {catchment_ID}...')

    #catchment_ID = outlet #s[slurm] #
    reach = catchment_ID
    pfaf = str(catchment_ID)[0:2] 

    #DA
    DA_path = f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/{model}/{catchment_ID}/da/output/c'
    outfilepath = os.path.join(DA_path)
    infilepath = outfilepath
    qr = const_q_all(infilepath)

    #plot multiyear hydrograph
    s = '2004-1-1'
    e = '2019-12-31'

    hrrr = f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{catchment_ID}/hrrid.csv'
    hrrr = pd.read_csv(hrrr)
    hrrid = hrrr[hrrr.COMID==reach].HRRID.values[0]

    #load river vector
    gdf = gpd.read_file(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{catchment_ID}/gis/{catchment_ID}.shp')

    #plot seasonal average results
    #iyears = [2004,2005,2006,2007,2008,2009]
    seasons = ['MAM','JJA','SON']
    #for iyear in iyears:
    for i, year in enumerate(np.arange(2004,2019+1)):
    #spring (March-April-May), summer (June-July-August), and fall (September-October-November)     

        s = f'{year}-1-1'
        e = f'{year}-12-31'

        for season in seasons:

            if (i == 0) & (season=='MAM'):
                #qdf = qr.sel(time=slice(s,e)).mean(dim='time').DA.to_dataframe().reset_index().rename(columns={'reach':'HRRID'})
                qdf = qr.DA.sel(time=slice(s,e)).groupby('time.season').mean('time').sel(season=season).to_dataframe().reset_index()
                qdf = qdf.rename(columns={'reach':'HRRID','DA': f'DA_{year}_{season}'}).drop(['season'],axis=1)
                qdf = qdf.merge(hrrr,how='right',on='HRRID').drop(['WID'],axis=1)
                #qdf = qdf.rename(columns={'DA':f'DA_{year}'})
                main = qdf#.copy
                #print(main)
            else:
                qdf = qr.DA.sel(time=slice(s,e)).groupby('time.season').mean('time').sel(season=season).to_dataframe().reset_index()
                qdf = qdf.rename(columns={'reach':'HRRID','DA': f'DA_{year}_{season}'}).drop(['season'],axis=1)
                qdf = qdf.merge(hrrr,how='right',on='HRRID').drop(['HRRID','WID'],axis=1)
                main = main.merge(qdf,how='right',on='COMID')
    main = main.reindex(sorted(main.columns), axis=1)
    main.to_csv(f'./figs/{catchment_ID}/main_{model}.csv')
            
def getFile(filepath, eNum, ncFile):
    pathdir = os.path.join(filepath, "{0:02d}")
    file = os.path.join(pathdir.format(eNum), ncFile)
    #print(file)
    return file

def const_q_all(filepath):
    eTot = 20
    daFile = 'dischargeAssim.nc'
    olFile = 'discharge.nc'
    for  eNum in range(eTot):
        q_da = xr.open_dataset(getFile(filepath,eNum,daFile),engine='netcdf4')#,chunks = {'time': 10})
        q_ol = xr.open_dataset(getFile(filepath,eNum,olFile),engine='netcdf4')#,chunks = {'time': 10})
        #print(q_da)
        if eNum == 0:
            ems_mean_da = q_da
            ems_mean_ol = q_ol
        else:
            ems_mean_da = q_da + ems_mean_da
            ems_mean_ol = q_ol + ems_mean_ol 
        
    hrr = xr.open_dataset(getFile(filepath,0,olFile),engine='netcdf4')#,chunks = {'time': 10})
    q = hrr['discharge']
    qq = q.to_dataframe()
    #print(qq.head())
        
    q_all = hrr
    #q_all['HRR'] = hrr['discharge']
    q_all['openloop'] = ems_mean_ol['discharge']/eTot
    q_all['DA'] = ems_mean_da['discharge']/eTot
    q_all['baseline'] = hrr['discharge']
    return q_all

if __name__ == "__main__": 

    outlets = [46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223] #catchment_IDs
    basin_name = ['AmuDarya','Indus','Ganges/Brahmaputra','Irawaddy','Salween','Mekong','Yangtze','Yellow'] 
    
    with mp.Pool(len(outlets)) as p:
        p.map(mprun, outlets)
            