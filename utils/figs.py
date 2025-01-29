import xarray as xr
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd
import os
import geopandas as gpd
import matplotlib as mpl
import matplotlib.dates as mdates

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

    #load dataset
    model = 'grfr' #'noah'
    outlets = [46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223] 
    basin_name = ['AmuDarya','Indus','Ganges/Brahmaputra','Irawaddy','Salween','Mekong','Yangtze','Yellow'] 
    slurm = 4 #int(os.environ['SLURM_ARRAY_TASK_ID'])
    catchment_ID = outlets[slurm] #
    reach = catchment_ID
    pfaf = str(catchment_ID)[0:2] 

    #output path
    out_path = f'./figs/{catchment_ID}'
    os.makedirs(out_path, exist_ok=True)

    #DA
    DA_path = f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/{model}/{catchment_ID}/da/output/c'
    outfilepath = os.path.join(DA_path)
    infilepath = outfilepath
    qr = const_q_all(infilepath)

    #DA noah
    model = 'noah' #'noah'
    DA_path = f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/{model}/{catchment_ID}/da/output/c'
    outfilepath = os.path.join(DA_path)
    infilepath = outfilepath
    qr_noah = const_q_all(infilepath)

    #GRFR-hydroDL, dongmei
    gdl = xr.open_dataset('/work/pi_cjgleason_umass_edu/jonathan/GRADES_hydroDL/output_pfaf_04_1979_2023.nc')
    dfr = xr.open_dataset(f'/nas/cee-water/cjgleason/jonathan/data/HiMAT/GRADES/DA_results_all/pfaf{pfaf}_all.nc')


    #plot multiyear hydrograph
    s = '2004-1-1'
    e = '2019-12-31'

    hrrr = f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{catchment_ID}/hrrid.csv'
    hrrr = pd.read_csv(hrrr)
    hrrid = hrrr[hrrr.COMID==reach].HRRID.values[0]

    da = qr.sel(reach=hrrid,time =slice(s,e)).DA
    da_noah = qr_noah.sel(reach=hrrid,time =slice(s,e)).DA

    base = qr.sel(reach=hrrid,time =slice(s,e)).baseline
    base_noah = qr_noah.sel(reach=hrrid,time =slice(s,e)).baseline

    grdl= gdl.sel(rivid=reach,time=slice(s,e)).Qout
    df = dfr.sel(COMID=reach,time=slice(s,e)).discharge


    fig, axs = plt.subplots(2, 2,figsize=(40,10),dpi=200 , squeeze=True)
    #plt.rcParams.update({'font.size': 80})
    axs = axs.ravel()

    
    #axs[0, 0].plot(x, y)
    #plt.plot(base.time,base,'-k',linewidth=.75)
    axs[0].plot(base.time,base,'-g',linewidth=.75)
    axs[0].plot(base_noah.time,base_noah,'b',linewidth=.75)
    #plt.plot(lat.time,lat,'g',linewidth=.75)
    axs[0].plot(df.time,df,'m',linewidth=.75)
    axs[0].plot(grdl.time,grdl,'r',linewidth=.75)
    axs[0].plot(da.time,da, '--g',linewidth=.75)
    axs[0].plot(da_noah.time,da_noah,'--b',linewidth=.75) 

    #plt.legend(['HRR Baseline VIC (GRFR)','HRR Baseline (NOAHMP)','Dongmei GRADES','GRFR-hydroDL','HRR+DA Landsat (GRFR)','HRR+DA Landsat (NOAHMP)']) 
    #plt.title(label=f'{basin_name[slurm]} Basin Outlet, COMID: {reach}')
    #axs[0].set_ylabel('Discharge, m3/s')
    axs[0].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    axs[0].tick_params(axis='both', which='major', labelsize=24)


    #plot hydrographs by year
    years = [2004,2010,2019]

    for i, year in enumerate(years):
        print(axs[i+1])
        s = f'{year}-1-1'
        e = f'{year}-12-31'

        hrrr = f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{catchment_ID}/hrrid.csv'
        hrrr = pd.read_csv(hrrr)
        hrrid = hrrr[hrrr.COMID==reach].HRRID.values[0]

        da = qr.sel(reach=hrrid,time =slice(s,e)).DA
        da_noah = qr_noah.sel(reach=hrrid,time =slice(s,e)).DA

        base = qr.sel(reach=hrrid,time =slice(s,e)).baseline
        base_noah = qr_noah.sel(reach=hrrid,time =slice(s,e)).baseline

        grdl= gdl.sel(rivid=reach,time=slice(s,e)).Qout
        df = dfr.sel(COMID=reach,time=slice(s,e)).discharge

        lw = 1
        
        axs[i+1].plot(df.time,df,'m',linewidth=lw)
        axs[i+1].plot(grdl.time,grdl,'r',linewidth=lw)
        axs[i+1].plot(base.time,base,'-g',linewidth=lw)
        axs[i+1].plot(base_noah.time,base_noah,'b',linewidth=lw)
        axs[i+1].plot(da.time,da, '--g',linewidth=lw)
        axs[i+1].plot(da_noah.time,da_noah,'--b',linewidth=lw) 
        #axs[i+1].set_ylabel('Discharge, m3/s')
        axs[i+1].yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
        axs[i+1].tick_params(axis='both', which='major', labelsize=24)
        
        #plt.setp(axs[i+1].get_xticklabels(), rotation=45)
        
        '''axs[i+1].set_xticks(df.time[::120])
        date_format = mpl.dates.DateFormatter('%Y-%m-%d')  # Customize the date format as needed
        axs[i+1].xaxis.set_major_formatter(date_format)'''

        '''first_day_of_month = df.time.where(df.time.dt.day == 1, drop=True)
        axs[i+1].set_xticks(first_day_of_month)
        date_format = mdates.DateFormatter('%Y-%m-%d')
        axs[i+1].xaxis.set_major_formatter(date_format)'''
        
        '''# Extract the first day of each month
        first_day_of_month = df.time.where(df.time.dt.day == 1).unique()
        interval = 3
        interval_first_day_of_month = first_day_of_month[::interval]
        axs[i+1].set_xticks(interval_first_day_of_month)
        date_format = mdates.DateFormatter('%Y-%m-%d')
        axs[i+1].xaxis.set_major_formatter(date_format)'''
        
        axs[i+1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axs[i+1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

        
        #if i == 0:
            #axs[i+1].legend(['GRADES+Landsat DA','GRFR-hydroDL','HRR Baseline (GRFR)','HRR Baseline (NOAHMP)','HRR+Landsat DA (GRFR)','HRR+Landsat DA (NOAHMP)'],fontsize='large')
        
    plt.tight_layout()
    plt.savefig(f'{out_path}/{catchment_ID}.jpg')