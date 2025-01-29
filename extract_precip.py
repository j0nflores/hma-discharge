#extract monthly precip data of MERIT subcatchments
#from hourly HIMAT forcing data with exact extract

import os
import glob
import time
import xarray as xr
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio as rio
from datetime import datetime
from exactextract import exact_extract
from functions import get_ds


if __name__ == '__main__':
    
    #get target subcatchment geometry
    melt_rivup_subs_rivids = pd.read_csv('./output/upstream/master.csv').COMID#[0:20000]

    #subcatchments to extract
    gdf_catch = gpd.read_file('/nas/cee-water/cjgleason/jonathan/data/MERIT/cat_pfaf_4_MERIT_Hydro_v07_Basins_v01.shp')
    gdf_catch = gdf_catch.set_crs('EPSG:4326')
    #gdf_catch = gdf_catch[gdf_catch.COMID.isin(melt_rivup_subs_rivids)]

    #year months for nc data
    years = [str(x) for x in np.arange(2004,2020)]
    months = ['01','02','03','04','05','06','07','08','09','10','11','12'] 
    year_months = [year+month for year in years for month in months]
    year_month = year_months[int(os.environ['SLURM_ARRAY_TASK_ID'])]

    #output directories
    output_folder = f'./output/precip_subs2/{year_month}'
    os.makedirs(output_folder,exist_ok=True)

    #load precip data and write temporary nc data (direct call for faster eextract ~1.4s/1000 features)
    nc_temp_folder = f'./output/precip_subs2/nc_temp'
    nc_temp_file = f'{nc_temp_folder}/{year_month}.nc'
    os.makedirs(nc_temp_folder,exist_ok=True)
    if not os.path.exists(nc_temp_file):
        ds = get_ds('/nas/cee-water/cjgleason/jonathan/data/hma_forcing/rev/',year_month)
        if ds.rio.crs != gdf_catch.crs:
            ds = ds.rio.write_crs(gdf_catch.crs)
        ds = ds[['Prec']].sum('time')
        ds.to_netcdf(nc_temp_file)

    #extract data by batch 
    # Define batch size & split the series into batches of size 1000
    batch_size = 1000
    batches = [melt_rivup_subs_rivids[i:i+batch_size] 
               for i in range(0, len(melt_rivup_subs_rivids), batch_size)]

    for i,batch in enumerate(batches):
        
        output_file = f'{output_folder}/{year_month}_{i}.csv'
        
        if not os.path.exists(output_file):

            start_time = time.time()
            gdf = gdf_catch[gdf_catch.COMID.isin(batch)].copy()

            try:
                exact_extract(nc_temp_file, gdf, ["mean"], include_cols = 'COMID', output = "pandas").set_index('COMID').to_csv(output_file)
                print(f'{year_month}, batch {i} done') 
            except:
                print(f'{year_month}, batch {batch} failed') 

            elapsed = time.time() - start_time
            print(elapsed, ' seconds \n')
            #os.remove(nc_temp_file)

        else:
            print(f'{year_month}, batch {batch} already done') 