import geopandas as gpd
import json
import pandas as pd
import xarray as xr
import os
import glob
import numpy as np
from functions import prewhiten, mann_kendall_trend_test_xr
from functions import calculate_percent_change_per_year


if __name__ == '__main__':

    #load datasets
    gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
    alldf_dams = pd.read_csv('./output/alldf_dams.csv') #dams
    ds_p = xr.open_dataset('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/melt/annual_precip_rivers.nc')
    ds_m = xr.open_dataset('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/melt/annual_melt_rivers_old.nc')
    ds_merged = xr.merge([ds_m,ds_p],join='left')

    #calculate gmp ratio
    ds_merged['precip_km3'] = ds_merged['precip'] * ds_merged['unitarea']/1e6
    ds_merged['gp_ratio'] = ds_merged['melt']/ds_merged['precip_km3']
    ds = ds_merged[['gp_ratio']]


    #prep dataframe
    var = 'gp_ratio'
    mean_annual = ds.mean('time').to_dataframe().reset_index()
    mean_annual.columns = ['COMID','melt_mean_annual']

    mean_annual_std = ds.std('time').to_dataframe().reset_index()
    mean_annual_std.columns = ['COMID','melt_mean_annual_std']

    # %change, slope
    change_slope = calculate_percent_change_per_year(ds,'COMID',var)

    # MK trend test
    trend_test_results = mann_kendall_trend_test_xr(ds[var],'COMID','time')
    trendstat = trend_test_results.to_dataframe().reset_index().rename(columns={'reach':'COMID'})

    dfs = [mean_annual, mean_annual_std, change_slope, trendstat]

    alldf = dfs[0]
    for df in dfs[1:]:
        alldf = pd.merge(alldf, df, how='left', on='COMID')
    alldf.merge(alldf_dams, how='left',on='COMID').to_csv(f'./output/alldf_{var}.csv')

    
    #Export trends and dataframe
    exportdf = gdf_riv[['COMID','geometry']].merge(alldf,on='COMID')
    exportdf.to_file('./output/alldf_gpratio.shp')