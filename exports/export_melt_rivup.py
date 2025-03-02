##Extract area-weighted glacier melt of intersecting upstream MERIT subcatchments and RGI glaciers
##Trend analysis and change per year of glacier melt

import geopandas as gpd
import json
import pandas as pd
import xarray as xr
import os
import glob
import numpy as np
from scipy.stats import linregress, norm, theilslopes
from functions import mann_kendall_trend_test_xr, calculate_percent_change_per_year

if __name__ == '__main__':
    
    rgi = gpd.read_file('/nas/cee-water/cjgleason/jonathan/data/GIS/HMA_RGI.shp')
    gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
    gdf_catch = gpd.read_file('/nas/cee-water/cjgleason/jonathan/data/MERIT/cat_pfaf_4_MERIT_Hydro_v07_Basins_v01.shp')

    with open('./output/riv20_ids.json') as f:
        riv20 = json.load(f)

    outlets = [46015164,46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223]
    basinname = ['SyrDarya','AmuDarya','Indus','Ganges/Brahmaputra','Irawaddy','Salween','Mekong','Yangtze','Yellow'] 

    ## Extract area-weighted glacier melt of intersecting upstream MERIT subcatchments per basin
    if not os.path.exists('./output/melt_precip/melt_rivups.nc'):
        
        for b,basin in enumerate(outlets): 

            ids = pd.read_csv(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{basin}/hrrid.csv').COMID
            rivcatch = gdf_catch[gdf_catch.COMID.isin(ids)].copy() ##.plot()
            rivcatch.crs = gdf_riv.crs

            # Load the shapefiles
            gdf1 = rivcatch.copy()
            gdf2 = rgi.copy()

            # Ensure both GeoDataFrames use the same CRS
            if gdf1.crs != gdf2.crs:
                gdf2 = gdf2.to_crs(gdf1.crs)

            # Reproject to the Asia North Albers Equal Area Conic projection
            #https://spatialreference.org/ref/esri/102025/
            gdf1 = gdf1.to_crs('ESRI:102025') #asia_albers_crs
            gdf2 = gdf2.to_crs('ESRI:102025')

            # Perform the spatial join to get intersecting polygons
            sjoin_df = gpd.sjoin(gdf1, gdf2, how='inner', predicate='intersects')

            # Compute the intersection geometries
            sjoin_df['intersection'] = sjoin_df.apply(
                lambda row: row['geometry'].intersection(gdf2.loc[row['index_right'], 'geometry']), axis=1)

            # Calculate the area of MERIT and RGI intersections in m2 and m3
            sjoin_df['intersection_area_m2'] = sjoin_df['intersection'].area
            sjoin_df['intersection_area_km2'] = sjoin_df['intersection_area_m2'] / 1e6
            sjoin_df['%glac_area'] = sjoin_df.apply(lambda x: x['intersection_area_km2']/x['Area'] 
                                                    if x['intersection_area_km2']/x['Area'] <= 1 else 1, axis=1) 
            main_df = sjoin_df[['COMID','RGIId','Area','intersection_area_m2', 'intersection_area_km2', '%glac_area']]

            #Extract melt data for each intersected glaciers
            comids = list(main_df.COMID.unique())
            for i,comid in enumerate(comids): 
                riv_main_df = main_df[main_df.COMID==comid].copy()
                riv_main_df['rgi_reg'] = riv_main_df.apply(lambda x: str(x['RGIId'][6:8]),axis=1)

                for j, rgi_reg in enumerate(riv_main_df.rgi_reg.unique()):
                    melt = xr.open_dataset(f'/nas/cee-water/cjgleason/jonathan/data/HiMAT/Melt/R{rgi_reg}_glac_melt_monthly_aggregated.nc')
                    if j == 0:
                        annual_melt_riv = melt.where(melt.RGIId.isin(riv_main_df.RGIId),drop=True)
                    else:
                        annual_melt_riv = xr.concat([annual_melt_riv, melt.where(melt.RGIId.isin(riv_main_df.RGIId),drop=True)], dim="glacier")

                annual_melt_riv = annual_melt_riv.sel(time=slice('2004-01-01','2019-12-31'))
                annual_melt_riv = annual_melt_riv.mean('model').resample(time='YS').sum().drop_vars('crs')
                combo_df = annual_melt_riv.to_dataframe().reset_index().merge(riv_main_df, on='RGIId')
                combo_df['%glac_melt'] = combo_df['glac_melt_monthly'] * combo_df['%glac_area']
                combo_df = combo_df.groupby('time').sum()['%glac_melt'].reset_index()
                combo_df.time= pd.to_datetime(combo_df.time.astype('str'))

                #concatenate xarray data for export
                if (b == 0)&(i == 0):
                    ds = xr.Dataset(
                    {"melt": (["time", "COMID"], combo_df['%glac_melt'].values.reshape(-1, 1))},
                    coords={"time": combo_df.time, "COMID": [comid]})
                else:
                    ds = xr.concat([ds, xr.Dataset(
                    {"melt": (["time", "COMID"], combo_df['%glac_melt'].values.reshape(-1, 1))},
                    coords={"time": combo_df.time, "COMID": [comid]})], dim="COMID")
        
        #export annual melt data per MERIT reach
        ds.to_netcdf('./output/melt_precip/melt_rivups.nc')

    else:
        ds = xr.open_dataset('./output/melt_precip/melt_rivups.nc')
        
    
    ##export change trends for mapping
    var = 'melt'

    #annual mean and std
    mean_annual = ds.mean('time').to_dataframe().reset_index()
    mean_annual.columns = ['COMID','melt_mean_annual']
    mean_annual_std = ds.std('time').to_dataframe().reset_index()
    mean_annual_std.columns = ['COMID','melt_mean_annual_std']

    # %change, slope
    change_slope = calculate_percent_change_per_year(ds,'COMID',var)

    # MK trend test
    trend_test_results = mann_kendall_trend_test_xr(ds[var],'COMID','time')
    trendstat = trend_test_results.to_dataframe().reset_index().rename(columns={'reach':'COMID'})

    #merge export dataframes and shapefiles
    dfs = [mean_annual, mean_annual_std, change_slope, trendstat]
    alldf = dfs[0]
    for df in dfs[1:]:
        alldf = pd.merge(alldf, df, how='left', on='COMID')
    alldf.to_csv('./output/alldf_melt_rev.csv')
