#calculate and export area-weighted reach precip (mm) with catchment area (km2)
#and monthly glacier/precip ratio per reach as nc files
import os
import pandas as pd
import geopandas as gpd
import glob
import numpy as np
import xarray as xr
from functions import export_trend_test

def get_precip(input_folder,output_file,ref_catchment_fn):
    
    #load extracted precip data per subcatchment from extract_precip.py
    if not os.path.exists(output_file):
        dfs = []
        for year_month in year_months:
            files = glob.glob(f'{input_folder}/{year_month}/{year_month}*.csv')
            df = pd.concat([pd.read_csv(file) for file in files], ignore_index=True).rename(columns={'mean':year_month})
            df_long = df.melt(id_vars=['COMID'], var_name='time', value_name='precip')
            df_long['time'] = pd.to_datetime(df_long['time'], format='%Y%m')
            dfs.append(df_long)
        combined_df = pd.concat(dfs)

        ##get total precip of subbasins with upstream drainage, convert to km3
        bouts = [46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223,46015164]
        combined_df_sub = pd.DataFrame()

        for bout in bouts: #[0]

            print(bout)

            brivs = pd.read_csv(f'{ref_catchment_fn}_{bout}.csv')
            brivs = brivs[['COMID','WID']]

            #compute precip with subcatchment area, km3
            temp_df = combined_df[combined_df.COMID.isin(brivs.COMID.unique())]
            temp_df = temp_df.merge(gdf_catch[gdf_catch.COMID.isin(brivs.COMID.unique())],on='COMID')
            #temp_df['precip'] = temp_df.apply(lambda x: x['precip']/1e6 * x['unitarea'], axis=1)

            #outlet reach
            #area = temp_df.groupby('time').sum().unitarea.values[0]
            outlet_df = temp_df[['COMID','time','precip']].groupby('time').mean()
            outlet_df['unitarea'] = temp_df.groupby('time').sum(numeric_only=True).unitarea.values[0]
            outlet_df['COMID'] = bout 

            #glacierized reaches
            upstream = melt_rivs[melt_rivs.isin(brivs.COMID)]
            upstream_df = temp_df[temp_df.COMID.isin(upstream)]
            upstream_df = upstream_df[['COMID','time','precip','unitarea']].set_index('time')

            #merge the dataframe (aggregation due to earlier preprocessing)
            combined_df_sub = pd.concat([combined_df_sub,upstream_df,outlet_df])

            #reaches with upstream drainage networks
            for w in brivs.WID.unique():
                wids = brivs[brivs.WID==w].COMID
                middle_df = temp_df[temp_df.COMID.isin(wids)]

                if len(middle_df) >= 1:
                    area = middle_df.groupby('time').sum(numeric_only=True).unitarea.values[0]
                    middle_df = middle_df[['COMID','time','precip']].groupby('time').mean()
                    middle_df['COMID'] = brivs[brivs.WID==w].iloc[0].COMID
                    middle_df['unitarea'] = area
                    combined_df_sub = pd.concat([combined_df_sub,middle_df])

        ds = xr.Dataset.from_dataframe(combined_df_sub.reset_index().set_index(['COMID','time']))
        ds = ds.resample(time='MS').sum()
        ds['unitarea'] = xr.DataArray(combined_df_sub.groupby('COMID')['unitarea'].first())
        ds.to_netcdf(output_file)

    else:
        ds = xr.open_dataset(output_file)
    
    return ds

    
if __name__ == '__main__':

    precip_nc = './output/melt/monthly_precip_rivers_all.nc'
    
    if not os.path.exists(precip_nc):
        
        #year months for nc data
        years = [str(x) for x in np.arange(2004,2020)]
        months = ['01','02','03','04','05','06','07','08','09','10','11','12'] 
        year_months = [year+month for year in years for month in months]

        #load subcatchment data
        gdf_catch = gpd.read_file('/nas/cee-water/cjgleason/jonathan/data/MERIT/cat_pfaf_4_MERIT_Hydro_v07_Basins_v01.shp')
        melt_rivs = pd.read_csv('./output/alldf_melt.csv').COMID

        #get precip for glacierized channels
        ds_p1 = get_precip('./output/precip_subs2',
                           './output/melt/monthly_precip_rivers.nc',
                            f'./output/upstream/melt_rivup')

        #get precip for non-glacierized channels
        ds_p2 = get_precip('./output/precip_subs2',
                           './output/melt/monthly_precip_rivers_noglac.nc',
                            f'./output/upstream/melt_rivup_noglac')

        #merge to get precip for all rivers and export
        ds_p = xr.concat([ds_p1,ds_p2],dim='COMID').drop_duplicates(dim='COMID')
        ds_p['precip_km3'] = ds_p['precip'] * ds_p['unitarea']/1e6 #calculate precip volume per reach
        ds_p.to_netcdf(precip_nc)
    
    else:
        ds_p = xr.open_dataset(precip_nc)

    #export_trend_test(ds_p,'precip_km3','./output')
