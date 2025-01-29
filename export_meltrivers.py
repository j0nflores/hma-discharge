#extract glacier melt proportion downstream assuming no change in storage/time

import pandas as pd
import geopandas as gpd
import json
import os
import xarray as xr
from functions import get_down

if __name__ == '__main__':
    
    
    melt_rivups = xr.open_dataset('./output/melt_precip/monthly_melt_rivups.nc')
    melt_src = melt_rivups.mean('time').melt/1e9 #convert to km3
 
    #river downstream per glacier catchment
    if not os.path.exists('./data/rivdown.json'):
        
        print('getting downstream rivers...')
        gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
        
        #list downstream reach ids of glacierized catchments
        melt_all = {}
        for comid in melt_src.COMID:
            melt_all[int(comid)] = get_down(comid)

        # Serialize to JSON and export
        json_data = json.dumps(melt_all)
        with open("./data/rivdown.json", "w") as file:
            file.write(json_data)    #downstream rivers per source catchment 
    else:
        with open("./data/rivdown.json", "r") as file:
            melt_all = json.load(file)

    #distribute melt downstream
    print('getting melt downstream...')
    all_melt_df = pd.DataFrame()
    for comid in list(melt_all.keys()):
        print('\t', comid)
        for ym in melt_rivups.time.values:
            temp_df = pd.DataFrame(melt_all[comid])
            temp_df.columns = ['COMID']
            temp_df['time'] = ym
            temp_df['melt'] = melt_rivups.sel(COMID=int(comid),time=ym).melt.values/1e9 
            all_melt_df = pd.concat([all_melt_df,temp_df])
    all_melt_df = all_melt_df.groupby(['COMID','time']).sum()


    #Partition by reaches to total basin melt
    ds_melt = xr.Dataset.from_dataframe(all_melt_df)
    outlets = [46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223,46015164]
    outlet = outlets[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    try:
        print(f'getting melt partition... {outlet}')
        ids = pd.read_csv(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{outlet}/hrrid.csv').COMID
        valid_ids = [comid for comid in ids if comid in ds_melt.COMID]
        temp_ds = ds_melt.sel(COMID=valid_ids)
        temp_ds['perc_melt'] = temp_ds['melt']/temp_ds.sel(COMID=outlet)['melt']
        if i == 0:
            annual_melt_rivers = temp_ds
            print(f'\t basin: {outlet} done')
        else:
            annual_melt_rivers = xr.concat([annual_melt_rivers,temp_ds],dim='COMID')
            print(f'\t basin: {outlet} done')
    except:
        print(f'\t basin: {outlet} error')
    annual_melt_rivers.to_netcdf(f'./output/melt_precip/monthly_melt_{outlet}.nc')
