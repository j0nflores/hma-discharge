#glacier melt proportion downstream assuming no change in storage/time

import pandas as pd
import geopandas as gpd
import json
import os

#list downstream reach ids
def get_down(comid):
    down_all = [int(comid)]  # List to store all COMIDs downstream
    index = 0
    while index < len(down_all):
        current_comid = down_all[index]  
        down_id = gdf_riv[gdf_riv['COMID'] == current_comid]['NextDownID']  # Get 'NextDownID' for the current COMID
        if not down_id.empty:
            down_temp = down_id.values[0]  # Get the first 'NextDownID' value
            while down_temp != 0:
                if down_temp not in down_all:  # Only add to down_all if it's not already there
                    down_all.append(int(down_temp))  # Append the current downstream COMID to the list
                    down_id = gdf_riv[gdf_riv['COMID'] == down_temp]['NextDownID']  # Get the next 'NextDownID'
                    if not down_id.empty:
                        down_temp = down_id.values[0]  
                    else:
                        break  
                else:
                    break  
        index += 1 
    return down_all

if __name__ == '__main__':
    
    rgi = gpd.read_file('/nas/cee-water/cjgleason/jonathan/data/GIS/HMA_RGI.shp')
    gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')

    melt_riv = pd.read_csv('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/alldf_melt.csv')
    melt_riv['melt_mean_annual'] = melt_riv['melt_mean_annual'].apply(lambda x: 0 if x<0 else x)

    #river downstream per glacier catchment
    if not os.path.exists('./data/rivdown.json'):
        #list downstream reach ids of glacierized catchments
        melt_all = {}
        for comid in melt_riv.COMID:
            melt_all[int(comid)] = get_down(comid)

        # Serialize to JSON and save
        json_data = json.dumps(melt_all)
        with open("./data/rivdown.json", "w") as file:
            file.write(json_data)    #downstream rivers per source catchment 
    else:
        with open("./data/rivdown.json", "r") as file:
            melt_all = json.load(file)


    #get annual melt per reach
    all_melt_df = pd.DataFrame()
    for comid in melt_riv.COMID:
        main_df = pd.DataFrame(melt_all[str(comid)])
        main_df.columns = ['COMID']
        main_df['melt_annual'] = melt_riv[melt_riv.COMID ==comid]['melt_mean_annual'].values[0]
        all_melt_df = pd.concat([all_melt_df,main_df])
    all_melt_df['melt_annual'] = all_melt_df['melt_annual'] /1e9
    all_melt_df = all_melt_df.groupby('COMID').sum().reset_index()

    #calculate melt proportion of reaches based total melt at outlet
    outlets = [46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223,46015164]
    pmelt_df = pd.DataFrame()
    for i, outlet in enumerate(outlets):
        ids = pd.read_csv(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{outlet}/hrrid.csv').COMID
        xdf = all_melt_df.copy()
        xdf = xdf[xdf.COMID.isin(ids)]
        melt_tot = xdf[xdf.COMID==outlet].melt_annual.values[0]
        xdf['basin'] = outlet
        xdf['%melt'] = xdf.apply(lambda x: x['melt_annual']/melt_tot, axis=1)
        pmelt_df = pd.concat([pmelt_df,xdf])

    gdf_riv[gdf_riv.COMID.isin(all_melt_df.COMID)].merge(pmelt_df, how='left', on= 'COMID').to_file('./data/melt_rivdown.shp')
    
    
    
    #melt proportion downstream
    melt_rivdown = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/data/melt_rivdown.shp')
    melt_rivdown = melt_rivdown[['COMID','%melt']]

    #annual melt data from glacierized reach catchments
    dm = xr.open_dataset('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/melt/melt_rivups.nc')

    outlets = [46015164,46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223]

    for i,basin in enumerate(outlets): 

        ids = pd.read_csv(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{basin}/hrrid.csv').COMID
        melt_rivdown_basin = melt_rivdown[melt_rivdown.COMID.isin(ids)]

        melt_valid_ids = [idx for idx in ids if idx in dm.COMID.values]

        temp_ts = xr.Dataset()
        temp_ts['melt'] = xr.Dataset.from_dataframe(melt_rivdown_basin.set_index('COMID'))['%melt'] * dm.sel(COMID=melt_valid_ids).sum('COMID')['melt'] / 1e9
        if i == 0:
            meltdown_ts = temp_ts.copy()
        else:
            meltdown_ts = xr.concat([meltdown_ts,temp_ts],dim='COMID')

    meltdown_ts.to_netcdf('./output/melt/annual_melt_rivers.nc')