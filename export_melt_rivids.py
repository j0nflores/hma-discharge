#find upstream network for individual reaches within glacier melt route

import geopandas as gpd
import pandas as pd
import os
from functions import makeHRRID

if __name__ == '__main__':
    
    version = 'glacierized'
    gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
    
    #basin out
    bouts = [46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223,46015164]
    bout = bouts[int(os.environ['SLURM_ARRAY_TASK_ID'])]
    ids = pd.read_csv(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{bout}/hrrid.csv').COMID
        
    if version == 'glacierized':
        
        #load dataset
        melt_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/data/melt_rivdown.shp')
        glac_riv = pd.read_csv('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/alldf_melt.csv').COMID

        #filter only for glacier melt path excluding upstream and outlet, per basin
        melt_riv_filter = melt_riv[~melt_riv.COMID.isin(glac_riv)].copy()
        melt_riv_filter = melt_riv_filter[~melt_riv_filter.COMID.isin(bouts)]
        melt_riv_filter = melt_riv_filter[melt_riv_filter.COMID.isin(ids)]
        melt_riv_filter_ids = list(melt_riv_filter.COMID)
        
        #run upstream connection search
        df = makeHRRID(gdf_riv,melt_riv_filter_ids)
        df.to_csv(f'./output/upstream/melt_rivup_{bout}.csv')
    
    
    else:
        
        #load dataset
        gpratio = pd.read_csv('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/alldf_gp_ratio.csv').COMID
        
        #filter only for glacier melt path excluding upstream and outlet, per basin
        gdf_riv_filter = gdf_riv[~gdf_riv.COMID.isin(gpratio)].copy()
        gdf_riv_filter = gdf_riv_filter[gdf_riv_filter.COMID.isin(ids)]
        gdf_riv_filter_ids = list(gdf_riv_filter.COMID)

        #run upstream connection search
        df = makeHRRID(gdf_riv,gdf_riv_filter_ids)
        df.to_csv(f'./output/upstream/melt_rivup_noglac_{bout}.csv')
    
    


