import pandas as pd
import geopandas as gpd

def get_trend(x):
    if ((x['trend_y'] == 'Significant') & (x['percent_change_per_year_y'] > 0)):
        return 'Increasing' 
    elif ((x['trend_y'] == 'Significant') & (x['percent_change_per_year_y'] < 0)):
        return 'Decreasing'
    else:
        return 'Not Significant'
    
#Discharge, streampower trend tests 
alldf = pd.read_csv('./output/alldf.csv')

#Merge dams dataframe to discharge trends
dams = pd.read_csv('./output/alldf_dams.csv')
dams_merged = alldf.merge(dams,how='left', on='COMID')
dams_merged['power_trend'] = dams_merged.apply(get_trend,axis=1)
dams_merged = dams_merged[['COMID', 'mean_flow', 'mean_annual', 'mean_annual_std',
       'percent_change_per_year_x', 'slope_x', 'p_value_x',
       'power_mean_flow', 'power_mean_annual', 'power_mean_annual_std',
       'percent_change_per_year_y', 'slope_y', 'p_value_y', 'trend_y', 'infra',
       'power_trend']]

#Precip trend test
precip_df = pd.read_csv('./output/advances/alldf_precip_km3.csv')#,index_col='COMID')
precip_df.columns = ['COMID', 'precip_mean', 'precip_std',
       'precip_change', 'precip_slope', 'precip_p']

#Glacier melt trend test
melt_df = pd.read_csv('./output/advances/alldf_melt.csv')
melt_df.columns = ['COMID', 'melt_mean', 'melt_std',
       'melt_change', 'melt_slope', 'melt_p']

#GMP ratio trend test
df_ratio = pd.read_csv('./output/advances/alldf_gmp_ratio.csv')
df_ratio.columns = ['COMID', 'gmp_mean', 'gmp_std',
       'gmp_change', 'gmp_slope', 'gmp_p']

#Basin name attribute
outlets = [46015164,46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223]
basinname = ['1SyrDarya','2AmuDarya','3Indus','4GBM','5Irrawaddy','6Salween','7Mekong','8Yangtze','9Yellow']
basins = []
for i, outlet in enumerate(outlets):
    ids = pd.read_csv(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{outlet}/hrrid.csv')#.COMID
    ids['basin'] = basinname[i]
    ids = ids[['COMID','basin']]
    basins.append(ids)
basins = pd.concat(basins)

#Ensemble models trend uncertainty
mk_shp = gpd.read_file('./output/advances/mk.shp')
mk_shp = mk_shp[['COMID','p','p_std',
                 'ci_low','ci_up','mk_se_up','mk_se_low']]

#Merge and prepare results dataframe for export
resultdf = dams_merged.merge(df_ratio,how='left', on='COMID')
resultdf = resultdf.merge(precip_df,how='left', on='COMID')
resultdf = resultdf.merge(melt_df,how='left', on='COMID')
resultdf = resultdf.merge(basins,how='left', on='COMID')
resultdf = resultdf.merge(mk_shp,how='left', on='COMID')


#Fill NA 
resultdf[['percent_change_per_year_y', 'gmp_ratio_mean', 'gmp_ratio_std',
       'gmp_ratio_change', 'gmp_ratio_slope',  'precip_km3_mean', 'precip_km3_std',
       'precip_km3_change', 'precip_km3_slope', 'melt_mean', 'melt_std',
       'melt_change', 'melt_slope' ]] = resultdf[['percent_change_per_year_y', 'gmp_ratio_mean', 'gmp_ratio_std',
       'gmp_ratio_change', 'gmp_ratio_slope',  'precip_km3_mean', 'precip_km3_std',
       'precip_km3_change', 'precip_km3_slope', 'melt_mean', 'melt_std',
       'melt_change', 'melt_slope' ]].fillna(0)
resultdf[['precip_km3_p','gmp_ratio_p','melt_p']] = resultdf[['precip_km3_p','gmp_ratio_p','melt_p']].fillna(1)
resultdf[['infra']] = resultdf[['infra']].fillna('No dams')

#Export results 
resultdf.set_index('COMID').to_csv('./data/resultdf_rev.csv')

#Export shapefile for mapping
gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
gdf_riv = gdf_riv[gdf_riv.COMID.isin(resultdf.COMID)]
gdf_riv = gdf_riv[['COMID', 'lengthkm', 'slope', 'uparea', 'order', 'strmDrop_t', 'slope_taud', 
                   'mwth_mean', 'mwth_max', 'gwth_mean', 'gwth_max', 'geometry']]
export_df = gdf_riv.merge(resultdf, on='COMID', how='left')
export_df.to_file('./data/resultdf_rev.shp')