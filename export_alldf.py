#Calculate and export annual trend analysis of discharge and 
#power change per year for individual reach
#also plots cdf

import xarray as xr
import pandas as pd
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
from functions import prewhiten, mann_kendall_trend_test_xr
from functions import calculate_percent_change_per_year
    
if __name__ == '__main__':
    
    ##DISCHARGE ###########################################
    var = 'LandsatPlanet'

    #load ensemble means data
    ens_path = f'/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/ens_{var}.nc'
    mean_ensemble_by_coordinate_id = xr.open_dataset(ens_path).sel(time=slice('2004-01-01','2019-12-31'))

    #all time mean 
    mean_streamflow_all = mean_ensemble_by_coordinate_id.mean('time') #daily mean, m3/s
    mean_all = mean_streamflow_all.to_dataframe().reset_index()
    mean_all.columns = ['COMID','mean_flow']

    #annual mean and std & convert to km3
    mean_streamflow = (mean_ensemble_by_coordinate_id.groupby('time.year').sum('time')*24*3600/1e9)  #total annual, km3/yr
    mean_annual = mean_streamflow.mean('year').to_dataframe().reset_index()
    mean_annual.columns = ['COMID','mean_annual']

    mean_annual_std = mean_streamflow.std('year').to_dataframe().reset_index()
    mean_annual_std.columns = ['COMID','mean_annual_std']

    # %change, slope
    change_slope = calculate_percent_change_per_year(mean_streamflow)

    # MK trend test
    trend_test_results = mann_kendall_trend_test(mean_streamflow[var])
    trendstat = trend_test_results.to_dataframe().reset_index().rename(columns={'reach':'COMID'})


    ##POWER ###########################################
    var = 'power'
    gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
    df = gdf_riv[['COMID','slope']].rename(columns={'COMID':'reach'})#.set_index('reach')
    slope_ds = xr.DataArray(data=df['slope'].values, dims=['reach'], coords={'reach': df['reach'].values})
    mean_ensemble_by_coordinate_id['slope'] = slope_ds
    mean_ensemble_by_coordinate_id['power'] = 1e3*9.8*mean_ensemble_by_coordinate_id['slope']*mean_ensemble_by_coordinate_id['LandsatPlanet']

    #mean 
    power_mean_streamflow_all = mean_ensemble_by_coordinate_id[['power']].mean('time') #daily mean, W/m
    power_mean_all = power_mean_streamflow_all.to_dataframe().reset_index()
    power_mean_all.columns = ['COMID','power_mean_flow']

    #annual mean std
    power_mean_streamflow = mean_ensemble_by_coordinate_id[['power']].groupby('time.year').mean('time') #annual mean, W/m
    power_mean_streamflow_std = mean_ensemble_by_coordinate_id[['power']].groupby('time.year').std('time')

    power_mean_annual = power_mean_streamflow.mean('year').to_dataframe().reset_index()
    power_mean_annual.columns = ['COMID','power_mean_annual']

    power_mean_annual_std = power_mean_streamflow_std.std('year').to_dataframe().reset_index()
    power_mean_annual_std.columns = ['COMID','power_mean_annual_std']

    # %change, slope
    power_change_slope = calculate_percent_change_per_year(power_mean_streamflow,var=var)

    # MK trend test
    power_trend_test_results = mann_kendall_trend_test(power_mean_streamflow[var])
    power_trendstat = power_trend_test_results.to_dataframe().reset_index().rename(columns={'reach':'COMID'})

    dfs = [mean_all, mean_annual, mean_annual_std, change_slope, trendstat, 
           power_mean_all, power_mean_annual,power_mean_annual_std, power_change_slope, power_trendstat]

    alldf = dfs[0]
    for df in dfs[1:]:
        alldf = pd.merge(alldf, df, how='left', on='COMID')
    alldf.to_csv('./output/alldfx.csv')

    
    #plot cdfs
    outlets = [46015164,46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223]
    basinname = ['SyrDarya','AmuDarya','Indus','Ganges-Brahmaputra','Irawaddy','Salween','Mekong','Yangtze','Yellow'] 
    var_plot = 'percent_change_per_year_x'
    #gdf_all = gdf_riv.merge(alldf[['COMID',var_plot,'trend_x','trend_y']],on='COMID')

    for i,outlet in enumerate(outlets): #@= outlets[0]
        ids = pd.read_csv(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{outlet}/hrrid.csv').COMID
        basindf = alldf[alldf.COMID.isin(ids)].copy() #&(gdf_all.trend_x=='Significant')
        lab = f'{basinname[i]} (n={len(basindf)})'
        sns.ecdfplot(basindf[var_plot], linestyle='-',label=lab)
        plt.xlabel(var_plot)
        plt.ylabel('ECDF')
        plt.grid(True)
        plt.xlim(-5,20)

    plt.legend()
    plt.savefig('./figs/trend/cdf_all.jpg',dpi=120)
    plt.show()