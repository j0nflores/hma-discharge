import numpy as np
import xarray as xr
import pymannkendall as mk
import geopandas as gpd
import multiprocessing as mp
import pandas as pd

# Load data
hmadd = xr.open_dataset('./output/advances/hmadd.nc')
hmadd_yr = hmadd.groupby('time.year').mean('time') 

# Extract years
years = hmadd_yr.year.values

# Define function for multiprocessing
def process_reach(reach):
    hmadd_yr_unit = hmadd_yr.sel(reach=reach)
    
    # Extract streamflow and standard deviation
    true_streamflow = hmadd_yr_unit["discharge"].values
    streamflow_std = hmadd_yr_unit["discharge_std"].values
    initial_discharge = hmadd_yr_unit.sel(year=2004).discharge.values # for %change calculation 
    
    # Monte Carlo simulations
    num_sim = 1000
    mc_simulations = np.random.normal(loc=true_streamflow[:, None], scale=streamflow_std[:, None], size=(len(years), num_sim))

    trend_results = [mk.pre_whitening_modification_test(mc_simulations[:, i]) for i in range(num_sim)]

    keep = {}
    for j, mk_results in enumerate(trend_results):
        change = mk_results.slope * 100 / initial_discharge
        keep[j] = {'p': mk_results.p, 'slope': mk_results.slope, 
                   'change': change, 'trend': mk_results.trend} #.dir()

    #extract mc sim results
    temp_df = pd.DataFrame(keep).T
    mc_p = temp_df['p'].mean()
    mc_p_se = temp_df['p'].std()/np.sqrt(num_sim)
    mc_slope = temp_df['slope'].mean()
    mc_slope_se = temp_df['slope'].std()/np.sqrt(num_sim)
    mc_change = temp_df['change'].mean() 
    mc_change_se = temp_df['change'].std()/np.sqrt(num_sim)
    mc_trend = temp_df.trend.describe().top
    mc_prop = temp_df.trend.describe().freq/1000 #proportion of most trend

    return reach, mc_p, mc_p_se, mc_slope, mc_slope_se, mc_change, mc_change_se, mc_trend, mc_prop

# Use multiprocessing to process reaches in parallel
with mp.Pool(mp.cpu_count()) as pool:
    results = pool.map(process_reach, hmadd_yr.reach.values)


# Convert results to dictionary
results_dict = {k: np.array(v) for k, v in zip(
    ['reach', 'p_value', 'p_value_se', 'slope', 'slope_se', 
     'change', 'change_se', 'mc_trend', 'mc_proportion'], zip(*results)
)}

hmadd_yr_test = hmadd_yr.sel(reach=hmadd_yr.reach)

# Add results to dataset
for key, value in results_dict.items():
    hmadd_yr[key] = ("reach", value)

# Save to NetCDF
hmadd_yr.to_netcdf('./output/advances/hmadd_mkmc_mp.nc')

#export shp file
hmadd_mkdf = hmadd_yr[['p_value','slope','change', 'mc_trend', 'mc_proportion']].to_dataframe().reset_index().rename(columns={'reach':'COMID'})
gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
gdf_riv = gdf_riv[gdf_riv.COMID.isin(hmadd_mkdf.COMID)]
export = gdf_riv.merge(hmadd_mkdf, on='COMID', how='left')
export.to_file('./output/advances/hmadd_mkmc_mp.shp')