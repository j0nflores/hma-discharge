#dask implement
import numpy as np
import xarray as xr
import pymannkendall as mk
import dask
from dask.diagnostics import ProgressBar

import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt

hmadd = xr.open_dataset('./output/advances/hmadd.nc')
hmadd_yr = hmadd.groupby('time.year').mean('time') #*24*3600/1e9 #annual discharge, km3/yr


# Simulated Years (16 years)
years = hmadd_yr.year

#dask implement
import numpy as np
import xarray as xr
import pymannkendall as mk
import dask
from dask.diagnostics import ProgressBar

import numpy as np
import pymannkendall as mk
import matplotlib.pyplot as plt

hmadd = xr.open_dataset('./output/advances/hmadd.nc')
hmadd_yr = hmadd.groupby('time.year').mean('time') 

# Define a function to apply Mann-Kendall test and Monte Carlo simulation for each reach
def process_reach(reach_idx, hmadd_yr_unit, years, num_sim=1000):
    # Monte Carlo Simulation: Generate samples based on the mean and std
    mc_simulations = np.random.normal(
        loc=hmadd_yr_unit["discharge"].values[:, None], 
        scale=hmadd_yr_unit["discharge_std"].values[:, None], 
        size=(len(years), num_sim)
    )

    # Compute Ensemble Mean and 95% Confidence Interval
    ensemble_mean_mc = np.mean(mc_simulations, axis=1)
    #ci_lower_mc, ci_upper_mc = np.percentile(mc_simulations, [2.5, 97.5], axis=1)

    # Add the computed values back into the dataset
    #hmadd_yr_unit["mc_mean"] = (["year"], ensemble_mean_mc)
    
    # Perform Mann-Kendall Trend Test on the Monte Carlo Ensemble Mean
    mk_result_mean = mk.pre_whitening_modification_test(ensemble_mean_mc)
    
    # Collect trend results and p-values for each Monte Carlo realization
    trend_results = []
    p_values = []

    # Perform Mann-Kendall test on each realization
    for i in range(num_sim):
        mk_result = mk.pre_whitening_modification_test(mc_simulations[:, i])
        trend_results.append(mk_result.trend)
        p_values.append(mk_result.p)
    
    # Determine the direction of the trend across all simulations
    increasing_trends = trend_results.count('increasing')
    decreasing_trends = trend_results.count('decreasing')
    no_trend = trend_results.count('no trend')

    #Extract results    
    trend = mk_result_mean.trend
    p_value = mk_result_mean.p
    slope = mk_result_mean.slope
    change = mk_result_mean.slope * 100 / hmadd_yr_unit.sel(year=2004).discharge #initial

    if (increasing_trends > decreasing_trends) & (increasing_trends > no_trend):
        mc_trend = 'increasing'
        mc_proportion = increasing_trends/num_sim
    elif (decreasing_trends > increasing_trends) & (decreasing_trends > no_trend):
        mc_trend = 'decreasing'
        mc_proportion = decreasing_trends/num_sim
    else:
        mc_trend = 'no trend'
        mc_proportion = no_trend/num_sim

    return trend, p_value, slope, change, mc_trend, mc_proportion

# Define a function to process all reaches in parallel
@dask.delayed
def process_all_reaches(hmadd_yr, years, reach_idx):
    hmadd_yr_unit = hmadd_yr.sel(reach=reach_idx)
    return process_reach(reach_idx, hmadd_yr_unit, years)

# Create delayed tasks for each reach
delayed_tasks = [process_all_reaches(hmadd_yr, years, reach_idx) for reach_idx in hmadd_yr.reach]

# Compute results in parallel
with ProgressBar():
    results = dask.compute(*delayed_tasks)

# Unpack results (returned as a list of tuples for each reach)
trends, p_values, slopes, changes, mc_trends, mc_proportions = zip(*results)
hmadd_yr["trend"] = ("reach", np.array(trends))
hmadd_yr["p_value"] = ("reach", np.array(p_values))
hmadd_yr["slope"] = ("reach", np.array(slopes))
hmadd_yr["change"] = ("reach", np.array(changes))
hmadd_yr["mc_trend"] = ("reach", np.array(mc_trends))
hmadd_yr["mc_proportion"] = ("reach", np.array(mc_proportions))
hmadd_yr.to_netcdf('./output/advances/hmadd_mkmc_dask.nc')


#export shp file
hmadd_mkdf = hmadd_yr[['p_value','slope','change', 'mc_trend', 'mc_proportion']].to_dataframe().reset_index().rename(columns={'reach':'COMID'})
gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
gdf_riv = gdf_riv[gdf_riv.COMID.isin(hmadd_mkdf.COMID)]
export = gdf_riv.merge(hmadd_mkdf, on='COMID', how='left')
export.to_file('./output/advances/hmadd_mkmc_dask.shp')