import xarray as xr
import pymannkendall as mk
import numpy as np
import geopandas as gpd

hmadd = xr.open_dataset('./output/advances/hmadd.nc')
hmadd_yr = hmadd.groupby('time.year').sum('time')*24*3600/1e9 #annual discharge, km3/yr
hmadd_yr

# Create arrays to store Mann-Kendall test results
trend_results = np.full(len(hmadd.reach), None, dtype=object)
p_values = np.full(len(hmadd.reach), np.nan)
slope = np.full(len(hmadd.reach), np.nan)
change = np.full(len(hmadd.reach), np.nan)

years = hmadd_yr.year

# Apply the Monte Carlo and Mann-Kendall test to each reachID
for i, reach in enumerate(hmadd_yr.reach):
    
    hmadd_yr_unit = hmadd_yr.sel(reach=reach)
    
    # Number of Monte Carlo simulations
    num_sim = 10000

    # Monte Carlo Simulation: Generate 1000 samples per year based on the mean and std
    mc_simulations = np.random.normal(
        loc=hmadd_yr_unit["discharge"].values[:, None], 
        scale=hmadd_yr_unit["discharge_std"].values[:, None], 
        size=(len(years), num_sim)
    )

    # Compute the Ensemble Mean and 95% Confidence Interval
    ensemble_mean_mc = np.mean(mc_simulations, axis=1)
    ci_lower_mc, ci_upper_mc = np.percentile(mc_simulations, [2.5, 97.5], axis=1)

    # Add the computed values back into the dataset
    hmadd_yr_unit["mc_mean"] = (["year"], ensemble_mean_mc)
    #hmadd_yr_unit["mc_ci_lower"] = (["year"], ci_lower_mc)
    #hmadd_yr_unit["mc_ci_upper"] = (["year"], ci_upper_mc)

    # Perform Mann-Kendall Trend Test on the Monte Carlo Ensemble Mean
    mk_result = mk.pre_whitening_modification_test(hmadd_yr_unit["mc_mean"].values)
    
    # Store the results
    trend_results[i] = mk_result.trend
    p_values[i] = mk_result.p
    slope[i] = mk_result.slope
    change[i] = mk_result.slope / hmadd_yr_unit.sel(year=2004).discharge #initial
    
# Add the Mann-Kendall results to the dataset as new variables
hmadd_yr["trend"] = ("reach", trend_results)
hmadd_yr["p_value"] = ("reach", p_values)
hmadd_yr["slope"] = ("reach", slope)
hmadd_yr["change"] = ("reach", change)

hmadd_yr.to_netcdf('./output/advances/hmadd_mk.nc')

#export shp file
hmadd_mkdf = hmadd_yr[['p_value','slope','change']].to_dataframe().reset_index().rename(columns={'reach':'COMID'})
gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
gdf_riv = gdf_riv[gdf_riv.COMID.isin(hmadd_mkdf.COMID)]
export = gdf_riv.merge(hmadd_mkdf, on='COMID', how='left')
export.to_file('./data/hmadd_mkmc.shp')