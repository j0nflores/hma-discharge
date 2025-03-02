import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import contextily as ctx
import glob
import os
import geopandas as gpd

#calculate or load ensemble means data
for var in ['Landsat','Planet']:
    lspl = {}
    ens_path = f'/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/ens_{var}.nc'

    for model in ['noah','grfr','wbm','gfdl']:
        files = glob.glob(f'/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/DA_merged/*{model}.nc')
        ds = [xr.open_dataset(file) for file in files]
        ds = xr.concat(ds, dim='reach')
        lspl[model] = ds[var]

    # Combine datasets along a new dimension representing ensemble members
    ensemble_mean = xr.concat([lspl['noah'],lspl['grfr'],lspl['wbm'],lspl['gfdl']], dim='ensemble')
    # Calculate the mean along the ensemble dimension
    mean_ensemble_by_coordinate_id = ensemble_mean.mean(dim='ensemble')
    mean_ensemble_by_coordinate_id.to_netcdf(f'/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/ens_{var}.nc')
