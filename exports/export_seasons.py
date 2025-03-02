import xarray as xr
import pandas as pd
import numpy as np
from scipy.stats import theilslopes #, kendalltau #faster
from functions import calculate_percent_change_per_year
from functions import prewhiten, mann_kendall_trend_test_xr
from functions import seasonal_trend_test

if __name__ == '__main__':
    
    var = 'precip_km3'

    #load nc data
    ds_path = f'./output/melt/monthly_precip_rivers_all.nc'
    ds = xr.open_dataset(ds_path).sel(time=slice('2004-01-01','2019-12-31'))
    ds = ds.rename({'COMID':'reach'})
    #ds = ds.sel(reach=[45035011,43018620,45008653]) #test
    
    seasonal_df = pd.DataFrame()
    for reach in ds.reach.values:
        result = seasonal_trend_test(ds.sel(reach=reach), var=var)
        seasonal_df = pd.concat([seasonal_df,result])

    seasonal_df.to_csv(f'./output/seasonal/seasonal_{var}.csv')

