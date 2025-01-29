import xarray as xr
import pandas as pd
import glob
import multiprocessing as mp


def model_valprep(model):    
    syear='2003-02-01'
    eyear='2019-12-31'
    
    if model == 'wbm':
        eyear='2018-12-31'

    #load gauge data
    val_merit = pd.read_csv('/nas/cee-water/cjgleason/jonathan/data/gauge/val_merit_rev.csv')
    gauge = xr.open_dataset('/nas/cee-water/cjgleason/jonathan/data/gauge/hma_gauge.nc')

    files = glob.glob(f'/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/DA_merged/*{model}.nc')
    ds_concat = [xr.open_dataset(file) for file in files]
    ds_concat = xr.concat(ds_concat, dim='reach')


    gaugex = gauge.sel(date=slice(syear,eyear))#.dropna(dim='date')
    reach = val_merit.COMID.values
    a = ds_concat.sel(reach=ds_concat['reach'].isin(val_merit.COMID.values),time=gaugex.date).drop('time')
    a.to_netcdf(f'/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/{model}.nc')
    
    ds_concat.close()
    
models = ['grfr','wbm','noah','gfdl']

'''with mp.Pool(len(models)*3) as p:
    p.map(model_valprep,models)'''

for model in models:
    model_valprep(model)
    