import xarray as xr
import glob

#load ensemble data
for var in ['LandsatPlanet']:
    lspl = {}
    ens_path = f'/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/ens_{var}.nc'

    for model in ['noah','grfr','wbm','gfdl']:
        files = glob.glob(f'/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/DA_merged/*{model}.nc')
        ds = [xr.open_dataset(file) for file in files]
        ds = xr.concat(ds, dim='reach')
        lspl[model] = ds[var]

    # Combine datasets along a new dimension representing ensemble members
    ensemble = xr.concat([lspl['noah'],lspl['grfr'],lspl['wbm'],lspl['gfdl']], dim='ensemble')
    ensemble.coords['ensemble'] = ['noah','grfr','wbm','gfdl']
    #ensemble.to_netcdf('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/output/advances/all_models_xr.nc')
    
#Free up resources
del lspl, ds

#Calculate mean annual stream flow 
ensemble_ds = ensemble.sel(time=slice('2004-01-01','2019-12-31'))
#mean_ens = mean_streamflow.groupby('time.year').sum('time')*24*3600/1e9#total annual, km3/yr
#mean_ens = mean_ens.where(mean_ens != 0)
ensemble_ds = ensemble_ds.to_dataset(name='discharge')
hmadd = ensemble_ds.mean('ensemble')#
hmadd['discharge_std'] = ensemble_ds.std('ensemble').discharge

hmadd.to_netcdf('./output/advances/hmadd.nc')