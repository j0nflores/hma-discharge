import geopandas as gpd

bout = 45070322

ganges = gpd.read_file(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{bout}/gis/{bout}.shp')
nepal = gpd.read_file('/nas/cee-water/cjgleason/jonathan/data/GIS/nepal/Nepal.shp')
nepal = nepal.to_crs(ganges.crs)

nepal_riv = ganges.sjoin(nepal, predicate='intersects')

import xarray as xr
import pandas as pd

ids = pd.read_csv(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/grfr/{bout}/hrrid.csv').COMID

s,e = '2004-01-01', '2019-12-31'
ds = xr.open_dataset('./output/ens_LandsatPlanet.nc')
ds = ds.sel(reach=list(nepal_riv.COMID),time=slice(s,e))
ds = ds.rename({'LandsatPlanet':'discharge'})
ds.to_netcdf('nepal.nc')#,engine='netcdf4')