import geopandas as gpd
import pandas as pd
import glob
import xarray as xr

files = glob.glob('/nas/cee-water/cjgleason/jonathan/data/gauge/#prep/*.csv')

start_date = '2000-01-01'
end_date = '2019-12-31'

main_df = pd.DataFrame()
for file in files:
    print(file)
    df = pd.read_csv(file) #,index_col=0)#, encoding='latin-1')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').resample('D').mean()#.dropna()
    df = df.loc[(df.index >= start_date) & (df.index <= end_date)]
    main_df = pd.concat([main_df,df],axis=1)
    print(len(df),len(df.columns))#.head(2))

'''cfs_list = ['akoora', 'batakoot', 'khanabal', 'verinag', 'muniwar',
       'indus_tarbela', 'kalabagh', 'taunsa', 'sukkur', 'jhelum_mangla',
       'panjnad', 'kabul_nowshera', 'chashma', 'guddu', 'kotri',
       'chenab_marala', 'bahadurabad', 'hardings bridge', 'chatara',
       'tpdc', 'babai', 'tinau', 'dumre khola', 'west rapti', 'chameliya']'''

cfs_list = ['akoora', 'batakoot', 'khanabal', 'verinag', 'muniwar',
       'indus_tarbela', 'kalabagh', 'taunsa', 'sukkur', 'jhelum_mangla',
       'panjnad', 'kabul_nowshera', 'chashma', 'guddu', 'kotri',
       'chenab_marala','bahadurabad', 'chameliya']

main_df[cfs_list] = main_df[cfs_list] * 0.0283168 #convert cfs to cms

# Convert DataFrame to xarray Dataset
ds = xr.Dataset({'discharge': (('date', 'gauge'), main_df.values)},
                coords={'date': pd.to_datetime(main_df.index), 
                        'gauge': main_df.columns})

ds.to_netcdf('/nas/cee-water/cjgleason/jonathan/data/gauge/hma_gauge.nc')