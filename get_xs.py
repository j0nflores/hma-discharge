#Function to generate cross-section points from river centerlines

import numpy as np
import pandas as pd
import geopandas as gpd


#get xs for all lines from the input shape file
def get_shape_xs(input_path,output_path,n_xs,ids=None):
    '''n_xs = number of desire cross-sections'''
    df_all = gpd.GeoDataFrame()
    shape_df = gpd.read_file(input_path)
    if ids:
        shape_df = shape_df[shape_df.COMID.isin(ids)]
    else:
        pass
    shape_df.mwth_mean = shape_df.mwth_mean.apply(lambda x: 10 if x == -1 else x)
    for i in range(len(shape_df)):
        df_line = line_xs(shape_df.iloc[i],n_xs)
        df_all = pd.concat([df_all,df_line])
    df_all.crs = shape_df.crs
    df_all = df_all.reset_index()
    df_all = df_all.rename(columns = {'index': 'xs_n'})
    df_all['xsID'] = df_all.COMID.astype(str) + '_' + df_all.xs_n.astype(str)
    df_all = df_all.reset_index()
    df_all = df_all.rename(columns = {'index': 'mindex'})
    df_all.to_file(output_path)
    print('Done.')

#get a number of xs for a single reach
def line_xs(line,n_xs):
    distances = np.linspace(0, line.geometry.length, n_xs)
    points = [line.geometry.interpolate(distance) for distance in distances]
    df = gpd.GeoDataFrame(line).T
    df = pd.concat([df.iloc[[-1]*n_xs]])
    df = df.reset_index(drop=True)
    df.geometry = gpd.GeoSeries(points)
    df = df.infer_objects()
    return df


#comid_list = [46045476,44001563] 
n = 30 #number of cross-sections
merit_path = r'C:\Users\fa002\OneDrive - University of Massachusetts\Git\hma-discharge\data\merit\merit20.shp'
out_path = './data/merit/merit20_xs.shp'
get_shape_xs(merit_path,out_path,n)
