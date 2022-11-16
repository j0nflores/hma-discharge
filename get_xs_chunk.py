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

    for i in range(7):
        if i < 6:
            merit_out = shape_df.iloc[i*1000:(i+1)*1000]
            dfout = df_all[df_all.COMID.isin(merit_out.COMID)]
        else:
            merit_out = shape_df.iloc[i*1000:]
            dfout = df_all[df_all.COMID.isin(merit_out.COMID)]

        output_path_xs = f'{output_path}merit20_xs{i}.shp'
        output_path_merit = f'{output_path}merit20_{i}.shp'
        dfout.to_file(output_path_xs)
        merit_out.to_file(output_path_merit)

    #df_all.to_file()
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
out_path = './data/merit/'
get_shape_xs(merit_path,out_path,n)
