#Function to generate cross-section points from river centerlines

import numpy as np
import pandas as pd
import geopandas as gpd


#get xs for all lines from the input shape file
def get_shape_xs(input_path,output_path,n_xs=10):
    '''n_xs = number of desire cross-sections'''
    df_all = gpd.GeoDataFrame()
    shape_df = gpd.read_file(input_path)[0:3]
    for i in range(len(shape_df)):
        df_line = line_xs(shape_df.iloc[i],n_xs)
        df_all = pd.concat([df_all,df_line])
    df_all.crs = shape_df.crs
    df_all = df_all.reset_index()
    df_all.to_file(output_path)

#get a number of xs for a single reach
def line_xs(line,n_xs=10):
    distances = np.linspace(0, line.geometry.length, n_xs)
    points = [line.geometry.interpolate(distance) for distance in distances]
    df = gpd.GeoDataFrame(line).T
    df = pd.concat([df.iloc[[-1]*n_xs]])
    df = df.reset_index(drop=True)
    df.geometry = gpd.GeoSeries(points)
    df = df.infer_objects()
    return df

n = 10 #number of cross-sections
merit_path = r'C:\Users\fa002\OneDrive - University of Massachusetts\Git\hma-discharge\data\merit\merit20.shp'
out_path = './data/merit/merit20_xs.shp'
get_shape_xs(merit_path,out_path,n)
