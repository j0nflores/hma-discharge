#Functions used for analysis

import os
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from statsmodels.tsa.stattools import ccf
from scipy.stats import norm, linregress, theilslopes, kendalltau
import matplotlib.pyplot as plt
from matplotlib import rcParams

#plot font settings
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Palatino'] + rcParams['font.serif']

#preprocess multiple nc files
def get_ds(path,year_month):
    def add_time_dim(xda):
        '''creates arbitrary dates to time dimension'''
        xda = xda.expand_dims(time = [datetime.now()])
        return xda
    filenames = glob.glob(f'{path}/{year_month[0:4]}/ERA5_LPM_5km_{year_month}*')
    dates = [datetime.strptime(fn[-13:-3],"%Y%m%d%H") for fn in sorted(filenames)]
    data = xr.open_mfdataset(sorted(filenames), concat_dim="time", combine="nested",
                      data_vars='minimal', coords='minimal', compat='override',preprocess = add_time_dim)
    data['time'] = dates
    return data

#Perform prewhitening on time series data
def prewhiten(X):
    N = len(X)
    acov = np.correlate(X, X, mode='full') / N
    acov = acov[N-1:]
    rho = acov / acov[0]  # Autocorrelation
    sigma = np.sqrt(np.cumsum(rho))
    Z = (X - np.mean(X)) / np.sqrt(np.var(X))  # Standardize data
    Y = Z / sigma
    return Y

#Mann-Kendall function
def mann_kendall_trend_test(X, alpha=0.05):
    X = prewhiten(X)
    N = len(X)
    S = 0
    for i in range(N - 1):
        for j in range(i + 1, N):
            S += np.sign(X[j] - X[i])

    varS = (N * (N - 1) * (2 * N + 5)) / 18
    if S > 0:
        Z = (S - 1) / np.sqrt(varS)
    elif S < 0:
        Z = (S + 1) / np.sqrt(varS)
    else:
        Z = 0

    p = 2 * (1 - norm.cdf(abs(Z)))
    if p < alpha:
        trend = 'Significant'
    else:
        trend = 'Not Significant'
    return Z, p, trend

# Mann-Kendall function - xarray ds
def mann_kendall_trend_test_xr(ds,idn='reach',date_n='year', alpha=0.05,plabel=''):
    N = len(ds[date_n])
    S = np.zeros_like(ds[idn], dtype=float)  
    p_values = np.zeros_like(ds[idn], dtype=float)  
    trends = np.empty_like(ds[idn], dtype='object')  
    
    for i, reach in enumerate(ds[idn]):  
        if idn!='reach':
            x = ds.sel(COMID=reach)
        else:
            x = ds.sel(reach=reach) 
        X = prewhiten(x.values)  

        for j in range(N - 1):
            for k in range(j + 1, N):
                S[i] += np.sign(X[k] - X[j])

        varS = (N * (N - 1) * (2 * N + 5)) / 18
        if S[i] > 0:
            Z = (S[i] - 1) / np.sqrt(varS)
        elif S[i] < 0:
            Z = (S[i] + 1) / np.sqrt(varS)
        else:
            Z = 0

        p = 2 * (1 - norm.cdf(abs(Z)))

        if p < alpha:
            trends[i] = 'Significant'
        else:
            trends[i] = 'Not Significant'
        p_values[i] = p

    if plabel:
        pval = f'{plabel}_p'
    else:
        pval = 'p'
    #print(ds)
    return xr.Dataset({
        #'Mann_Kendall_Test_Statistic': ((idn), S),
        pval : ((idn), p_values),
        #'trend': ((idn), trends)
    }, coords={idn: ds[idn]})


#%Change and slope per year
def calculate_percent_change_per_year(ds, idn='reach', var='LandsatPlanet', method='thiel'):

    river_ids = ds[idn].values
    percent_changes = []
    trend_stats = {}

    for river_id in river_ids:
        
        # Extract data for the reach
        if (var == 'LandsatPlanet') | (var == 'power'):# | (var == 'precip_km3') :
            river_data = ds.sel(reach=river_id)
            years = river_data['year'].year 
        elif (var == 'week') | (var == 'gmp'):
            river_data = ds.sel(reach=river_id)
            years = river_data['time']#.year 
        else:
            river_data = ds.sel(COMID=river_id) #
            years = river_data['time'].dt.year 
        streamflow = river_data[var]


        if method == 'linear':
            slope, intercept, r_value, p_value, std_err  = linregress(years, streamflow)
        else:
            slope, intercept, low, high = theilslopes(streamflow, years)
            
        # Calculate the percent change per year
        initial_value = streamflow[0].item()  #first year streamflow as initial value
        percent_change_per_year = (slope / initial_value) * 100

        if method == 'linear':
            trend_stats[river_id] = {f'{var}_change': percent_change_per_year,
                                     f'{var}_slope': slope } #, 'r': r_value, 'p': p_value, 'std_err': std_err}
        else:
            trend_stats[river_id] = {f'{var}_change': percent_change_per_year,
                                     f'{var}_slope': slope } #'low': low, 'high': high
    output_df = pd.DataFrame(trend_stats).T
    output_df.index.name = 'COMID'
    return output_df.reset_index()



def seasonal_trend_test(ds, var):
    
    # Convert the xarray Dataset to a DataFrame
    df = ds[var].to_dataframe().reset_index()

    # Add season and year columns to the DataFrame
    df['Season'] = df['time'].dt.to_period('Q').apply(lambda x: x.quarter).map({1: 'winter', 2: 'spring', 3: 'summer', 4: 'fall'})
    #df['year'] = 

    # Extract unique seasons from the DataFrame
    seasons = df['Season'].unique()

    for start,season in enumerate(seasons):
        # Filter data for the current season
        seasonal_data = df[df['Season'] == season].reset_index(drop=True)

        ds_season = seasonal_data.set_index(['reach','time']).to_xarray()
        ds_season = ds_season.groupby('time.year').sum()
        print(ds_season)

        trend = mann_kendall_trend_test_xr(ds_season[var],'reach','year',plabel=season).to_dataframe().reset_index()
        change = calculate_percent_change_per_year(ds_season, var=var)

        print(change)
        change.columns = ['reach',f'{season}_change_yr',f'{season}_slope']

        if start == 0:
            result_df = trend.merge(change,how='left',on='reach')
        else:
            result_df = result_df.merge(trend,on='reach').merge(change,on='reach')

    return result_df

def export_trend_test(ds,var,output_folder):

    ds = ds[[var]]

    #prep dataframe
    mean_annual = ds.mean('time').to_dataframe().reset_index()
    mean_annual.columns = ['COMID',f'{var}_mean']

    mean_annual_std = ds.std('time').to_dataframe().reset_index()
    mean_annual_std.columns = ['COMID',f'{var}_std']

    # %change, slope
    change_slope = calculate_percent_change_per_year(ds,'COMID',var)

    # MK trend test
    trend_test_results = mann_kendall_trend_test_xr(ds[var],'COMID','time',plabel=var)
    trendstat = trend_test_results.to_dataframe().reset_index().rename(columns={'reach':'COMID'})

    dfs = [mean_annual, mean_annual_std, change_slope, trendstat] #mean_annual_std, 

    #merge dataframes
    alldf = dfs[0]
    for df in dfs[1:]:
        alldf = pd.merge(alldf, df, how='left', on='COMID')
    #alldf = alldf.merge(alldf_dams, how='left',on='COMID')
    alldf.set_index('COMID').to_csv(f'{output_folder}/alldf_{var}.csv')

    #Export dataframe as shapefile
    #gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
    #exportdf = gdf_riv[['COMID','geometry']].merge(alldf,on='COMID')
    #exportdf.to_file(f'{output_folder}/alldf_{var}.shp')

    return alldf


def nse(observed, simulated):
    """
    Nash-Sutcliffe Efficiency (NSE) calculation.
    :param observed: Array of observed values
    :param simulated: Array of simulated values
    :return: Nash-Sutcliffe Efficiency (NSE) value
    """
    numerator = np.sum((observed - simulated) ** 2)
    denominator = np.sum((observed - np.mean(observed)) ** 2)
    nse_value = 1 - (numerator / denominator)
    return nse_value

def kge(observed, simulated):
    """
    Kling-Gupta Efficiency (KGE) calculation.
    :param observed: Array of observed values
    :param simulated: Array of simulated values
    :return: Kling-Gupta Efficiency (KGE) value
    """
    r = np.corrcoef(observed, simulated)[0, 1]
    alpha = np.mean(simulated) / np.mean(observed)
    beta = np.sum((simulated - alpha * observed) ** 2) / np.sum((alpha * observed - np.mean(alpha * observed)) ** 2)
    gamma = np.std(simulated) / np.mean(simulated)
    kge_value = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return kge_value


#Get upstream reaches
def makeHRRID(gdf,outlets):
    
    def findup(reach, riv_table):
        ups = []
        gdf = riv_table.loc[riv_table['COMID']==reach,]
        for i in ['up1','up2','up3','up4']:
            if gdf[i].values>0:
                ups.extend(gdf[i].values)
        return ups

    print('there are {} watersheds'.format(len(outlets)))
    df = pd.DataFrame(columns=['COMID','WID'])#,'HRRID'])
    WID = 1
    #HRR_st = 0
    for outlet in outlets:
        outlet = int(outlet)
        watershed = [outlet]
        upstreams = findup(outlet,gdf)
        while len(upstreams)>0:
            watershed.extend(upstreams)
            ups = []
            for upstream in upstreams:
                if gdf.loc[gdf['COMID']==upstream,]['maxup'].values>0:
                    ups.extend(findup(upstream,gdf))
            upstreams = ups

        df2 = pd.DataFrame({'COMID':watershed,'WID':[WID]*len(watershed)})
                            #'HRRID':[int(x) for x in range(HRR_st+len(watershed),HRR_st,-1)]})
        df = df.append(df2,ignore_index=True)
        #HRR_st = HRR_st + len(watershed)
        WID = WID + 1
    #df['HRRID'] = range(len(df),0,-1)
    #print(df)
    return df
 

#Get downstream reaches
def get_down(comid):
    down_all = [int(comid)]  #list all COMIDs downstream
    index = 0
    while index < len(down_all):
        current_comid = down_all[index]  
        down_id = gdf_riv[gdf_riv['COMID'] == current_comid]['NextDownID']  
        if not down_id.empty:
            down_temp = down_id.values[0]  
            while down_temp != 0:
                if down_temp not in down_all:  #add to down_all if it's not already there
                    down_all.append(int(down_temp))  #append the current downstream COMID to the list
                    down_id = gdf_riv[gdf_riv['COMID'] == down_temp]['NextDownID']  
                    if not down_id.empty:
                        down_temp = down_id.values[0]  
                    else:
                        break  
                else:
                    break  
        index += 1 
    return down_all


#Calculate and plot trends
def trend_plot(streamflow_data, glacier_melt_data, basin, frequency, axs):

    monthly_data = pd.concat([streamflow_data, glacier_melt_data], axis=1)#.dropna(inplace=True)
    monthly_data = monthly_data[monthly_data.index >='2004-01-01']
    monthly_data.columns = ['Streamflow', 'GlacierMelt']
    
    # Drop any rows with missing values
    #monthly_data.dropna(inplace=True)
    
    # Convert to seasonal or annual data if specified
    if frequency == 'seasonal':
        monthly_data = monthly_data.resample('QS').sum()  # Assuming quarterly seasons
    elif frequency == 'annual':
        monthly_data = monthly_data.resample('YS').sum()

    # Trend analysis for streamflow
    streamflow_result = mann_kendall_trend_test(monthly_data['Streamflow'])
    streamflow_slope, streamflow_intercept, _, _ = theilslopes(monthly_data['Streamflow'])

    # Trend analysis for glacier melt
    glacier_melt_result = mann_kendall_trend_test(monthly_data['GlacierMelt'])
    glacier_melt_slope, glacier_melt_intercept, _, _ = theilslopes(monthly_data['GlacierMelt'])

    # Print results
    print(f"\n {basin}, streamflow Mann-Kendall Test:, {streamflow_result}")
    print("Streamflow Sen's Slope:", streamflow_slope)
    
    # Create trend series
    streamflow_trend = streamflow_slope * np.arange(len(monthly_data)) + streamflow_intercept
    glacier_melt_trend = glacier_melt_slope * np.arange(len(monthly_data)) + glacier_melt_intercept
    
    # Add trend series to the DataFrame
    monthly_data['Streamflow_Trend'] = streamflow_trend
    monthly_data['GlacierMelt_Trend'] = glacier_melt_trend
    
    #Plot Streamflow & Glacier Melt Trends
    ax1 = axs
    ax1.plot(monthly_data.index, monthly_data['Streamflow'], linestyle='-',label='Streamflow', color='black')
    ax1.plot(monthly_data.index, monthly_data['Streamflow_Trend'], label='Streamflow Trend', linestyle='--', color='black',linewidth=1)
    
    if frequency == 'monthly':
        streamflow_ma = monthly_data['Streamflow'].rolling(window=12).mean()
        ax1.plot(monthly_data.index, streamflow_ma, label='Streamflow 12-Month MA', linestyle='-.', color='red')
    
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Streamflow, km3/year')
    ax1.tick_params(axis='y')

    # Create the second y-axis
    ax2 = ax1.twinx()
    ax2.plot(monthly_data.index, monthly_data['GlacierMelt'],  linestyle='-', label='Glacier Melt', color='orange')
    if frequency == 'annual':
        ax2.plot(monthly_data.index, monthly_data['GlacierMelt_Trend'], label='Glacier Melt Trend', linestyle='--', color='orange',linewidth=1)

    ax2.set_ylabel('Glacier Melt, km3/year')
    ax2.tick_params(axis='y')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    plt.title(f'{basin}')# {frequency.capitalize()})# Streamflow and Glacier Melt Trends')
    

#Calculate and plot correlation
def corr_plot(streamflow_data, comparison_data, variable_name='Variable', basin='Basin', frequency='monthly', ax=None): 
    
    # Normalize the data
    def normalize(series):
        return (series - series.mean()) / series.std()

    streamflow = normalize(streamflow_data)
    comparison = normalize(comparison_data)
    
    # Compute cross-correlation
    lags = np.arange(0, 13)  
    ccf_values = [ccf(streamflow, comparison, adjusted=False)[lag] for lag in lags]
    
    # Plot the cross-correlation function
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.stem(lags, ccf_values, use_line_collection=True)
    ax.set_xlabel('Lag (months)')
    ax.set_ylabel('Cross-Correlation')
    ax.set_title(f'{basin}')