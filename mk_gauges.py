from functions import mann_kendall_trend_test_xr, nse, kge
from functions import calculate_percent_change_per_year
from scipy.stats import norm, linregress, theilslopes
import numpy as np

# Mann-Kendall function - xarray ds
def mann_kendall_trend_test_xr(ds,idn='reach',date_n='year', alpha=0.05,plabel=''):
    N = len(ds[date_n])
    S = np.zeros_like(ds[idn], dtype=float)  
    p_values = np.zeros_like(ds[idn], dtype=float)  
    trends = np.empty_like(ds[idn], dtype='object')  
    
    for i, reach in enumerate(ds[idn]):
        
        #try:
        if idn=='gauge':
            x = ds.sel(gauge=reach)#.dropna(dim='year')
            #print(x.values)

        elif idn!='reach':
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
        #except:
            #p_values[i] = np.nan

    if plabel:
        pval = f'{plabel}_p'
    else:
        pval = 'p'
    #print(ds)
    return xr.Dataset({
        #'Mann_Kendall_Test_Statistic': ((idn), S),
        pval : ((idn), p_values),
        'trend': ((idn), trends)
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
        elif (var == 'discharge'):
            river_data = ds.sel(gauge=river_id)
            years = river_data['year']#.year 
        else:
            river_data = ds.sel(COMID=river_id) #
            years = river_data['time'].dt.year 
            
        streamflow = river_data[var]
        streamflow = streamflow.where(streamflow != 0).dropna(dim='year')
        years = streamflow['year']#.dt.year 
        #print(streamflow)


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

def prewhiten(X):
    N = len(X)
    acov = np.correlate(X, X, mode='full') / N
    acov = acov[N-1:]
    rho = acov / acov[0]  # Autocorrelation
    sigma = np.sqrt(np.cumsum(rho))
    Z = (X - np.mean(X)) / np.sqrt(np.var(X))  # Standardize data
    Y = Z / sigma
    return Y

#load gauge ds
sy = 2004
ey = 2019
val_merit = pd.read_csv('/nas/cee-water/cjgleason/jonathan/data/gauge/val_merit_rev2.csv')
gauge = xr.open_dataset('/nas/cee-water/cjgleason/jonathan/data/gauge/hma_gauge.nc')
gauge_obs = gauge.sel(gauge=list(val_merit.gauge), date=slice(f'{sy}-01-01',f'{ey}-12-31'))#.dropna(dim='date')
gauge_obs = gauge_obs.groupby('date.year').sum('date')*24*3600/1e9


#mk test
var = 'discharge'
slope = calculate_percent_change_per_year(gauge_obs, idn='gauge', var=var, method='thiel')
increasing = slope[slope.discharge_slope>0].COMID

mk = mann_kendall_trend_test_xr(gauge_obs.discharge,idn='gauge',date_n='year', alpha=0.05,plabel='gauge')
mk_df = mk.to_dataframe().reset_index()
mk_df = mk_df[mk_df.gauge.isin(increasing)]
mk_df = mk_df[mk_df.gauge_p<0.05]



print(len(mk_df)/77)
print(slope[slope.discharge_slope>0].describe())


#Plot annual trends #######################################################
import matplotlib.pyplot as plt

sy = 2004
ey = 2019

for i,gauge_id in enumerate(increasing): 
    gauge_obs = gauge.sel(gauge=gauge_id, date=slice(f'{sy}-01-01',f'{ey}-12-31')).dropna(dim='date').groupby('date.year').sum('date')*24*3600/1e9
    gauge_obs = gauge_obs.where(gauge_obs != 0)
    ax = sns.regplot(x=gauge_obs.year, y=gauge_obs.discharge, color='black', label = gauge_id, 
                     line_kws={'linestyle': 'dashed', 'color': 'black', 'linewidth': 2})

    ax.set_ylabel(r'Annual Discharge, km$^3$/yr/yr')
    plt.legend()
    plt.show()