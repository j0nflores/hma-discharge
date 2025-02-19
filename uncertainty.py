import xarray as xr
import geopandas as gpd
import pandas as pd
from functions import mann_kendall_trend_test_xr, nse, kge
import numpy as np
import scipy.stats as stats
import glob
import os


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
mean_streamflow = ensemble.sel(time=slice('2004-01-01','2019-12-31'))
mean_ens = mean_streamflow.groupby('time.year').sum('time')*24*3600/1e9#total annual, km3/yr
mean_ens = mean_ens.where(mean_ens != 0)

#Mann-Kendal Trend Test
mk = mann_kendall_trend_test_xr(mean_ens.sel(ensemble='noah'),idn='reach',date_n='year', alpha=0.05,plabel='noah')
mk_gfdl = mann_kendall_trend_test_xr(mean_ens.sel(ensemble='gfdl'),idn='reach',date_n='year', alpha=0.05,plabel='gfdl')
mk_grfr = mann_kendall_trend_test_xr(mean_ens.sel(ensemble='grfr'),idn='reach',date_n='year', alpha=0.05,plabel='grfr')
mk_wbm = mann_kendall_trend_test_xr(mean_ens.sel(ensemble='wbm',year=slice(2004,2018)),idn='reach',date_n='year', alpha=0.05,plabel='wbm')
mk['grfr_p'] = mk_grfr.grfr_p
mk['gfdl_p'] = mk_gfdl.gfdl_p
mk['wbm_p'] = mk_wbm.wbm_p


#Merge xr data arrays
mk_xr = xr.concat([mk.noah_p,mk.grfr_p,mk.wbm_p,mk.gfdl_p.where(mk.gfdl_p!=1)],dim='ensemble')
mk_xr.coords['ensemble'] = ['noah','grfr','wbm','gfdl']


#Calculate ensemble mean and std into dataframe
mk_df = mk_xr.mean('ensemble',skipna=True).to_dataframe().rename(columns={'noah_p':'p'})
mk_df['p_std'] = mk_xr.std('ensemble',skipna=True).to_dataframe().rename(columns={'noah_p':'p_std'})


#Calculate uncertainty of ensemble
n= 4
margin_of_error = stats.t.ppf(0.975, df=n-1) * (mk_df.p_std / np.sqrt(n)) # For 95% CI
mk_df['ci_low'] = mk_df.p - margin_of_error
mk_df['ci_low'] = mk_df['ci_low'].apply(lambda x: 0 if x < 0 else x)
mk_df['ci_up'] = mk_df.p + margin_of_error
mk_df['mk_se_up']  = mk_df.p + mk_df.p_std/np.sqrt(n) 
mk_df['mk_se_low'] = mk_df.p - mk_df.p_std/np.sqrt(n)
mk_df['mk_se_low'] = mk_df['mk_se_low'].apply(lambda x: 0 if x < 0 else x)
print(margin_of_error)


#Export shapefiles for mapping
resultgdf = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/analysis/data/resultgdf.shp')
mk_shp = resultgdf.merge(mk_df.reset_index().rename(columns={'reach':'COMID'}), on='COMID')
mk_shp = mk_shp[['COMID','mean_flow', 'mean_annua', 'mean_ann_1', 'percent_ch',
                 'slope_x', 'p_value_x', 'trend_x', 'power_mean', 'percent__1',
                 'slope_y', 'p_value_y', 'trend_y', 'infra', 'basin','p','p_std',
                 'ci_low','ci_up','mk_se_up','mk_se_low','geometry']]
mk_shp.to_file('./output/advances/mk.shp')


#Export stat summary, signifiicant trend tests
mk_mean_stat = mk_shp[mk_shp['p']<=0.05][['mean_annua','percent_ch','power_mean','percent__1']].describe().round(2)
mk_stat_up = mk_shp[mk_shp['mk_se_up']<=0.05][['mean_annua','percent_ch','power_mean','percent__1']].describe().round(2)
mk_stat_low = mk_shp[mk_shp['mk_se_low']<=0.05][['mean_annua','percent_ch','power_mean','percent__1']].describe().round(2)
pd.concat([mk_mean_stat,mk_stat_up,mk_stat_low]).to_csv('./output/advances/mk_stats_table.csv') #ensemble mean, mean+std, mean-std


#Plot sample hydrographs with spreads ###################################################
from matplotlib import rcParams
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Palatino'] + rcParams['font.serif']
#rcParams['font.size'] = 12

#Load gauge data
val_merit = pd.read_csv('/nas/cee-water/cjgleason/jonathan/data/gauge/val_merit_rev2.csv')
gauge = xr.open_dataset('/nas/cee-water/cjgleason/jonathan/data/gauge/hma_gauge.nc')

paper_gauges = ['indus','2260100','chiang_saen','16363']
annot_gauges = ['Indus', '2260100','Chiang Saen','16363']

fig, axes = plt.subplots(2,2, figsize=(12,12))
axes = axes.flatten()

for i, gauge_id in enumerate(paper_gauges):
    
    sy = 2016
    ey = 2019
    
    ax = axes[i]

    reach = val_merit[val_merit.gauge==gauge_id].COMID.values[0]
    gauge_obs = gauge.sel(gauge=gauge_id, date=slice(f'{sy}-01-01',f'{ey}-12-31')).dropna(dim='date')#.resample(date='MS').mean()

    #Calculate ensemble mean and spread
    ensemble_mean = ensemble.sel(reach=reach, time=gauge_obs.date).mean(dim='ensemble')#.resample(time='MS').mean()
    ensemble_std = ensemble.sel(reach=reach, time=gauge_obs.date).std(dim='ensemble')/np.sqrt(n)#.resample(time='MS').mean()
    ensemble_std = ensemble_std.where(ensemble_std>0, other=0)
    

    # Plot ensemble mean
    ax.plot(ensemble_mean.time, gauge_obs.discharge, "--k", label=f"Gauge ({annot_gauges[i]})" )
    ax.plot(ensemble_mean.time, ensemble_mean, label="Ensemble Mean", color="red")


    #Standard Deviation 
    ax.fill_between(ensemble_mean.time, 
                     ensemble_mean - ensemble_std, 
                     ensemble_mean + ensemble_std, 
                     color="gray", alpha=0.5, label="Ensemble Spread")

    ax.set_xlabel("Time")
    ax.set_ylabel("Discharge, cms")
    ax.set_ylim(0,None)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    
    ax.legend()
    
plt.tight_layout()
plt.savefig(f'./figs/advances/hgraph_ens_spread.jpg')


#Plot annual trends #######################################################
paper_gauges = ['2260100','2260120','chiang_saen',
                'jhelum',  '16363',  '16350', ] 
annot_gauges = ['2260100','2260120','Chiang Saen',
                'Jhelum',  '16363', '16350']

rcParams['font.size'] = 10
fig, axes = plt.subplots(2,3,figsize=(12,8))
axes = axes.flatten()

sy = 2004
ey = 2018

for i,gauge_id in enumerate(paper_gauges): #['indus','chenab','kabul']: #
    
    ax = axes[i]
    reach = val_merit[val_merit.gauge==gauge_id].COMID.values[0]
    gauge_obs = gauge.sel(gauge=gauge_id, date=slice(f'{sy}-01-01',f'{ey}-12-31')).dropna(dim='date').groupby('date.year').sum('date')*24*3600/1e9
    gauge_obs = gauge_obs.where(gauge_obs != 0)#.resample(date='MS').mean()
    ensx_mean = mean_ens.sel(reach=reach, year=slice(sy,ey))
    ax = sns.regplot(x=gauge_obs.year, y=gauge_obs.discharge, ax=ax, color='black',label=f'Gauge: {annot_gauges[i]}',
                     line_kws={'linestyle': 'dashed', 'color': 'black', 'linewidth': 2})
    ax = sns.regplot(x=ensx_mean.year, y=ensx_mean.mean('ensemble'), ax=ax, color='red',label=f'Ensemble Mean',
                     line_kws={'linestyle': 'dashed', 'color': 'red', 'linewidth': 2})
    
    ax.set_ylabel(r'Annual Discharge, km$^3$/yr/yr')
    ax.legend()
    ax.xaxis.set_major_locator(MultipleLocator(2)) 
    
plt.tight_layout()
plt.savefig(f'./figs/advances/annual_trends.jpg')


#Plot infras Figure 4 ##################################
fig, axes = plt.subplots(1,3,figsize=(10,3.5))
axes = axes.flatten()

labels = ['Planned HPP','Existing HPP','GeoDAR Dam']

for i, infra in enumerate(['hppp','hppe','dams']):
    
    ax = axes[i]
    plot_df = mk_shp.copy()
    plot_df = plot_df[~plot_df['infra'].isna()]
    plot_df = plot_df[plot_df['infra']==infra]

    #data_mean = plot_df[plot_df['p_value_y']<=0.05]['percent__1']
    data1 = plot_df[plot_df['mk_se_low']<=0.05]['percent__1'] #up or low
    data2 = plot_df[plot_df['mk_se_up']<=0.05]['percent__1']
    print(min(pd.concat([data1,data2])))

    # Define histogram bins
    if i == 0:
        binterval = 1.5
        bins = np.arange(min(pd.concat([data1,data2])), max(pd.concat([data1,data2]))+1, binterval)
        binmax = 60 #None #bins.max()+35
    if i == 1:
        binterval = 0.5
        bins = np.arange(min(pd.concat([data1,data2])), max(pd.concat([data1,data2]))+1, binterval)
        binmax = bins.max()
    elif i == 2:
        binterval = 2
        bins = np.arange(min(pd.concat([data1,data2])), max(pd.concat([data1,data2]))+1, binterval)
        binmax = None #bins.max()+40

    # Set up Seaborn style
    sns.set_context("paper")#, font_scale=1.75)

    # Plot histograms with Seaborn #'#ff7f00' blue
    sns.histplot(data1, bins=bins, ax=ax, color='gray',  
                 label=f"Ensemble Mean - SE (n={len(data1)})", kde=False, alpha=0.5, edgecolor=None)
    sns.histplot(data2, bins=bins, ax=ax, color='blue', 
                 label=f"Ensemble Mean + SE (n={len(data2)})", kde=False, alpha=0.5, edgecolor=None)

    # Labels and title
    ax.set_ylim(0,binmax)
    ax.set_xlabel("Stream power change per year, %")
    ax.set_ylabel("Frequency")
    ax.legend(title=f'{labels[i]} \nwith significant trends', fontsize=8, frameon=False)


plt.tight_layout()
plt.savefig(f'./figs/advances/Fig4_infra.jpg')
