import pandas as pd
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
import time 
import imageio
import glob
import cv2

model = 'noah'
iyear = 2009
print(iyear)

path = '/nas/cee-water/cjgleason/jonathan/himat_routing/figs'
outlets = [46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223] 
basin_name = ['AmuDarya','Indus','Ganges/Brahmaputra','Irawaddy','Salween','Mekong','Yangtze','Yellow'] 
seasons = ['MAM','JJA','SON']
year = 2019

#Plot seasons
df = pd.DataFrame()
for outlet in outlets:
    seasonals = f'{path}/{outlet}/main_{model}.csv' #{iyear}
    df_temp = pd.read_csv(seasonals,index_col=0)
    df = pd.concat([df,df_temp],axis=0)
df = df.reset_index(drop=True)

gdf_riv = gpd.read_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma.shp')
main = df.copy()

for season in seasons:
    main[f'diff_{iyear}_{year}_{season}'] = 100*(main[f'DA_{year}_{season}'] - main[f'DA_{iyear}_{season}'])/main[f'DA_{iyear}_{season}']
    gdf_da = gdf_riv.merge(main,how='right',on='COMID') 

#plot output folder
out_path = f'./figs/{model}_test'
os.makedirs(out_path, exist_ok=True)
    
for season in seasons:
    
    #if not os.path.exists(f'{out_path}/diff_{iyear}{season}.jpg'):
    print(season)

    gdf_all = gdf_da.copy() #[~(gdf_da[f'diff_2009_2019_{season}']>500)&~(gdf_da[f'diff_2009_2019_{season}']<-500)]
    gdf_all[f'diff_{iyear}_{year}_{season}'] = gdf_all[f'diff_{iyear}_{year}_{season}'].apply(lambda x: 100 if x >=100 else x)
    gdf_all[f'diff_{iyear}_{year}_{season}'] = gdf_all[f'diff_{iyear}_{year}_{season}'].apply(lambda x: -100 if x <=-100 else x)

    # Calculate the bounding box of your shapefile
    shapefile_bounds = gdf_all.total_bounds

    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Set the axis limits to your shapefile's extent
    ax.set_xlim(shapefile_bounds[0], shapefile_bounds[2])
    ax.set_ylim(shapefile_bounds[1], shapefile_bounds[3])

    # Plot the satellite basemap
    ctx.add_basemap(ax, crs=gdf_all.crs,alpha=0.5,
                    source='https://tiles.maps.eox.at/wmts/1.0.0/s2cloudless-2018_3857/default/g/{z}/{y}/{x}.jpg')

    # Plot your shapefile within its extent
    gdf_all.plot(ax=ax, column=f'diff_{iyear}_{year}_{season}', cmap='RdYlBu', legend=True)

    # Set title and display the plot
    ax.set_title(f'HMA %Change in Streamflow {iyear} vs {year} - Season: {season}')
    plt.tight_layout()
    plt.savefig(f'{out_path}/diff_{iyear}{season}.jpg',dpi=200)
    #plt.show()   
    
#break
time.sleep(3)

#Generate gif 
#if not os.path.exists(f'{out_path}/diff_{iyear}.gif'):
imgps = glob.glob(f'{out_path}/diff_{iyear}*.jpg')
images = []
for imgp in imgps:
    img = imageio.imread(imgp)
    img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
    images.append(img)
gif_path = f'{out_path}/diff_{iyear}.gif'
imageio.mimsave(gif_path, images, fps=.5)