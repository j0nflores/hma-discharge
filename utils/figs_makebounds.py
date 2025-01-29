import geopandas as gpd
import pandas as pd


model = 'grfr'
outlets = [46032729,45059531,45070322,44017811,44017734,44017781,43052327,43003223,46015164] 

main = gpd.GeoDataFrame()

for outlet in outlets: #[4]

    #plt.figure(figsize=(12,8))

    print(outlet)
    gdf_riv = gpd.read_file(f'/nas/cee-water/cjgleason/jonathan/himat_routing/hrr_out/{model}/{outlet}/gis/{outlet}_bound.shp')
    gdf_riv['id'] = outlet
    main = pd.concat([main,gdf_riv],axis=0)
    
        #gdf_riv.plot()
        #plt.show()

print(main)
main.to_file('/nas/cee-water/cjgleason/jonathan/himat_routing/data/riv_hma_bound_man.shp')