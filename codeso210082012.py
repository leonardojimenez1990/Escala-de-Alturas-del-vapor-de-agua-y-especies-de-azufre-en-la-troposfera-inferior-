import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.feature import NaturalEarthFeature

with xr.open_dataset('sulphur_dioxideEne_dic2003_2020CMAS.nc') as ds:
    print(ds.so2.attrs)  # time slice

#so2_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01'))
#calculate and assign the height coordinate to the set.
so2_2003_2020 = ds.assign_coords(height =("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
                                                                  ds.level[8].values))))

#Swap the level and height dimensions
so2_2003_2020 = so2_2003_2020.swap_dims({"level": "height"})
selected = so2_2003_2020.where(lambda x: x.time.dt.year == 2003, drop=True)

for t in range(len(ds.time)):
    so24d = so2_2003_2020.so2.isel(time=t)
    so24dC = so24d.copy()
    #calculate so24d [z] / so24d [0]
    for h in range(len(so24d.height)):
        so24dC[h,:,:] = so24d.isel(height=h).values / so24d[8,:,:].values
    #so24d = so24d.sel(height= slice(5.476,-0.0))
    #so24d = so2_2003_2020.so2.sel(level = slice(500,1000)).polyfit(dim='level', deg=1)

    #calculo de la regresion lineal de los perfil vertical de la altura con minimos cuadrados
    Hx = -1 / (so24dC.polyfit(dim='height', deg=1, full=True, cov=True)) #, skipna=False
    
    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
    ax.coastlines()
    Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(range(1000000, 10000000)))  # Hx.polyfit_coefficients[1,:,:].plot.contourf()
    #Hx.polyfit_coefficients[1, :, :].plot.contourf(levels=(range(-10, 33))) # Hx.polyfit_coefficients[1,:,:].plot.contourf()
    ax.gridlines(draw_labels=True)
    plt.title("Catidad de SO2 a nivel mundial " + str(so24d.time.values))
    plt.show()
