import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.feature import NaturalEarthFeature

"""
with xr.open_dataset('sulphur_dioxideJun_ago2003_2020CMAS.nc') as ds:
    dahx = ds.to_array()
z = list()

# print( ds.values)
# calcula los valores de z (altitud) para cada nivel de presión
for i in range(len(ds.level.values)):
    z.append(-7.9 * np.log(ds.level[i].values / ds.level[8].values))
    print(z[i])

#calcular la regresión lineal por mínimos cuadrados en cada punto de regilla
for t in range(0, 53):#dimension tiempo
    for lat in range(0, 241):#dimension latitude
        for lon in range(0, 480):#dimencion longitude
            so24d = ds.so2.isel(time=t, latitude=lat, longitude=lon)#perfil vertical de so2
            print(so24d.values)
            kr = np.polyfit(z, np.log(so24d / so24d[0].values), 1)# Ajustar una línea a los datos
            print(kr, ds.longitude[lon].values, ds.latitude[lat].values)
            hx = -1 / kr[0]
            print(hx, ds.longitude[lon].values, ds.latitude[lat].values)
            dahx [t,lon,lat].values = kr[0] # Obtener la pendiente (primer elemento)
            print(dahx)

#Hx1 = ds.drop_dims('level')
#del Hx1['level']

#print(type(hx))
#print(Hx1.values())

#Hx1 = -11.2999857326264*(Hx1.so2.values)
#print(Hx1.values)"""

with xr.open_dataset('sulphur_dioxideEne_dic2003_2020CMAS.nc') as ds:
    print(ds.so2.attrs)  # time slice

#so2_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01'))
#calculate and assign the height coordinate to the set.
so2_2003_2020 = ds.assign_coords(height =("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values / ds.level[8].values))))
print(so2_2003_2020)
#Swap the level and height dimensions
so2_2003_2020 = so2_2003_2020.swap_dims({"level": "height"})
selected = so2_2003_2020.where(lambda x: x.time.dt.year == 2003, drop=True)
print(so2_2003_2020)

for t in range(len(ds.time)):
    so24d = so2_2003_2020.so2.isel(time=t)
    print("olikujyhtgrfedwsqa",so24d.values)
    so24d = so24d.sel(height= slice(5.476,-0.0))
    print("mjmhnggbfdcsxaz",so24d.values)
    #so24d = so2_2003_2020.so2.sel(level = slice(500,1000)).polyfit(dim='level', deg=1)

    #calculo de la regresion lineal de los perfil vertical de la altura con minimos cuadrados
    Hx = -1 / (so24d.polyfit(dim='height', deg=1, full=True, cov=True)) #, skipna=False
    print(Hx.polyfit_coefficients.values)
    print("\n\n\n",  (Hx.polyfit_coefficients[0,:,:].values))
    print("\n\n\n",  (Hx.polyfit_coefficients[1,:,:].values))
    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
    ax.coastlines()
    Hx.polyfit_coefficients[0, :, :].plot.contourf(
        levels=[-2.6453149994534474e+26, -2.6453149994534474e+12, -2.6453149994534474e+11, -2.6453149994534474e+10, 0,
                3.638123694600816e+10, 3.638123694600816e+11, 3.638123694600816e+12,
                3.638123694600816e+26])  # Hx.polyfit_coefficients[1,:,:].plot.contourf()
    #Hx.polyfit_coefficients[1, :, :].plot.contourf(
    #    levels=[-31241366794107.77, -312413667941.77, -2124136679.77, -21241366.77, 0,
    #            1804807283308.5613, 2804807283308.5613, 3804807283308.5613,
    #            4804807283308.5613])  # Hx.polyfit_coefficients[1,:,:].plot.contourf()
    ax.gridlines(draw_labels=True)
    plt.title("Catidad de SO2 a nivel mundial " + str(so24d.time.values))
    plt.show()

so24d = -1 / so24d.polyfit_coefficients[1, :, :, :].drop('degree')
# del so24d['degree']
so24d = so24d.isel(time=11)
print(so24d)

# plot SO2
fig = plt.figure(1, figsize=(15., 12.))
ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
ax.coastlines()
so24d.plot.contourf()
ax.gridlines(draw_labels=True)
plt.title("Catidad de SO2 a nivel mundial " + str(so24d.time.values))
plt.savefig(
    '/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
    + str(so24d.time.values) + '.png')
plt.show()
