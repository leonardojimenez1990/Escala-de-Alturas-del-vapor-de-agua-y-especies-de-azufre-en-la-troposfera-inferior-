import copy
import os
import imageio
from tqdm import tqdm

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import statistics as stat
from scipy import stats as st


# from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
# import cartopy.feature as cf
# from cartopy.feature import NaturalEarthFeature
# from xarray.plot.utils import legend_elements


# crear archivo gif
def gif():
    pach = '../Datos/salidas/0.5SulfurdioxideResultados10122021/'
    archivos = os.listdir(pach)
    img_arr = []

    for i in tqdm(range(len(archivos))):
        nomArchivos = archivos[i]
        dirArchivos = pach + nomArchivos

        leer_img = imageio.imread(dirArchivos)
        img_arr.append(leer_img)
        imageio.mimwrite('../Datos/salidas/'
                         '0.5SulfurdioxideResultados10122021/gif.gif', img_arr, 'GIF', duration='0.5')
# Graficos del analisis precursor
def creargraficoso2(ds, so2_2003_2020):
    dfgraficoso2 = ds.so2.to_dataframe()
    dfgraficoso2 = dfgraficoso2.reset_index()
    dfgraficoso2 = dfgraficoso2[['time', 'level', 'so2']].set_index('time')
    colors = np.random.rand(5000000)
    plt.scatter(dfgraficoso2.index[:5000000],dfgraficoso2.so2.values[:5000000],
                s=dfgraficoso2.level.values[:5000000], c=colors, alpha=0.5)
    #plt.scatter(dfgraficoso2.index,dfgraficoso2.level.values,s=dfgraficoso2.so2.values,alpha=0.5)
    # dfgraficoso2['so2'].plot()
    plt.show()

    dfgraficoso2 = dfgraficoso2.reset_index()
    dfgraficoso2 = dfgraficoso2[['time','so2']].set_index('time')
    plt.plot(dfgraficoso2)
    #dfgraficoso2['so2'].plot()
    plt.show()


def leersulphur_dioxide2003_2021CMAS():
    with xr.open_dataset('/home/leo/Documentos/proPython/proyectos/Vertical-scale-for-water-vapor-and-sulfur-compound-profiles-in-the-lower-troposphere-main/Data/sulphur_dioxide2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.so2.attrs)
        # Append the height dimension
        so2_2003_2020 = ds.expand_dims({"height": 9})

        # so2_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # seleccionar periodo de tiempo

        # calculate and assign the height coordinate to the set.
        # calculate height = H * ln(level0 / level) where H = -7.9
        so2_2003_2020 = so2_2003_2020.assign_coords(height=("height", (
                7.9 * np.log(ds.level[8].values / ds.level.sel(level=slice(500, 1000)).values))))

        # Calcular de los valores de so2 / valores de so2 en 950 HPa
        so2_2003_2020['so2'] = so2_2003_2020.so2 / (so2_2003_2020.so2.isel(height=7, level=7)+1)

        # Calcular el log de los datos de so2
        # log_so2_2003_2020 = np.log(so2_2003_2020.so2.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    return ds, so2_2003_2020


def meanMesessulphur_dioxide2003_2021CMAS(so2_2003_2020):
    # with xr.open_dataset('sulphur_dioxide2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #     print(ds.so2.attrs)
    # print(ds)
    #
    # # so2_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # # calculate and assign the height coordinate to the set.
    # so2_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                  ds.level[8].values))))
    listaDsMeses, listaMeanMeses = list(), list()
    # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # so2_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(so2_2003_2020.height.values)
    # Swap the level and height dimensions
    # so2_2003_2020 = so2_2003_2020.swap_dims({"level": "height"})
    # seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont = 0
    # Calcular el log de los datos de so2
    log_so2_2003_2020 = np.log(so2_2003_2020.so2.sel(level=slice(600, 950), height=slice(5.476, 0.4052)))
    print(log_so2_2003_2020.sel(time='2003' + '-' + '01' + '-01'))
    for i in range(1, 13):
        listaDsMeses.append(log_so2_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate so24d[z] / so24d[0] normalización
        so24d = listaMeanMeses[cont]

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (so24dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)

        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        print(Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot SO2
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values, alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=1, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')
        # ax.set_ylabel('Km')

        plt.title('Mean vertical profile scale Sulfur dioxide (SO2) of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('../Datos/salidas/'
                    + 'MediaMesSulfur dioxide' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1


def meanMesesDec_Febsulphur_dioxide2003_2021CMAS(so2_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2 para el perfil vertical de 600 - 950 HPa
    log_so2_2003_2020 = np.log(so2_2003_2020.so2.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [12, 1, 2]:
        listaDsMeses.append(log_so2_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                   transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                   linewidth=0.5, vmin=0, vmax=2.8, cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')


    plt.title('Mean vertical profile scale Sulfur dioxide (SO2) of months Dec, Jun, Feb: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesDec_FebSulfur dioxide' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesMar_Maysulphur_dioxide2003_2021CMAS(so2_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_so2_2003_2020 = np.log(so2_2003_2020.so2.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [3, 4, 5]:
        listaDsMeses.append(log_so2_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / (k.polyfit_coefficients[0] * 950 + 1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Sulfur dioxide (SO2) of months Mar, Apr, May: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesMar_MaySulfur dioxide' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesJun_Augsulphur_dioxide2003_2021CMAS(so2_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_so2_2003_2020 = np.log(so2_2003_2020.so2.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [6, 7, 8]:
        listaDsMeses.append(log_so2_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Sulfur dioxide (SO2) of months Jun, Jul, Aug: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesJun_Ago Sulfur dioxide' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesSep_Novsulphur_dioxide2003_2021CMAS(so2_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_so2_2003_2020 = np.log(so2_2003_2020.so2.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [9, 10, 11]:
        listaDsMeses.append(log_so2_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Sulfur dioxide (SO2) of months Sep, Oct, Nov: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesSep_Nov Sulfur dioxide' + str(i) + '.png')  # ,dpi=720

    plt.show()


def sulphur_dioxide2003_2021CMAS(ds, so2_2003_2020):
    # Calcular el log de los datos de so2
    log_so2_2003_2020 = np.log(so2_2003_2020.so2.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    meanMax1 = list()
    for t in range(len(ds.time)):
        so24d = so2_2003_2020.so2.isel(time=t)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # k = -1 / (so24dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = log_so2_2003_2020[0, t, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)
        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        print(Hx)

        meanMax1.append(Hx.max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("BuPu"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot SO2
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. Sulfur dioxide (SO2) " + str(so24d.time.values))
        plt.savefig('../Datos/salidas/'
                    + 'Sulfur dioxide' + str(so24d.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))


def leersulphate_aerosol_mixing_ratio2003_2021CMAS():
    with xr.open_dataset(
            'sulphate_aerosol_mixing_ratio2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.aermr11.attrs)

        # Append the height dimension
        aermr11_2003_2020 = ds.expand_dims({"height": 9})

        # aermr11_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # seleccionar periodo de tiempo

        # calculate and assign the height coordinate to the set.
        # calculate height = H * ln(level0 / level) where H = -7.9
        aermr11_2003_2020 = aermr11_2003_2020.assign_coords(height=("height", (
                7.9 * np.log(ds.level[8].values / ds.level.sel(level=slice(500, 1000)).values))))

        # Calcular de los valores de so2 / valores de aermr11 en 950 HPa
        aermr11_2003_2020['aermr11'] = aermr11_2003_2020.aermr11 / (aermr11_2003_2020.aermr11.isel(height=7, level=7)+1)
        print(aermr11_2003_2020)

        # Calcular el log de los datos de aermr11
        # log_aermr11_2003_2020 = np.log(aermr11_2003_2020.aermr11.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    return ds, aermr11_2003_2020


def meanMesessulphate_aerosol_mixing_ratio2003_2021CMAS(aermr11_2003_2020):
    # with xr.open_dataset('sulphur_dioxide2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #     print(ds.aermr11.attrs)
    # print(ds)
    #
    # # aermr11_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # # calculate and assign the height coordinate to the set.
    # aermr11_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                  ds.level[8].values))))
    listaDsMeses, listaMeanMeses = list(), list()
    # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # aermr11_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(aermr11_2003_2020.height.values)
    # Swap the level and height dimensions
    # aermr11_2003_2020 = aermr11_2003_2020.swap_dims({"level": "height"})
    # seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont = 0
    # Calcular el log de los datos de aermr11
    log_aermr11_2003_2020 = np.log(aermr11_2003_2020.aermr11.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    print(log_aermr11_2003_2020.sel(time='2003' + '-' + '01' + '-01'))
    for i in range(1, 13):
        listaDsMeses.append(log_aermr11_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate aermr11d[z] / aermr11d[0] normalización
        aermr11d = listaMeanMeses[cont]

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (aermr11dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = aermr11d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)

        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        print(Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot AERMR11
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')
        # ax.set_ylabel('Km')

        plt.title('Mean vertical profile scale Sulphate Aerosol Mixing Ratio (AERMR11) of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('../Datos/salidas/'
                    + 'MediaMesSulphate Aerosol Mixing Ratio' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1

def meanMesesDec_Febsulphate_aerosol_mixing_ratio2003_2021CMAS(aermr11_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_aermr11_2003_2020 = np.log(aermr11_2003_2020.aermr11.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [12, 1, 2]:
        listaDsMeses.append(log_aermr11_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Sulphate Aerosol Mixing Ratio (AERMR11) of months Dec, Ene, Feb: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesDec_FebSulphate Aerosol Mixing Ratio' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesMar_Maysulphate_aerosol_mixing_ratio2003_2021CMAS(aermr11_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_aermr11_2003_2020 = np.log(aermr11_2003_2020.aermr11.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [3, 4, 5]:
        listaDsMeses.append(log_aermr11_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Sulphate Aerosol Mixing Ratio (AERMR11) of months Mar, Apr, May: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesMar_MaySulphate Aerosol Mixing Ratio' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesJun_Augsulphate_aerosol_mixing_ratio2003_2021CMAS(aermr11_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_aermr11_2003_2020 = np.log(aermr11_2003_2020.aermr11.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [6, 7, 8]:
        listaDsMeses.append(log_aermr11_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Sulphate Aerosol Mixing Ratio (AERMR11) of months Jun, Jul, Aug: ' + str(i) + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesJun_Ago Sulphate Aerosol Mixing Ratio' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesSep_Novsulphate_aerosol_mixing_ratio2003_2021CMAS(aermr11_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_aermr11_2003_2020 = np.log(aermr11_2003_2020.aermr11.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [9, 10, 11]:
        listaDsMeses.append(log_aermr11_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Sulphate Aerosol Mixing Ratio (AERMR11) of months Sep, Oct, Nov: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesSep_Nov Sulphate Aerosol Mixing Ratio' + str(i) + '.png')  # ,dpi=720

    plt.show()

def sulphate_aerosol_mixing_ratio2003_2021CMAS(ds, aermr11_2003_2020):
    # Calcular el log de los datos de aermr11
    log_aermr11_2003_2020 = np.log(aermr11_2003_2020.aermr11.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    meanMax1 = list()
    for t in range(len(ds.time)):
        aermr11d = aermr11_2003_2020.aermr11.isel(time=t)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # k = -1 / (aermr11C.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = log_aermr11_2003_2020[0, t, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)
        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        print(Hx)

        meanMax1.append(Hx.max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("plasma")).reversed()
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot AERMR11
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. Sulphate Aerosol Mixing Ratio (AERMR11) " + str(aermr11d.time.values))
        plt.savefig('../Datos/salidas/'
                    + 'Sulphate Aerosol Mixing Ratio' + str(aermr11d.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))


def leerso2_precursor_mixing_ratio2003_2021CMAS():
    with xr.open_dataset('so2_precursor_mixing_ratio2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.aermr12.attrs)

        # Append the height dimension
        aermr12_2003_2020 = ds.expand_dims({"height": 9})

        # aermr12_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # seleccionar periodo de tiempo

        # calculate and assign the height coordinate to the set.
        # calculate height = H * ln(level0 / level) where H = -7.9
        aermr12_2003_2020 = aermr12_2003_2020.assign_coords(height=("height", (
                7.9 * np.log(ds.level[8].values / ds.level.sel(level=slice(500, 1000)).values))))

        # Calcular de los valores de so2 / valores de aermr12 en 950 HPa
        aermr12_2003_2020['aermr12'] = aermr12_2003_2020.aermr12 / (aermr12_2003_2020.aermr12.isel(height=7, level=7)+1)
        print(aermr12_2003_2020)

        # Calcular el log de los datos de aermr12
        # log_aermr12_2003_2020 = np.log(aermr12_2003_2020.aermr12.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    return ds, aermr12_2003_2020


def meanMesesso2_precursor_mixing_ratio2003_2021CMAS(aermr12_2003_2020):
    # with xr.open_dataset('sulphur_dioxide2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #     print(ds.aermr12.attrs)
    # print(ds)
    #
    # # aermr12_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # # calculate and assign the height coordinate to the set.
    # aermr12_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                  ds.level[8].values))))
    listaDsMeses, listaMeanMeses = list(), list()
    # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # aermr12_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(aermr12_2003_2020.height.values)
    # Swap the level and height dimensions
    # aermr12_2003_2020 = aermr12_2003_2020.swap_dims({"level": "height"})
    # seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont = 0
    # Calcular el log de los datos de aermr12
    log_aermr12_2003_2020 = np.log(aermr12_2003_2020.aermr12.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    print(log_aermr12_2003_2020.sel(time='2003' + '-' + '01' + '-01'))
    for i in range(1, 13):
        listaDsMeses.append(log_aermr12_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate aermr12d[z] / aermr12d[0] normalización
        aermr12d = listaMeanMeses[cont]
        print(aermr12d[:8, :8, :, :])

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (aermr12dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = aermr12d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)

        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        print(Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot AERMR12
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')
        # ax.set_ylabel('Km')

        plt.title('Mean vertical profile scale SO2 precursor mixing ratio (AERMR12) of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('../Datos/salidas/'
                    + 'MediaMesSO2 precursor mixing ratio' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1


def so2_precursor_mixing_ratio2003_2021CMAS(ds, aermr12_2003_2020):
    # Calcular el log de los datos de aermr12
    log_aermr12_2003_2020 = np.log(aermr12_2003_2020.aermr11.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    meanMax1 = list()
    for t in range(len(ds.time)):
        aermr12d = aermr12_2003_2020.aermr12.isel(time=t)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # k = -1 / (aermr12C.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = log_aermr12_2003_2020[0, t, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)
        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        print(Hx)

        meanMax1.append(Hx.max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("RdYlBu")).reversed()
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot AERMR11
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. SO2 precursor mixing ratio (AERMR12) " + str(aermr12d.time.values))
        plt.savefig('../Datos/salidas/'
                    + 'SO2 precursor mixing ratio' + str(aermr12d.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))


def leerrelative_humidity2003_2021CMAS(): #specific_humidity2003_2021CMAS.nc
    with xr.open_dataset(
            'relative_humidity2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.r.attrs)

        # Append the height dimension
        r_2003_2020 = ds.expand_dims({"height": 9})

        # r_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # seleccionar periodo de tiempo

        # calculate and assign the height coordinate to the set.
        # calculate height = H * ln(level0 / level) where H = -7.9
        r_2003_2020 = r_2003_2020.assign_coords(height=("height", (
                7.9 * np.log(ds.level[8].values / ds.level.sel(level=slice(500, 1000)).values))))

        # Calcular de los valores de so2 / valores de r en 950 HPa
        r_2003_2020['r'] = r_2003_2020.r / (r_2003_2020.r.isel(height=7, level=7)+1)
        print(r_2003_2020.r)

        # Calcular el log de los datos de r
        # log_r_2003_2020 = np.log(r_2003_2020.r.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    return ds, r_2003_2020


def meanMesesrelative_humidity2003_2021CMAS(r_2003_2020):
    # with xr.open_dataset('sulphur_dioxide2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #     print(ds.r.attrs)
    # print(ds)
    #
    # # r_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # # calculate and assign the height coordinate to the set.
    # r_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                  ds.level[8].values))))
    listaDsMeses, listaMeanMeses = list(), list()
    # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # r_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(r_2003_2020.height.values)
    # Swap the level and height dimensions
    # r_2003_2020 = r_2003_2020.swap_dims({"level": "height"})
    # seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont = 0
    # Calcular el log de los datos de r
    log_r_2003_2020 = np.log(r_2003_2020.r.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    print(log_r_2003_2020.sel(time='2003' + '-' + '01' + '-01'))
    for i in range(1, 13):
        listaDsMeses.append(log_r_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate rd[z] / rd[0] normalización
        rd = listaMeanMeses[cont]

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (rdC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = rd[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)

        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        # Hx = k.polyfit_coefficients[0]
        print(Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot r
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')
        # ax.set_ylabel('Km')

        plt.title('Mean Relative humidity (r) of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('../Datos/salidas/'
                    + 'MediaMesRelative humidity' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1

def meanMesesDec_Febrelative_humidity2003_2021CMAS(r_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2 2----------------------------3
    log_r_2003_2020 = np.log(r_2003_2020.r.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [12, 1, 2]:
        listaDsMeses.append(log_r_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950) +1)
    # Hx = ((k.polyfit_coefficients[0] * 950) + 1) / 8
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Relative humidity (r) of months Dec, Ene, Feb: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesDec_FebRelative humidity' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesMar_Mayrelative_humidity2003_2021CMAS(r_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2 3---------------------------------------------3
    log_r_2003_2020 = np.log(r_2003_2020.r.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [3, 4, 5]:
        listaDsMeses.append(log_r_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Relative humidity (r) of months Mar, Apr, May: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesMar_MayRelative humidity' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesJun_Augrelative_humidity2003_2021CMAS(r_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2 4------------------------------------------------5
    log_r_2003_2020 = np.log(r_2003_2020.r.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [6, 7, 8]:
        listaDsMeses.append(log_r_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Relative humidity (r) of months Jun, Jul, Aug: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesJun_Aug Relative humidity' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesSep_Novrelative_humidity2003_2021CMAS(r_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2 5---------------------------------------------6
    log_r_2003_2020 = np.log(r_2003_2020.r.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [9, 10, 11]:
        listaDsMeses.append(log_r_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                      linewidth=0.5, vmin=0, vmax=2.8,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Relative humidity (r) of months Sep, Oct, Nov: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/'
                    + 'MediaMesSep_Nov Relative humidity' + str(i) + '.png')  # ,dpi=720

    plt.show()


def relative_humidity2003_2021CMAS(ds, r_2003_2020):
    # Calcular el log de los datos de r 6--------------------------------------------7
    log_r_2003_2020 = np.log(r_2003_2020.r.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    meanMax1 = list()
    for t in range(len(ds.time)):
        rd = r_2003_2020.r.isel(time=t)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # k = -1 / (rC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = log_r_2003_2020[0, t, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)
        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        print(Hx)

        meanMax1.append(Hx.max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("RdYlBu")).reversed()
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot r
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values, alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. Relative humidity (r) " + str(rd.time.values))
        plt.savefig('../Datos/salidas/'
                    + 'Relative humidity' + str(rd.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))


def leerspecific_humidity2003_2021CMAS(): #specific_humidity2003_2021CMAS.nc
    with xr.open_dataset(
            'specific_humidity2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.q.attrs)

        # Append the height dimension
        q_2003_2020 = ds.expand_dims({"height": 9})

        # r_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # seleccionar periodo de tiempo

        # calculate and assign the height coordinate to the set.
        # calculate height = H * ln(level0 / level) where H = -7.9
        q_2003_2020 = q_2003_2020.assign_coords(height=("height", (
                7.9 * np.log(ds.level[8].values / ds.level.sel(level=slice(500, 1000)).values))))

        # Calcular de los valores de so2 / valores de q en 950 HPa
        q_2003_2020['q'] = q_2003_2020.q / (q_2003_2020.q.isel(height=7, level=7)+1)
        print(q_2003_2020.q)

        # Calcular el log de los datos de q
        # log_q_2003_2020 = np.log(q_2003_2020.q.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    return ds, q_2003_2020


def meanMesesspecific_humidity2003_2021CMAS(q_2003_2020):
    # with xr.open_dataset('sulphur_dioxide2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #     print(ds.q.attrs)
    # print(ds)
    #
    # # q_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # # calculate and assign the height coordinate to the set.
    # q_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                  ds.level[8].values))))
    listaDsMeses, listaMeanMeses = list(), list()
    # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # q_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(q_2003_2020.height.values)
    # Swap the level and height dimensions
    # q_2003_2020 = q_2003_2020.swap_dims({"level": "height"})
    # seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont = 0
    # Calcular el log de los datos de r
    log_q_2003_2020 = np.log(q_2003_2020.q.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    print(log_q_2003_2020.sel(time='2003' + '-' + '01' + '-01'))
    for i in range(1, 13):
        listaDsMeses.append(log_q_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate rd[z] / rd[0] normalización
        rd = listaMeanMeses[cont]

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (rdC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = rd[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)

        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        # Hx = k.polyfit_coefficients[0]
        print(Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot r
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 2.8, num=10)), vmin=0, vmax=2.8)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 2.8, num=10)),
                          linewidth=0.5, vmin=0, vmax=2.8,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')
        # ax.set_ylabel('Km')

        plt.title('Mean Specific humidity (q) of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('../Datos/salidas/0101/'
                    + 'MediaMesSpecific humidity' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1


def meanMesesDec_Febspecific_humidity2003_2021CMAS(q_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_q_2003_2020 = np.log(q_2003_2020.q.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [12, 1, 2]:
        listaDsMeses.append(log_q_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950) +1)
    # Hx = ((k.polyfit_coefficients[0] * 950) + 1) / 8
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 3, num=10)), vmin=0, vmax=3)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 3, num=10)),
                      linewidth=0.5, vmin=0, vmax=3,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Specific humidity (q) of months Dec, Ene, Feb: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/0101/'
                    + 'MediaMesDec_FebSpecific humidity' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesMar_Mayspecific_humidity2003_2021CMAS(q_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_q_2003_2020 = np.log(q_2003_2020.q.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [3, 4, 5]:
        listaDsMeses.append(log_q_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 3, num=10)), vmin=0, vmax=3)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 3, num=10)),
                      linewidth=0.5, vmin=0, vmax=3,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Specific humidity (q) of months Mar, Apr, May: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/0101/'
                    + 'MediaMesMar_MaySpecific humidity' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesJun_Augspecific_humidity2003_2021CMAS(q_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_q_2003_2020 = np.log(q_2003_2020.q.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [6, 7, 8]:
        listaDsMeses.append(log_q_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 3, num=10)), vmin=0, vmax=3)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 3, num=10)),
                      linewidth=0.5, vmin=0, vmax=3,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Specific humidity (q) of months Jun, Jul, Aug: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/0101/'
                    + 'MediaMesJun_Aug Specific humidity' + str(i) + '.png')  # ,dpi=720

    plt.show()


def meanMesesSep_Novspecific_humidity2003_2021CMAS(q_2003_2020):
    listaDsMeses, listaMeanMeses = list(), list()

    # Calcular el log de los datos de so2
    log_q_2003_2020 = np.log(q_2003_2020.q.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))

    # Seleccionar los periodos por meses y calcular las medias de los meses de los años
    cont, d, sumaDsAnho = 0, 0, 0
    for i in [9, 10, 11]:
        listaDsMeses.append(log_q_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        if i < 6:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4])):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        else:
            for j in range(int(str(listaDsMeses[cont].time.min().values)[0:4]),
                           int(str(listaDsMeses[cont].time.max().values)[0:4]) + 1):
                print(listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01'))
                sumaDsAnho = sumaDsAnho + listaDsMeses[cont].sel(time=str(j) + '-' + str(i) + '-01')
                d += 1
        cont += 1
    # Calcular la media de las matrices por meses
    print('Suma de los meses: [12, 1, 2] total: ' + str(d) + ' para todos los años es:\n ', sumaDsAnho)
    so24d = sumaDsAnho / d

    # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
    k = so24d[0, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
    print("Valores del mejor ajuste polyfit \n", k)

    Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
    print(Hx)

    # cmap = mpl.cm.jet  # seleccionar el color del mapa
    cmap = copy.copy(mpl.cm.get_cmap("jet"))
    cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
    cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

    # plot SO2
    fig = plt.figure(1, figsize=(15., 12.))
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

    CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values,  alpha=0.65,
                    transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                    levels=(np.linspace(0, 3, num=10)), vmin=0, vmax=3)

    CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                      transform=ccrs.PlateCarree(), levels=(np.linspace(0, 3, num=10)),
                      linewidth=0.5, vmin=0, vmax=3,
                      cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
    ax.set_extent((-180, 180, -90, 90))
    ax.gridlines(draw_labels=True)
    ax.coastlines(resolution='50m', linewidth=0.75)
    ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
    CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

    plt.title('Mean vertical profile scale Specific humidity (q) of months Sep, Oct, Nov: ' + ' 2003 - 2020 ')
    plt.savefig('../Datos/salidas/0101/'
                    + 'MediaMesSep_Nov Specific humidity' + str(i) + '.png')  # ,dpi=720

    plt.show()


def specific_humidity2003_2021CMAS(ds, q_2003_2020):
    # Calcular el log de los datos de q
    log_q_2003_2020 = np.log(q_2003_2020.q.sel(level=slice(600, 950), height=slice(4.036, 0.4052)))
    meanMax1 = list()
    for t in range(len(ds.time)):
        rd = q_2003_2020.q.isel(time=t)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # k = -1 / (rC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        k = log_q_2003_2020[0, t, :8, :, :].polyfit(dim='level', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", k)
        Hx = 8 / ((k.polyfit_coefficients[0] * 950)+1)
        print(Hx)

        meanMax1.append(Hx.max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("RdYlBu")).reversed()
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot r
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))

        CS = ax.contour(Hx.longitude.values, Hx.latitude.values, Hx.values, alpha=0.65,
                        transform=ccrs.PlateCarree(), cbar_kwargs={'label': 'Km'}, linewidth=0.5,
                        levels=(np.linspace(0, 3, num=10)), vmin=0, vmax=3)

        CS1 = ax.contourf(Hx.longitude.values, Hx.latitude.values, Hx.values,
                          transform=ccrs.PlateCarree(), levels=(np.linspace(0, 3, num=10)),
                          linewidth=0.5, vmin=0, vmax=3,
                          cmap=cmap)  # Hx.longitude.values, Hx.latitude.values, Hx.values
        ax.set_extent((-180, 180, -90, 90))
        ax.gridlines(draw_labels=True)
        ax.coastlines(resolution='50m', linewidth=0.75)
        ax.clabel(CS, inline=0.5, fontsize=12, colors='k', fmt='%.1f')
        CB = fig.colorbar(CS1, shrink=0.5, extend='both', orientation='vertical', label='Km', format='%.1f')

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. Specific humidity (q) " + str(rd.time.values))
        plt.savefig('../Datos/salidas/0101/'
                    + 'Specific humidity' + str(rd.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))