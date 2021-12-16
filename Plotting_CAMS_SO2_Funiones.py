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
    pach = '/home/leo/Documentos/Universidad/Trabajo_de_investigación/' \
           'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/0.5SulfurdioxideResultados10122021/'
    archivos = os.listdir(pach)
    img_arr = []

    for i in tqdm(range(len(archivos))):
        nomArchivos = archivos[i]
        dirArchivos = pach + nomArchivos

        leer_img = imageio.imread(dirArchivos)
        img_arr.append(leer_img)
        imageio.mimwrite('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                         'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                         '0.5SulfurdioxideResultados10122021/gif.gif', img_arr, 'GIF', duration='0.5')


def leersulphur_dioxide2003_2021CMAS():
    with xr.open_dataset('sulphur_dioxide2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.so2.attrs)
    # print(ds)

    # so2_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # calculate and assign the height coordinate to the set.
    # so2_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                 ds.level[8].values))))

    so2_2003_2020 = ds.assign_coords(height=("level", [np.percentile(ds.so2[0, 0, :, :].values, 87),
                                                       np.percentile(ds.so2[0, 1, :, :].values, 87),
                                                       np.percentile(ds.so2[0, 2, :, :].values, 87),
                                                       np.percentile(ds.so2[0, 3, :, :].values, 87),
                                                       np.percentile(ds.so2[0, 4, :, :].values, 87),
                                                       np.percentile(ds.so2[0, 5, :, :].values, 87),
                                                       np.percentile(ds.so2[0, 6, :, :].values, 87),
                                                       np.percentile(ds.so2[0, 7, :, :].values, 87),
                                                       np.percentile(ds.so2[0, 8, :, :].values, 87)]))
    # so2_2003_2020 = ds.assign_coords(height=("level", [1.56887836e-10,2.09070095e-10,2.78646439e-10,3.48222784e-10,
    #                                                   4.00405042e-10,4.70038231e-10,5.22220489e-10,5.74402748e-10,
    #                                                   6.09190920e-10]))

    print(so2_2003_2020.height.values)
    # Swap the level and height dimensions
    so2_2003_2020 = so2_2003_2020.swap_dims({"level": "height"})
    print(so2_2003_2020.so2[0, :, 0, 0].values)
    print(np.percentile(so2_2003_2020.so2[0, 0, :, :].values, 87))
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
    for i in range(1, 13):
        listaDsMeses.append(so2_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].so2.sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].so2.sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate so24d[z] / so24d[0] normalización
        so24d = listaMeanMeses[cont]
        print(so24d)
        so24dC = so24d.copy()
        for h in range(len(so24d.height)):
            so24dC[h, :, :] = so24d.isel(height=h).values / so24d.isel(height=8).values  # so24d[8, :, :].values
        print("Valores de los datos con los calculos de la normalizacion \n\n\n\n\n", type(so24dC), "\n", so24dC)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (so24dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        Hx = so24dC.polyfit(dim='height', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot SO2
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()

        Hx.polyfit_coefficients[1, :, :].plot.contourf(
            cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'},
            levels=(np.linspace(0.5, Hx.polyfit_coefficients[1, :, :].max().values,
                                num=10)), cmap=cmap, vmin=0.5,
            vmax=Hx.polyfit_coefficients[1, :, :].max().values)  # Hx.polyfit_coefficients[0,:,:].plot.contourf()

        ax.gridlines(draw_labels=True)

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title('Mean Sulfur dioxide of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                    + 'MediaMesSulfur dioxide' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1


def sulphur_dioxide2003_2021CMAS(ds, so2_2003_2020):
    # with xr.open_dataset('sulphur_dioxide2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #     print(ds.so2.attrs)
    # print(ds)
    #
    # # so2_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # # calculate and assign the height coordinate to the set.
    # so2_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                  ds.level[8].values))))
    meanMax1 = list()
    # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # so2_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(so2_2003_2020.height.values)
    # Swap the level and height dimensions
    # so2_2003_2020 = so2_2003_2020.swap_dims({"level": "height"})
    # selected = so2_2003_2020.where(lambda x: x.time.dt.year == 2020, drop=True) # seleccionar un periodo de
    # print("Valores de los datos con los calculos de altura y la seleccion de la dimension time por anho \n",selected)
    # print("Valores de los datos con los calculos de altura \n",so2_2003_2020.sel(time="2003-01-01"))
    for t in range(len(ds.time)):
        so24d = so2_2003_2020.so2.isel(time=t)
        so24dC = so24d.copy()
        print(so24d)
        # calculate so24d[z] / so24d[0] normalización
        for h in range(len(so24d.height)):
            so24dC[h, :, :] = so24d.isel(height=h).values / so24d[8, :, :].values
        print("Valores de los datos con los calculos de la normalizacion \n\n\n\n\n", type(so24dC), "\n", so24dC)
        # so24d = so24d.sel(height= slice(5.476,-0.0))
        # so24d = so2_2003_2020.so2.sel(level = slice(500,1000)).polyfit(dim='level', deg=1)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (so24dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        Hx = so24dC.polyfit(dim='height', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", Hx)

        # evaluar los resultados de la regresión lineal por ninimos cuadrados
        # HXPolyvald = xr.polyval(so24dC['height'], Hx.polyfit_coefficients[0, :, :], degree_dim='degree')
        # HXPolyvald = HXPolyvald*100
        # print(" Evaluacion de los Valores del mejor ajuste polyval \n", HXPolyvald)
        ##print(Hx.polyfit_coefficients)
        ##print("min()\n", Hx.polyfit_coefficients[0, :, :].min().values)
        ##print("max()\n", Hx.polyfit_coefficients[0, :, :].max().values)
        # print("min()\n", Hx.polyfit_coefficients[1, :, :].min().values)
        # print("max()\n", Hx.polyfit_coefficients[1, :, :].max().values)
        meanMax1.append(Hx.polyfit_coefficients[0, :, :].max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot SO2
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()
        # ax.set_ylabel('YLabel 1')
        # fig.align_ylabels()
        # ax.set_ylabel('verbosity coefficient')
        # ax.set_yticks([0,HXPolyvald['latitude'].max().values])
        # ax.set_xticks([0, HXPolyvald['longitude'].max().values])

        # HXPolyvald[1,:,:].plot.contourf(cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'},
        #                                levels=(np.linspace(0.5, HXPolyvald.max().values, num=15)))  # HXPolyvald[8,:,:].plot.contourf(levels=(np.linspace(0.5, 5, num=15)))
        # Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(np.linspace(0.5, 5, num=15)))  # Hx.polyfit_coefficients[0,:,:].plot.contourf()

        Hx.polyfit_coefficients[0, :, :].plot.contourf(
            cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'},
            levels=(np.linspace(0.5, Hx.polyfit_coefficients[0, :, :].max().values,
                                num=10)), cmap=cmap, vmin=0.5,
            vmax=Hx.polyfit_coefficients[0, :, :].max().values)  # Hx.polyfit_coefficients[0,:,:].plot.contourf()
        # Hx.polyfit_coefficients[1, :, :].plot.contourf(
        #    cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'}, levels=(
        #        np.linspace(0.5, Hx.polyfit_coefficients[1, :, :].max().values,
        #                    num=33)))  # Hx.polyfit_coefficients[1,:,:].plot.contourf()
        # Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(
        #    np.linspace(Hx.polyfit_coefficients[0,:,:].min().values, Hx.polyfit_coefficients[0,:,:].max().values,
        #    num=10000)))  # -1 / Hx.polyfit_coefficients[0,:,:].plot.contourf()
        # Hx.polyfit_coefficients[1, :, :].plot.contourf(levels=(
        #    np.linspace(Hx.polyfit_coefficients[1,:,:].min().values, Hx.polyfit_coefficients[1,:,:].max().values,
        #    num=200)))# -1 / Hx.polyfit_coefficients[1,:,:].plot.contourf()

        ax.gridlines(draw_labels=True)

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. Sulfur dioxide (SO2) " + str(so24d.time.values))
        plt.savefig('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                    + 'Sulfur dioxide' + str(so24d.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))


def leersulphate_aerosol_mixing_ratio2003_2021CMAS():
    with xr.open_dataset(
            'sulphate_aerosol_mixing_ratio2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.aermr11.attrs)
    # print(ds)

    # aermr11_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # calculate and assign the height coordinate to the set.
    # aermr11_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                     ds.level[8].values))))

    aermr11_2003_2020 = ds.assign_coords(height=("level", [np.percentile(ds.aermr11[0, 0, :, :].values, 87),
                                                           np.percentile(ds.aermr11[0, 1, :, :].values, 87),
                                                           np.percentile(ds.aermr11[0, 2, :, :].values, 87),
                                                           np.percentile(ds.aermr11[0, 3, :, :].values, 87),
                                                           np.percentile(ds.aermr11[0, 4, :, :].values, 87),
                                                           np.percentile(ds.aermr11[0, 5, :, :].values, 87),
                                                           np.percentile(ds.aermr11[0, 6, :, :].values, 87),
                                                           np.percentile(ds.aermr11[0, 7, :, :].values, 87),
                                                           np.percentile(ds.aermr11[0, 8, :, :].values, 87)]))

    print(aermr11_2003_2020.height.values)
    # Swap the level and height dimensions
    aermr11_2003_2020 = aermr11_2003_2020.swap_dims({"level": "height"})
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
    for i in range(1, 13):
        listaDsMeses.append(aermr11_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].aermr11.sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].aermr11.sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate aermr114d[z] / aermr114d[0] normalización
        aermr114d = listaMeanMeses[cont]
        print(aermr114d)
        aermr114dC = aermr114d.copy()
        for h in range(len(aermr114d.height)):
            aermr114dC[h, :, :] = aermr114d.isel(height=h).values / aermr114d.isel(
                height=8).values  # aermr114d[8, :, :].values
        print("Valores de los datos con los calculos de la normalizacion \n\n\n\n\n", type(aermr114dC), "\n",
              aermr114dC)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (aermr114dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        Hx = aermr114dC.polyfit(dim='height', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot aermr11
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()

        Hx.polyfit_coefficients[1, :, :].plot.contourf(
            cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'},
            levels=(np.linspace(0.5, Hx.polyfit_coefficients[1, :, :].max().values,
                                num=10)), cmap=cmap, vmin=0.5,
            vmax=Hx.polyfit_coefficients[1, :, :].max().values)  # Hx.polyfit_coefficients[0,:,:].plot.contourf()

        ax.gridlines(draw_labels=True)

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title('Mean Sulphate Aerosol Mixing Ratio of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                    + 'MediaMesSulphate Aerosol Mixing Ratio' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1


def sulphate_aerosol_mixing_ratio2003_2021CMAS(ds, aermr11_2003_2020):
    # with xr.open_dataset(
    #         'sulphate_aerosol_mixing_ratio2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #     print(ds.aermr11.attrs)  # time slice
    # print(ds)
    #
    # # aermr11_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01'))
    # # calculate and assign the height coordinate to the set.
    # aermr11_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                      ds.level[8].values))))
    meanMax1 = list()
    # # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # # aermr11_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(aermr11_2003_2020.height.values)
    # # Swap the level and height dimensions
    # aermr11_2003_2020 = aermr11_2003_2020.swap_dims({"level": "height"})
    # selected = aermr11_2003_2020.where(lambda x: x.time.dt.year == 2020, drop=True) # seleccionar un periodo de
    # print("Valores de los datos con los calculos de altura y la seleccion de la dimension time por anho \n",selected)
    # print("Valores de los datos con los calculos de altura \n",aermr11_2003_2020.sel(time="2003-01-01"))
    for t in range(len(ds.time)):
        aermr114d = aermr11_2003_2020.aermr11.isel(time=t)
        aermr114dC = aermr114d.copy()
        print(aermr114d)
        # calculate aermr114d[z] / aermr114d[0] normalización
        # for h in range(len(aermr114d.height)):
        #    aermr114dC[h, :, :] = aermr114d.isel(height=h).values / aermr114d[8, :, :].values
        print("Valores de los datos con los calculos de la normalizacion \n\n\n\n\n", type(aermr114dC), "\n",
              aermr114dC)
        # aermr114d = aermr114d.sel(height= slice(5.476,-0.0))
        # aermr114d = aermr11_2003_2020.aermr11.sel(level = slice(500,1000)).polyfit(dim='level', deg=1)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (aermr114dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        Hx = aermr114dC.polyfit(dim='height', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", Hx)

        HXPolyvald = xr.polyval(aermr114dC['height'], Hx.polyfit_coefficients[0, :, :], degree_dim='degree')
        # HXPolyvald = HXPolyvald*100
        print(" Evaluacion de los Valores del mejor ajuste polyval \n", HXPolyvald)
        ##print(Hx.polyfit_coefficients)
        ##print("min()\n", Hx.polyfit_coefficients[0, :, :].min().values)
        ##print("max()\n", Hx.polyfit_coefficients[0, :, :].max().values)
        # print("min()\n", Hx.polyfit_coefficients[1, :, :].min().values)
        # print("max()\n", Hx.polyfit_coefficients[1, :, :].max().values)
        meanMax1.append(Hx.polyfit_coefficients[0, :, :].max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot aermr11
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()
        # ax.set_ylabel('YLabel 1')
        # fig.align_ylabels()
        # ax.set_ylabel('verbosity coefficient')
        # ax.set_yticks([0,HXPolyvald['latitude'].max().values])
        # ax.set_xticks([0, HXPolyvald['longitude'].max().values])

        # HXPolyvald[1,:,:].plot.contourf(cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'},
        #                                levels=(np.linspace(0.5, HXPolyvald.max().values, num=15)))  # HXPolyvald[8,:,:].plot.contourf(levels=(np.linspace(0.5, 5, num=15)))
        # Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(np.linspace(0.5, 5, num=15)))  # Hx.polyfit_coefficients[0,:,:].plot.contourf()

        Hx.polyfit_coefficients[0, :, :].plot.contourf(
            cbar_kwargs={'label': 'kg kg ** - 1 Sulphate Aerosol Mixing Ratio'},
            levels=(np.linspace(0.5, Hx.polyfit_coefficients[0, :, :].max().values,
                                num=10)), cmap=cmap, vmin=0.5,
            vmax=Hx.polyfit_coefficients[0, :, :].max().values)  # Hx.polyfit_coefficients[0,:,:].plot.contourf()
        # Hx.polyfit_coefficients[1, :, :].plot.contourf(
        #    cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'}, levels=(
        #        np.linspace(0.5, Hx.polyfit_coefficients[1, :, :].max().values,
        #                    num=33)))  # Hx.polyfit_coefficients[1,:,:].plot.contourf()
        # Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(
        #    np.linspace(Hx.polyfit_coefficients[0,:,:].min().values, Hx.polyfit_coefficients[0,:,:].max().values,
        #    num=10000)))  # -1 / Hx.polyfit_coefficients[0,:,:].plot.contourf()
        # Hx.polyfit_coefficients[1, :, :].plot.contourf(levels=(
        #    np.linspace(Hx.polyfit_coefficients[1,:,:].min().values, Hx.polyfit_coefficients[1,:,:].max().values,
        #    num=200)))# -1 / Hx.polyfit_coefficients[1,:,:].plot.contourf()

        ax.gridlines(draw_labels=True)

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. Sulphate Aerosol Mixing Ratio (aermr11) " + str(aermr114d.time.values))
        plt.savefig('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                    + 'Sulphate Aerosol Mixing Ratio' + str(aermr114d.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))


def leerso2_precursor_mixing_ratio2003_2021CMAS():
    with xr.open_dataset('so2_precursor_mixing_ratio2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.aermr12.attrs)
    # print(ds)

    # aermr12_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # calculate and assign the height coordinate to the set.
    # aermr12_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                     ds.level[8].values))))

    aermr12_2003_2020 = ds.assign_coords(height=("level", [np.percentile(ds.aermr12[0, 0, :, :].values, 87),
                                                           np.percentile(ds.aermr12[0, 1, :, :].values, 87),
                                                           np.percentile(ds.aermr12[0, 2, :, :].values, 87),
                                                           np.percentile(ds.aermr12[0, 3, :, :].values, 87),
                                                           np.percentile(ds.aermr12[0, 4, :, :].values, 87),
                                                           np.percentile(ds.aermr12[0, 5, :, :].values, 87),
                                                           np.percentile(ds.aermr12[0, 6, :, :].values, 87),
                                                           np.percentile(ds.aermr12[0, 7, :, :].values, 87),
                                                           np.percentile(ds.aermr12[0, 8, :, :].values, 87)]))

    print(aermr12_2003_2020.height.values)
    # Swap the level and height dimensions
    aermr12_2003_2020 = aermr12_2003_2020.swap_dims({"level": "height"})
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
    for i in range(1, 13):
        listaDsMeses.append(aermr12_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].aermr12.sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].aermr12.sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate aermr124d[z] / aermr124d[0] normalización
        aermr124d = listaMeanMeses[cont]
        print(aermr124d)
        aermr124dC = aermr124d.copy()
        for h in range(len(aermr124d.height)):
            aermr124dC[h, :, :] = aermr124d.isel(height=h).values / aermr124d.isel(
                height=8).values  # aermr124d[8, :, :].values
        print("Valores de los datos con los calculos de la normalizacion \n\n\n\n\n", type(aermr124dC), "\n",
              aermr124dC)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (aermr124dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        Hx = aermr124dC.polyfit(dim='height', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot aermr12
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()

        Hx.polyfit_coefficients[1, :, :].plot.contourf(
            cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'},
            levels=(np.linspace(0.5, Hx.polyfit_coefficients[1, :, :].max().values,
                                num=10)), cmap=cmap, vmin=0.5,
            vmax=Hx.polyfit_coefficients[1, :, :].max().values)  # Hx.polyfit_coefficients[0,:,:].plot.contourf()

        ax.gridlines(draw_labels=True)

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title('Mean SO2 precursor mixing ratio of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                    + 'MediaMesSO2 precursor mixing ratio' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1


def so2_precursor_mixing_ratio2003_2021CMAS(ds, aermr12_2003_2020):
    # with xr.open_dataset('so2_precursor_mixing_ratio2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #    print(ds.aermr12.attrs)  # time slice
    # print(ds)

    # aermr12_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01'))
    # calculate and assign the height coordinate to the set.
    # aermr12_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                     ds.level[8].values))))
    meanMax1 = list()
    # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # aermr12_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(aermr12_2003_2020.height.values)
    # Swap the level and height dimensions
    # aermr12_2003_2020 = aermr12_2003_2020.swap_dims({"level": "height"})
    # selected = aermr12_2003_2020.where(lambda x: x.time.dt.year == 2020, drop=True) # seleccionar un periodo de
    # print("Valores de los datos con los calculos de altura y la seleccion de la dimension time por anho \n",selected)
    # print("Valores de los datos con los calculos de altura \n",aermr12_2003_2020.sel(time="2003-01-01"))
    for t in range(len(ds.time)):
        aermr124d = aermr12_2003_2020.aermr12.isel(time=t)
        aermr124dC = aermr124d.copy()
        print(aermr124d)
        # calculate aermr124d[z] / aermr124d[0] normalización
        # for h in range(len(aermr124d.height)):
        #    aermr124dC[h, :, :] = aermr124d.isel(height=h).values / aermr124d.isel(
        #        height=8).values  # aermr124d[8, :, :].values
        print("Valores de los datos con los calculos de la normalizacion \n\n\n\n\n", type(aermr124dC), "\n",
              aermr124dC)
        # aermr124d = aermr124d.sel(height= slice(5.476,-0.0))
        # aermr124d = aermr12_2003_2020.aermr12.sel(level = slice(500,1000)).polyfit(dim='level', deg=1)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (aermr124dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        Hx = aermr124dC.polyfit(dim='height', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", Hx)

        HXPolyvald = xr.polyval(aermr124dC['height'], Hx.polyfit_coefficients[0, :, :], degree_dim='degree')
        # HXPolyvald = HXPolyvald*100
        print(" Evaluacion de los Valores del mejor ajuste polyval \n", HXPolyvald)
        ##print(Hx.polyfit_coefficients)
        ##print("min()\n", Hx.polyfit_coefficients[0, :, :].min().values)
        ##print("max()\n", Hx.polyfit_coefficients[0, :, :].max().values)
        # print("min()\n", Hx.polyfit_coefficients[1, :, :].min().values)
        # print("max()\n", Hx.polyfit_coefficients[1, :, :].max().values)
        meanMax1.append(Hx.polyfit_coefficients[0, :, :].max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot aermr12
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()
        # ax.set_ylabel('YLabel 1')
        # fig.align_ylabels()
        # ax.set_ylabel('verbosity coefficient')
        # ax.set_yticks([0,HXPolyvald['latitude'].max().values])
        # ax.set_xticks([0, HXPolyvald['longitude'].max().values])

        # HXPolyvald[1,:,:].plot.contourf(cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'},
        #                                levels=(np.linspace(0.5, HXPolyvald.max().values, num=15)))  # HXPolyvald[8,:,:].plot.contourf(levels=(np.linspace(0.5, 5, num=15)))
        # Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(np.linspace(0.5, 5, num=15)))  # Hx.polyfit_coefficients[0,:,:].plot.contourf()

        Hx.polyfit_coefficients[0, :, :].plot.contourf(
            cbar_kwargs={'label': 'kg kg ** - 1 SO2 precursor mixing ratio'},
            levels=(np.linspace(0.5, Hx.polyfit_coefficients[0, :, :].max().values,
                                num=10)), cmap=cmap, vmin=0.5,
            vmax=Hx.polyfit_coefficients[0, :, :].max().values)  # Hx.polyfit_coefficients[0,:,:].plot.contourf()
        # Hx.polyfit_coefficients[1, :, :].plot.contourf(
        #    cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'}, levels=(
        #        np.linspace(0.5, Hx.polyfit_coefficients[1, :, :].max().values,
        #                    num=33)))  # Hx.polyfit_coefficients[1,:,:].plot.contourf()
        # Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(
        #    np.linspace(Hx.polyfit_coefficients[0,:,:].min().values, Hx.polyfit_coefficients[0,:,:].max().values,
        #    num=10000)))  # -1 / Hx.polyfit_coefficients[0,:,:].plot.contourf()
        # Hx.polyfit_coefficients[1, :, :].plot.contourf(levels=(
        #    np.linspace(Hx.polyfit_coefficients[1,:,:].min().values, Hx.polyfit_coefficients[1,:,:].max().values,
        #    num=200)))# -1 / Hx.polyfit_coefficients[1,:,:].plot.contourf()

        ax.gridlines(draw_labels=True)

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. SO2 precursor mixing ratio (aermr12) " + str(aermr124d.time.values))
        plt.savefig('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                    + 'SO2 precursor mixing ratio' + str(aermr124d.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))


def leerrelative_humidity2003_2021CMAS():
    with xr.open_dataset(
            'relative_humidity2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
        print(ds.r.attrs)
    # print(ds)

    # r_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01')) # time slice
    # calculate and assign the height coordinate to the set.
    # r_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                               ds.level[8].values))))

    r_2003_2020 = ds.assign_coords(height=("level", [np.percentile(ds.r[0, 0, :, :].values, 87),
                                                     np.percentile(ds.r[0, 1, :, :].values, 87),
                                                     np.percentile(ds.r[0, 2, :, :].values, 87),
                                                     np.percentile(ds.r[0, 3, :, :].values, 87),
                                                     np.percentile(ds.r[0, 4, :, :].values, 87),
                                                     np.percentile(ds.r[0, 5, :, :].values, 87),
                                                     np.percentile(ds.r[0, 6, :, :].values, 87),
                                                     np.percentile(ds.r[0, 7, :, :].values, 87),
                                                     np.percentile(ds.r[0, 8, :, :].values, 87)]))

    # r_2003_2020 = ds.assign_coords(
    #    height=("level", [77.201065, 75.67061, 79.01246, 77.544876, 79.79242, 84.234474, 87.02819, 90.14018, 99.33862]))

    print(r_2003_2020.height.values)
    # Swap the level and height dimensions
    r_2003_2020 = r_2003_2020.swap_dims({"level": "height"})
    print(r_2003_2020.r[0, :, :, :].values)
    print('mean', r_2003_2020.r[0, 0, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 0, :, :].values, 87))
    print('mean', r_2003_2020.r[0, 1, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 1, :, :].values, 87))
    print('mean', r_2003_2020.r[0, 2, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 2, :, :].values, 87))
    print('mean', r_2003_2020.r[0, 3, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 3, :, :].values, 87))
    print('mean', r_2003_2020.r[0, 4, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 4, :, :].values, 87))
    print('mean', r_2003_2020.r[0, 5, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 5, :, :].values, 87))
    print('mean', r_2003_2020.r[0, 6, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 6, :, :].values, 87))
    print('mean', r_2003_2020.r[0, 7, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 7, :, :].values, 87))
    print('mean', r_2003_2020.r[0, 8, :, :].mean().values)
    print('percentile', np.percentile(r_2003_2020.r[0, 8, :, :].values, 87))
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
    for i in range(1, 13):
        listaDsMeses.append(r_2003_2020.where(lambda x: x.time.dt.month == i, drop=True))
        contAnho, sumaDsAnho = 0, 0
        for j in range(int(str(listaDsMeses[cont].time[0].values)[0:4]),
                       int(str(listaDsMeses[0].time[-1].values)[0:4])):
            print(listaDsMeses[cont].r.sel(time=str(j) + '-' + str(i) + '-01'))
            sumaDsAnho = sumaDsAnho + listaDsMeses[cont].r.sel(time=str(j) + '-' + str(i) + '-01')
            contAnho += 1
        print('Suma de los meses: ' + str(i) + ' para todos los años es:\n ', sumaDsAnho)
        listaMeanMeses.append(sumaDsAnho / contAnho)
        print('Media de los meses: ' + str(i) + ' para todos los años es:\n ', listaMeanMeses[cont])

        # calculate r4d[z] / r4d[0] normalización
        r4d = listaMeanMeses[cont]
        print(r4d)
        r4dC = r4d.copy()
        for h in range(len(r4d.height)):
            r4dC[h, :, :] = r4d.isel(height=h).values / r4d.isel(
                height=8).values  # r4d[8, :, :].values
        print("Valores de los datos con los calculos de la normalizacion \n\n\n\n\n", type(r4dC), "\n",
              r4dC)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (r4dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        Hx = r4dC.polyfit(dim='height', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", Hx)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot r
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()

        Hx.polyfit_coefficients[1, :, :].plot.contourf(
            cbar_kwargs={'label': '% Relative humidity'},
            levels=(np.linspace(0.5, Hx.polyfit_coefficients[1, :, :].max().values,
                                num=10)), cmap=cmap, vmin=0.5,
            vmax=Hx.polyfit_coefficients[1, :, :].max().values)  # Hx.polyfit_coefficients[0,:,:].plot.contourf()

        ax.gridlines(draw_labels=True)

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title('Mean Relative humidity of months: ' + str(i) + ' 2003 - 2020 ')
        plt.savefig('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                    + 'MediaMesRelative humidity' + str(i) + '.png')  # ,dpi=720

        plt.show()

        cont += 1


def relative_humidity2003_2021CMAS(ds, r_2003_2020):
    # with xr.open_dataset(
    #         'relative_humidity2003_2021CMAS.nc') as ds:  # sulphur_dioxideEne_dic2003_2020CMAS.nc
    #     print(ds.r.attrs)  # time slice
    # print(ds)
    #
    # # r_2003_2020 = ds.sel(time=slice('2003-01-01', '2020-12-01'))
    # # calculate and assign the height coordinate to the set.
    # r_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
    #                                                                ds.level[8].values))))
    meanMax1 = list()
    # # altu = [0.5, 0.7, 1, 1.5, 1.75, 2, 2.5, 3, 4]
    # # r_2003_2020 = ds.assign_coords(height=('level',altu))#0.5, 0.7, 1, 1.25, 1,5, 1.75, 2, 2.25, 2.5, 3, 4
    # print(r_2003_2020.height.values)
    # # Swap the level and height dimensions
    # r_2003_2020 = r_2003_2020.swap_dims({"level": "height"})
    # selected = r_2003_2020.where(lambda x: x.time.dt.year == 2020, drop=True) # seleccionar un periodo de
    # print("Valores de los datos con los calculos de altura y la seleccion de la dimension time por anho \n",selected)
    # print("Valores de los datos con los calculos de altura \n",r_2003_2020.sel(time="2003-01-01"))
    for t in range(len(ds.time)):
        r4d = r_2003_2020.r.isel(time=t)
        r4dC = r4d.copy()
        print(r4d)
        # calculate r4d[z] / r4d[0] normalización
        # for h in range(len(r4d.height)):
        #    r4dC[h, :, :] = r4d.isel(height=h).values / r4d[8, :, :].values
        print("Valores de los datos con los calculos de la normalizacion \n\n\n\n\n", type(r4dC), "\n",
              r4dC)
        # r4d = r4d.sel(height= slice(5.476,-0.0))
        # r4d = r_2003_2020.r.sel(level = slice(500,1000)).polyfit(dim='level', deg=1)

        # calculo de la regresión lineal de los perfil vertical de la altura con minimos cuadrados
        # Hx = -1 / (r4dC.polyfit(dim='height', deg=1, full=True, cov=True))  # skipna=False
        Hx = r4dC.polyfit(dim='height', deg=1, full=True, cov=True)  # skipna=False
        print("Valores del mejor ajuste polyfit \n", Hx)

        HXPolyvald = xr.polyval(r4dC['height'], Hx.polyfit_coefficients[0, :, :], degree_dim='degree')
        # HXPolyvald = HXPolyvald*100
        print(" Evaluacion de los Valores del mejor ajuste polyval \n", HXPolyvald)
        ##print(Hx.polyfit_coefficients)
        ##print("min()\n", Hx.polyfit_coefficients[0, :, :].min().values)
        ##print("max()\n", Hx.polyfit_coefficients[0, :, :].max().values)
        # print("min()\n", Hx.polyfit_coefficients[1, :, :].min().values)
        # print("max()\n", Hx.polyfit_coefficients[1, :, :].max().values)
        meanMax1.append(Hx.polyfit_coefficients[0, :, :].max().values)

        # cmap = mpl.cm.jet  # seleccionar el color del mapa
        cmap = copy.copy(mpl.cm.get_cmap("jet"))
        cmap.set_over(color='indigo')  # seleccionar el valor maximo para traza en color blanco en el mapa
        cmap.set_under(color='w')  # seleccionar el valor minimo para traza en color blanco en el mapa

        # plot r
        fig = plt.figure(1, figsize=(15., 12.))
        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=0.0))
        ax.coastlines()
        # ax.set_ylabel('YLabel 1')
        # fig.align_ylabels()
        # ax.set_ylabel('verbosity coefficient')
        # ax.set_yticks([0,HXPolyvald['latitude'].max().values])
        # ax.set_xticks([0, HXPolyvald['longitude'].max().values])

        # HXPolyvald[1,:,:].plot.contourf(cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'},
        #                                levels=(np.linspace(0.5, HXPolyvald.max().values, num=15)))  # HXPolyvald[8,:,:].plot.contourf(levels=(np.linspace(0.5, 5, num=15)))
        # Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(np.linspace(0.5, 5, num=15)))  # Hx.polyfit_coefficients[0,:,:].plot.contourf()

        Hx.polyfit_coefficients[0, :, :].plot.contourf(
            cbar_kwargs={'label': '% Relative humidity'},
            levels=(np.linspace(0.5, Hx.polyfit_coefficients[0, :, :].max().values,
                                num=10)), cmap=cmap, vmin=0.5,
            vmax=Hx.polyfit_coefficients[0, :, :].max().values)  # Hx.polyfit_coefficients[0,:,:].plot.contourf()
        # Hx.polyfit_coefficients[1, :, :].plot.contourf(
        #    cbar_kwargs={'label': 'kg kg ** - 1 Sulfur dioxide mass_fraction_of_sulfur_dioxide_in_air'}, levels=(
        #        np.linspace(0.5, Hx.polyfit_coefficients[1, :, :].max().values,
        #                    num=33)))  # Hx.polyfit_coefficients[1,:,:].plot.contourf()
        # Hx.polyfit_coefficients[0, :, :].plot.contourf(levels=(
        #    np.linspace(Hx.polyfit_coefficients[0,:,:].min().values, Hx.polyfit_coefficients[0,:,:].max().values,
        #    num=10000)))  # -1 / Hx.polyfit_coefficients[0,:,:].plot.contourf()
        # Hx.polyfit_coefficients[1, :, :].plot.contourf(levels=(
        #    np.linspace(Hx.polyfit_coefficients[1,:,:].min().values, Hx.polyfit_coefficients[1,:,:].max().values,
        #    num=200)))# -1 / Hx.polyfit_coefficients[1,:,:].plot.contourf()

        ax.gridlines(draw_labels=True)

        # ax.set_ylabel('kg kg**-1 Sulphur dioxide mass_fraction_of_sulfur_dioxide_in_air')

        plt.title("Better fit. Relative humidity (r) " + str(r4d.time.values))
        plt.savefig('/home/leo/Documentos/Universidad/Trabajo_de_investigación/'
                    'PerfilesVerticalesContaminantesAtmosfera/Datos/salidas/'
                    + 'Relative humidity' + str(r4d.time.values) + '.png')  # ,dpi=720

        plt.show()

    print("Media de los Valores del mejor ajuste\n", np.mean(meanMax1))
