import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from matplotlib.cm import get_cmap
import cartopy.crs as ccrs
import cartopy.feature as cf
from cartopy.feature import NaturalEarthFeature
from xarray.plot.utils import legend_elements

# Example data
##To obtain the results of the research, the linear least squares regression model
##is used in the programming language python3.8.8, using the polyfit function
##from the Xarray0.19.0 library. The processes for using the model are described below.

##The data of the variables is obtained from the reanalysis (ERA5 monthly averaged
# data on single levels from 1979 to present, CAMS global reanalysis (EAC4) monthly
# averaged fields), in the netcdf file format type.

##Then the files are read before downloaded using the Xarray library,
# which is highly recommended for working with datasets and arrays of multiple
# dimensions. The files contain the data structure of a variable ('so2'),
# which contains four dimensions [time, pressure levels, latitude, longitude].

# create DataArray wish Xarray
"""
da = xr.DataArray(np.random.randn(2, 9, 241, 480),
                  dims=('time','level', 'latitude', 'longitude'),
                  coords={'time': np.array(['2003-01-01', '2003-02-01',], dtype='datetime64'),
                          'level':[500,600,700,800,850,900,925,950,1000],
                          'latitude': np.linspace(-90,90,241),
                          'longitude': np.linspace(0,359.2, 480)})
#print(da)
"""

# np.random.seed(0)
so2 = (10 ** -10) * np.random.randn(480, 241, 9, 2)
longitude = np.linspace(0, 359.2, 480)
latitude = np.linspace(90, -90, 241)
level = [500, 600, 700, 800, 850, 900, 925, 950, 1000]
time = pd.date_range("2003-01-01", periods=2)
reference_time = pd.Timestamp("2003-01-01")

ds = xr.Dataset(
    data_vars=dict(
        so2=(["longitude", "latitude", 'level', "time"], so2),
    ),
    coords=dict(
        longitude=(["longitude"], longitude),
        latitude=(["latitude"], latitude),
        level=(["level"], level),
        time=time,
        # reference_time=reference_time,
    ),
    attrs=dict(
        description="Weather related data.",
        units='kg kg**-1',
        long_name='Sulphur dioxide',
        standard_name='mass_fraction_of_sulfur_dioxide_in_air'),
)

##After obtaining the data from the files, the height is calculated using
# the inverse of the logarithmic formulation dependent on the pressure in the height.

# calculate and assign the height coordinate to the set.
#so2_2003_2020 = ds.assign_coords(height=("level", (-7.9 * np.log(ds.level.sel(level=slice(500, 1000)).values /
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

##Then the height values are introduced to the datasets exchanging the
# values in the dimension [pressure levels], (pressure levels for the calculated height values).

#print(so2_2003_2020.height.values)
# Swap the level and height dimensions
so2_2003_2020 = so2_2003_2020.swap_dims({"level": "height"})
#selected = so2_2003_2020.where(lambda x: x.time.dt.year == 2003, drop=True)

for t in range(len(ds.time)):
    so24d = so2_2003_2020.so2.isel(time=t)
    so24dC = so24d.copy()

    ##The values of the variable in the datasets are normalized by dividing the value
    # of the variable at each longitude and latitude at each elevation level by
    # the values of the variable on the surface.
    # calculate so24d[z] / so24d[0] normalización

    for h in range(len(so24d.height)):
        so24dC[:, :, h] = so24d.isel(height=h).values / so24d[:, :, 8].values

    ##When obtaining the datasets with the normalized values, the least squares
    # regression is calculated using the polyfit () method of degree 1,
    # which obtains the best fit for the resolution of the polynomial.
    ##`Hx` is a` Dataset`, which contains the variable `polyfit_coefficients`.
    ##The variable `polyfit_coefficients` of the` Dataset` `Hx` contains the
    # coefficients of the best fit for the resolution of the polynomial.

    Hx = so24dC.polyfit(dim='height', deg=1, full=True, cov=True)

    ##The polyval () method is used to evaluate the polynomial of the results
    # obtained from the polyfit () method. You     ##get a 3-dimensional array
    # `DataArray` (height, latitude, longitude).

    # HXPolyvald = xr.polyval(so24dC['height'], Hx.polyfit_coefficients[0, :, :], degree_dim='degree')
    # print(HXPolyvald[0, :, :])
    
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
            vmax=Hx.polyfit_coefficients[1, :, :].max().values)

    ax.gridlines(draw_labels=True)

    plt.title("Concentration SO2 " + str(so24d.time.values))

    plt.show()
