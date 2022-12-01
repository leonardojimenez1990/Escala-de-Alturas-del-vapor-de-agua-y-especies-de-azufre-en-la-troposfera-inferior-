# Vertical scale for water vapor and sulfur compound profiles in the lower troposphere

This Python 3 code processes the data from the CAMS Global Reanalysis (EAC4) monthly average fields to calculate the Vertical Scales for profiles of water vapor and sulfur compounds in the lower troposphere in the period 2003 - 2020 and include multiple variables (specific humidity, sulfur dioxide, sulfate aerosol mixing ratio, SO2 precursor mixing ratio). With the help of the linear regression method, the relationship between the concentration of pollutants SO2 and the vertical coordinate (pressure in isobaric coordinates) will be analyzed.

Python libraries used:
-numpy
-xarray
-matplotlib
-cartopy
## Descripción del método utilizado

The height of the pressure levels of the data sets was calculated using the Earth's Atmosphere Scale Height equation.

$$ P = P0 * exp (z / -H)$$ 

The polynomial fit of the least squares of the data sets of each variable was calculated. With the implementation of the library xarray and the funtion polyfit().
