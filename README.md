# Vertical scale for water vapor and sulfur compound profiles in the lower troposphere

This Python 3.8 code processes the data from the CAMS Global Reanalysis (EAC4) monthly average fields to calculate the Vertical Scales for profiles of water vapor and sulfur compounds in the lower troposphere in the period 2003 - 2020 and include multiple variables (specific humidity, sulfur dioxide, sulfate aerosol mixing ratio, SO2 precursor mixing ratio). With the help of the linear regression method, the relationship between the concentration of pollutants SO2 and the vertical coordinate (pressure in isobaric coordinates) will be analyzed.

Python libraries used:
-numpy
-xarray
-matplotlib
-cartopy
## Descripción del método utilizado

Se calculo la altura de los niveles de presion de los conjunto de datos utilizando la ecuacion de la Altura de escala de la atmosfera terrestre.

$$ P = P0 * exp (z / -H)$$ 

Se calculó el ajuste polinomial de los mínimos cuadrados de los conjuntos de datos de cada variable. Como resultado de la Implementanción de la librería xarray y la función polyfit().
