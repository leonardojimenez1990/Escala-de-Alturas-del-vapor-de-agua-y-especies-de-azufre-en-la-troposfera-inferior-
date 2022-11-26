# Vertical scale for water vapor and sulfur compound profiles in the lower troposphere

This Python 3.8 code processes the data from the CAMS Global Reanalysis (EAC4) monthly average fields to calculate the Vertical Scales for profiles of water vapor and sulfur compounds in the lower troposphere in the period 2003 - 2020 and include multiple variables (specific humidity, sulfur dioxide, sulfate aerosol mixing ratio, SO2 precursor mixing ratio). With the help of the linear regression method, the relationship between the concentration of pollutants SO2 and the vertical coordinate (pressure in isobaric coordinates) will be analyzed.

Python libraries used:
-numpy
-xarray
-matplotlib
-cartopy
