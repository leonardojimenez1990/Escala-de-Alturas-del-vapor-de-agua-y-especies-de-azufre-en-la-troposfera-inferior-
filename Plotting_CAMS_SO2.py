from Plotting_CAMS_SO2_Funiones import *

# crear gif
# gif()

# leer los datos de sulphur_dioxide2003_2021CMAS.nc
ds, so2_2003_2020 = leersulphur_dioxide2003_2021CMAS()
meanMesessulphur_dioxide2003_2021CMAS(so2_2003_2020)
#sulphur_dioxide2003_2021CMAS(ds,so2_2003_2020)


# leer los datos de sulphate_aerosol_mixing_ratio2003_2021CMAS.nc
ds, aermr11_2003_2020 = leersulphate_aerosol_mixing_ratio2003_2021CMAS()
meanMesessulphate_aerosol_mixing_ratio2003_2021CMAS(aermr11_2003_2020)
#sulphate_aerosol_mixing_ratio2003_2021CMAS(ds, aermr11_2003_2020)

# leer los datos de so2_precursor_mixing_ratio2003_2021CMAS.nc
ds, aermr12_2003_2020 = leerso2_precursor_mixing_ratio2003_2021CMAS()
meanMesesso2_precursor_mixing_ratio2003_2021CMAS(aermr12_2003_2020)
#so2_precursor_mixing_ratio2003_2021CMAS(ds, aermr12_2003_2020)

# leer los datos de relative_humidity2003_2021CMAS.nc
ds, r_2003_2020 = leerrelative_humidity2003_2021CMAS()
meanMesesrelative_humidity2003_2021CMAS(r_2003_2020)
#relative_humidity2003_2021CMAS(ds, r_2003_2020)
