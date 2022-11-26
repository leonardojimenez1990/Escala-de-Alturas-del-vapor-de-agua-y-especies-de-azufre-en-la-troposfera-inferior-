import cdsapi
import yaml

with open('CMAS', 'r') as f:
    credentials = yaml.safe_load(f)

c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

c.retrieve(
    'cams-global-reanalysis-eac4-monthly',
    {
        'variable': 'relative_humidity',
        'pressure_level': [
            '500','600', '700',
            '800','850', '900',
            '925','950', '1000',
        ],
        'year': [
            '2003', '2004', '2005',
            '2006', '2007', '2008',
            '2009', '2010', '2011',
            '2012', '2013', '2014',
            '2015', '2016', '2017',
            '2018', '2019', '2020',
            '2021',
        ],
        'month': [
            '09', '10', '11',
        ],
        'product_type': 'monthly_mean',
        'format': 'netcdf',
    },
    'relative_humiditySep_nov2003_2020CMAS.nc')
