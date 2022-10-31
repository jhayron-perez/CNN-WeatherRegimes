import cdsapi
from calendar import monthrange
import numpy as np
import os

years = np.arange(1989,2022).astype(str)
for year in years:
    print(year)
    path_output = f'/glade/work/jhayron/Weather_Regimes/ERA5/SL/'

    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'volumetric_soil_water_layer_1', 'volumetric_soil_water_layer_2',
                'volumetric_soil_water_layer_3',
            ],
            'area': [
                70, -150, 10,
                -40,
            ],
            'time': [
                '00:00', '01:00', '02:00',
                '03:00', '04:00', '05:00',
                '06:00', '07:00', '08:00',
                '09:00', '10:00', '11:00',
                '12:00', '13:00', '14:00',
                '15:00', '16:00', '17:00',
                '18:00', '19:00', '20:00',
                '21:00', '22:00', '23:00',
            ],
            'day': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
                '13', '14', '15',
                '16', '17', '18',
                '19', '20', '21',
                '22', '23', '24',
                '25', '26', '27',
                '28', '29', '30',
                '31',
            ],
            'month': [
                '01', '02', '03',
                '04', '05', '06',
                '07', '08', '09',
                '10', '11', '12',
            ],
            'year': year,
        },
        f'{path_output}ERA5_SL_SOIL_{year}.nc')