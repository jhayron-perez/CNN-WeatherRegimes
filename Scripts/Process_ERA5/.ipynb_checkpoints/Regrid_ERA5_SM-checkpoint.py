import os
long_name = 'SM'
short_name = 'swvl'
for year in range(1959,2022):
    print(year)
    if year==1959:
        os.system(f'python /glade/u/home/jhayron/WeatherRegimes/Scripts/Handle_ERA5/regrid_year.py {year} True {long_name} {short_name}')
    else:
        os.system(f'python /glade/u/home/jhayron/WeatherRegimes/Scripts/Handle_ERA5/regrid_year.py {year} False {long_name} {short_name}')