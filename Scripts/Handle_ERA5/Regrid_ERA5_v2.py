import os

for year in range(1965,2022):
    print(year)
    os.system(f'python /glade/u/home/jhayron/WeatherRegimes/Scripts/Handle_ERA5/regrid_year.py {year} False')