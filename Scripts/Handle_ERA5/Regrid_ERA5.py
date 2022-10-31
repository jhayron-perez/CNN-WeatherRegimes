import numpy as np
import xarray as xr
import pandas as pd
import datetime as dt
import glob
import os
import matplotlib.pyplot as plt
import xesmf as xe

def regrid(ds, variable, reuse_weights=False,filename_weights = None):
    """
    Function to regrid onto coarser ERA5 grid (0.25-degree).
    Args:
        ds (xarray dataset): file.
        variable (str): variable.
        reuse_weights (boolean): Whether to use precomputed weights to speed up calculation.
                                 Defaults to ``False``.
        filename_weights (str): if reuse_weights is True, then a string for the weights path is needed.
    Returns:
        Regridded data file for use with machine learning model.
    """
    # ds.lon = ds.longitude
    # ds.lat = ds.latitude
    ds = ds.rename({'latitude':'lat',
                   'longitude':'lon'})
    ds_out = xe.util.grid_2d(lon0_b=0-0.5,   lon1_b=360-0.5, d_lon=1., 
                             lat0_b=-90-0.5, lat1_b=90,      d_lat=1.)

    if reuse_weights == False:
        regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights)
        return regridder(ds[variable]),regridder
    else:
        regridder = xe.Regridder(ds, ds_out, method='nearest_s2d', reuse_weights=reuse_weights,
            filename = filename_weights)
        return regridder(ds[variable])
    
#variables = ['SM', 'U10', 'Z500']
#variables_names_datasets = ['swvl','u','z']

variables = ['Z500']
variables_names_datasets = ['z']

for variable,variable_ds_name in zip(variables,variables_names_datasets):
    print(variable,variable_ds_name)
    
    path_data_025 = f'/glade/work/jhayron/Weather_Regimes/ERA5/Daily/{variable}/'
    files_in_path = np.sort(glob.glob(f'{path_data_025}*.nc'))

    path_data_1degree = f'/glade/work/jhayron/Weather_Regimes/ERA5/Daily_1degree/{variable}/'
    if not os.path.exists(path_data_1degree):
        os.makedirs(path_data_1degree)
    
    for ifile in range(5000,len(files_in_path)):
        if ifile%1000==0:
            print(ifile,len(files_in_path))
        ds_temp = xr.open_dataset(files_in_path[ifile])
        if ifile == 0:
            ds_temp, regridder = regrid(ds_temp,variable_ds_name,False)
            regridder.to_netcdf(f'/glade/work/jhayron/Weather_Regimes/ERA5/regridders/regrid_{variable}.nc')
        else:
            ds_temp = regrid(ds_temp,variable_ds_name,True,\
                f'/glade/work/jhayron/Weather_Regimes/ERA5/regridders/regrid_{variable}.nc')
        ds_temp = ds_temp.to_dataset(name=variable_ds_name)

        name_file_nc = files_in_path[ifile].split('/')[-1]
        ds_temp.to_netcdf(f'{path_data_1degree}{name_file_nc}')
        ds_temp.close()