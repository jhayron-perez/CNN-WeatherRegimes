import copy
import glob
import pickle
import warnings
from datetime import datetime, timedelta
from itertools import product
import joblib

import cartopy
import cartopy.crs as ccrs
import cartopy.feature
import cartopy.feature as cfeature
import cartopy.feature as cf
import cartopy.io.shapereader as shpreader
import matplotlib as mpl
import matplotlib.path as mpath
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely.geometry as sgeom
import xarray as xr
from scipy import stats
from scipy.spatial.distance import cdist
from shapely import geometry
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def get_cold_indx(ds, mo_init=9, mo_end=2):
    """
    Extract indices for cold season.
    Grabbing Sept thru February init, for Oct thru March predictions.
    """
    dt_array = pd.to_datetime(ds['date_range'])
    # return dt_array
    return xr.where((dt_array.month>=mo_init) | (dt_array.month<=mo_end), True, False)

path_z_files = '/glade/work/jhayron/Weather_Regimes/ERA5/Daily_1degree/Z500/'

files = np.sort(glob.glob(f'{path_z_files}*.nc'))

lat = xr.open_dataset(files[0]).lat.values
lon = xr.open_dataset(files[0]).lon.values

data = []
dates = []
for i in range(len(files)):
    if i%1000==0:
        print(i,len(files))
    vals_temp = xr.open_dataset(files[i]).z.values
    data.append(vals_temp)
    dates.append(datetime.strptime(files[i].split('/')[-1],'Z500_%Y_%m_%d.nc').date())
    
data = np.array(data)
dates = np.array(dates)

ds_resampled = xr.Dataset({
             'z500': (['time','lat','lon'], data),
            },
             coords =
            {'time': (['time'], pd.to_datetime(dates)),
             'lat' : (['lat'], lat[:,0]),
             'lon' : (['lon'], lon[0])
            },
            attrs = 
            {'File Author' : 'Jhayron S. Pérez-Carrasquilla','units':'m2/s2'})

ds_resampled.to_netcdf('/glade/work/jhayron/Weather_Regimes/ERA5/Daily_1degree/netcdf_final/z500.nc')