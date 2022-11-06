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

names_variables = ['ttr','swvl','sst','skt','u']
names = ['OLR', 'SM', 'SST', 'ST', 'U10']
units = ['J/m2','m3/m3','K','K','m/s']

for iname in range(len(names)):
    print(names[iname])
    path_files = f'/glade/work/jhayron/Weather_Regimes/ERA5/Daily_1degree/{names[iname]}/'
    files = np.sort(glob.glob(f'{path_files}*.nc'))
    
    lat = xr.open_dataset(files[0]).lat.values
    lon = xr.open_dataset(files[0]).lon.values
    
    data = []
    dates = []
    for i in range(len(files)):
        vals_temp = xr.open_dataset(files[i])[names_variables[iname]].values
        data.append(vals_temp)
        dates.append(datetime.strptime(files[i].split('/')[-1],f'{names[iname]}_%Y_%m_%d.nc').date())
        
    data = np.array(data)
    dates = np.array(dates)
    
    ds_resampled = xr.Dataset({
                 names[iname].lower(): (['time','lat','lon'], data),
                },
                 coords =
                {'time': (['time'], pd.to_datetime(dates)),
                 'lat' : (['lat'], lat[:,0]),
                 'lon' : (['lon'], lon[0])
                },
                attrs = 
                {'File Author' : 'Jhayron S. Pérez-Carrasquilla','units':units[iname]})
    ds_resampled.to_netcdf(f'/glade/work/jhayron/Weather_Regimes/ERA5/Daily_1degree/netcdf_final/{names[iname].lower()}.nc')
    