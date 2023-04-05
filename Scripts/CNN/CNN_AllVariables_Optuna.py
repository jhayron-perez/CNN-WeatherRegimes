import copy
import glob
import pickle
import warnings
from datetime import datetime, timedelta
from itertools import product
import joblib
import os

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
# np.random.seed(42)
import pandas as pd
import shapely.geometry as sgeom
import xarray as xr
from scipy import stats
from scipy.spatial.distance import cdist
from shapely import geometry
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight

# import cluster_analysis, narm_analysis, som_analysis

import keras
from keras.models import Sequential,Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, AveragePooling2D, Dropout, BatchNormalization,SpatialDropout2D
from keras.utils import to_categorical
from keras.layers import LeakyReLU
from keras.layers import ReLU
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score

# import visualkeras
# import tensorflow as tf

# ## GLOBAL SEED ##    
# np.random.seed(42)
# tf.random.set_seed(42)

import optuna
from optuna.samplers import TPESampler
from optuna.integration import TFKerasPruningCallback

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('integers', metavar='N', type=int, nargs='+',
                    help='an integer for the accumulator')

parser = argparse.ArgumentParser(description='Running optuna for an specific week with all variables')

parser.add_argument("--week",choices=[1, 2, 3, 4, 5, 6, 7, 8, 9],required=True,type=int, 
                    help="Week (lead time) predicted.")
args=parser.parse_args()


optuna.logging.set_verbosity(optuna.logging.WARN)

wks = f'week{args.week}'
print(wks)

### LOAD DATA ###

week1_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week1_wr_v3.csv',\
                      index_col = 0, parse_dates = True)
week2_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week2_wr_v3.csv',\
                      index_col = 0, parse_dates = True)
week3_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week3_wr_v3.csv',\
                      index_col = 0, parse_dates = True)
week4_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week4_wr_v3.csv',\
                      index_col = 0, parse_dates = True)
week5_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week5_wr_v3.csv',\
                      index_col = 0, parse_dates = True)
week6_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week6_wr_v3.csv',\
                      index_col = 0, parse_dates = True)
week7_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week7_wr_v3.csv',\
                      index_col = 0, parse_dates = True)
week8_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week8_wr_v3.csv',\
                      index_col = 0, parse_dates = True)
week9_wr = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/week9_wr_v3.csv',\
                      index_col = 0, parse_dates = True)

df_wr = pd.concat([week1_wr,week2_wr,week3_wr,week4_wr,week5_wr,week6_wr,week7_wr,week8_wr,week9_wr],axis=1)
df_wr.columns = ['week1','week2','week3','week4','week5','week6','week7','week8','week9']

df_wr_2 = pd.read_csv('/glade/work/jhayron/Weather_Regimes/weekly_wr/weekly_wr_mean_geop_v3.csv',
                     index_col=0,parse_dates=True)

df_wr_2 = df_wr_2.dropna()
df_wr = df_wr.dropna()

variables = ['z500','olr', 'sst', 'u10', 'sm_region', 'st_region']
name_var = ['z500','olr', 'sst', 'u10', 'sm', 'st']
units = ['m2/s2','J/m2','K','m/s','m3/m3','K']

dic_vars = {}
for var_short, variable,unit in zip(name_var,variables,units):
# for var_short, variable,unit in zip(['sst'],['sst'],['K']):
    path_w_anoms = '/glade/work/jhayron/Weather_Regimes/weekly_anomalies/'
    week1_anoms = xr.open_dataset(f'{path_w_anoms}week1_{variable}_anoms_v3.nc')
    # week1_anoms = week1_anoms.sel(time=df_wr_2.index)
    if variable=='z500':
        week1_anoms = week1_anoms.where(week1_anoms.lat>-30,drop=True)
    # week1_anoms = week1_anoms.sel(time=df_wr.index)
    week1_anoms = week1_anoms.sel(time=df_wr_2.index)
    dic_vars[variable] = week1_anoms
    
## BUILD MODELS ##

def create_model(ks,ps,type_pooling,stc,stp,do,bn,md,nfilters,activation):
    num_classes = 4
    
    if activation == 'LeakyReLU':
        activation_conv= LeakyReLU()
    elif activation == 'ReLU':
        activation_conv= ReLU()
        
    padding_type = 'same'
    model = Sequential()
    
    model.add(Conv2D(nfilters, kernel_size=(ks, ks),activation=activation_conv,
        input_shape=X_train.shape[1:],padding=padding_type,strides=stc))
    
    if type_pooling == 'Max':
        model.add(MaxPooling2D((ps, ps),padding=padding_type,strides=stp))
    elif type_pooling == 'Average':
        model.add(AveragePooling2D((ps, ps),padding=padding_type,strides=stp))
        
    model.add(Dropout(do))
    if bn==True:
        model.add(BatchNormalization())
        
    model.add(Conv2D(nfilters*2, (ks, ks), activation=activation_conv,padding=padding_type,strides=stc))
    
    if type_pooling == 'Max':
        model.add(MaxPooling2D((ps, ps),padding=padding_type,strides=stp))
    elif type_pooling == 'Average':
        model.add(AveragePooling2D((ps, ps),padding=padding_type,strides=stp))
        
    model.add(SpatialDropout2D(do))
    if bn==True:
        model.add(BatchNormalization())
    
    model.add(Flatten())
    model.add(Dense(num_classes*md*md, activation=activation_conv))
    model.add(Dense(num_classes*md, activation=activation_conv))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy, 
        optimizer=keras.optimizers.Adam(lr=0.001),metrics=['accuracy'])
    return model

class Objective(object):
    def __init__(self, X_train, y_train, X_val, y_val,
                 path_models, variable, week, d_class_weights):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.path_models = path_models
        self.variable = variable
        self.week = week
 
    def __call__(self, trial):    
        keras.backend.clear_session()
                
        ks = trial.suggest_categorical('ks',[3,5,7,9,11])
        ps = trial.suggest_categorical('ps',[2,4,6,8])
        type_pooling = trial.suggest_categorical('type_pooling',['Average','Max'])
        stc = trial.suggest_categorical('stc',[1,2,3,4])
        stp = trial.suggest_categorical('stp',[1,2,3,4])
        do = trial.suggest_categorical('do',[0.3,0.4,0.5])
        bn = trial.suggest_categorical('bn',[True,False])
        md = trial.suggest_categorical('md',[2,4,8,16])
        nfilters = trial.suggest_categorical('nfilters',[4,8,16,32])
        activation = trial.suggest_categorical('activation',['LeakyReLU','ReLU'])
        bs = trial.suggest_categorical('bs',[16,32,64])
        
        dict_params = {'ks':ks,
                       'ps':ps,
                       'type_pooling':type_pooling,
                       'stc':stc,
                       'stp':stp,
                       'do':do,
                       'bn':bn,
                       'md':md,
                       'nfilters':nfilters,
                       'activation':activation,
                       # 'lr':lr,
                       'bs':bs}
                                              
        # instantiate and compile model
        cnn_model = create_model(dict_params['ks'],
                                 dict_params['ps'],
                                 dict_params['type_pooling'],
                                 dict_params['stc'],
                                 dict_params['stp'],
                                 dict_params['do'],
                                 dict_params['bn'],
                                 dict_params['md'],
                                 dict_params['nfilters'],
                                 dict_params['activation'],
                                 # dict_params['lr'],
                                )
        
        epochs = 50
        earlystop = EarlyStopping(monitor='val_loss', patience=5)
        try:
            os.mkdir(f'{self.path_models}{self.variable}')
        except: pass
        filepath = f'{self.path_models}{self.variable}/model_{self.week}_v7.h5'
        checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, 
                                     mode='auto',save_weights_only=False)
        h = cnn_model.fit(self.X_train, self.y_train, batch_size=dict_params['bs'],\
            epochs=epochs,verbose=0,validation_data=(self.X_val, self.y_val), \
            callbacks=[checkpoint,earlystop],class_weight = d_class_weights) #TFKerasPruningCallback(trial, "val_loss")

        validation_loss = np.min(h.history['val_loss'])
        
        return validation_loss
    
def logging_callback(study, frozen_trial):
    previous_best_value = study.user_attrs.get("previous_best_value", None)
    if previous_best_value != study.best_value:
        study.set_user_attr("previous_best_value", study.best_value)
        print(
            "Trial {} finished with best value: {} and parameters: {}. ".format(
            frozen_trial.number,
            frozen_trial.value,
            frozen_trial.params,
            )
        )
        
path_models = '/glade/work/jhayron/Weather_Regimes/models/CNN/weights_variables_v9/'


dic_metrics = {}

for var_short, variable in zip(name_var,variables):
    print('********************************************************************************************')
    print(variable)
    print('********************************************************************************************')
    loss_weeks_model = []
    loss_weeks_persistence = []
    acc_weeks_model = []
    acc_weeks_persistence = []

    for week in [wks]:
        print(week)
        #### ORGANIZE DATA ####
        week_output_wr = df_wr_2[week].values.astype(int)
        # Make Y categorical
        serie_wr_categorical = to_categorical(week_output_wr,num_classes=4)
        week1_anoms = copy.deepcopy(dic_vars[variable])
        
        # # Scale by min-max
        Min = week1_anoms[f'{var_short}_anomalies'].min(dim='time')
        Max = week1_anoms[f'{var_short}_anomalies'].max(dim='time')
        scaled_x = (week1_anoms[f'{var_short}_anomalies']) / (Max - Min)

        indices = np.arange(len(serie_wr_categorical))
        #Reshape X
        scaled_x = scaled_x.data.reshape(-1, scaled_x.shape[1],scaled_x.shape[2], 1)

        indices_train = np.where(df_wr_2.week2.index.year<=2001)[0]
        indices_val = np.where((df_wr_2.week2.index.year>2001)&(df_wr_2.week2.index.year<=2010))[0]
        indices_test = np.where(df_wr_2.week2.index.year>2010)[0]

        X_test = scaled_x[indices_test]
        y_test = serie_wr_categorical[indices_test]

        X_train = scaled_x[indices_train]
        y_train = serie_wr_categorical[indices_train]

        X_val = scaled_x[indices_val]
        y_val = serie_wr_categorical[indices_val]

        wr_persistence = df_wr_2.week1.values.astype(int)[indices_test]
        serie_wr_persistence_categorical = to_categorical(wr_persistence)
        
        y_train_integers = np.argmax(y_train, axis=1)
        class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(y_train_integers),
                                             y = y_train_integers)
        d_class_weights = dict(enumerate(class_weights))
        
        #### TRAIN ####
        # def print_best_callback(study, trial):
        #     print(f"Best value: {study.best_value}, Best params: {study.best_trial.params}")

        optimizer_direction = 'minimize'
        number_of_random_points = 30  # random searches to start opt process
        maximum_time = 0.75*60*60  # seconds
        objective = Objective(X_train,y_train,X_val,y_val,path_models,variable,week,d_class_weights)

        results_directory = f'/glade/work/jhayron/Weather_Regimes/models/CNN/results_optuna/{week}/'
        
        study_name = f'study_{week}_v9'
        storage_name = f'sqlite:///{study_name}.db'
        
        
        # optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction=optimizer_direction,
                sampler=TPESampler(n_startup_trials=number_of_random_points),
                study_name=study_name, storage=storage_name,load_if_exists=True)
        
        study.optimize(objective, timeout=maximum_time, gc_after_trial=True,callbacks=[logging_callback],)
        
        # save results
        df_results = study.trials_dataframe()
        df_results.to_pickle(results_directory + f'df_optuna_results_{var_short}_v4.pkl')
        df_results.to_csv(results_directory + f'df_optuna_results_{var_short}_v4.csv')
        #save study
        joblib.dump(study, results_directory + f'optuna_study_{var_short}_v4.pkl')
        
        