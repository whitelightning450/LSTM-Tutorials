#basic packagers
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from time import process_time 
import joblib

#Model evaluation metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error
import hydroeval as he

#modeling packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.autograd import Variable
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#plotting function
def plot(df, cols):
    #plot data
    fig, ax = plt.subplots()
    for col in cols:
            ax.plot(df.index, df[col], label = col)
    ax.set(xlabel='Datetime (yr)', ylabel='Streamflow (cfs)',
            title='Streamflow')
    ax.grid()
    ax.legend()
    plt.show()

    ModelMetrics = [col for col in cols if 'Test' in col]
    if len(ModelMetrics) > 0:
        #calculate model skill on testing data
        cols = ['USGS_flow', 'Test']
        dfTest = df[cols]
        dfTest.dropna(inplace = True)
        dfTest.head()
        nse = he.evaluator(he.nse, dfTest['USGS_flow'], dfTest['Test'])
        rmse = round(mean_squared_error(dfTest['USGS_flow'], dfTest['Test'], squared=False),0)
        maxerror = round(max_error(dfTest['USGS_flow'], dfTest['Test']),0)
        MAPE = round(mean_absolute_percentage_error(dfTest['USGS_flow'], dfTest['Test'])*100,0)
        kge, r, alpha, beta = he.evaluator(he.kge,dfTest['USGS_flow'], dfTest['Test'])
        kge = round(kge[0],2)
        print("LSTM: NSE %.4f, RMSE %.4f, MaxError %.4f, MAPE %.4f, KGE %.4f " % (nse, rmse, maxerror, MAPE, kge))

        if 'NWM_flow' in df.columns:
             #calculate model skill on NWM data
            cols = ['USGS_flow', 'NWM_flow']
            dfTest = df[cols]
            dfTest.dropna(inplace = True)
            dfTest.head()
            nse = he.evaluator(he.nse, dfTest['USGS_flow'], dfTest['NWM_flow'])
            rmse = round(mean_squared_error(dfTest['USGS_flow'], dfTest['NWM_flow'], squared=False),0)
            maxerror = round(max_error(dfTest['USGS_flow'], dfTest['NWM_flow']),0)
            MAPE = round(mean_absolute_percentage_error(dfTest['USGS_flow'], dfTest['NWM_flow'])*100,0)
            kge, r, alpha, beta = he.evaluator(he.kge,dfTest['USGS_flow'], dfTest['NWM_flow'])
            kge = round(kge[0],2)
            print("NWM: NSE %.4f, RMSE %.4f, MaxError %.4f, MAPE %.4f, KGE %.4f " % (nse, rmse, maxerror, MAPE, kge))

#plotting function
def Models_Eval_plot(df, evalcols):
    cols = evalcols.copy()
    #plot data
    fig, ax = plt.subplots()
    for col in cols:
            ax.plot(df.index, df[col], label = col)
    ax.set(xlabel='Datetime (yr)', ylabel='Streamflow (cfs)',
            title='Streamflow')
    ax.grid()
    ax.legend()
    plt.show()

    cols.remove('USGS_flow')
    for col in cols:

        #calculate model skill on testing data
        nse = he.evaluator(he.nse, df['USGS_flow'], df[col])
        rmse = round(mean_squared_error(df['USGS_flow'], df[col], squared=False),0)
        maxerror = round(max_error(df['USGS_flow'], df[col]),0)
        MAPE = round(mean_absolute_percentage_error(df['USGS_flow'], df[col])*100,0)
        kge, r, alpha, beta = he.evaluator(he.kge,df['USGS_flow'], df[col])
        kge = round(kge[0],2)
        print(f"{col}", ": NSE %.4f, RMSE %.4f, MaxError %.4f, MAPE %.4f, KGE %.4f " % (nse, rmse, maxerror, MAPE, kge))

        



def model_eval(df, cols, model, y_scaler_path, lookback, X_train, X_test, trainratio):

    #load the scaler
    yscaler = joblib.load(y_scaler_path) 
    #adjust df to different df sizes
    timeseries = df.to_numpy()[:,0].reshape(len(df), 1)
    trainsize = int(len(df)*trainratio) # 67% of data for training

    #must detach from GPU and put to CPU to calculate model performance
    with torch.no_grad():
        if DEVICE.type == 'cuda':
            y_pred_train = model(X_train).detach().cpu().numpy()
            y_pred_test = model(X_test).detach().cpu().numpy()
        else:
            y_pred_train = model(X_train)
            y_pred_test = model(X_test)

        #select the current prediction and rescale
        y_pred_train = y_pred_train[:, -1, :]
        yscaler = joblib.load(y_scaler_path) 
        y_pred_train = yscaler.inverse_transform(y_pred_train)
        y_pred_test = y_pred_test[:, -1, :]
        y_pred_test = yscaler.inverse_transform(y_pred_test)


        #make data for plot
        train_plot = np.ones_like(timeseries) * np.nan
        train_plot[lookback:trainsize] = y_pred_train
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[trainsize+lookback:len(timeseries)] = y_pred_test

    #add training and testing predictions to df
    df['Train'] = train_plot
    df['Test'] = test_plot

    #add physical constraints to model (i.e., cannot predict negative streamflow)
    df['Train'][df['Train']<0] = 0
    df['Test'][df['Test']<0] = 0

    #Plot the results
    plot(df, cols)

    return df

def model_eval_univariate(df, cols, model, lookback, X_train, X_test, trainratio):

    #load the scaler
    #adjust df to different df sizes
    timeseries = df.to_numpy()[:,0].reshape(len(df), 1)
    trainsize = int(len(df)*trainratio) # 67% of data for training

    #must detach from GPU and put to CPU to calculate model performance
    with torch.no_grad():
        if DEVICE.type == 'cuda':
            y_pred_train = model(X_train).detach().cpu().numpy()
            y_pred_test = model(X_test).detach().cpu().numpy()
        else:
            y_pred_train = model(X_train)
            y_pred_test = model(X_test)

        #select the current prediction and rescale
        y_pred_train = y_pred_train[:, -1, :]
        y_pred_test = y_pred_test[:, -1, :]
     
        #make data for plot
        train_plot = np.ones_like(timeseries) * np.nan
        train_plot[lookback:trainsize] = y_pred_train
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[trainsize+lookback:len(timeseries)] = y_pred_test

    #add training and testing predictions to df
    df['Train'] = train_plot
    df['Test'] = test_plot

    #add physical constraints to model (i.e., cannot predict negative streamflow)
    df['Train'][df['Train']<0] = 0
    df['Test'][df['Test']<0] = 0

    #Plot the results
    plot(df, cols)

    return df