#basic packagers
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from time import process_time 

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

#plotting function
def plot(df):
    #plot data
    cols = df.columns
    fig, ax = plt.subplots()
    for col in cols:
            ax.plot(df.index, df[col], label = col)
    ax.set(xlabel='Datetime (yr)', ylabel='Streamflow (cfs)',
            title='Streamflow')
    ax.grid()
    ax.legend()
    plt.show()

    ModelMetrics = [col for col in df.columns if 'Test' in col]
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



def model_eval(df, model,lookback, X_train, X_test, trainratio):

    #adjust df to different df sizes
    timeseries = df.to_numpy()[:,0].reshape(len(df), 1)
    trainsize = int(len(df)*trainratio) # 67% of data for training

    #must detach from GPU and put to CPU to calculate model performance
    with torch.no_grad():
        y_pred_train = model(X_train)
        y_pred_test = model(X_test)

        #make data for plot
        train_plot = np.ones_like(timeseries) * np.nan
        train_plot[lookback:trainsize] = y_pred_train[:, -1, :].detach().cpu().numpy()
        # shift test predictions for plotting
        test_plot = np.ones_like(timeseries) * np.nan
        test_plot[trainsize+lookback:len(timeseries)] = y_pred_test[:, -1, :].detach().cpu().numpy()

    #add training and testing predictions to df
    df['Train'] = train_plot
    df['Test'] = test_plot


    #Plot the results
    plot(df)

    return df