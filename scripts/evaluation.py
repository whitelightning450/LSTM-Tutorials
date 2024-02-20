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
        print("NSE %.4f, RMSE %.4f, MaxError %.4f, MAPE %.4f, KGE %.4f " % (nse, rmse, maxerror, MAPE, kge))

def model_eval(df, model,lookback, X_train, X_test, y_train, y_test, timeseries, trainsize, loss_fn):

    #must detach from GPU and put to CPU to calculate model performance
    with torch.no_grad():
        y_pred_train = model(X_train)
        #train_rmse = np.sqrt(loss_fn(y_pred_train, y_train).detach().cpu().numpy())
        y_pred_test = model(X_test)
        #test_rmse = np.sqrt(loss_fn(y_pred_test, y_test).detach().cpu().numpy())

        #print("Train RMSE %.4f, test RMSE %.4f" % (train_rmse, test_rmse))

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