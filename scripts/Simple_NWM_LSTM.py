#basic packagers
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from time import process_time 

#modeling packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

#Model evaluation metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import max_error
from sklearn.metrics import mean_absolute_percentage_error
import hydroeval as he

#packages to load AWS data
import boto3
import os
from botocore import UNSIGNED 
from botocore.client import Config
import os
os.environ['AWS_NO_SIGN_REQUEST'] = 'YES'

#Set Global Variables
ACCESS_KEY = pd.read_csv('AWSaccessKeys.csv')

#AWS Data Connectivity
#start session
SESSION = boto3.Session(
    aws_access_key_id=ACCESS_KEY['Access key ID'][0],
    aws_secret_access_key=ACCESS_KEY['Secret access key'][0]
)
s3 = SESSION.resource('s3')

BUCKET_NAME = 'streamflow-app-data'
BUCKET = s3.Bucket(BUCKET_NAME) 
S3 = boto3.resource('s3')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



#read parquet file
def readdata(filepath):
    obj = BUCKET.Object(filepath)
    body = obj.get()['Body']
    df = pd.read_csv(body)

    return df

#Combine DFs
def df_combine(USGS, NWM):
    #changes date as str to datetime object
    #USGS
    USGS['Datetime'] = USGS['Datetime'].astype('datetime64[ns]')
    #set index to datetime
    USGS.set_index('Datetime', inplace = True)
    #select streamflow
    cols =['USGS_flow']
    USGS = USGS[cols]
    #remove NaN values
    USGS.dropna(inplace = True)

    #NWM
    NWM['Datetime'] = NWM['Datetime'].astype('datetime64[ns]')
    #set index to datetime
    NWM.set_index('Datetime', inplace = True)
    #select streamflow
    cols =['NWM_flow']
    NWM = NWM[cols]
    #remove NaN values
    NWM.dropna(inplace = True)

    #combine NWM and USGS DF by datetime
    df = pd.concat([USGS, NWM], axis =1)
    df.dropna(inplace = True)

    return df

#create tensors/lookback out of training data for pytorch
def create_tensors(dataset, lookback):
    '''
    Transform a time series into a prediction dataset
    Args:
        dataset - a numpy array of time series, first dimension is the time step
        lookback -  size of window for prediction
    '''
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
        
    return torch.tensor(X).to(DEVICE), torch.tensor(y).to(DEVICE)

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
    
#build a simple LSTM model
class Simple_LSTM(nn.Module):
    def __init__(self, 
                 input_size: int = 1, 
                 hidden_size: int = 50, 
                 num_layers: int = 1, 
                 batch_first: bool = True):
        
        super().__init__()
        self.lstm = nn.LSTM(input_size = input_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            batch_first = batch_first
                            )
        self.linear = nn.Linear(hidden_size,1)

    def forward(self, X):
        X, _ = self.lstm(X)
        X = self.linear(X)
        return X
    

#Weight reset of param function
def weight_reset(m):
    if isinstance(m, nn.LSTM) or isinstance(m, nn.Linear):
        m.reset_parameters()

#LSTM training function
def train_LSTM(parameters):

    #set parameters
    model = parameters['model']
    loader =parameters['loader']
    X_train = parameters['X_train']
    y_train = parameters['y_train']
    X_test = parameters['X_test']
    y_test = parameters['y_test']
    optimizer = parameters['optimizer'] 
    loss_fn = parameters['loss_fn'] 

    t1_start = process_time()
    for epoch in range(parameters['n_epochs']):
        model.train()
        for X_batch, y_batch in loader:
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Print training data error as the model trains itself
        if epoch % 100 != 0:
            continue
        model.eval()

        if parameters['test_score'] == True:
            with torch.no_grad():
                y_pred = model(X_train)
                #must detach from GPU and put to CPU to calculate model performance
                train_rmse = np.sqrt(loss_fn(y_pred, y_train).detach().cpu().numpy())
                y_pred = model(X_test)
                test_rmse = np.sqrt(loss_fn(y_pred, y_test).detach().cpu().numpy())
            print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))
        
        else:
            with torch.no_grad():
                y_pred = model(X_train)
                #must detach from GPU and put to CPU to calculate model performance
                train_rmse = np.sqrt(loss_fn(y_pred, y_train).detach().cpu().numpy())
                print("Epoch %d: train RMSE %.4f" % (epoch, train_rmse))
    t1_stop = process_time()
    print("Model training took:", t1_stop-t1_start, ' seconds') 

    if parameters['save_model'] == True:
        #save model
        best_model = deepcopy(model.state_dict())
        torch.save(best_model,parameters['model_path'] )
        print(f"Model training complete, model saved as {parameters['model_path']}")
    else:
        print('Model training complete, model NOT saved.')


def LSTM_load(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model = model.cuda()

    return model

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