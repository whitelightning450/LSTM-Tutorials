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
from torch.autograd import Variable


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
def create_lookback_univariate(dataset, lookback):
    '''
    Transform a time series into a prediction dataset
    Args:
        dataset - a numpy array of time series, first dimension is the time step
        lookback -  size of window for prediction
    '''
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature, target = dataset[i:i+lookback], dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
        
    return torch.tensor(X).to(DEVICE), torch.tensor(y).to(DEVICE)

def create_lookback_multivariate(dataset, lookback):
    X, y = [],[]
    for i in range(len(dataset)-lookback):
        # find the end of this pattern
        end_ix = i + lookback
        if end_ix > len(dataset):
            break
        features, targets = dataset[i:i+lookback, :-1], dataset[i+lookback, -1]
        X.append(features)
        y.append(targets)
    return np.array(X), np.array(y)

# split a multivariate sequences into train/test
def Multivariate_DataProcessing(df, input_columns, target, lookback, trainratio):
    # define input sequence
    #inputs
    in_seq = {}
    for col in input_columns:
        in_seq[col] = np.array(df[col])
        in_seq[col] = in_seq[col].reshape((len(in_seq[col]), 1))

    #Outputs
    out_seq = np.array(df[target])
    out_seq = out_seq.reshape((len(out_seq), 1))

    # horizontally stack columns
    in_seq = np.concatenate([v for k,v in sorted(in_seq.items())], 1)
    dataset = np.hstack((in_seq, out_seq))

    #split data into train/test - note, for LSTM there is not need to randomize the split
    trainsize = int(len(df)*trainratio) # 67% of data for training
    testsize = len(df)-trainsize # remaining (~33%) data for testing
    train, test = dataset[:trainsize,:], dataset[-testsize:,:]

    X_train, y_train = create_lookback_multivariate(train, lookback)
    X_test, y_test = create_lookback_multivariate(test, lookback)


    #need to convert to float32, tensors of the expected shape, and make sure they are on the device
    X_train = Variable(torch.from_numpy(X_train).float(), requires_grad=False).to(DEVICE)
    X_test = Variable(torch.from_numpy(X_test).float(), requires_grad=False).to(DEVICE)
    y_train = Variable(torch.from_numpy(y_train).float(), requires_grad=False).to(DEVICE)
    y_test = Variable(torch.from_numpy(y_test).float(), requires_grad=False).to(DEVICE)
  
    return X_train, X_test, y_train, y_test, dataset