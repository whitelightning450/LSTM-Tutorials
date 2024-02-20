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
