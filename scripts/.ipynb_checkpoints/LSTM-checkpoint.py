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
        if epoch % 10 != 0:
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
    if DEVICE =='cuda':
        model = model.cuda()

    return model

