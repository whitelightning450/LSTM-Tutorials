"""
# LSTM - Tutorial

## Author
Name: Savalan Naser Neisary
Email: savalan.neisary@gmail.com | snaserneisary@crimson.ua.edu
Affiliation: PhD Student, University of Alabama
GitHub: savalann

## Code Description
Creation Date: 03-08-2024
Description: This code gets the dataset, scales it, and adds the lookback needed for the LSTM model.
It also divides the data into training and testing dataset based on the year.
"""




from sklearn.preprocessing import MinMaxScaler
import numpy as np


def Data_Preprocessing_MultiStation(data, length_lookback, split_year):
    station_list = data['station_id'].drop_duplicates()
    # Save the column
    column = data.pop('flow_cfs')
    # Insert the column at the end
    data['flow_cfs'] = column
    data.reset_index(drop=True, inplace=True)
    data_temp_00 = data.drop(columns=['datetime', 'station_id']).reset_index(drop=True)
    scaler_x = MinMaxScaler()
    scaler_y = MinMaxScaler()
    data_temp_01 = scaler.fit_transform(data_temp_00)
    data_temp_01 = np.concatenate((data['station_id'].to_numpy().reshape(-1, 1), data_temp_01), axis=1)

    
    
    final_result = {}
    for data_type in ['train','test']:
        data_x_all, data_y_all = [], []
        for station_number in station_list:
    
 
            if data_type == 'train':
                row_data = (data.station_id == station_number) & (data.datetime < f'01-01-{split_year}')
            if data_type == 'test':
                row_data = (data.station_id == station_number) & (data.datetime >= f'01-01-{split_year}') 
            
            
            
            data_temp_02 = data_temp_01[row_data]
            data_x, data_y = [], []
            for i in range(len(data_temp_02)-length_lookback-2):
                # find the end of this pattern
                features, targets = data_temp_02[i:i+length_lookback, :-1], data_temp_02[i+length_lookback:i+length_lookback, -1]
                data_x.append(features)
                data_y.append(targets)
            data_x_all.extend(data_x)
            data_y_all.extend(data_y)
        if data_type == 'test': 
            final_result['test_station_list'] = np.array(data_x_all)[:, 0, 0]
        
        final_result[f'x_{data_type}'] = torch.Tensor(np.delete(np.array(data_x_all).astype(np.float64), 0, axis=2))   
        final_result[f'y_{data_type}'] = torch.Tensor(np.array(data_y_all).astype(np.float64))  
    final_result['scaler_fit'] = scaler
    return final_result

