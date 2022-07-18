import pandas_datareader as pdr
import matplotlib
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from keras.models import load_model
from numpy import array


#Normalizing the data.
def scaling_data(dataset):
    scaler = MinMaxScaler(feature_range=(0,1))
    df1 = scaler.fit_transform(np.array(dataset['Close']).reshape(-1,1))
    return df1, scaler

def split_data(data): 

    '''
    A function to split the data into  train and validation sets.
    '''
    train_size = int(len(data) * 0.70)
    test_size = int(len(data) - train_size)
    test_data = data.iloc[train_size:len(data),:1]
    return test_data

def loading_model():
    model = load_model("C:\Codes\Time-Series\lstm_model2.h5")
    return model

def last_output(y,model,n_steps=23):
    x_input=np.array(y[354:]).reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    i=0
    while(i<30):
        if(len(temp_input))>n_steps:
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, 23, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1,23,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    return lst_output   

def visuals(data,scaler,output):
    df3=data.tolist()
    df3.extend(output)
    df3=scaler.inverse_transform(df3).tolist()
    return df3


if __name__ == "__main__":

    #Getting user input and the corresponding data.
    user_input = input("Enter a stock ticker = ").upper()
    df = pdr.get_data_yahoo((user_input))

    #Showing closing prices.
    fig = plt.figure(figsize=(12,6))
    plt.plot(df['Close'], color ='red')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Closing price')
    plt.show()

    test_data = split_data(df)
    df1,scaler = scaling_data(df)
    model = loading_model()
    lst_output = last_output(test_data, model, 23)
    df3 = visuals(df1, scaler, lst_output)

    fig2 = plt.figure(figsize=(12,6))
    plt.plot(df3)
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Forecasted Closing price')
    plt.show()
