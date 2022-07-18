
import streamlit as st
import pandas_datareader as pdr
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import tensorflow as tf
from keras.models import load_model
from numpy import array


header=st.container()
box=st.container()
features=st.container()
data=st.container()


#Normalizing the data.
def data_preprocess_and_split(data):

    #Scaling the data.
    scaler = MinMaxScaler(feature_range=(0,1))
    df1 = scaler.fit_transform(np.array(data['Close']).reshape(-1,1))

    train_size = int(len(data) * 0.75)
    test_size = int(len(data) - train_size)
    train_data, test_data = df1[0:train_size,:],df1[train_size:len(df1),:1]
    print(train_size, test_size)
    return scaler, train_data, test_data, df1


def dataset_matrix(dataset, timestep=23):
    '''
    Function to generate a data matrix depending upon the time stamps taken into consideration for the forecasting.
    '''
    X_data, Y_data = [], []
    for i in range(len(dataset)-timestep-1):
        flag_X = dataset[i:(i+timestep), 0]
        flag_Y = dataset[(i+timestep),0]
        X_data.append(flag_X)
        Y_data.append(flag_Y)
    Xtrain, Xtest = np.array(X_data), np.array(Y_data)
    Xtrain = Xtrain.reshape(Xtrain.shape[0], Xtrain.shape[1], 1) 
    Xtest = Xtest.reshape(Xtest.shape[0], Xtest.shape[1], 1)
    return Xtrain,Xtest
   
def last_output(y_data):
    model = load_model("lstm_model3.h5")
    x_input=np.array(y_data[291:]).reshape(1,-1)
    temp_input=list(x_input)
    temp_input=temp_input[0].tolist()
    lst_output=[]
    n_steps=23
    i=0
    while(i<30):
        
        if(len(temp_input))>23:
            x_input=np.array(temp_input[1:])
            x_input=x_input.reshape(1,-1)
            x_input = x_input.reshape((1, n_steps, 1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            lst_output.extend(yhat.tolist())
            i=i+1
        else:
            x_input = x_input.reshape((1, n_steps,1))
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            lst_output.extend(yhat.tolist())
            i=i+1
    return lst_output


def visuals(dataset1, predicted_data, scaler):
    predicted_data=dataset1.tolist()
    predicted_data.extend(lst_output)
    predicted_data=scaler.inverse_transform(predicted_data).tolist()
    return predicted_data
 

st.markdown(
  """
  <style>
  .main{
    background-color:#F5F5F5
  }
  </style>
  """,
  unsafe_allow_html=True
)
#________________________________________________________________________________________________________________
with header:
    #Title.
    title_alignment = st.markdown("<h1 style='text-align: center; color:#008080;'>Stock Forecast</h1>", unsafe_allow_html=True)
   
    #Getting user input and the corresponding data.
    st.subheader("Search")

    user_input = st.text_input("Search by entering the Stock ticker of the Company you are looking for:  ", "INFY")
    button_clicked = st.button("OK")
    df = pdr.get_data_yahoo((user_input))

with box:
    st.subheader("Suggestion Box:")
    option = st.selectbox('if you cannot makeup your mind, checkout the suggestions here instead:',('MSFT', 'AAPL', 'JPM','AMZN','BBN'))
    df = pdr.get_data_yahoo((option))

with features:
      
        st.subheader('Previous Trend')   
        sel_col,disp_col=st.columns(2)

        #Showing the recent 5 days data.
        sel_col.text("Past 5 days data")
        sel_col.write(df.tail(5))

        #Showing closing prices.
        disp_col.text('Data Visualisation')
        fig = plt.figure(figsize=(12,6))
        plt.plot(df['Close'], color ='#FF2216')
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Closing price')
        disp_col.pyplot(fig)



with data:
   
        m_scaler, train_data, test_data, df1 = data_preprocess_and_split(df)
        lst_output = last_output(test_data)
        df3 = visuals(df1, lst_output, m_scaler)

        st.subheader('Forecasted Data')
        fig2 = plt.figure(figsize=(12,6))
        plt.plot(df3)
        plt.grid(True)
        plt.xlabel('Time')
        plt.ylabel('Forecasted Closing price')
        st.pyplot(fig2)

 