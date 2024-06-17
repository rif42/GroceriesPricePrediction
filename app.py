import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import matplotlib.dates as mdates
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, GRU, Bidirectional
from keras.optimizers import SGD
from sklearn.metrics import mean_squared_error
from sklearn.svm import SVR
from tensorflow.keras.models import load_model
import pickle
import openpyxl
import tensorflow
from sklearn.metrics import mean_squared_error, r2_score
import plotly.express as px
import plotly.graph_objects as go

@st.cache(allow_output_mutation=True)

def preprocess_data(dataset):
    dataset = dataset.T
    dataset.columns = dataset.iloc[0]
    dataset = dataset.drop(dataset.index[0])
    dataset['Average'] = round(dataset.mean(axis=1),2)
    dataset = dataset.drop(dataset.columns[0:3], axis=1)
    return dataset

def dataset_split(dataset):
    train_data = dataset.iloc[0:791] 
    validation_data = dataset.iloc[791:995] 
    test_data = dataset.iloc[995:1023] 
    train_data = train_data.values
    validation_data = validation_data.values
    return train_data, validation_data, test_data
  
def visualize_initial(dataset):
  # Prepare dates for the predicted stock prices
  date_range = pd.date_range(start='2021-03-10', periods=len(dataset), freq='D')  # 'B' for business day frequency
  # Visualizing Results with Month and Year on X-axis
  plt.figure(figsize=(10, 6))
  plt.plot(date_range, dataset['Average'], color='black', label='Grocery Price')
  plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))  # Show tick marks for every 3 months
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))  # Format x-axis labels as 'Jan 2023', 'Feb 2023', etc.
  plt.title('Grocery Price Prediction')
  plt.xlabel('Time')
  plt.ylabel('Grocery Price')
  plt.legend()
  return plt

def plotlytest(dataset):
  fig = px.line(dataset, x=dataset.index, y='Average', title='Grocery Price Prediction')
  return fig

def process_model(model):
  rawdataset = pd.read_excel('grocery_price.xlsx', index_col=None)
  dataset = rawdataset.T
  dataset.columns = dataset.iloc[0]
  dataset = dataset.drop(dataset.index[0])
  dataset['Average'] = round(dataset.mean(axis=1),2)
  dataset = dataset.drop(dataset.columns[0:3], axis=1)
  dataset = dataset.values
  
  sc = MinMaxScaler(feature_range=(0,1))
  scaled_dataset = sc.fit_transform(dataset)
  
  def create_sequences(data, seq_length=60):
    x=[]
    y=[]
    for i in range(seq_length, len(data)):
        x.append(data[i-seq_length:i,0])
        y.append(data[i,0])
    return np.array(x), np.array(y)
  x_test, y_test = create_sequences(scaled_dataset)
  x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
  
  if selection == "SVR":
    x_pred = x_test.reshape(x_test.shape[0], x_test.shape[1])
    predicted_stock_price = model.predict(x_pred)
    index = np.arange(len(predicted_stock_price))
    predicted_stock_price = np.column_stack((index, predicted_stock_price))
    y_test = np.column_stack((index, y_test))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=y_test[:,0], y=y_test[:,1], mode='lines', name='Grocery Price', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=predicted_stock_price[:,0], y=predicted_stock_price[:,1], mode='lines', name='Predicted Grocery Price', line=dict(color='green')))
    mse = mean_squared_error(y_test, predicted_stock_price)
    r2 = r2_score(y_test, predicted_stock_price)
  else:
    # Predict the stock price
    predicted_stock_price = model.predict(x_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)
    # round the values to 2 decimal places
    predicted_stock_price = np.round(predicted_stock_price,2)
    
    newdataset = preprocess_data(rawdataset)
    newdataset = newdataset[60:]
    newchart = newdataset.copy()
    newchart['Predicted'] = predicted_stock_price

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=newchart.index, y=newchart['Average'], mode='lines', name='Grocery Price', line=dict(color='red')))
    fig.add_trace(go.Scatter(x=newchart.index, y=newchart['Predicted'], mode='lines', name='Predicted Grocery Price', line=dict(color='green')))
    
    mse = mean_squared_error(dataset[60:], predicted_stock_price)
    r2 = r2_score(dataset[60:], predicted_stock_price)
  return fig,mse,r2

dataset = pd.read_excel('grocery_price.xlsx', index_col=None)
st.title("Grocery Price Prediction")
st.write("Prediksi rata-rata harga sembako menggunakan metode LSTM, GRU, dan SVR")

st.write("Dataset:")
st.write(dataset)

st.write("Dataset Preprocessing:")
preprocessed = preprocess_data(dataset)
st.write(preprocessed)

st.write("Dataset Visualization:")
visualized = visualize_initial(preprocessed)
st.pyplot(visualized)

plottest = plotlytest(preprocessed)
st.plotly_chart(plottest)

selection = st.selectbox("Pilih Model", ["LSTM", "GRU", "SVR"])
if st.button("Mulai Prediksi"):
    if selection == "LSTM":
      lstm_model = tensorflow.keras.models.load_model('LSTM.h5', compile=False)
      result,mse,r2 = process_model(lstm_model)
      st.plotly_chart(result)
      st.write("MSE: ", mse)
      st.write("Accuracy: ", r2)
      

    elif selection == "GRU":
      gru_model = tensorflow.keras.models.load_model('GRU.h5', compile=False)
      result,mse,r2 = process_model(gru_model)
      st.plotly_chart(result)
      st.write("MSE: ", mse)
      st.write("Accuracy: ", r2)

    elif selection == "SVR":
          with open('SVR.pkl', 'rb') as f:
            svr_model = pickle.load(f)
          result,mse,r2 = process_model(svr_model)
          st.plotly_chart(result)
          st.write("MSE: ", mse)
          st.write("Accuracy: ", r2)