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
from keras.models import load_model

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

def process_model(model):
  dataset = pd.read_excel('grocery_price.xlsx', index_col=None)
  dataset = dataset.T
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
  
  # Predict the stock price
  predicted_stock_price = model.predict(x_test)
  predicted_stock_price = sc.inverse_transform(predicted_stock_price)
  
  # Visualizing the result
  date_range = pd.date_range(start='2021-03-10', periods=len(dataset), freq='D')  # 'B' for business day frequency
  plt.figure(figsize=(10, 6))
  plt.plot(date_range[60:], dataset[60:], color='black', label='SPY Stock Price')
  plt.plot(date_range[60:], predicted_stock_price, color='green', label='Predicted SPY Stock Price')
  plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
  plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
  plt.title('SPY Stock Price Prediction')
  plt.xlabel('Time')
  plt.ylabel('SPY Stock Price')
  plt.legend()
  plt.show()
  return plt
  

lstm_model = load_model('./LSTM.h5', compile=False)
gru_model = load_model('./GRU.h5', compile=False)

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

selection = st.selectbox("Pilih Model", ["LSTM", "GRU"])
if st.button("Mulai Prediksi"):
    if selection == "LSTM":
      result = process_model(lstm_model)
      st.pyplot(result)

    elif selection == "GRU":
      result = process_model(gru_model)
      st.pyplot(result)





