import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Fetching stock data
@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, start='2010-01-01', end='2023-12-31')
    data.reset_index(inplace=True)
    return data

st.title('Stock Market Predictions')
ticker = st.text_input('Enter Stock Ticker', 'AAPL')
data = load_data(ticker)

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_raw_data()

# Forecasting with Exponential Smoothing
df_train = data[['Date','Close']]
df_train = df_train.set_index('Date')

model = ExponentialSmoothing(df_train['Close'], trend='add', seasonal='add', seasonal_periods=365)
fit = model.fit()

# Predicting future values
pred_periods = 365
forecast = fit.forecast(pred_periods)
forecast_dates = pd.date_range(start=df_train.index[-1], periods=pred_periods + 1, closed='right')

forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast': forecast})
forecast_df.reset_index(drop=True, inplace=True)

# Plot forecast data
def plot_forecast_data():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='Stock Close'))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name='Forecast'))
    fig.layout.update(title_text='Forecasted Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

plot_forecast_data()
