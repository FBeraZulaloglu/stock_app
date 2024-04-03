import streamlit as st
import pandas as pd
import yfinance as yf
from ta.trend import EMAIndicator
from ta.momentum import RSIIndicator
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


st.set_page_config(
    page_title="KeenSight - Stock Analysis",
    layout='wide')

@st.cache_resource
def download_data(symbol, start_date, end_date):
    df = yf.download(symbol, start=start_date, end=end_date)
    return df


input_symbols = st.sidebar.text_input(
    'Enter up to 4 symbols (comma-separated)', 'AAPL,MSFT,SPY,WMT')
symbols = [s.strip() for s in input_symbols.split(',') if s.strip()]

# Select up to 4 symbols
selected_symbols = symbols[:4]
start_date = st.sidebar.date_input(
    'Start date', datetime.date.today() - datetime.timedelta(days=700))
end_date = st.sidebar.date_input('End date', datetime.date.today())
if start_date > end_date:
    st.sidebar.error('Error: End date must fall after start date')
    st.stop()


data = {}
for symbol in selected_symbols:
    df = download_data(symbol, start_date, end_date)
    df['EMA'] = EMAIndicator(df['Adj Close'], window=14).ema_indicator()
    df['RSI'] = RSIIndicator(df['Adj Close'], window=14).rsi()
    data[symbol] = df[['Adj Close', 'EMA', 'RSI']]

combined_data = pd.concat(data, axis=1)
# Calculate correlation matrix for 'Adj Close' only
correlation_matrix = combined_data.xs('Adj Close', axis=1, level=1).corr()

st.sidebar.write("### Correlation matrix (Adjusted Close only)")
st.sidebar.table(correlation_matrix)

scaler = StandardScaler()


def indicators():
    st.subheader('Adjusted Close')
    st.line_chart(combined_data.xs('Adj Close', axis=1, level=1))

    st.subheader('EMA')
    st.line_chart(combined_data.xs('EMA', axis=1, level=1))

    st.subheader('RSI')
    st.line_chart(combined_data.xs('RSI', axis=1, level=1))


def prediction():
    st.subheader('Predicting stock price')
    num = st.number_input('How many days prediction?', value=3)
    num = int(num)
    engine = LinearRegression()
    for symbol in selected_symbols:
        model_engine(engine, num, symbol)


def model_engine(model, num, symbol):
    # getting only the closing price
    data_symbol = data[symbol]
    df = data_symbol[['Adj Close']].copy()
    # shifting the closing price based on number of days forecast
    df['preds'] = data_symbol[['Adj Close']].shift(-num)
    # scaling the data
    x = df.drop(['preds'], axis=1).values
    x = scaler.fit_transform(x)
    # storing the last num_days data
    x_forecast = x[-num:]
    # selecting the required values for training
    x = x[:-num]
    # getting the preds column
    y = df.preds.values
    # selecting the required values for training
    y = y[:-num]

    # spliting the data
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=.2, random_state=7)
    # training the model
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    st.write(f"""###### {symbol} price prediction""")
    st.text(f' R-squared score: {r2_score(y_test, preds)} \
            \nMean Absolute Error: {mean_absolute_error(y_test, preds)}')
    # predicting stock price based on the number of days
    forecast_pred = model.predict(x_forecast)
    day = 1
    for i in forecast_pred:
        st.text(f'Day {day}: {i}')
        day += 1


def main():
    st.image("logo.png", width=200)
    indicators()
    prediction()


if __name__ == '__main__':
    main()
