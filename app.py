import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np

# Fetch BTC historical data using yfinance
btc_data = yf.download('BTC-USD', start='2020-01-01', end='2024-07-01')
btc_data['Date'] = btc_data.index
btc_data.reset_index(drop=True, inplace=True)

# Ensure required columns are present
required_columns = ['High', 'Low', 'Close', 'Volume']
for column in required_columns:
    if column not in btc_data.columns:
        st.error(f"Column '{column}' is missing in the downloaded data.")
        st.stop()

# Define indicator calculation functions
def calculate_rsi(data, window):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_stoch_k(data, window, smooth_k, smooth_d):
    low_min = data['Low'].rolling(window=window).min()
    high_max = data['High'].rolling(window=window).max()
    stoch_k = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    stoch_k = stoch_k.rolling(window=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d).mean()
    return stoch_k, stoch_d

def calculate_cci(data, window):
    tp = (data['High'] + data['Low'] + data['Close']) / 3
    tp_sma = tp.rolling(window=window).mean()
    md = (tp - tp_sma).abs().rolling(window=window).mean()
    cci = (tp - tp_sma) / (0.015 * md)
    return cci

def calculate_adx(data, window):
    high_diff = data['High'].diff()
    low_diff = data['Low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr1 = pd.DataFrame(data['High'] - data['Low'])
    tr2 = pd.DataFrame(abs(data['High'] - data['Close'].shift(1)))
    tr3 = pd.DataFrame(abs(data['Low'] - data['Close'].shift(1)))
    tr = pd.concat([tr1, tr2, tr3], axis=1, join='inner').max(axis=1)
    atr = tr.rolling(window=window).mean()
    plus_di = 100 * (pd.Series(plus_dm).rolling(window=window).mean() / atr)
    minus_di = 100 * (pd.Series(minus_dm).rolling(window=window).mean() / atr)
    dx = (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100
    adx = dx.rolling(window=window).mean()
    return adx

def calculate_momentum(data, window):
    momentum = data['Close'].diff(window)
    return momentum

def calculate_macd(data, fast, slow, signal):
    fast_ema = data['Close'].ewm(span=fast, min_periods=1).mean()
    slow_ema = data['Close'].ewm(span=slow, min_periods=1).mean()
    macd = fast_ema - slow_ema
    signal_line = macd.ewm(span=signal, min_periods=1).mean()
    macd_diff = macd - signal_line
    return macd, signal_line, macd_diff

def calculate_stoch_rsi(data, window):
    rsi = calculate_rsi(data, window)
    stoch_rsi = (rsi - rsi.rolling(window=window).min()) / (rsi.rolling(window=window).max() - rsi.rolling(window=window).min())
    return stoch_rsi * 100

def calculate_williams_r(data, window):
    high_max = data['High'].rolling(window=window).max()
    low_min = data['Low'].rolling(window=window).min()
    williams_r = (high_max - data['Close']) / (high_max - low_min) * -100
    return williams_r

def calculate_awesome_oscillator(data):
    median_price = (data['High'] + data['Low']) / 2
    ao = median_price.rolling(window=5).mean() - median_price.rolling(window=34).mean()
    return ao

def calculate_bull_bear_power(data):
    ema_13 = data['Close'].ewm(span=13).mean()
    bull_bear_power = data['Close'] - ema_13
    return bull_bear_power

def calculate_ultimate_oscillator(data):
    bp = data['Close'] - data[['Low', 'Close']].shift().min(axis=1)
    tr = data[['High', 'Close']].shift().max(axis=1) - data[['Low', 'Close']].shift().min(axis=1)
    avg7 = (bp.rolling(7).sum() / tr.rolling(7).sum())
    avg14 = (bp.rolling(14).sum() / tr.rolling(14).sum())
    avg28 = (bp.rolling(28).sum() / tr.rolling(28).sum())
    uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / 7
    return uo

def calculate_ichimoku_base_line(data):
    nine_period_high = data['High'].rolling(window=9).max()
    nine_period_low = data['Low'].rolling(window=9).min()
    base_line = (nine_period_high + nine_period_low) / 2
    return base_line

def calculate_vwma(data, window):
    vwma = (data['Close'] * data['Volume']).rolling(window=window).sum() / data['Volume'].rolling(window=window).sum()
    return vwma

def calculate_hull_moving_average(data, window):
    wma_half = 2 * data['Close'].rolling(window=int(window / 2)).mean()
    wma_full = data['Close'].rolling(window=window).mean()
    raw_hma = wma_half - wma_full
    hma = raw_hma.rolling(window=int(np.sqrt(window))).mean()
    return hma

def calculate_stoch_rsi_fast(data, rsi_window, k_window, d_window):
    rsi = calculate_rsi(data, rsi_window)
    stoch_rsi = (rsi - rsi.rolling(window=rsi_window).min()) / (rsi.rolling(window=rsi_window).max() - rsi.rolling(window=rsi_window).min())
    stoch_rsi_k = stoch_rsi.rolling(window=k_window).mean()
    stoch_rsi_d = stoch_rsi_k.rolling(window=d_window).mean()
    return stoch_rsi_k * 100, stoch_rsi_d * 100

def calculate_ema(data, window):
    ema = data['Close'].ewm(span=window, adjust=False).mean()
    return ema

def calculate_sma(data, window):
    sma = data['Close'].rolling(window=window).mean()
    return sma

# Calculate indicators
btc_data['RSI'] = calculate_rsi(btc_data, window=14)
btc_data['STOCH_K'], btc_data['STOCH_D'] = calculate_stoch_k(btc_data, window=14, smooth_k=3, smooth_d=3)
btc_data['CCI'] = calculate_cci(btc_data, window=20)
btc_data['ADX'] = calculate_adx(btc_data, window=14)
btc_data['MOM'] = calculate_momentum(btc_data, window=10)
btc_data['MACD'], btc_data['MACD_Signal'], btc_data['MACD_Hist'] = calculate_macd(btc_data, fast=12, slow=26, signal=9)
btc_data['STOCHRSI_K'], btc_data['STOCHRSI_D'] = calculate_stoch_rsi_fast(btc_data, rsi_window=14, k_window=3, d_window=3)
btc_data['WILLR'] = calculate_williams_r(btc_data, window=14)
btc_data['AO'] = calculate_awesome_oscillator(btc_data)
btc_data['Bull_Bear_Power'] = calculate_bull_bear_power(btc_data)
btc_data['Ultimate_Oscillator'] = calculate_ultimate_oscillator(btc_data)
btc_data['Ichimoku_Base_Line'] = calculate_ichimoku_base_line(btc_data)
btc_data['VWMA'] = calculate_vwma(btc_data, 20)
btc_data['HMA'] = calculate_hull_moving_average(btc_data, 9)

# Calculate EMAs and SMAs for various periods
for period in [10, 20, 30, 50, 100, 200]:
    btc_data[f'EMA_{period}'] = calculate_ema(btc_data, window=period)
    btc_data[f'SMA_{period}'] = calculate_sma(btc_data, window=period)

# Streamlit App layout and logic
st.title('BTC Trading Strategy Playground')

# Sidebar for Buy/Sell Conditions
st.sidebar.title('Buy Conditions')
buy_conditions = {}
buy_values = {}
for indicator in ['RSI', 'STOCH_K', 'CCI', 'ADX', 'MOM', 'MACD', 'STOCHRSI_K', 'WILLR', 'AO', 'Bull_Bear_Power', 'Ultimate_Oscillator', 'Ichimoku_Base_Line', 'VWMA', 'HMA']:
    if st.sidebar.checkbox('Buy ' + indicator, key='buy_' + indicator):  # Ensuring unique key by prepending "buy_"
        buy_conditions[indicator] = st.sidebar.selectbox(f"Buy if {indicator}", ["greater than", "less than"], key=f"buy_cond_{indicator}")
        buy_values[indicator] = st.sidebar.number_input(f"Value for {indicator}", value=0, key=f"buy_value_{indicator}")

st.sidebar.title('Sell Conditions')
sell_conditions = {}
sell_values = {}
for indicator in ['RSI', 'STOCH_K', 'CCI', 'ADX', 'MOM', 'MACD', 'STOCHRSI_K', 'WILLR', 'AO', 'Bull_Bear_Power', 'Ultimate_Oscillator', 'Ichimoku_Base_Line', 'VWMA', 'HMA']:
    if st.sidebar.checkbox('Sell ' + indicator, key='sell_' + indicator):  # Ensuring unique key by prepending "sell_"
        sell_conditions[indicator] = st.sidebar.selectbox(f"Sell if {indicator}", ["greater than", "less than"], key=f"sell_cond_{indicator}")
        sell_values[indicator] = st.sidebar.number_input(f"Value for {indicator}", value=0, key=f"sell_value_{indicator}")

# Determine Buy/Sell signals
def determine_signals(data, buy_conditions, buy_values, sell_conditions, sell_values):
    data['Buy'] = np.nan
    data['Sell'] = np.nan
    for indicator in buy_conditions:
        condition = buy_conditions[indicator]
        value = buy_values[indicator]
        if condition == "greater than":
            data.loc[data[indicator] > value, 'Buy'] = data['Close']
        elif condition == "less than":
            data.loc[data[indicator] < value, 'Buy'] = data['Close']
    for indicator in sell_conditions:
        condition = sell_conditions[indicator]
        value = sell_values[indicator]
        if condition == "greater than":
            data.loc[data[indicator] > value, 'Sell'] = data['Close']
        elif condition == "less than":
            data.loc[data[indicator] < value, 'Sell'] = data['Close']
    return data

# Apply conditions and plot data
btc_data = determine_signals(btc_data, buy_conditions, buy_values, sell_conditions, sell_values)

# Setting up dark background style for matplotlib
plt.style.use('dark_background')

# Plotting function with dark background
def plot_data(data):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close'], label='Price', color='white')  # Adjust line color for visibility
    plt.scatter(data['Date'], data['Buy'], color='green', label='Buy Signal', marker='^', alpha=1)
    plt.scatter(data['Date'], data['Sell'], color='red', label='Sell Signal', marker='v', alpha=1)
    plt.xlabel('Date', color='white')  # Adjust label color
    plt.ylabel('Price', color='white')
    plt.title('BTC Price with Buy/Sell Signals', color='white')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, color='grey')  # Adjust grid color and style for visibility
    st.pyplot(plt)

# Display the plot in Streamlit
plot_data(btc_data)
