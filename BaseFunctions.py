import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
import pandas_ta as ta
from tqdm import tqdm


def getMarketDetails(client):
    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']
    usdt_pairs = [
        s['symbol'] for s in symbols
        if s['quoteAsset'] == 'USDT' and 
        s['status'] == 'TRADING' and
        any('SPOT' in permissionSets for permissionSets in s['permissionSets'])
    ]
    # usdt_pairs = usdt_pairs[usdt_pairs['permissionSets'].apply(lambda x: any('SPOT' in s for s in x))]
    return pd.DataFrame({'pair': usdt_pairs})
    # return symbols

def getBalance(client, symbol):
    asset_balance = client.get_asset_balance(asset='USDT')
    return(float(asset_balance['free']))

def get_candles_data(symbol, interval, limit, backtime, client):
    """
    Fetches candlestick data from Binance.
    
    Args:
        symbol (str): Trading pair symbol, e.g., 'BTCUSDT'.
        interval (str): Candlestick interval, e.g., '1m', '5m', '1h', '1d'.
        limit (int): Number of candlesticks to retrieve.
        backtime (int): End timestamp in milliseconds (when to stop fetching data).
        api_key (str): Binance API key.
        api_secret (str): Binance API secret.

    Returns:
        pandas.DataFrame: DataFrame with candlestick data.
    """

    # Fetch candlestick data ending at backtime
    backtime = int(backtime)
    candles = client.get_klines(
        symbol=symbol,
        interval=interval,
        limit=limit,
        endTime=backtime
    )
    
    import pandas as pd
    # Convert to DataFrame
    df = pd.DataFrame(candles, columns=[
        'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
        'Close Time', 'Quote Asset Volume', 'Number of Trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    
    # Convert timestamp to datetime
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Asset Volume', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    df[columns_to_convert] = df[columns_to_convert].astype('float64')

    df['pair'] = symbol
    
    return df

def calculate_indicators(df1, CandlesFlag):
    # Calculate EMAs
    df1['EMA_1'] = ta.ema(df1['Close'], length=5)
    df1['EMA_2'] = ta.ema(df1['Close'], length=12)
    df1['EMA_3'] = ta.ema(df1['Close'], length=96)
    df1['EMA_4'] = ta.ema(df1['Close'], length=288)

    # # Calculate MFI
    df1['MFI_1'] = ta.mfi(df1['High'], df1['Low'], df1['Close'], df1['Volume'], length=12)

    # # Calculate RSI
    df1['RSI_1'] = ta.rsi(df1['Close'], length=12)

    # Calculate MACD
    macd = df1.ta.macd(close='Close', fast=12, slow=26, signal=9)

    # Add MACD results to the DataFrame
    df1 = pd.concat([df1, macd], axis=1)

    if CandlesFlag == True:
        # Calculate all candlestick patterns
        candlestick_patterns = df1.ta.cdl_pattern(name="all")
        # Add patterns to the original DataFrame
        df1 = pd.concat([df1, candlestick_patterns], axis=1)
    
    return df1

# Function to calculate MACD
def calculate_macd(data, short_window=12, long_window=26, signal_window=9):
    data['EMA_short'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_long'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['MACD_Signal'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
    return data

# Function to interpret MACD and maintain flags and counters
def interpret_macd(data):
    data['Scenario'] = 'Neutral'
    data['MACD_Counter'] = 0
    current_scenario = None
    counter = 0

    for i in range(1, len(data)):
        # Using .loc for setting cell values
        index_i = data.index[i]
        # Update histogram characteristics
        histogram_current = data.loc[index_i, 'MACD_Histogram']
        histogram_previous = data.loc[data.index[i - 1], 'MACD_Histogram']

        # Determine scenario based on histogram and MACD comparison
        if histogram_current > 0:
            if data.loc[index_i, 'MACD'] > data.loc[index_i, 'MACD_Signal']:
                scenario = 'Bullish Increasing'
            else:
                scenario = 'Warning'
        elif histogram_current < 0:
            if data.loc[index_i, 'MACD'] < data.loc[index_i, 'MACD_Signal']:
                scenario = 'Bearish Increasing'
            else:
                scenario = 'Warning'
        else:
            scenario = 'Neutral'

        # Determine trend based on histogram change
        if histogram_current > histogram_previous:
            trend = 'Increasing'
        elif histogram_current < histogram_previous:
            trend = 'Decreasing'
        else:
            trend = 'Neutral'

        # Combine scenario and trend
        full_scenario = f"{scenario} ({trend})"

        # Track scenario changes
        if full_scenario == current_scenario:
            counter += 1
        else:
            current_scenario = full_scenario
            counter = 1

        # Assign scenario and counter to the DataFrame
        data.loc[index_i, 'Scenario'] = full_scenario
        data.loc[index_i, 'MACD_Counter'] = counter

    return data
    
# Function to calculate metrics for a single coin
def calculate_metrics(df, pair):

    df = df[df['pair']==pair]

    if df.empty:
        print(f"No data for pair: {pair}")
        return None

    df['EMA_1'] = ta.ema(df['Close'], length=5)
    df['EMA_2'] = ta.ema(df['Close'], length=12)
    df['EMA_3'] = ta.ema(df['Close'], length=96)
    df['EMA_4'] = ta.ema(df['Close'], length=288)

    df = df.sort_values(by='Close Time', ascending=True)

    # Technical indicators using pandas_ta
    df['Vsma1'] = ta.sma(df['Volume'], length=2)
    df['Vsma2'] = ta.sma(df['Volume'], length=12)
    df['Vsma3'] = ta.sma(df['Volume'], length=96)
    df['Vsma4'] = ta.sma(df['Volume'], length=288)
    df['Per_Volume'] = df['Vsma1'] / df['Vsma3']
    
    # Calculate MACD and Crossover Distance 
    df = calculate_macd(df)
    df = interpret_macd(df)

    # Latest values
    latest = df.iloc[-1].copy()

    Per_Volume = latest['Per_Volume']

    DyingCoinFlag_EMA = 0
    if latest['EMA_1'] < latest['EMA_2'] and latest['EMA_2'] < latest['EMA_3'] and latest['EMA_3'] < latest['EMA_4']: 
        DyingCoinFlag_EMA = 1
    else: DyingCoinFlag_EMA = 0

    Latest_MACD_Scenario = latest['Scenario']
    Latest_MACD_ScenarioCounter = latest['MACD_Counter']
    
    if Latest_MACD_Scenario == 'Bullish Increasing (Increasing)':
        MACD_Score = 1
    elif Latest_MACD_Scenario == 'Bearish Increasing (Increasing)':
        MACD_Score = 0
    elif Latest_MACD_Scenario == 'Bullish Increasing (Decreasing)':
        MACD_Score = 0
    elif Latest_MACD_Scenario == 'Bearish Increasing (Decreasing)':
        MACD_Score = 0
    else:
        MACD_Score = 0
    
    MACD_Scored = MACD_Score * Latest_MACD_ScenarioCounter

    Metric_df = pd.DataFrame({
        'pair': pair,
        'Per_Volume': Per_Volume,
        'Latest_MACD_Scenario': Latest_MACD_Scenario,
        'Latest_MACD_ScenarioCounter': Latest_MACD_ScenarioCounter,
        'MACD_Score': MACD_Scored,
        'DyingCoinFlag_EMA': DyingCoinFlag_EMA
    }, index=[0])


    return Metric_df

    # return latest

# Function to rank coins and select top 10
def rank_pairs(metrics_list):
    # Create DataFrame for ranking
    df_results = pd.DataFrame(metrics_list)

    # Normalize and score
    # Volume: Higher is better (scale 0-100)
    df_results['volume_score'] = (df_results['volume'].rank(pct=True) * 10).clip(0, 10)

    # Crossover distance: Closer to 0 is better (scale 0-100)
    df_results['MACD_Scored'] = (1-df_results['MACD_Score'].rank(pct=True))*10
    df_results['MACD_Scored'] = df_results['MACD_Scored'].clip(0, 10)
    
    df_results['total_score'] = np.where(
    (df_results['DyingCoinFlag_EMA'] == 0) &
    (df_results['Latest_MACD_Scenario'] == 'Bullish Increasing (Increasing)'),
    0.40 * df_results['volume_score'] + 0.60 * df_results['MACD_Scored'],
    0)

    # Filtering df_results for total_score >0
    df_results = df_results[df_results['total_score'] > 0]

    # Select top 10 coins
    top_10 = df_results.sort_values(by='total_score', ascending=False).head(10)

    return top_10
