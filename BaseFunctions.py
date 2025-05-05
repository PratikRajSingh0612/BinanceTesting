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

def calculate_fib_cluster(df, num_swings=3, fib_levels=[0.236, 0.382, 0.5, 0.618, 0.786]):
    """
    Calculate Fibonacci cluster levels from multiple swing points
    Parameters:
        num_swings: Number of recent swing highs/lows to consider
        fib_levels: List of Fibonacci ratios to calculate
    Returns:
        DataFrame with 'Fib_Cluster' column added
    """
    df = df.copy()
    highs = df['High']
    lows = df['Low']
    
    # Find recent swing highs and lows
    swing_highs = highs[(highs.shift(1) < highs) & (highs.shift(-1) < highs)]
    swing_lows = lows[(lows.shift(1) > lows) & (lows.shift(-1) > lows)]
    
    # Get last n swings
    last_highs = swing_highs.tail(num_swings)
    last_lows = swing_lows.tail(num_swings)
    
    # Calculate all Fibonacci levels
    all_fib_levels = []
    for high, low in zip(last_highs, last_lows):
        for level in fib_levels:
            fib_value = high - (high - low) * level
            all_fib_levels.append(fib_value)
    
    # Find cluster zones (price areas with most Fibonacci levels)
    if all_fib_levels:
        hist, bin_edges = np.histogram(all_fib_levels, bins=20)
        cluster_zones = bin_edges[:-1][hist > (max(hist) * 0.7)]  # Get densest 70% areas
        
        # Assign nearest cluster to each row
        def get_nearest_cluster(price):
            if len(cluster_zones) > 0:
                return cluster_zones[np.argmin(np.abs(cluster_zones - price))]
            return np.nan
        
        df['Fib_Cluster'] = df['Close'].apply(get_nearest_cluster)
    else:
        df['Fib_Cluster'] = np.nan
    
    return df

def rsi_multi_timeframe_convergence(row):
    try:
        if (pd.notna(row['RSI_4H']) and pd.notna(row['RSI_1D']) and
            row['RSI_4H'] > 50 and row['RSI_1D'] > 50):
            return 1
        return 0
    except KeyError:
        return 0

def add_fibonacci_support(df, lookback_periods):
    try:
        if 'Time' not in df.columns or 'High' not in df.columns or 'Low' not in df.columns:
            print("Error: Required columns missing in DataFrame")
            df['Fib_Support'] = 0
            return df
        
        # Initialize Fib_Support
        df['Fib_Support'] = 0.0
        
        # Iterate through the DataFrame to find swings and compute Fibonacci levels
        for i in range(lookback_periods, len(df)):
            # Get lookback window
            window = df.iloc[i-lookback_periods:i]
            
            # Find swing high and low
            swing_high = window['High'].max()
            swing_low = window['Low'].min()
            swing_range = swing_high - swing_low
            
            # Calculate 61.8% Fibonacci retracement level
            fib_618 = swing_high - 0.618 * swing_range
            
            # Assign Fib_Support if valid
            if swing_range > 0:
                df['Fib_Support'].iloc[i] = fib_618
        
        # Forward-fill Fib_Support and default to 0
        df['Fib_Support'] = df['Fib_Support'].replace(0, method='ffill').fillna(0)
        
        return df
    except Exception as e:
        print(f"Error adding Fib_Support: {e}")
        df['Fib_Support'] = 0
        return df