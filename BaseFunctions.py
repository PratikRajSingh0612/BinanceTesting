import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
import pandas_ta as ta
from tqdm import tqdm
from scipy.stats import linregress
from concurrent.futures import ThreadPoolExecutor, as_completed


def getMarketDetails(client):
    # Fetch exchange info
    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']
    
    # Get all USDT trading pairs with full data
    usdt_pairs = [
        s for s in symbols
        if s['quoteAsset'] == 'USDT' and 
        s['status'] == 'TRADING' and
        any('SPOT' in permissionSets for permissionSets in s.get('permissionSets', []))
    ]
    
    # Convert to DataFrame
    df = pd.DataFrame(usdt_pairs)
    
    # Explode nested structures for better analysis
    if 'filters' in df.columns:
        # Extract filter values into separate columns
        # Handle cases where filters might be empty or missing
        df['filters'] = df['filters'].apply(lambda x: x if isinstance(x, list) and x else [])
        filters_df = pd.json_normalize(df['filters'].explode())
        if not filters_df.empty:
            filters_df = filters_df.groupby(filters_df.index).first()
            df = pd.concat([df.drop('filters', axis=1), filters_df], axis=1)
        else:
            df = df.drop('filters', axis=1)
    
    if 'permissionSets' in df.columns:
        # Handle permissions with proper null checking
        def process_permissions(x):
            if isinstance(x, (list, tuple)) and x:
                # Flatten nested lists if they exist
                try:
                    return ','.join([item for sublist in x for item in (sublist if isinstance(sublist, (list, tuple)) else [sublist])])
                except:
                    return ','.join(map(str, x))
            return ''  # Return empty string for None, NaN, or empty lists
        
        df['permissions'] = df['permissionSets'].apply(process_permissions)
        df = df.drop('permissionSets', axis=1)
    
    # Clean up column names
    df.columns = df.columns.str.lower()
    
    # Remove rows with all missing or blank values
    df = df.dropna(how='all')  # Drop rows where ALL values are NaN/None
    # Optionally, drop rows with any missing values in critical columns
    critical_columns = ['symbol', 'baseasset', 'quoteasset']  # Adjust based on needs
    if all(col in df.columns for col in critical_columns):
        df = df.dropna(subset=critical_columns, how='any')
    
    # Remove rows where all non-critical columns are empty strings or NaN
    non_critical_columns = [col for col in df.columns if col not in critical_columns]
    if non_critical_columns:
        df = df.loc[~((df[non_critical_columns].replace('', pd.NA).isna()).all(axis=1))]
    
    # Reset index to ensure clean row numbering
    df = df.reset_index(drop=True)
    
    return df

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
        'Close Time', 'Quote Volume', 'Number of Trades',
        'Taker buy base asset volume', 'Taker buy quote asset volume', 'Ignore'
    ])
    
    # Convert timestamp to datetime
    df['Open Time'] = pd.to_datetime(df['Open Time'], unit='ms')
    df['Close Time'] = pd.to_datetime(df['Close Time'], unit='ms')

    columns_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume', 'Quote Volume', 'Taker buy base asset volume', 'Taker buy quote asset volume']
    df[columns_to_convert] = df[columns_to_convert].astype('float64')

    df['Symbol'] = symbol
    
    return df

def get_CandlesData_AllPairs(pairs, client, interval, endTime, limit):
    """
    Fetch candlestick data for a list of trading pairs in parallel.
    
    Parameters:
    - pairs: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT']).
    - client: Binance API client (e.g., from python-binance).
    - interval: Candlestick interval (e.g., '1h' for 1-hour, '1d' for 1-day).
    - limit: Number of candles to fetch per pair (max 1000 for Binance API).
    
    Returns:
    - pandas DataFrame with columns: Symbol, Timestamp, Open, High, Low, Close, Volume, Quote Volume.
    """
    candles_data = []
    with ThreadPoolExecutor(max_workers=9) as executor:
        # Submit tasks for each pair
        futures = [
            executor.submit(client.get_klines, symbol=symbol, interval=interval, limit=limit, endTime=endTime)
            for symbol in pairs
        ]
        # Process completed tasks
        for future in as_completed(futures):
            try:
                klines = future.result()
                # Extract symbol from the first kline (assuming all klines are for the same pair)
                symbol = pairs[futures.index(future)]  # Note: This is approximate; see notes
                # Process each kline (candle)
                for kline in klines:
                    candles_data.append({
                        'Symbol': symbol,
                        'Open Time': pd.to_datetime(kline[0], unit='ms'),  # Convert timestamp to datetime
                        'Open': float(kline[1]),
                        'High': float(kline[2]),
                        'Low': float(kline[3]),
                        'Close': float(kline[4]),
                        'Volume': float(kline[5]),
                        'Close Time': pd.to_datetime(kline[6], unit='ms'),  # Convert timestamp to datetime
                        'Quote Volume': float(kline[7]),  # Quote asset volume
                        'Taker buy base asset volume': float(kline[8]),  # Taker buy base asset volume - Taker buy base asset volume" measures how much of the asset was bought by takers
                        'Taker buy quote asset volume': float(kline[9])  # Taker buy quote asset volume - Taker buy quote asset volume" measures the equivalent amount in the quote currency spent to make those buys

                    })
            except Exception as e:
                # Log error and continue with other pairs
                print(f"Error fetching candles for a pair: {e}")
    
    # Convert to DataFrame
    df = pd.DataFrame(candles_data)
    return df

def get_24hr_stats(pairs, client):
    ticker_data = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [
            executor.submit(client.get_ticker, symbol=symbol)
            for symbol in pairs
        ]
        for future in as_completed(futures):  # Use as_completed directly
            try:
                ticker = future.result()
                ticker_data.append({
                    'Symbol': ticker['symbol'],
                    'Last Price': float(ticker['lastPrice']),
                    'Price Change': float(ticker['priceChange']),
                    'Price Change %': float(ticker['priceChangePercent']),
                    'High': float(ticker['highPrice']),
                    'Low': float(ticker['lowPrice']),
                    'Volume': float(ticker['volume']),
                    'Quote Volume': float(ticker['quoteVolume'])
                })
            except Exception as e:
                print(f"Error processing a symbol: {e}")
    return pd.DataFrame(ticker_data)