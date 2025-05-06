import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
import pandas_ta as ta
from tqdm import tqdm
from scipy.stats import linregress


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
