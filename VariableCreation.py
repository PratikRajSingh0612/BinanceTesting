import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
import pandas_ta as ta
from tqdm import tqdm
from scipy.stats import linregress
from ta.momentum import RSIIndicator
from ta.trend import MACD

import importlib
import BaseFunctions
importlib.reload(BaseFunctions)
from BaseFunctions import *

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
        if 'High' not in df.columns or 'Low' not in df.columns:
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
    
def detect_bearish_trendlines(df, lookback=20, min_touches=3, tolerance=0.01):
    highs = df['High'].values
    dates = np.arange(len(highs))

    # Initialize output columns
    df['Trendline'] = np.nan
    df['Trendline_Touches'] = 0
    df['Trendline_Slope'] = np.nan

    for i in range(lookback, len(df)):
        window_highs = highs[i-lookback:i]
        window_dates = dates[i-lookback:i]
        
        # Find potential lower highs
        peak_indices = []
        for j in range(1, len(window_highs)-1):
            if window_highs[j] > window_highs[j-1] and window_highs[j] > window_highs[j+1]:
                peak_indices.append(j)
        
        if len(peak_indices) >= min_touches:
            # Try to fit trendline to the peaks
            peak_highs = window_highs[peak_indices]
            peak_dates = window_dates[peak_indices]
            
            slope, intercept, _, _, _ = linregress(peak_dates, peak_highs)
            
            # Count touches (including nearby points)
            touch_count = 0
            for j in range(len(window_highs)):
                trendline_val = intercept + slope * window_dates[j]
                if abs(window_highs[j] - trendline_val) <= tolerance * window_highs[j]:
                    touch_count += 1
            
            if touch_count >= min_touches:
                # Project trendline to current point
                current_trendline = intercept + slope * dates[i]
                df.at[df.index[i], 'Trendline'] = current_trendline
                df.at[df.index[i], 'Trendline_Touches'] = touch_count
                df.at[df.index[i], 'Trendline_Slope'] = slope

    # Forward fill the trendline values
    df['Trendline'].fillna(method='ffill', inplace=True)
    df['Trendline_Touches'].fillna(method='ffill', inplace=True)
    df['Trendline_Slope'].fillna(method='ffill', inplace=True)

    # Fill any remaining NaNs
    df['Trendline'].fillna(np.inf, inplace=True)  # No trendline = infinite resistance
    df['Trendline_Touches'].fillna(0, inplace=True)
    df['Trendline_Slope'].fillna(0, inplace=True)

    return df

def calculate_atr(df, period):
    df['Previous_Close'] = df['Close'].shift(1)
    df['High_Low'] = df['High'] - df['Low']
    df['High_PrevClose'] = abs(df['High'] - df['Previous_Close'])
    df['Low_PrevClose'] = abs(df['Low'] - df['Previous_Close'])
    df['True_Range'] = df[['High_Low', 'High_PrevClose', 'Low_PrevClose']].max(axis=1)
    atr = df['True_Range'].rolling(window=period).mean()
    return atr


def create_variables(df, symbol, client, BackTime):
    
    # ===== 1. Basic Price Transformations =====
    df['Close_prev'] = df['Close'].shift(1)
    df['Open_prev'] = df['Open'].shift(1)
    df['High_prev'] = df['High'].shift(1)
    df['Low_prev'] = df['Low'].shift(1)
    df['Volume_prev'] = df['Volume'].shift(1)
    
    period = 20
    # Calculate Money Flow Multiplier (MFM)
    df['MFM'] = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    
    # Calculate Money Flow Volume (MFV)
    df['MFV'] = df['MFM'] * df['Volume']
    
    # Calculate Chaikin Money Flow (CMF)
    df['CMF'] = df['MFV'].rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    
    # Shift to get the previous CMF value
    df['CMF_prev'] = df['CMF'].shift(1)

    # Calculate Typical Price
    df['Typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Calculate SMA of Typical Price
    df['SMA_Typical_price'] = df['Typical_price'].rolling(window=20).mean()
    
    # Calculate Mean Deviation
    df['Mean_Deviation'] = df['Typical_price'].rolling(window=20).apply(
        lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
    )
    
    df['TD'] = ((df['Typical_price'] > df['Typical_price'].shift(1)) * 2 - 1)
    
    # Calculate Volume Force (VF)
    df['VF'] = df['TD'] * df['Volume'] * (2 * (df['Close'] - df['Low'] - (df['High'] - df['Close'])) / (df['High'] - df['Low']))
    
    # Calculate the short and long EMA of VF
    df['Short_EMA'] = df['VF'].ewm(span=34, adjust=False).mean()
    df['Long_EMA'] = df['VF'].ewm(span=55, adjust=False).mean()
    
    # Calculate KVO
    df['KVO'] = df['Short_EMA'] - df['Long_EMA']
    
    # Calculate the Signal Line as EMA of KVO
    df['KVO_Signal'] = df['KVO'].ewm(span=13, adjust=False).mean()
    
    # Shift to get previous KVO values
    df['KVO_prev'] = df['KVO'].shift(1)
    df['KVO_Signal_prev'] = df['KVO_Signal'].shift(1)

    # Calculate CCI
    df['CCI'] = (df['Typical_price'] - df['SMA_Typical_price']) / (0.015 * df['Mean_Deviation'])
    df['CCI_prev'] = df['CCI'].shift(1).fillna(0)
    
    # Calculate TRIX - handle different output formats
    trix_data = ta.trix(df['Close'], length=15)
    
    # Handle multi-column output
    if isinstance(trix_data, pd.DataFrame):
        # Some libraries return DataFrame with multiple columns
        df['Trix'] = trix_data.iloc[:, 0]  # First column is TRIX line
        if trix_data.shape[1] > 1:
            df['Trix_signal'] = trix_data.iloc[:, 1]  # Second column is signal line
    else:
        # Assume single Series output
        df['Trix'] = trix_data
    
    # Create previous values
    df['Trix_prev'] = df['Trix'].shift(1)
    
    # Create individual history columns
    for i in range(1, 3 + 1):
        df[f'Trix_prev_{i}'] = df['Trix'].shift(i)
    
    # Create confirmation flag
    df['Trix_above_zero'] = True
    for i in range(1, 3 + 1):
        df['Trix_above_zero'] = (df['Trix_above_zero'] & 
                                (df[f'Trix_prev_{i}'] > 0))
    
    # Fill NA values
    cols_to_fill = ['Trix', 'Trix_prev'] + \
                    [f'Trix_prev_{i}' for i in range(1, 3 + 1)]
    for col in cols_to_fill:
        df[col].fillna(0, inplace=True)
    df['Trix_above_zero'] = df['Trix_above_zero'].fillna(False, inplace=True)

    # Multi-period lookbacks
    for i in range(2, 6):
        df[f'Close_prev{i}'] = df['Close'].shift(i)
        df[f'Open_prev{i}'] = df['Open'].shift(i)
        df[f'High_prev{i}'] = df['High'].shift(i)
        df[f'Low_prev{i}'] = df['Low'].shift(i)
        df[f'Volume_prev{i}'] = df['Volume'].shift(i)

    # ===== 2. Moving Averages =====
    ma_periods = [10, 20, 50, 200]
    for period in ma_periods:
        df[f'MA_{period}'] = df['Close'].rolling(period).mean()
        df[f'EMA_{period}'] = ta.ema(df['Close'], length=period)
        df[f'MA_{period}_prev'] = df[f'MA_{period}'].shift(1)
        df[f'EMA_{period}_prev'] = df[f'EMA_{period}'].shift(1)
        df[f'EMA_{period}_HTF'] = df['Close'].ewm(span=period).mean()
        df[f'EMA_{period}_HTF_prev'] = df[f'EMA_{period}_HTF'].shift(1)

    # Slope calculations
    df['MA_50_slope'] = df['MA_50'].diff(5) / 5
    df['MA_50_slope_prev'] = df['MA_50_slope'].shift(1)

    # ===== 3. Volume Indicators =====
    df['Volume_20_SMA'] = df['Volume'].rolling(20).mean()
    df['OBV'] = ta.obv(df['Close'], df['Volume'])
    df['OBV_prev'] = df['OBV'].shift(1)

    # ===== 4. Oscillators =====
    # RSI
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['RSI_prev'] = df['RSI'].shift(1)
    df['RSI_prev2'] = df['RSI'].shift(2)
    df['RSI_Trendline'] = df['RSI'].rolling(window=5).mean()  # Example trendline
    
    # Fetch 4-hour candles
    candles_4h = get_candles_data(symbol, '4h', 1000, BackTime, client)
    candles_4h['RSI_4H'] = RSIIndicator(candles_4h['Close'], window=14).rsi()
    macd_4h = MACD(candles_4h['Close'], window_fast=12, window_slow=26, window_sign=9)
    candles_4h['MACD_4H'] = macd_4h.macd()
    candles_4h['MACD_Signal_4H'] = macd_4h.macd_signal()
    candles_4h['High_4H'] = candles_4h['High']
    candles_4h['High_4H_prev'] = candles_4h['High'].shift(1)
    candles_4h['Low_4H'] = candles_4h['Low']
    candles_4h['Low_4H_prev'] = candles_4h['Low'].shift(1)

    # Fetch daily candles
    candles_1d = get_candles_data(symbol, '1d', 1000, BackTime, client)
    candles_1d['RSI_1D'] = RSIIndicator(candles_1d['Close'], window=14).rsi()
    macd_1d = MACD(candles_1d['Close'], window_fast=12, window_slow=26, window_sign=9)
    candles_1d['MACD_1D'] = macd_1d.macd()
    candles_1d['MACD_Signal_1D'] = macd_1d.macd_signal()
    candles_1d['High_1D'] = candles_1d['High']
    candles_1d['High_1D_prev'] = candles_1d['High'].shift(1)
    candles_1d['Low_1D'] = candles_1d['Low']
    candles_1d['Low_1D_prev'] = candles_1d['Low'].shift(1)
    
    # Merge RSI values into original DataFrame (align by timestamp)
    df = df.merge(candles_4h[['Close Time', 'RSI_4H', 'MACD_4H', 'MACD_Signal_4H', 'High_4H', 'High_4H_prev', 'Low_4H', 'Low_4H_prev']], on='Close Time', how='left')
    df = df.merge(candles_1d[['Close Time', 'RSI_1D', 'MACD_1D', 'MACD_Signal_1D', 'High_1D', 'High_1D_prev', 'Low_1D', 'Low_1D_prev']], on='Close Time', how='left')
    
    # Forward-fill missing values and default to 0
    for col in ['RSI_4H', 'RSI_1D', 'MACD_4H', 'MACD_Signal_4H', 'MACD_1D', 'MACD_Signal_1D']:
        df[col] = df[col].fillna(method='ffill').fillna(0)

    # Forward-fill missing RSI values
    df['RSI_4H'] = df['RSI_4H'].fillna(method='ffill')
    df['RSI_1D'] = df['RSI_1D'].fillna(method='ffill')

    # MACD
    macd = MACD(df['Close'], window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Histogram'] = macd.macd_diff()
    df['MACD_prev'] = df['MACD'].shift(1)
    df['MACD_Signal_prev'] = df['MACD_Signal'].shift(1)
    df['MACD_Histogram_prev'] = df['MACD_Histogram'].shift(1)


    # Stochastic
    # Calculate Stochastic RSI
    stoch_rsi = ta.stochrsi(
        close=df['Close'],
        window=14,
        smooth_window=14,
        smooth_k=3,
        smooth_d=3
    )

    df['StochRSI_K'] = stoch_rsi['STOCHRSIk_14_14_3_3']
    df['StochRSI_D'] = stoch_rsi['STOCHRSId_14_14_3_3']
    df['StochRSI_prev'] = df['StochRSI_K'].shift(1)

    # ADX
    adx = ta.adx(df['High'], df['Low'], df['Close'], length=14)
    df['ADX'] = adx['ADX_14']
    df['DMI+'] = adx['DMP_14']
    df['DMI-'] = adx['DMN_14']
    df['DMI+_prev'] = df['DMI+'].shift(1)
    df['DMI-_prev'] = df['DMI-'].shift(1)

    # ===== 5. Support/Resistance =====
    df['Support'] = df['Low'].rolling(20).min()
    df['Resistance'] = df['High'].rolling(20).max()
    df['Swing_High'] = df['High'].rolling(50).max()
    df['Swing_Low'] = df['Low'].rolling(50).min()
    df['Support_Tests'] = df['Low'].rolling(20).apply(lambda x: (x == x.min()).sum())
    df['Is_Swing_High'] = (df['High'] == df['High'].rolling(window=5, center=True, min_periods=1).max())

    # Calculate the Williams %R
    df['WilliamsR'] = (df['Resistance'] - df['Close']) / (df['Resistance'] - df['Support']) * -100
    
    # Shift to get the previous Williams %R
    df['WilliamsR_prev'] = df['WilliamsR'].shift(1)
    

    # Get recent swing high and low
    recent_high = df['High'].rolling(20).max()
    recent_low = df['Low'].rolling(20).min()
    # Calculate Fibonacci fan support levels (38.2%, 50%, 61.8%)
    df['Fib_Fan_Support1'] = recent_high - (recent_high - recent_low) * 0.382
    df['Fib_Fan_Support2'] = recent_high - (recent_high - recent_low) * 0.5
    df['Fib_Fan_Support3'] = recent_high - (recent_high - recent_low) * 0.618
    df['Fib_Fan_Support4'] = recent_high - (recent_high - recent_low) * 0.65
    
    # Find the closest support level below price
    supports = df[['Fib_Fan_Support1', 'Fib_Fan_Support2', 'Fib_Fan_Support3']]
    df['Fib_Fan_Support'] = supports.where(supports.lt(df['Close']), np.nan).max(axis=1)

    df = calculate_fib_cluster(df)

    vol_thresh = 1.5
    # Vectorized breakout detection
    swing_highs = df['High'].where(df['Is_Swing_High']).ffill()
    swing_highs = df['High'].where(df['Is_Swing_High']).ffill()
    vol_condition = df['Volume'] > vol_thresh * df['Volume_20_SMA']
    df['Breakout'] = np.where((df['Close'] > swing_highs) & vol_condition, 
                           swing_highs, 0)
    
    # Dynamic breakout zone with expiration
    df['BreakoutZone'] = df['Breakout'].replace(0, method='ffill')
    df['BreakoutZone'] = df['BreakoutZone'].where(
        df['Close'] > df['BreakoutZone'], 0)
    
    
    # Calculate prior resistance levels
    lookback_period = 20
    df['Prior_Resistance'] = df['High'].rolling(lookback_period).max()
    # Identify breakout (price closing above resistance)
    df['Breakout_Occurred'] = (df['Close'] > df['Prior_Resistance']).astype(int)
    # Find breakout zones that became support
    df['Potential_Support_Zone'] = df['Prior_Resistance'].where(df['Breakout_Occurred'] == 1)
    volume_multiplier = 1.5
    df['Volume_Spike'] = (df['Volume'] > volume_multiplier * df['Volume_20_SMA']).astype(int)
    # Price rejection criteria
    df['Candle_Rejection'] = ((df['Close'] - df['Low']) / (df['High'] - df['Low']) > 0.67).astype(int)
    # Signal conditions
    conditions = (
        (df['Close'] <= df['Potential_Support_Zone'] * 1.01) &  # Within 1% of support
        (df['Close'] >= df['Potential_Support_Zone'] * 0.99) &
        (df['Volume_Spike'] == 1) &
        (df['Candle_Rejection'] == 1) &
        (df['Close'] > df['Open'])  # Bullish candle
    )
    
    df['Volume_Support_Signal'] = np.where(conditions, 1, 0)

    period=25
    df['AroonUp'] = df['High'].rolling(window=period + 1).apply(lambda x: float(x.argmax()) / period * 100, raw=True)
    df['AroonDown'] = df['Low'].rolling(window=period + 1).apply(lambda x: float(x.argmin()) / period * 100, raw=True)
    
    # Shift to get the previous values for comparison
    df['AroonUp_prev'] = df['AroonUp'].shift(1)
    df['AroonDown_prev'] = df['AroonDown'].shift(1)


    # ===== 6. Fibonacci Levels =====
    df['Fib_382'] = df['Close'] * 0.382
    df['Fib_50'] = df['Close'] * 0.5
    df['Fib_618'] = df['Close'] * 0.618
    df['Fib_786'] = df['Close'] * 0.786
    df['Fib_Extension_1'] = df['Close'] * 1.618
    df['Fib_Extension_2'] = df['Close'] * 2.618
    df = add_fibonacci_support(df, 50)
    

    # ===== 7. Candlestick Features =====
    df['Candle_Body'] = abs(df['Close'] - df['Open'])
    df['Upper_Wick'] = df['High'] - df[['Close','Open']].max(axis=1)
    df['Lower_Wick'] = df[['Close','Open']].min(axis=1) - df['Low']
    df['Is_Bullish'] = (df['Close'] > df['Open']).astype(int)
    df['Is_Bearish'] = (df['Close'] < df['Open']).astype(int)
    df['Is_Doji'] = (df['Candle_Body'] < 0.001 * df['Close']).astype(int)

    # Calculate raw money flow
    df['money_flow'] = df['Typical_price'] * df['Volume']
    
    # Get direction of money flow
    df['positive_flow'] = (df['Typical_price'] > df['Typical_price'].shift(1)) * df['money_flow']
    df['negative_flow'] = (df['Typical_price'] < df['Typical_price'].shift(1)) * df['money_flow']
    
    # Calculate MFI
    df['positive_sum'] = df['positive_flow'].rolling(period).sum()
    df['negative_sum'] = df['negative_flow'].rolling(period).sum()
    df['money_ratio'] = df['positive_sum'] / df['negative_sum']
    df['MFI'] = 100 - (100 / (1 + df['money_ratio']))
    
    # Create previous value
    df['MFI_prev'] = df['MFI'].shift(1)
    
    # Fill NA values
    df['MFI'].fillna(50, inplace=True)  # Neutral value
    df['MFI_prev'].fillna(50, inplace=True)

    # Calculate channel boundaries
    df['Channel_High'] = df['High'].rolling(20).max()
    df['Channel_Low'] = df['Low'].rolling(20).min()
    
    # Calculate midline and width
    df['Channel_Midline'] = (df['Channel_High'] + df['Channel_Low']) / 2
    df['Channel_Width'] = df['Channel_High'] - df['Channel_Low']
    
    # Create previous values
    df['Channel_Midline_prev'] = df['Channel_Midline'].shift(1)
    df['Channel_High_prev'] = df['Channel_High'].shift(1)
    df['Channel_Low_prev'] = df['Channel_Low'].shift(1)
    
    # Fill NA values
    cols = ['Channel_High', 'Channel_Low', 'Channel_Midline', 'Channel_Width',
            'Channel_Midline_prev', 'Channel_High_prev', 'Channel_Low_prev']
    for col in cols:
        df[col].fillna(method='bfill', inplace=True)
    
    df = detect_bearish_trendlines(df, lookback=20, min_touches=3)

    consolidation_period = 10
    lookback = 20
    df['Range_High'] = df['High'].rolling(consolidation_period).max()
    df['Range_Low'] = df['Low'].rolling(consolidation_period).min()
    
    # Calculate additional metrics
    df['Range_Midpoint'] = (df['Range_High'] + df['Range_Low']) / 2
    df['Range_Width'] = df['Range_High'] - df['Range_Low']
    
    # Create previous values
    df['Range_High_prev'] = df['Range_High'].shift(lookback)
    df['Range_Low_prev'] = df['Range_Low'].shift(lookback)

    # Calculate ATR
    df['ATR'] = calculate_atr(df, 14)

    df = df.iloc[-1:]

    return pd.DataFrame(df)

    
def CreateVars_AllPairs(pairs, df, client, BackTime):
    """
    Run create_variablesV2 in parallel for a list of trading pairs.
    
    Parameters:
    - pairs: List of trading pairs (e.g., ['BTCUSDT', 'ETHUSDT']).
    - df: Input DataFrame, assumed to contain data for all pairs (filtered by symbol).
    - client: Binance API client (e.g., from python-binance).
    - BackTime: Lookback period for create_variablesV2 (e.g., timestamp, '24h').
    
    Returns:
    - pandas DataFrame combining results from all pairs.
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=10) as executor:
        # Submit tasks for each pair
        futures = [
            executor.submit(create_variables, df[df['Symbol'] == symbol] if 'Symbol' in df.columns else df, symbol, client, BackTime)
            for symbol in pairs
        ]
        # Process completed tasks
        for future in as_completed(futures):
            try:
                result_df = future.result()
                if result_df is not None and not result_df.empty:
                    results.append(result_df)
            except Exception as e:
                symbol = pairs[futures.index(future)]  # Approximate symbol
                print(f"Error processing {symbol}: {e}")
    
    # Combine results into a single DataFrame
    if results:
        final_df = pd.concat(results, ignore_index=True)
        return final_df
    

def CreateVars_AllPairsv2(pairs, df, client, BackTime):
    """
    Run create_variablesV2 sequentially for a list of trading pairs with a progress bar.
    """
    results = []
    for symbol in tqdm(pairs, desc='Processing pairs'):
        # Filter df for the current symbol if 'Symbol' column exists
        print(f"Processing {symbol}")
        df_symbol = df if 'Symbol' not in df.columns else df[df['Symbol'] == symbol]
        result_df = create_variables(df_symbol, symbol, client, BackTime)
        if result_df is not None and not result_df.empty:
            results.append(result_df)
    
    # Concatenate results
    if results:
        final_df = pd.concat(results, ignore_index=True)
        return final_df
