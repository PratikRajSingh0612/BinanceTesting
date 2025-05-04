import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
import pandas_ta as ta
from tqdm import tqdm


def allstrategies(df):
    # Calculate moving averages
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    df['MA_200'] = df['Close'].rolling(window=200).mean()
    df['EMA_20'] = ta.ema(df['Close'], length=20)
    df['EMA_50'] = ta.ema(df['Close'], length=50)
    df['EMA_200'] = ta.ema(df['Close'], length=200)
    df['EMA_10'] = ta.ema(df['Close'], length=10)

    # Calculate previous values for comparison
    df['MA_50_prev'] = df['MA_50'].shift(1)
    df['MA_200_prev'] = df['MA_200'].shift(1)
    df['EMA_20_prev'] = df['EMA_20'].shift(1)
    df['EMA_50_prev'] = df['EMA_50'].shift(1)
    df['EMA_200_prev'] = df['EMA_200'].shift(1)
    df['EMA_10_prev'] = df['EMA_10'].shift(1)

    df['Volume_20_SMA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_prev'] = df['Volume'].shift(1)
    df['Close_prev'] = df['Close'].shift(1)
    df['OBV'] = (df['Volume'] * ((df['Close'] > df['Close_prev']).astype(int) - (df['Close'] < df['Close_prev']).astype(int))).cumsum()
    df['OBV_prev'] = df['OBV'].shift(1)
    df['Resistance'] = df['High'].rolling(window=20).max()  # Example resistance level
    df['BreakoutZone'] = df['High'].rolling(window=50).max()  # Example breakout zone

    # Add required columns for RSI-based strategies
    df['RSI'] = ta.rsi(df['Close'], length=14)
    df['RSI_prev'] = df['RSI'].shift(1)
    df['RSI_prev2'] = df['RSI'].shift(2)
    df['Low_prev'] = df['Low'].shift(1)
    df['RSI_4H'] = ta.rsi(df['Close'], length=14)  # Example for 4H timeframe
    df['RSI_1D'] = ta.rsi(df['Close'], length=14)  # Example for 1D timeframe
    df['RSI_Trendline'] = df['RSI'].rolling(window=5).mean()  # Example trendline

    # Add required columns for MACD-based strategies
    df['MACD_prev'] = df['MACD'].shift(1)
    df['MACD_Signal_prev'] = df['MACD_Signal'].shift(1)
    df['MACD_Histogram_prev'] = df['MACD_Histogram'].shift(1)
    df['Close_prev'] = df['Close'].shift(1)

    # Assuming higher timeframe MACD values are precomputed
    df['MACD_4H'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD']
    df['MACD_Signal_4H'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACDs']
    df['MACD_1D'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACD']
    df['MACD_Signal_1D'] = ta.macd(df['Close'], fast=12, slow=26, signal=9)['MACDs']

    # Add required columns for chart pattern-based strategies
    df['Resistance'] = df['High'].rolling(window=20).max()  # Example resistance level
    df['Low_prev'] = df['Low'].shift(1)
    df['Low_prev2'] = df['Low'].shift(2)
    df['High_prev'] = df['High'].shift(1)
    df['Volume_20_SMA'] = df['Volume'].rolling(window=20).mean()

    # Add required columns for support/resistance-based strategies
    df['Support'] = df['Low'].rolling(window=20).min()  # Example support level
    df['Resistance'] = df['High'].rolling(window=20).max()  # Example resistance level
    df['Support_Tests'] = df['Low'].rolling(window=20).apply(lambda x: (x == x.min()).sum(), raw=True)
    df['Weekly_Support'] = df['Low'].rolling(window=140).min()  # Weekly support (assuming 7 days * 20 periods/day)
    df['Fib_Support'] = df['Close'] * 0.618  # Example Fibonacci retracement level (61.8%)

    # Add required columns for Fibonacci-based strategies
    df['Fib_618'] = df['Close'] * 0.618  # Example 0.618 Fib level
    df['Fib_382'] = df['Close'] * 0.382  # Example 0.382 Fib level
    df['Fib_50'] = df['Close'] * 0.5    # Example 0.5 Fib level
    df['Fib_Extension_1'] = df['Close'] * 1.618  # Example Fib extension target
    df['Fib_Extension_2'] = df['Close'] * 2.618  # Example Fib extension target
    df['Fib_Fan_Support'] = df['Close'] * 0.786  # Example Fib fan support
    df['Fib_Cluster'] = (df['Fib_618'] + df['Fib_382']) / 2  # Example Fib cluster
    df['Fib_65'] = df['Close'] * 0.65  # Golden pocket upper bound

    # Add required columns for indicators & oscillators-based strategies
    df['StochRSI'] = ta.stochrsi(df['Close'], length=14)
    df['StochRSI_prev'] = df['StochRSI'].shift(1)
    df['ADX'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['ADX_14']
    df['DI+'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMP_14']
    df['DI-'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMN_14']
    df['CCI'] = ta.cci(df['High'], df['Low'], df['Close'], length=20)
    df['CCI_prev'] = df['CCI'].shift(1)
    df['WilliamsR'] = ta.willr(df['High'], df['Low'], df['Close'], length=14)
    df['WilliamsR_prev'] = df['WilliamsR'].shift(1)
    df['AroonUp'] = ta.aroon(df['High'], df['Low'], length=25)['AROOND_25']
    df['AroonDown'] = ta.aroon(df['High'], df['Low'], length=25)['AROONU_25']
    df['AroonUp_prev'] = df['AroonUp'].shift(1)
    df['AroonDown_prev'] = df['AroonDown'].shift(1)
    df['CMF'] = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    df['CMF_prev'] = df['CMF'].shift(1)
    df['KVO'] = ta.kvo(df['High'], df['Low'], df['Close'], df['Volume'])['KVO']
    df['KVO_Signal'] = ta.kvo(df['High'], df['Low'], df['Close'], df['Volume'])['KVOs']
    df['KVO_prev'] = df['KVO'].shift(1)
    df['KVO_Signal_prev'] = df['KVO_Signal'].shift(1)
    df['Trix'] = ta.trix(df['Close'], length=15)
    df['Trix_prev'] = df['Trix'].shift(1)
    df['MFI'] = ta.mfi(df['High'], df['Low'], df['Close'], df['Volume'], length=14)
    df['MFI_prev'] = df['MFI'].shift(1)
    df['DMI+'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMP_14']
    df['DMI-'] = ta.adx(df['High'], df['Low'], df['Close'], length=14)['DMN_14']
    df['DMI+_prev'] = df['DMI+'].shift(1)
    df['DMI-_prev'] = df['DMI-'].shift(1)

    # Add required columns for trend & structure-based strategies
    df['High_prev'] = df['High'].shift(1)
    df['Low_prev'] = df['Low'].shift(1)
    df['Trendline'] = df['Close'].rolling(window=20).mean()  # Example trendline
    df['Resistance'] = df['High'].rolling(window=20).max()
    df['Support'] = df['Low'].rolling(window=20).min()
    df['Swing_High'] = df['High'].rolling(window=50).max()
    df['Channel_Midline'] = (df['High'] + df['Low']) / 2
    df['Range_High'] = df['High'].rolling(window=30).max()
    df['High_4H'] = df['High'].rolling(window=48).max()  # Example for 4H timeframe
    df['Low_4H'] = df['Low'].rolling(window=48).min()
    df['High_4H_prev'] = df['High_4H'].shift(1)
    df['Low_4H_prev'] = df['Low_4H'].shift(1)
    df['High_1D'] = df['High'].rolling(window=288).max()  # Example for 1D timeframe
    df['Low_1D'] = df['Low'].rolling(window=288).min()
    df['High_1D_prev'] = df['High_1D'].shift(1)
    df['Low_1D_prev'] = df['Low_1D'].shift(1)


    # Calculate slopes
    df['MA_50_slope'] = (df['MA_50'] - df['MA_50'].shift(5)) / 5

    # Apply moving average strategy functions
    strategies = {
        # Moving Average Strategies
        'Golden Cross': golden_cross,
        'Price Bounces Off 200 EMA': price_bounces_off_200_ema,
        'Price Crosses Above 20 EMA': price_crosses_above_20_ema,
        'EMA 10 Crosses Above 50': ema_10_crosses_above_50,
        'Price Above All Major MAs': price_above_all_major_mas,
        'MA Ribbon Compression': ma_ribbon_compression,
        'Price Retests MA': price_retests_ma,
        'Death Cross Avoidance': death_cross_avoidance,
        'EMA Crossover High Timeframe': ema_crossover_high_timeframe,
        'MA Slope Turning Upward': ma_slope_turning_upward,
        # Volume-based Strategies
        'Volume Spike Green Candle': volume_spike_green_candle,
        'Volume Breakout Above 20 Day Avg': volume_breakout_above_20_day_avg,
        'Bullish Price Volume Divergence': bullish_price_volume_divergence,
        'Climax Volume Reversal': climax_volume_reversal,
        'Volume Supporting Resistance Breakout': volume_supporting_resistance_breakout,
        'Volume Rising Price Consolidates': volume_rising_price_consolidates,
        'Accumulation Volume Pattern': accumulation_volume_pattern,
        'Low Volume Pullback After High Volume Rally': low_volume_pullback_after_high_volume_rally,
        'Volume Rising OBV Rising': volume_rising_obv_rising,
        'Volume Supporting Prior Breakout Zone': volume_support_prior_breakout_zone,
        # RSI-based Strategies
        'RSI Oversold': rsi_oversold,
        'RSI Bullish Divergence': rsi_bullish_divergence,
        'RSI Crosses Above 30': rsi_crosses_above_30,
        'RSI Higher Lows vs Price Lower Lows': rsi_higher_lows_vs_price_lower_lows,
        'RSI Moving Out of Bear Control': rsi_moving_out_of_bear_control,
        'RSI Above 50': rsi_above_50,
        'RSI Trendline Breakout': rsi_trendline_breakout,
        'RSI Multi-timeframe Convergence': rsi_multi_timeframe_convergence,
        'RSI Double Bottom': rsi_double_bottom,
        'RSI Holding Bullish Structure': rsi_holding_bullish_structure,
        # MACD-based strategies
        'MACD Bullish Crossover': macd_bullish_crossover,
        'MACD Histogram Flips Green': macd_histogram_flips_green,
        'MACD Bullish Divergence': macd_bullish_divergence,
        'MACD Crossover Above Zero': macd_crossover_above_zero,
        'MACD Support at Zero': macd_support_at_zero,
        'MACD Trending Upward with Price': macd_trending_upward_with_price,
        'MACD Crossover Multiple Timeframes': macd_crossover_multiple_timeframes,
        'MACD Histogram Expanding': macd_histogram_expanding,
        'MACD Fakeout Signal': macd_fakeout_signal,
        'MACD Wide Separation Forming': macd_wide_separation_forming,
        # Chart Pattern-based strategies
        'Cup and Handle Breakout': cup_and_handle_breakout,
        'Inverse Head and Shoulders': inverse_head_and_shoulders,
        'Ascending Triangle Breakout': ascending_triangle_breakout,
        'Falling Wedge Breakout': falling_wedge_breakout,
        'Bull Flag Breakout': bull_flag_breakout,
        'Symmetrical Triangle Breakout': symmetrical_triangle_breakout,
        'Double Bottom Confirmation': double_bottom_confirmation,
        'Rounded Bottom Formation': rounded_bottom_formation,
        'Breakout from Consolidation Range': breakout_from_consolidation_range,
        'Expanding Triangle Breakout Upward': expanding_triangle_breakout_upward,
        # support/resistance-based strategies
        'Bounce from Long Term Support': bounce_from_long_term_support,
        'Price Flips Resistance into Support': price_flips_resistance_into_support,
        'Confluence of Support': confluence_of_support,
        'Horizontal Level Tested Multiple Times': horizontal_level_tested_multiple_times,
        'Strong Psychological Level': strong_psychological_level,
        'Support from Weekly Respected on Daily': support_from_weekly_respected_on_daily,
        'Price Rejects Breakdown from Key Support': price_rejects_breakdown_from_key_support,
        'Price Retests Prior Breakout Zone': price_retests_prior_breakout_zone,
        'Support with Bullish Divergence': support_with_bullish_divergence,
        'Support Aligning with Fib Retracement': support_aligning_with_fib_retracement,
        # Fibonacci-based strategies
        'Buy at 618 Fib Retracement': buy_at_618_fib_retracement,
        'Buy near 382 Bounce': buy_near_382_bounce,
        'Confluence of Fib Levels': confluence_of_fib_and_support,
        'Buy When Price Reclaims 0.5 Fib': buy_when_price_reclaims_05_fib,
        'Fib Extension Targets Reached': fib_extension_targets_reached,
        'Fib Fan Support Bounce': fib_fan_support_bounce,
        'Fib Cluster Zone with Indicators': fib_cluster_zone_with_indicators,
        'Price Consolidates Between Fib Levels': price_consolidates_between_fib_levels,
        'Fib Retracement After Parabolic Move': fib_retracement_after_parabolic_move,
        'Buy at Golden Pocket Zone': buy_at_golden_pocket_zone,
        # indicators & oscillators-based strategies
        'Stochastic RSI Oversold Cross Up': stochastic_rsi_oversold_cross_up,
        'ADX > 25 + Bullish DI': adx_bullish_di,
        'CCI Crossing Above -100': cci_crossing_above_minus_100,
        'Williams %R Coming Out of Oversold': williams_r_coming_out_of_oversold,
        'Aroon Up Crosses Above Aroon Down': aroon_up_crosses_above_aroon_down,
        'Chaikin Money Flow Turns Positive': chaikin_money_flow_turns_positive,
        'Klinger Volume Oscillator Bullish Cross': klinger_volume_oscillator_bullish_cross,
        'Trix Indicator Crosses Zero': trix_indicator_crosses_zero,
        'Money Flow Index < 20 and Rising': money_flow_index_rising,
        'DMI+ Crosses Above DMI-': dmi_plus_crosses_above_dmi_minus,
        # trend & structure-based strategies
        'Higher High and Higher Low Confirmation': higher_high_higher_low,
        'Break of Bearish Trendline': break_of_bearish_trendline,
        'Retest of Broken Trendline': retest_of_broken_trendline,
        'Trend Continuation After Consolidation': trend_continuation_after_consolidation,
        'Bullish Market Structure on Multiple Timeframes': bullish_market_structure_multiple_timeframes,
        'Break of Previous Swing High': break_of_previous_swing_high,
        'Rising Channel Midline Support Bounce': rising_channel_midline_support_bounce,
        'Price Reclaiming Previous Range High': price_reclaiming_previous_range_high,
        'Support Formed Above Previous Resistance': support_formed_above_previous_resistance,
        'Low Volatility + Range Breakout': low_volatility_range_breakout,


    }

    for name, func in strategies.items():
        df[name] = df.apply(func, axis=1)

    return df

####################### Define Moving Average Strategy #######################
def golden_cross(row):
    if row['MA_50'] > row['MA_200'] and row['MA_50_prev'] <= row['MA_200_prev']:
        return 1
    return 0

def price_bounces_off_200_ema(row):
    if abs(row['Close'] - row['EMA_200']) / row['EMA_200'] < 0.01:  # Within 1% of EMA
        return 1
    return 0

def price_crosses_above_20_ema(row):
    if row['Close'] > row['EMA_20'] and row['Close_prev'] <= row['EMA_20_prev']:
        return 1
    return 0

def ema_10_crosses_above_50(row):
    if row['EMA_10'] > row['EMA_50'] and row['EMA_10_prev'] <= row['EMA_50_prev']:
        return 1
    return 0

def price_above_all_major_mas(row):
    if row['Close'] > row['EMA_20'] and row['Close'] > row['EMA_50'] and row['Close'] > row['EMA_200']:
        return 1
    return 0

def ma_ribbon_compression(row):
    if abs(row['EMA_10'] - row['EMA_20']) < 0.01 * row['Close'] and abs(row['EMA_20'] - row['EMA_50']) < 0.01 * row['Close']:
        return 1
    return 0

def price_retests_ma(row):
    if row['Close'] > row['EMA_20'] and row['Low'] < row['EMA_20']:
        return 1
    return 0

def death_cross_avoidance(row):
    if row['MA_50'] < row['MA_200'] and row['MA_50_prev'] >= row['MA_200_prev']:
        return 1
    return 0

def ema_crossover_high_timeframe(row):
    if row['EMA_50_HTF'] > row['EMA_200_HTF'] and row['EMA_50_HTF_prev'] <= row['EMA_200_HTF_prev']:
        return 1
    return 0

def ma_slope_turning_upward(row):
    if row['MA_50_slope'] > 0 and row['MA_50_slope_prev'] <= 0:
        return 1
    return 0

######################## Define volume-based strategy functions #######################
def volume_spike_green_candle(row):
    if row['Volume'] > 2 * row['Volume_20_SMA'] and row['Close'] > row['Open']:
        return 1
    return 0

def volume_breakout_above_20_day_avg(row):
    if row['Volume'] > row['Volume_20_SMA']:
        return 1
    return 0

def bullish_price_volume_divergence(row):
    if row['Close'] > row['Close_prev'] and row['Volume'] < row['Volume_prev']:
        return 1
    return 0

def climax_volume_reversal(row):
    if row['Volume'] > 2 * row['Volume_20_SMA'] and row['Close'] < row['Open']:
        return 1
    return 0

def volume_supporting_resistance_breakout(row):
    if row['Close'] > row['Resistance'] and row['Volume'] > row['Volume_20_SMA']:
        return 1
    return 0

def volume_rising_price_consolidates(row):
    if row['High'] - row['Low'] < 0.01 * row['Close'] and row['Volume'] > row['Volume_prev']:
        return 1
    return 0

def accumulation_volume_pattern(row):
    if row['Close'] > row['Open'] and row['Volume'] > row['Volume_prev']:
        return 1
    return 0

def low_volume_pullback_after_high_volume_rally(row):
    if row['Volume'] < row['Volume_20_SMA'] and row['Close'] < row['Close_prev']:
        return 1
    return 0

def volume_rising_obv_rising(row):
    if row['Volume'] > row['Volume_prev'] and row['OBV'] > row['OBV_prev']:
        return 1
    return 0

def volume_support_prior_breakout_zone(row):
    if row['Close'] > row['BreakoutZone'] and row['Volume'] > row['Volume_20_SMA']:
        return 1
    return 0

######################## Define RSI-based strategy functions #######################
def rsi_oversold(row):
    if row['RSI'] < 30:
        return 1
    return 0

def rsi_bullish_divergence(row):
    if row['RSI'] > row['RSI_prev'] and row['Close'] < row['Close_prev']:
        return 1
    return 0

def rsi_crosses_above_30(row):
    if row['RSI'] > 30 and row['RSI_prev'] <= 30:
        return 1
    return 0

def rsi_higher_lows_vs_price_lower_lows(row):
    if row['RSI'] > row['RSI_prev'] and row['Low'] < row['Low_prev']:
        return 1
    return 0

def rsi_moving_out_of_bear_control(row):
    if row['RSI'] > 40 and row['RSI_prev'] <= 40:
        return 1
    return 0

def rsi_above_50(row):
    if row['RSI'] > 50:
        return 1
    return 0

def rsi_trendline_breakout(row):
    if row['RSI'] > row['RSI_Trendline']:
        return 1
    return 0

def rsi_multi_timeframe_convergence(row):
    if row['RSI_4H'] > 50 and row['RSI_1D'] > 50:
        return 1
    return 0

def rsi_double_bottom(row):
    if row['RSI'] > row['RSI_prev'] and row['RSI_prev'] < row['RSI_prev2']:
        return 1
    return 0

def rsi_holding_bullish_structure(row):
    if 40 < row['RSI'] < 70:
        return 1
    return 0

######################## Define MACD-based strategy functions #######################
def macd_bullish_crossover(row):
    if row['MACD'] > row['MACD_Signal'] and row['MACD_prev'] <= row['MACD_Signal_prev']:
        return 1
    return 0

def macd_histogram_flips_green(row):
    if row['MACD_Histogram'] > 0 and row['MACD_Histogram_prev'] <= 0:
        return 1
    return 0

def macd_bullish_divergence(row):
    if row['MACD'] > row['MACD_prev'] and row['Close'] < row['Close_prev']:
        return 1
    return 0

def macd_crossover_above_zero(row):
    if row['MACD'] > 0 and row['MACD_Signal'] > 0 and row['MACD_prev'] <= 0:
        return 1
    return 0

def macd_support_at_zero(row):
    if abs(row['MACD']) < 0.01 and row['MACD_Histogram'] > 0:
        return 1
    return 0

def macd_trending_upward_with_price(row):
    if row['MACD'] > row['MACD_prev'] and row['Close'] > row['Close_prev']:
        return 1
    return 0

def macd_crossover_multiple_timeframes(row):
    if row['MACD_4H'] > row['MACD_Signal_4H'] and row['MACD_1D'] > row['MACD_Signal_1D']:
        return 1
    return 0

def macd_histogram_expanding(row):
    if abs(row['MACD_Histogram']) > abs(row['MACD_Histogram_prev']):
        return 1
    return 0

def macd_fakeout_signal(row):
    if row['MACD'] < row['MACD_Signal'] and row['MACD_prev'] > row['MACD_Signal_prev']:
        return 1
    return 0

def macd_wide_separation_forming(row):
    if abs(row['MACD'] - row['MACD_Signal']) > abs(row['MACD_prev'] - row['MACD_Signal_prev']):
        return 1
    return 0


######################## Define chart pattern-based strategy functions #######################

def cup_and_handle_breakout(row):
    if row['Close'] > row['Resistance'] and row['Volume'] > row['Volume_20_SMA']:
        return 1
    return 0

def inverse_head_and_shoulders(row):
    if row['Low'] < row['Low_prev'] and row['Low_prev2'] > row['Low_prev']:
        return 1
    return 0

def ascending_triangle_breakout(row):
    if row['Close'] > row['Resistance'] and row['Low'] > row['Low_prev']:
        return 1
    return 0

def falling_wedge_breakout(row):
    if row['Close'] > row['Resistance'] and row['High'] < row['High_prev']:
        return 1
    return 0

def bull_flag_breakout(row):
    if row['Close'] > row['Resistance'] and row['High'] > row['High_prev']:
        return 1
    return 0

def symmetrical_triangle_breakout(row):
    if row['Close'] > row['Resistance'] and abs(row['High'] - row['Low']) < 0.01 * row['Close']:
        return 1
    return 0

def double_bottom_confirmation(row):
    if row['Low'] == row['Low_prev2'] and row['Close'] > row['Resistance']:
        return 1
    return 0

def rounded_bottom_formation(row):
    if row['Low'] > row['Low_prev'] and row['Low_prev'] > row['Low_prev2']:
        return 1
    return 0

def breakout_from_consolidation_range(row):
    if row['Close'] > row['Resistance'] and abs(row['High'] - row['Low']) < 0.01 * row['Close']:
        return 1
    return 0

def expanding_triangle_breakout_upward(row):
    if row['Close'] > row['Resistance'] and row['High'] > row['High_prev'] and row['Low'] < row['Low_prev']:
        return 1
    return 0


######################## Define support/resistance-based strategy functions #######################
def bounce_from_long_term_support(row):
    if abs(row['Close'] - row['Support']) / row['Support'] < 0.01:  # Within 1% of support
        return 1
    return 0

def price_flips_resistance_into_support(row):
    if row['Close'] > row['Resistance'] and row['Low'] < row['Resistance']:
        return 1
    return 0

def confluence_of_support(row):
    if abs(row['Close'] - row['Support']) / row['Support'] < 0.01 and row['Close'] > row['EMA_50']:
        return 1
    return 0

def horizontal_level_tested_multiple_times(row):
    if row['Support_Tests'] >= 3:
        return 1
    return 0

def strong_psychological_level(row):
    if row['Close'] % 10 == 0 or row['Close'] % 100 == 0:
        return 1
    return 0

def support_from_weekly_respected_on_daily(row):
    if abs(row['Close'] - row['Weekly_Support']) / row['Weekly_Support'] < 0.01:
        return 1
    return 0

def price_rejects_breakdown_from_key_support(row):
    if row['Low'] < row['Support'] and row['Close'] > row['Support']:
        return 1
    return 0

def price_retests_prior_breakout_zone(row):
    if abs(row['Close'] - row['BreakoutZone']) / row['BreakoutZone'] < 0.01:
        return 1
    return 0

def support_with_bullish_divergence(row):
    if row['Close'] > row['Support'] and row['RSI'] > row['RSI_prev'] and row['MACD'] > row['MACD_prev']:
        return 1
    return 0

def support_aligning_with_fib_retracement(row):
    if abs(row['Close'] - row['Fib_Support']) / row['Fib_Support'] < 0.01:
        return 1
    return 0

######################## Define Fibonacci-based strategy functions #######################
def buy_at_618_fib_retracement(row):
    if abs(row['Close'] - row['Fib_618']) / row['Fib_618'] < 0.01:  # Within 1% of 0.618 Fib
        return 1
    return 0

def buy_near_382_bounce(row):
    if abs(row['Close'] - row['Fib_382']) / row['Fib_382'] < 0.01:  # Within 1% of 0.382 Fib
        return 1
    return 0

def confluence_of_fib_and_support(row):
    if abs(row['Close'] - row['Fib_618']) / row['Fib_618'] < 0.01 and abs(row['Close'] - row['Support']) / row['Support'] < 0.01:
        return 1
    return 0

def buy_when_price_reclaims_05_fib(row):
    if row['Close'] > row['Fib_50'] and row['Close_prev'] <= row['Fib_50']:
        return 1
    return 0

def fib_extension_targets_reached(row):
    if row['Close'] > row['Fib_Extension_1'] and row['Close'] < row['Fib_Extension_2']:
        return 1
    return 0

def fib_fan_support_bounce(row):
    if abs(row['Close'] - row['Fib_Fan_Support']) / row['Fib_Fan_Support'] < 0.01:
        return 1
    return 0

def fib_cluster_zone_with_indicators(row):
    if abs(row['Close'] - row['Fib_Cluster']) / row['Fib_Cluster'] < 0.01 and row['RSI'] > 50 and row['MACD'] > row['MACD_Signal']:
        return 1
    return 0

def price_consolidates_between_fib_levels(row):
    if row['Close'] > row['Fib_382'] and row['Close'] < row['Fib_618'] and abs(row['High'] - row['Low']) < 0.01 * row['Close']:
        return 1
    return 0

def fib_retracement_after_parabolic_move(row):
    if abs(row['Close'] - row['Fib_618']) / row['Fib_618'] < 0.01 and row['Volume'] > row['Volume_20_SMA']:
        return 1
    return 0

def buy_at_golden_pocket_zone(row):
    if row['Fib_618'] <= row['Close'] <= row['Fib_65']:
        return 1
    return 0

######################## Define indicators & oscillators-based strategy functions #######################
def stochastic_rsi_oversold_cross_up(row):
    if row['StochRSI'] < 20 and row['StochRSI'] > row['StochRSI_prev']:
        return 1
    return 0

def adx_bullish_di(row):
    if row['ADX'] > 25 and row['DI+'] > row['DI-']:
        return 1
    return 0

def cci_crossing_above_minus_100(row):
    if row['CCI'] > -100 and row['CCI_prev'] <= -100:
        return 1
    return 0

def williams_r_coming_out_of_oversold(row):
    if row['WilliamsR'] > -80 and row['WilliamsR_prev'] <= -80:
        return 1
    return 0

def aroon_up_crosses_above_aroon_down(row):
    if row['AroonUp'] > row['AroonDown'] and row['AroonUp_prev'] <= row['AroonDown_prev']:
        return 1
    return 0

def chaikin_money_flow_turns_positive(row):
    if row['CMF'] > 0 and row['CMF_prev'] <= 0:
        return 1
    return 0

def klinger_volume_oscillator_bullish_cross(row):
    if row['KVO'] > row['KVO_Signal'] and row['KVO_prev'] <= row['KVO_Signal_prev']:
        return 1
    return 0

def trix_indicator_crosses_zero(row):
    if row['Trix'] > 0 and row['Trix_prev'] <= 0:
        return 1
    return 0

def money_flow_index_rising(row):
    if row['MFI'] < 20 and row['MFI'] > row['MFI_prev']:
        return 1
    return 0

def dmi_plus_crosses_above_dmi_minus(row):
    if row['DMI+'] > row['DMI-'] and row['DMI+_prev'] <= row['DMI-_prev']:
        return 1
    return 0

######################## Define trend & structure-based strategy functions #######################
def higher_high_higher_low(row):
    if row['High'] > row['High_prev'] and row['Low'] > row['Low_prev']:
        return 1
    return 0

def break_of_bearish_trendline(row):
    if row['Close'] > row['Trendline'] and row['Close_prev'] <= row['Trendline']:
        return 1
    return 0

def retest_of_broken_trendline(row):
    if row['Close'] > row['Trendline'] and row['Low'] < row['Trendline']:
        return 1
    return 0

def trend_continuation_after_consolidation(row):
    if row['Close'] > row['Resistance'] and abs(row['High'] - row['Low']) < 0.01 * row['Close']:
        return 1
    return 0

def bullish_market_structure_multiple_timeframes(row):
    if row['High_4H'] > row['High_4H_prev'] and row['Low_4H'] > row['Low_4H_prev'] and row['High_1D'] > row['High_1D_prev'] and row['Low_1D'] > row['Low_1D_prev']:
        return 1
    return 0

def break_of_previous_swing_high(row):
    if row['High'] > row['Swing_High']:
        return 1
    return 0

def rising_channel_midline_support_bounce(row):
    if abs(row['Close'] - row['Channel_Midline']) / row['Channel_Midline'] < 0.01:
        return 1
    return 0

def price_reclaiming_previous_range_high(row):
    if row['Close'] > row['Range_High'] and row['Close_prev'] <= row['Range_High']:
        return 1
    return 0

def support_formed_above_previous_resistance(row):
    if row['Support'] > row['Resistance']:
        return 1
    return 0

def low_volatility_range_breakout(row):
    if abs(row['High'] - row['Low']) < 0.01 * row['Close'] and row['Close'] > row['Resistance']:
        return 1
    return 0


########################### Candle Stick Pattern Strategies #########################
# 1. Single-Candle Patterns (Bullish Reversal)
def hammer(row):
    if row['Close'] > row['Open'] and (row['Low'] < row['Open'] - 2 * abs(row['Close'] - row['Open'])):
        return 1
    return 0

def inverse_hammer(row):
    if row['Close'] > row['Open'] and (row['High'] > row['Close'] + 2 * abs(row['Close'] - row['Open'])):
        return 1
    return 0

def bullish_marubozu(row):
    if row['Close'] > row['Open'] and abs(row['High'] - row['Close']) < 0.001 * row['Close'] and abs(row['Open'] - row['Low']) < 0.001 * row['Close']:
        return 1
    return 0

def dragonfly_doji(row):
    if abs(row['Close'] - row['Open']) < 0.001 * row['Close'] and (row['Low'] < row['Open'] - 2 * abs(row['High'] - row['Low'])):
        return 1
    return 0

def spinning_top_bullish(row):
    if abs(row['Close'] - row['Open']) < 0.3 * abs(row['High'] - row['Low']) and row['Close'] > row['Open']:
        return 1
    return 0

def hanging_man_in_uptrend(row):
    if row['Close'] < row['Open'] and (row['Low'] < row['Open'] - 2 * abs(row['Close'] - row['Open'])):
        return 1
    return 0

# Define two-candle pattern strategies (Bullish Reversal)

def bullish_engulfing(row):
    if row['Close'] > row['Open'] and row['Close_prev'] < row['Open_prev'] and row['Close'] > row['Open_prev'] and row['Open'] < row['Close_prev']:
        return 1
    return 0

def piercing_line(row):
    if row['Close_prev'] < row['Open_prev'] and row['Close'] > row['Open_prev'] and row['Close'] > (row['Open_prev'] + row['Close_prev']) / 2:
        return 1
    return 0

def tweezer_bottom(row):
    if row['Low'] == row['Low_prev'] and row['Close'] > row['Open']:
        return 1
    return 0

def kicking_bullish(row):
    if row['Open_prev'] < row['Close_prev'] and row['Open'] > row['Close'] and row['Open'] > row['Close_prev']:
        return 1
    return 0

def matching_low(row):
    if row['Low'] == row['Low_prev']:
        return 1
    return 0

def last_engulfing_bottom(row):
    if row['Close_prev'] > row['Open_prev'] and row['Close'] < row['Open'] and row['Close'] < row['Close_prev'] and row['Open'] > row['Open_prev']:
        return 1
    return 0

def on_neck_line_bullish(row):
    if row['Close_prev'] < row['Open_prev'] and row['Close'] == row['Low'] and row['Close'] > row['Close_prev']:
        return 1
    return 0

# Define three-candle pattern strategies (Bullish Reversal)

def morning_star(row):
    if row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] < row['Open_prev'] and row['Close'] > row['Open_prev2']:
        return 1
    return 0

def morning_doji_star(row):
    if row['Close_prev2'] < row['Open_prev2'] and abs(row['Close_prev'] - row['Open_prev']) < 0.001 * row['Close_prev'] and row['Close'] > row['Open_prev2']:
        return 1
    return 0

def abandoned_baby_bullish(row):
    if row['Close_prev2'] < row['Open_prev2'] and abs(row['Close_prev'] - row['Open_prev']) < 0.001 * row['Close_prev'] and row['Close'] > row['Open_prev2'] and row['Low_prev'] > row['High_prev2']:
        return 1
    return 0

def three_inside_up(row):
    if row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] > row['Open_prev'] and row['Close'] > row['Close_prev']:
        return 1
    return 0

def three_outside_up(row):
    if row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] > row['Open_prev'] and row['Close_prev'] > row['Close_prev2'] and row['Close'] > row['Close_prev']:
        return 1
    return 0

def stick_sandwich_bullish(row):
    if row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] > row['Open_prev'] and row['Close'] == row['Close_prev2']:
        return 1
    return 0

def tri_star_bullish(row):
    if abs(row['Close_prev2'] - row['Open_prev2']) < 0.001 * row['Close_prev2'] and abs(row['Close_prev'] - row['Open_prev']) < 0.001 * row['Close_prev'] and abs(row['Close'] - row['Open']) < 0.001 * row['Close']:
        return 1
    return 0

# Define multi-candle pattern strategies (Bullish Continuation/Reversal)

def three_white_soldiers(row):
    if row['Close_prev2'] > row['Open_prev2'] and row['Close_prev'] > row['Open_prev'] and row['Close'] > row['Open'] and row['Close_prev2'] < row['Open_prev'] and row['Close_prev'] < row['Open']:
        return 1
    return 0

def rising_three_methods(row):
    if row['Close_prev4'] > row['Open_prev4'] and row['Close_prev3'] < row['Open_prev3'] and row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] < row['Open_prev'] and row['Close'] > row['Open_prev4']:
        return 1
    return 0

def separating_lines_bullish(row):
    if row['Close_prev2'] < row['Open_prev2'] and row['Open_prev'] == row['Open_prev2'] and row['Close'] > row['Open']:
        return 1
    return 0

def mat_hold(row):
    if row['Close_prev4'] > row['Open_prev4'] and row['Close_prev3'] < row['Open_prev3'] and row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] < row['Open_prev'] and row['Close'] > row['Close_prev4']:
        return 1
    return 0

def side_by_side_white_lines(row):
    if row['Close_prev2'] > row['Open_prev2'] and row['Close_prev'] > row['Open_prev'] and row['Close'] > row['Open'] and row['Open_prev'] == row['Open_prev2']:
        return 1
    return 0

def upside_tasuki_gap(row):
    if row['Close_prev2'] > row['Open_prev2'] and row['Open_prev'] > row['Close_prev2'] and row['Close_prev'] > row['Open_prev'] and row['Open'] < row['Close_prev'] and row['Close'] > row['Open_prev']:
        return 1
    return 0

# Define bullish breakout pattern strategies

def bullish_flag_breakout(row):
    if row['Close'] > row['Resistance'] and row['High'] > row['High_prev'] and row['Low'] > row['Low_prev']:
        return 1
    return 0

def bullish_pennant_breakout(row):
    if row['Close'] > row['Resistance'] and abs(row['High'] - row['Low']) < 0.01 * row['Close']:
        return 1
    return 0

def falling_wedge_breakout(row):
    if row['Close'] > row['Resistance'] and row['High'] < row['High_prev']:
        return 1
    return 0

def symmetrical_triangle_breakout_upward(row):
    if row['Close'] > row['Resistance'] and abs(row['High'] - row['Low']) < 0.01 * row['Close']:
        return 1
    return 0

def ascending_triangle_breakout(row):
    if row['Close'] > row['Resistance'] and row['Low'] > row['Low_prev']:
        return 1
    return 0

def double_bottom_breakout(row):
    if row['Low'] == row['Low_prev2'] and row['Close'] > row['Resistance']:
        return 1
    return 0

def triple_bottom_breakout(row):
    if row['Low'] == row['Low_prev3'] and row['Low_prev2'] == row['Low_prev3'] and row['Close'] > row['Resistance']:
        return 1
    return 0

def rounding_bottom_breakout(row):
    if row['Low'] > row['Low_prev'] and row['Low_prev'] > row['Low_prev2'] and row['Close'] > row['Resistance']:
        return 1
    return 0

def cup_and_handle_breakout(row):
    if row['Close'] > row['Resistance'] and row['Volume'] > row['Volume_20_SMA']:
        return 1
    return 0

# Define gaps & exhaustion pattern strategies (Bullish)

def breakaway_gap_bullish(row):
    if row['Open'] > row['High_prev'] and row['Close'] > row['Open']:
        return 1
    return 0

def runaway_gap_bullish(row):
    if row['Open'] > row['Close_prev'] and row['Close'] > row['Open']:
        return 1
    return 0

def island_reversal_bullish(row):
    if row['Low_prev'] > row['High_prev2'] and row['Low'] > row['High_prev']:
        return 1
    return 0

def exhaustion_gap_bullish(row):
    if row['Open'] > row['High_prev'] and row['Close'] < row['Open']:
        return 1
    return 0

# Define rare but high-probability pattern strategies (Bullish)

def bullish_counterattack_line(row):
    if row['Close_prev'] < row['Open_prev'] and row['Close'] == row['Open_prev']:
        return 1
    return 0

def ladder_bottom(row):
    if row['Close_prev4'] < row['Open_prev4'] and row['Close_prev3'] < row['Open_prev3'] and row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] > row['Open_prev'] and row['Close'] > row['Open']:
        return 1
    return 0

def unique_three_river_bottom(row):
    if row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] < row['Open_prev'] and row['Close'] > row['Open'] and row['Low_prev'] < row['Low_prev2'] and row['Low'] > row['Low_prev']:
        return 1
    return 0

def concealing_baby_swallow(row):
    if row['Close_prev2'] < row['Open_prev2'] and row['Close_prev'] < row['Open_prev'] and row['Open'] < row['Low_prev'] and row['Close'] > row['Open_prev']:
        return 1
    return 0

# Define confirmation pattern strategies (Bullish)

def hammer_rsi_oversold(row):
    if hammer(row) == 1 and row['RSI'] < 30:
        return 1
    return 0

def bullish_engulfing_macd_crossover(row):
    if bullish_engulfing(row) == 1 and macd_bullish_crossover(row) == 1:
        return 1
    return 0

def morning_star_volume_spike(row):
    if morning_star(row) == 1 and row['Volume'] > 2 * row['Volume_20_SMA']:
        return 1
    return 0

def three_white_soldiers_adx(row):
    if three_white_soldiers(row) == 1 and row['ADX'] > 25:
        return 1
    return 0