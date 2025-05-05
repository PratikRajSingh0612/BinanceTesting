import pandas as pd
import numpy as np
from binance.client import Client
from datetime import datetime, timedelta
import os
import pandas_ta as ta
from tqdm import tqdm
from ta.momentum import RSIIndicator
from ta.trend import MACD

import importlib
import BaseFunctions
importlib.reload(BaseFunctions)
from BaseFunctions import *


    
def allstrategiesv2(df, symbol, client, BackTime):
    # ===== 1. Basic Price Transformations =====
    df['Close_prev'] = df['Close'].shift(1)
    df['Open_prev'] = df['Open'].shift(1)
    df['High_prev'] = df['High'].shift(1)
    df['Low_prev'] = df['Low'].shift(1)
    df['Volume_prev'] = df['Volume'].shift(1)
    
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

    # Fetch daily candles
    candles_1d = get_candles_data(symbol, '4h', 1000, BackTime, client)
    candles_1d['RSI_1D'] = RSIIndicator(candles_1d['Close'], window=14).rsi()
    macd_1d = MACD(candles_1d['Close'], window_fast=12, window_slow=26, window_sign=9)
    candles_1d['MACD_1D'] = macd_1d.macd()
    candles_1d['MACD_Signal_1D'] = macd_1d.macd_signal()
    
    # Merge RSI values into original DataFrame (align by timestamp)
    df = df.merge(candles_4h[['Close Time', 'RSI_4H', 'MACD_4H', 'MACD_Signal_4H']], on='Close Time', how='left')
    df = df.merge(candles_1d[['Close Time', 'RSI_1D', 'MACD_1D', 'MACD_Signal_1D']], on='Close Time', how='left')
    
    # Forward-fill missing values and default to 0
    for col in ['RSI_4H', 'RSI_1D', 'MACD_4H', 'MACD_Signal_4H', 'MACD_1D', 'MACD_Signal_1D']:
        df[col] = df[col].fillna(method='ffill').fillna(0)

    # Forward-fill missing RSI values
    df['RSI_4H'] = df['RSI_4H'].fillna(method='ffill')
    df['RSI_1D'] = df['RSI_1D'].fillna(method='ffill')

    # MACD
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    df['MACD'] = macd['MACD_12_26_9']
    df['MACD_Signal'] = macd['MACDs_12_26_9']
    df['MACD_Histogram'] = macd['MACDh_12_26_9']
    df['MACD_prev'] = df['MACD'].shift(1)
    df['MACD_Signal_prev'] = df['MACD_Signal'].shift(1)
    df['MACD_Histogram_prev'] = df['MACD_Histogram'].shift(1)


    # Stochastic
    # Calculate StochRSI
    stochrsi_k, stochrsi_d = ta.stochrsi(
        df['Close'], 
        timeperiod=14,
        fastk_period=3,
        fastd_period=3,
        fastd_matype=0
    )
    
    df['StochRSI_K'] = stochrsi_k
    df['StochRSI_D'] = stochrsi_d
    df['StochRSI_prev'] = df['StochRSI_K'].shift(1)

    # ADX
    adx = ta.adx(df['High'], df['Low'], df['Close'])
    df['ADX'] = adx['ADX_14']
    df['DI+'] = adx['DMP_14']
    df['DI-'] = adx['DMN_14']

    # ===== 5. Support/Resistance =====
    df['Support'] = df['Low'].rolling(20).min()
    df['Resistance'] = df['High'].rolling(20).max()
    df['Swing_High'] = df['High'].rolling(50).max()
    df['Swing_Low'] = df['Low'].rolling(50).min()
    df['Support_Tests'] = df['Low'].rolling(20).apply(lambda x: (x == x.min()).sum())
    df['Is_Swing_High'] = (df['High'] == df['High'].rolling(window=5, center=True, min_periods=1).max())

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

    # ===== 8. Apply All 138 Strategies =====
    strategy_mapping = {
    # ===== 1. Moving Average Strategies (10) =====
    'GoldenCross': golden_cross,
    'PriceBounce200EMA': price_bounces_off_200_ema,
    'PriceCross20EMA': price_crosses_above_20_ema,
    'EMA10Cross50': ema_10_crosses_above_50,
    'PriceAboveAllMAs': price_above_all_major_mas,
    'MARibbonCompression': ma_ribbon_compression,
    'PriceRetestsMA': price_retests_ma,
    'DeathCrossSignal': death_cross_avoidance,
    'EMACrossoverHTF': ema_crossover_high_timeframe,
    'MASlopeUpward': ma_slope_turning_upward,
    
    # ===== 2. Volume Strategies (10) =====
    'VolumeSpikeGreen': volume_spike_green_candle,
    'VolumeBreakout20SMA': volume_breakout_above_20_day_avg,
    'BullishVolumeDivergence': bullish_price_volume_divergence,
    'ClimaxVolumeReversal': climax_volume_reversal,
    'VolumeSupportResistance': volume_supporting_resistance_breakout,
    'VolumeRisingConsolidation': volume_rising_price_consolidates,
    'AccumulationPattern': accumulation_volume_pattern,
    'LowVolumePullback': low_volume_pullback_after_high_volume_rally,
    'OBVRising': volume_rising_obv_rising,
    'VolumeSupportBreakout': volume_support_prior_breakout_zone,
    
    # # ===== 3. RSI Strategies (10) =====
    'RSIOversold': rsi_oversold,
    'RSIBullishDivergence': rsi_bullish_divergence,
    'RSICrossAbove30': rsi_crosses_above_30,
    'RSIHigherLows': rsi_higher_lows_vs_price_lower_lows,
    'RSIExitBearControl': rsi_moving_out_of_bear_control,
    'RSIAbove50': rsi_above_50,
    'RSITrendlineBreak': rsi_trendline_breakout,
    'RSIMultiTFConvergence': rsi_multi_timeframe_convergence,
    'RSIDoubleBottom': rsi_double_bottom,
    'RSIBullishStructure': rsi_holding_bullish_structure,
    
    # ===== 4. MACD Strategies (10) =====
    'MACDBullishCross': macd_bullish_crossover,
    'MACDHistGreen': macd_histogram_flips_green,
    'MACDBullishDivergence': macd_bullish_divergence,
    'MACDCrossAboveZero': macd_crossover_above_zero,
    'MACDZeroSupport': macd_support_at_zero,
    'MACDTrendUp': macd_trending_upward_with_price,
    'MACDMultiTFCross': macd_crossover_multiple_timeframes,
    'MACDHistExpanding': macd_histogram_expanding,
    'MACDFakeout': macd_fakeout_signal,
    'MACDWideSeparation': macd_wide_separation_forming,
    
    # ===== 5. Chart Pattern Strategies (10) =====
    'CupHandleBreakout': cup_and_handle_breakout,
    'InverseHnS': inverse_head_and_shoulders,
    'AscTriangleBreak': ascending_triangle_breakout,
    'FallingWedgeBreak': falling_wedge_breakout,
    'BullFlagBreakout': bull_flag_breakout,
    'SymTriangleBreak': symmetrical_triangle_breakout,
    'DoubleBottomConfirm': double_bottom_confirmation,
    'RoundedBottom': rounded_bottom_formation,
    'ConsolidationBreak': breakout_from_consolidation_range,
    'ExpTriangleBreak': expanding_triangle_breakout_upward,
    
    # ===== 6. Support/Resistance Strategies (10) =====
    'BounceLongTermSupport': bounce_from_long_term_support,
    'FlipResistToSupport': price_flips_resistance_into_support,
    'ConfluenceSupport': confluence_of_support,
    'MultiTestLevel': horizontal_level_tested_multiple_times,
    'PsychLevel': strong_psychological_level,
    # 'WeeklySupportDaily': support_from_weekly_respected_on_daily,
    'RejectBreakdown': price_rejects_breakdown_from_key_support,
    'RetestBreakoutZone': price_retests_prior_breakout_zone,
    'SupportWithDivergence': support_with_bullish_divergence,
    'FibConfluenceSupport': support_aligning_with_fib_retracement,
    
    # ===== 7. Fibonacci Strategies (10) =====
    'Buy618Fib': buy_at_618_fib_retracement,
    'Buy382Bounce': buy_near_382_bounce,
    'FibSupportConfluence': confluence_of_fib_and_support,
    'Reclaim50Fib': buy_when_price_reclaims_05_fib,
    'FibExtensionTarget': fib_extension_targets_reached,
    'FibFanBounce': fib_fan_support_bounce,
    'FibClusterConfirm': fib_cluster_zone_with_indicators,
    'FibConsolidation': price_consolidates_between_fib_levels,
    'FibParabolicRetrace': fib_retracement_after_parabolic_move,
    'GoldenPocket': buy_at_golden_pocket_zone,
    
    # ===== 8. Other Indicator Strategies (10) =====
    'StochRSIOversold': stochastic_rsi_oversold_cross_up,
    'ADXBullishDI': adx_bullish_di,
    'CCIAboveNeg100': cci_crossing_above_minus_100,
    'WilliamsROversold': williams_r_coming_out_of_oversold,
    'AroonUpCross': aroon_up_crosses_above_aroon_down,
    'ChaikinPositive': chaikin_money_flow_turns_positive,
    'KlingerBullCross': klinger_volume_oscillator_bullish_cross,
    'TrixZeroCross': trix_indicator_crosses_zero,
    'MFIRising': money_flow_index_rising,
    'DMIPlusCross': dmi_plus_crosses_above_dmi_minus,
    
    # ===== 9. Trend/Structure Strategies (10) =====
    'HHHL': higher_high_higher_low,
    'BreakBearTrendline': break_of_bearish_trendline,
    'RetestTrendline': retest_of_broken_trendline,
    'TrendContinuation': trend_continuation_after_consolidation,
    'MultiTFStructure': bullish_market_structure_multiple_timeframes,
    'BreakSwingHigh': break_of_previous_swing_high,
    'ChannelBounce': rising_channel_midline_support_bounce,
    'ReclaimRangeHigh': price_reclaiming_previous_range_high,
    'SupportAboveResist': support_formed_above_previous_resistance,
    'LowVolBreakout': low_volatility_range_breakout,
    
    # ===== 10. Candlestick Patterns (48) =====
    'Hammer': hammer,
    'InverseHammer': inverse_hammer,
    'BullMarubozu': bullish_marubozu,
    'DragonflyDoji': dragonfly_doji,
    'SpinningTopBull': spinning_top_bullish,
    'HangingManBull': hanging_man_in_uptrend,
    'BullishEngulfing': bullish_engulfing,
    'PiercingLine': piercing_line,
    'TweezerBottom': tweezer_bottom,
    'KickingBullish': kicking_bullish,
    'MatchingLow': matching_low,
    'OnNeckBullish': on_neck_line_bullish,
    'MorningStar': morning_star,
    'MorningDojiStar': morning_doji_star,
    'AbandonedBabyBull': abandoned_baby_bullish,
    'ThreeInsideUp': three_inside_up,
    'ThreeOutsideUp': three_outside_up,
    'StickSandwichBull': stick_sandwich_bullish,
    'TriStarBullish': tri_star_bullish,
    'ThreeWhiteSoldiers': three_white_soldiers,
    'RisingThreeMethods': rising_three_methods,
    'SeparatingLinesBull': separating_lines_bullish,
    'MatHold': mat_hold,
    'SideWhiteLines': side_by_side_white_lines,
    'UpsideTasukiGap': upside_tasuki_gap,
    'BullFlagPattern': bullish_flag_breakout,
    'BullPennantBreak': bullish_pennant_breakout,
    'FallingWedgeBull': falling_wedge_breakout,
    'SymTriangleBull': symmetrical_triangle_breakout_upward,
    'AscTriangleBull': ascending_triangle_breakout,
    'DoubleBottomBull': double_bottom_breakout,
    'TripleBottomBull': triple_bottom_breakout,
    'RoundingBottomBull': rounding_bottom_breakout,
    'CupHandleBull': cup_and_handle_breakout,
    'BreakawayGapBull': breakaway_gap_bullish,
    'RunawayGapBull': runaway_gap_bullish,
    'IslandReversalBull': island_reversal_bullish,
    'BullCounterattack': bullish_counterattack_line,
    'LadderBottom': ladder_bottom,
    'ThreeRiverBottom': unique_three_river_bottom,
    'ConcealingSwallow': concealing_baby_swallow,
    'HammerRSI': hammer_rsi_oversold,
    'EngulfMACD': bullish_engulfing_macd_crossover,
    'MorningStarVolume': morning_star_volume_spike,
    'SoldiersADX': three_white_soldiers_adx
    }

    for strategy_name, strategy_func in strategy_mapping.items():
        df[strategy_name] = df.apply(strategy_func, axis=1)

    # ===== 9. Composite Signals =====
    df['BullishConfidenceScore'] = (
        df['GoldenCross'] * 0.15 +
        df['BullishEngulfing'] * 0.2 +
        df['RSIOversold'] * 0.1 +
        df['MACDBullishCrossover'] * 0.15 +
        df['VolumeSpikeGreen'] * 0.1 +
        df['PriceAboveAllMAs'] * 0.1 +
        df['HigherHighHigherLow'] * 0.2
    )

    strategy_names = list(strategy_mapping.keys())
    df['ScoreAll'] = df[strategy_names].fillna(0).sum(axis=1)

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
    if row['MA_50'] > row['MA_200'] and row['MA_50_prev'] <= row['MA_200_prev']:
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
    if row['High'] - row['Low'] < 0.05 * row['Close'] and row['Volume'] > row['Volume_prev']:
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
    """
    Row-by-row version for your strategy mapping
    """
    support_zone = max(row['High_prev'], row['High_prev2'], row['High_prev3'])
    volume_avg = (row['Volume_prev'] + row['Volume_prev2'] + row['Volume_prev3']) / 3
    rejection = ((row['Close'] - row['Low']) / (row['High'] - row['Low']) > 0.67)
    
    conditions = (
        (row['Close'] <= support_zone * 1.01) and
        (row['Close'] >= support_zone * 0.99) and
        (row['Volume'] > 1.5 * volume_avg) and
        rejection and
        (row['Close'] > row['Open'])
    )
    return 1 if conditions else 0

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
    # Ensure required columns exist and are not NaN
    required_cols = ['RSI', 'RSI_prev', 'RSI_prev2']
    if any(col not in row or pd.isna(row[col]) for col in required_cols):
        return 0
    
    # Simplified trendline breakout logic
    if (row['RSI_prev2'] > row['RSI_prev'] and  # Declining RSI trend
        row['RSI'] < 50 and  # Below midpoint
        row['RSI'] > row['RSI_prev']):  # RSI rising (breakout)
        # Approximate trendline
        avg_decline = (row['RSI_prev2'] - row['RSI_prev'])
        trendline_value = row['RSI_prev'] - avg_decline
        return 1 if row['RSI'] > trendline_value else 0
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
    if (pd.notna(row['MACD']) and pd.notna(row['MACD_prev']) and 
            pd.notna(row['Close']) and pd.notna(row['Close_prev']) and 
            row['MACD'] > row['MACD_prev'] and row['Close'] < row['Close_prev']):
            return 1
    return 0

def macd_histogram_flips_green(row):
    if (pd.notna(row['MACD']) and pd.notna(row['MACD_Signal']) and 
            pd.notna(row['MACD_prev']) and pd.notna(row['MACD_Signal_prev']) and 
            row['MACD'] > row['MACD_Signal'] and row['MACD_prev'] <= row['MACD_Signal_prev'] and 
            row['MACD'] > 0 and row['MACD_Signal'] > 0):
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
    if (pd.notna(row['MACD_Histogram']) and pd.notna(row['MACD_Histogram_prev']) and 
            row['MACD_Histogram'] > 0 and 
            abs(row['MACD_Histogram']) > abs(row['MACD_Histogram_prev'])):
            return 1
    return 0

def macd_fakeout_signal(row):
    if (pd.notna(row['MACD']) and pd.notna(row['MACD_Signal']) and 
            pd.notna(row['MACD_prev']) and pd.notna(row['MACD_Signal_prev']) and 
            pd.notna(row['Close']) and pd.notna(row['Close_prev']) and 
            row['MACD'] < row['MACD_Signal'] and row['MACD_prev'] > row['MACD_Signal_prev'] and 
            row['Close'] > row['Close_prev']):  # Bullish price confirmation
            return 1
    return 0

def macd_wide_separation_forming(row):
    if (pd.notna(row['MACD']) and pd.notna(row['MACD_Signal']) and 
            pd.notna(row['MACD_prev']) and pd.notna(row['MACD_Signal_prev']) and 
            row['MACD'] > row['MACD_Signal'] and 
            abs(row['MACD'] - row['MACD_Signal']) > abs(row['MACD_prev'] - row['MACD_Signal_prev'])):
            return 1
    return 0


######################## Define chart pattern-based strategy functions #######################

def cup_and_handle_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Volume']) and pd.notna(row['Volume_20_SMA']) and 
            row['Close'] > row['Resistance'] and row['Volume'] > row['Volume_20_SMA']):
            return 1
    return 0

def inverse_head_and_shoulders(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev']) and pd.notna(row['Low_prev2']) and 
            pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            row['Low'] < row['Low_prev'] and row['Low_prev2'] > row['Low_prev'] and 
            row['Close'] > row['Resistance']):  # Breakout confirmation
            return 1
    return 0

def ascending_triangle_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            row['Close'] > row['Resistance'] and row['Low'] > row['Low_prev']):
            return 1
    return 0

def falling_wedge_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['High']) and pd.notna(row['High_prev']) and 
            row['Close'] > row['Resistance'] and row['High'] > row['High_prev']):  # Higher high for breakout
            return 1
    return 0

def bull_flag_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['High']) and pd.notna(row['High_prev']) and 
            row['Close'] > row['Resistance'] and row['High'] > row['High_prev']):
            return 1
    return 0

def symmetrical_triangle_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['High']) and pd.notna(row['Low']) and 
            row['High'] != row['Low'] and  # Avoid invalid range
            row['Close'] > row['Resistance'] and 
            abs(row['High'] - row['Low']) < 0.01 * row['Close']):
            return 1
    return 0

def double_bottom_confirmation(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev']) and pd.notna(row['Low_prev2']) and 
            pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            abs(row['Low'] - row['Low_prev2']) < 0.01 * row['Low'] and  # Similar lows
            row['Low_prev'] > row['Low'] and  # Trough between
            row['Close'] > row['Resistance']):
            return 1
    return 0

def rounded_bottom_formation(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev']) and pd.notna(row['Low_prev2']) and 
            row['Low'] > row['Low_prev'] and row['Low_prev'] > row['Low_prev2']):
            return 1
    return 0

def breakout_from_consolidation_range(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['High']) and pd.notna(row['Low']) and 
            row['High'] != row['Low'] and 
            row['Close'] > row['Resistance'] and 
            abs(row['High'] - row['Low']) < 0.01 * row['Close']):
            return 1
    return 0

def expanding_triangle_breakout_upward(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['High']) and pd.notna(row['High_prev']) and 
            pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            row['Close'] > row['Resistance'] and row['High'] > row['High_prev'] and 
            row['Low'] < row['Low_prev']):
            return 1
    return 0


######################## Define support/resistance-based strategy functions #######################
def bounce_from_long_term_support(row):
    if (pd.notna(row['Close']) and pd.notna(row['Support']) and 
            row['Support'] != 0 and 
            abs(row['Close'] - row['Support']) / row['Support'] < 0.01 and 
            row['Close'] > row['Support']):  # Confirm price above support
            return 1
    return 0

def price_flips_resistance_into_support(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Low']) and 
            row['Close'] > row['Resistance'] and 
            row['Low'] <= row['Resistance'] and 
            abs(row['Low'] - row['Resistance']) / row['Resistance'] < 0.01):  # Retest within 1%
            return 1
    return 0

def confluence_of_support(row):
    if (pd.notna(row['Close']) and pd.notna(row['Support']) and 
            pd.notna(row['EMA_50']) and 
            row['Support'] != 0 and 
            abs(row['Close'] - row['Support']) / row['Support'] < 0.01 and 
            row['Close'] > row['Support'] and row['Close'] > row['EMA_50']):
            return 1
    return 0

def horizontal_level_tested_multiple_times(row):
    if (pd.notna(row['Support_Tests']) and pd.notna(row['Close']) and 
            pd.notna(row['Support']) and 
            row['Support_Tests'] >= 3 and 
            abs(row['Close'] - row['Support']) / row['Support'] < 0.01 and  # Price near support
            row['Close'] > row['Support']):  # Bullish stance
            return 1
    return 0

def strong_psychological_level(row):
    if (pd.notna(row['Close']) and pd.notna(row['Support']) and row['Close'] > 0):
        nearest_10 = round(row['Close'] / 10) * 10
        nearest_100 = round(row['Close'] / 100) * 100
        
        if (
            (abs(row['Close'] - nearest_10) / row['Close'] < 0.01 or 
             abs(row['Close'] - nearest_100) / row['Close'] < 0.01)
            and (row['Close'] > row['Support'])
        ):
            return 1
    return 0

def support_from_weekly_respected_on_daily(row):
    if (pd.notna(row['Close']) and pd.notna(row['Weekly_Support']) and 
            row['Weekly_Support'] != 0 and 
            abs(row['Close'] - row['Weekly_Support']) / row['Weekly_Support'] < 0.01 and 
            row['Close'] > row['Weekly_Support']):
            return 1
    return 0

def price_rejects_breakdown_from_key_support(row):
    if (pd.notna(row['Low']) and pd.notna(row['Support']) and 
            pd.notna(row['Close']) and 
            row['Low'] < row['Support'] and row['Close'] > row['Support']):
            return 1
    return 0

def price_retests_prior_breakout_zone(row):
    if (pd.notna(row['Close']) and pd.notna(row['Breakout_Occurred']) and 
            row['Breakout_Occurred'] != 0 and 
            abs(row['Close'] - row['Breakout_Occurred']) / row['Breakout_Occurred'] < 0.01 and 
            row['Close'] > row['Breakout_Occurred']):
            return 1
    return 0

def support_with_bullish_divergence(row):
    if (pd.notna(row['Close']) and pd.notna(row['Support']) and 
            pd.notna(row['RSI']) and pd.notna(row['RSI_prev']) and 
            pd.notna(row['MACD']) and pd.notna(row['MACD_prev']) and 
            row['Close'] > row['Support'] and 
            row['RSI'] > row['RSI_prev'] and row['MACD'] > row['MACD_prev']):
            return 1
    return 0

def support_aligning_with_fib_retracement(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_Support']) and 
            row['Fib_Support'] != 0 and 
            abs(row['Close'] - row['Fib_Support']) / row['Fib_Support'] < 0.01 and 
            row['Close'] > row['Fib_Support']):
            return 1
    return 0

######################## Define Fibonacci-based strategy functions #######################
def buy_at_618_fib_retracement(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_618']) and 
            row['Fib_618'] != 0 and 
            abs(row['Close'] - row['Fib_618']) / row['Fib_618'] < 0.01 and 
            row['Close'] > row['Fib_618']):  # Bullish confirmation
            return 1
    return 0

def buy_near_382_bounce(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_382']) and 
            row['Fib_382'] != 0 and 
            abs(row['Close'] - row['Fib_382']) / row['Fib_382'] < 0.01 and 
            row['Close'] > row['Fib_382']):
            return 1
    return 0

def confluence_of_fib_and_support(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_618']) and pd.notna(row['Support']) and 
            row['Fib_618'] != 0 and row['Support'] != 0 and 
            abs(row['Close'] - row['Fib_618']) / row['Fib_618'] < 0.01 and 
            abs(row['Close'] - row['Support']) / row['Support'] < 0.01 and 
            row['Close'] > row['Fib_618'] and row['Close'] > row['Support']):
            return 1
    return 0

def buy_when_price_reclaims_05_fib(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_50']) and pd.notna(row['Close_prev']) and 
            row['Close'] > row['Fib_50'] and row['Close_prev'] <= row['Fib_50']):
            return 1
    return 0

def fib_extension_targets_reached(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_Extension_1']) and 
            pd.notna(row['Fib_Extension_2']) and 
            row['Close'] > row['Fib_Extension_1'] and row['Close'] < row['Fib_Extension_2'] and 
            row['Close'] > row['Close_prev']):  # Bullish confirmation
            return 1
    return 0

def fib_fan_support_bounce(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_Fan_Support']) and 
            row['Fib_Fan_Support'] != 0 and 
            abs(row['Close'] - row['Fib_Fan_Support']) / row['Fib_Fan_Support'] < 0.01 and 
            row['Close'] > row['Fib_Fan_Support']):
            return 1
    return 0

def fib_cluster_zone_with_indicators(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_Cluster']) and 
            pd.notna(row['RSI']) and pd.notna(row['MACD']) and pd.notna(row['MACD_Signal']) and 
            row['Fib_Cluster'] != 0 and 
            abs(row['Close'] - row['Fib_Cluster']) / row['Fib_Cluster'] < 0.01 and 
            row['Close'] > row['Fib_Cluster'] and 
            row['RSI'] > 50 and row['MACD'] > row['MACD_Signal']):
            return 1
    return 0

def price_consolidates_between_fib_levels(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_382']) and pd.notna(row['Fib_618']) and 
            pd.notna(row['High']) and pd.notna(row['Low']) and 
            row['Close'] != 0 and row['High'] != row['Low'] and 
            row['Close'] > row['Fib_382'] and row['Close'] < row['Fib_618'] and 
            abs(row['High'] - row['Low']) < 0.01 * row['Close'] and 
            row['Close'] > row['Close_prev']):  # Bullish confirmation
            return 1
    return 0

def fib_retracement_after_parabolic_move(row):
    if (pd.notna(row['Close']) and pd.notna(row['Fib_618']) and 
            pd.notna(row['Volume']) and pd.notna(row['Volume_20_SMA']) and 
            row['Fib_618'] != 0 and 
            abs(row['Close'] - row['Fib_618']) / row['Fib_618'] < 0.01 and 
            row['Close'] > row['Fib_618'] and 
            row['Volume'] > row['Volume_20_SMA']):
            return 1
    return 0

def buy_at_golden_pocket_zone(row, 
                            rsi_threshold=40, 
                            volume_multiplier=1.2, 
                            candle_confirmation=True):
    """
    Enhanced Golden Pocket Zone strategy with:
    - Price between 61.8% and 65% Fib levels (Golden Pocket)
    - Bullish price momentum
    - RSI confirmation
    - Volume confirmation
    - Optional candle pattern confirmation
    
    Parameters:
        rsi_threshold: Minimum RSI value (default 40)
        volume_multiplier: Volume spike requirement (default 1.2x avg)
        candle_confirmation: Require bullish candle (default True)
    """
    # Check required values exist
    required_cols = ['Close', 'Fib_618', 'Fib_65', 'Close_prev', 'RSI', 'Volume', 'Volume_MA']
    if not all(col in row and pd.notna(row[col]) for col in required_cols):
        return 0
    
    # Golden Pocket Zone condition
    in_golden_pocket = (row['Fib_618'] <= row['Close'] <= row['Fib_65'])
    
    # Momentum confirmation
    price_momentum = (row['Close'] > row['Close_prev'])
    
    # Indicator confirmations
    rsi_confirm = (row['RSI'] > rsi_threshold)
    volume_confirm = (row['Volume'] > volume_multiplier * row['Volume_MA'])
    
    # Optional candle confirmation
    candle_confirm = (not candle_confirmation) or (row['Close'] > row['Open'])
    
    # MACD bullish crossover (optional)
    macd_confirm = ('MACD' not in row) or ('MACD_Signal' not in row) or (row['MACD'] > row['MACD_Signal'])
    
    return 1 if (in_golden_pocket and price_momentum and rsi_confirm 
                and volume_confirm and candle_confirm and macd_confirm) else 0

######################## Define indicators & oscillators-based strategy functions #######################
def stochastic_rsi_oversold_cross_up(row):
    """
    Identifies when the Stochastic RSI crosses above the oversold threshold (typically 20)
    after being in oversold territory, suggesting a potential bullish reversal.
    
    Rules:
    1. StochRSI %K must be below oversold threshold (default: 20)
    2. StochRSI %K crosses above %D line
    3. Current %K crosses above oversold threshold
    4. Confirmation: RSI > 30 (optional but recommended)
    
    Returns: 1 when conditions are met, 0 otherwise
    """
    # Default parameters
    oversold_threshold = 20
    rsi_confirmation = 30
    
    current_k = float(row.get('StochRSI_K', 50))
    current_d = float(row.get('StochRSI_D', 50))
    prev_k = float(row.get('StochRSI_prev', 50))  # Make sure you have this column from shift(1)
    
    # Check if we have valid numerical values
    if None in (prev_k, current_k, current_d):
        return 0
    
    # Current RSI for confirmation (optional)
    current_rsi = float(row.get('RSI', 100))  # Default to 100 if RSI not calculated
    
    # Strategy conditions
    condition1 = prev_k < oversold_threshold  # Was in oversold
    condition2 = current_k > current_d        # K crosses above D
    condition3 = current_k > oversold_threshold  # K crosses above oversold line
    condition4 = current_rsi > rsi_confirmation  # Additional RSI confirmation
    
    return 1 if (condition1 and condition2 and condition3 and condition4) else 0

def adx_bullish_di(row):
    if (pd.notna(row['ADX']) and pd.notna(row['DI+']) and pd.notna(row['DI-']) and 
            row['ADX'] > 25 and row['DI+'] > row['DI-']):
            return 1
    return 0

def cci_crossing_above_minus_100(row):
    if (pd.notna(row['CCI']) and pd.notna(row['CCI_prev']) and 
            row['CCI'] > -100 and row['CCI_prev'] <= -100):
            return 1
    return 0

def williams_r_coming_out_of_oversold(row):
    if (pd.notna(row['WilliamsR']) and pd.notna(row['WilliamsR_prev']) and 
            row['WilliamsR'] > -80 and row['WilliamsR_prev'] <= -80):
            return 1
    return 0

def aroon_up_crosses_above_aroon_down(row):
    if (pd.notna(row['AroonUp']) and pd.notna(row['AroonDown']) and 
            pd.notna(row['AroonUp_prev']) and pd.notna(row['AroonDown_prev']) and 
            row['AroonUp'] > row['AroonDown'] and row['AroonUp_prev'] <= row['AroonDown_prev']):
            return 1
    return 0

def chaikin_money_flow_turns_positive(row):
    if (pd.notna(row['CMF']) and pd.notna(row['CMF_prev']) and 
            row['CMF'] > 0 and row['CMF_prev'] <= 0):
            return 1
    return 0

def klinger_volume_oscillator_bullish_cross(row):
    if (pd.notna(row['KVO']) and pd.notna(row['KVO_Signal']) and 
            pd.notna(row['KVO_prev']) and pd.notna(row['KVO_Signal_prev']) and 
            row['KVO'] > row['KVO_Signal'] and row['KVO_prev'] <= row['KVO_Signal_prev']):
            return 1
    return 0

def trix_indicator_crosses_zero(row):
    if (pd.notna(row['Trix']) and pd.notna(row['Trix_prev']) and 
            row['Trix'] > 0 and row['Trix_prev'] <= 0):
            return 1
    return 0

def money_flow_index_rising(row):
    if (pd.notna(row['MFI']) and pd.notna(row['MFI_prev']) and 
            row['MFI'] > 20 and row['MFI_prev'] <= 20):  # Cross above oversold
            return 1
    return 0

def dmi_plus_crosses_above_dmi_minus(row):
    if (pd.notna(row['DMI+']) and pd.notna(row['DMI-']) and 
            pd.notna(row['DMI+_prev']) and pd.notna(row['DMI-_prev']) and 
            row['DMI+'] > row['DMI-'] and row['DMI+_prev'] <= row['DMI-_prev']):
            return 1
    return 0

######################## Define trend & structure-based strategy functions #######################
def higher_high_higher_low(row):
    if (pd.notna(row['High']) and pd.notna(row['High_prev']) and 
            pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            row['High'] > row['High_prev'] and row['Low'] > row['Low_prev']):
            return 1
    return 0

def break_of_bearish_trendline(row):
    if (pd.notna(row['Close']) and pd.notna(row['Trendline']) and 
            pd.notna(row['Close_prev']) and 
            row['Close'] > row['Trendline'] and row['Close_prev'] <= row['Trendline']):
            return 1
    return 0

def retest_of_broken_trendline(row):
    if (pd.notna(row['Close']) and pd.notna(row['Trendline']) and 
            pd.notna(row['Low']) and row['Trendline'] != 0 and 
            row['Close'] > row['Trendline'] and 
            row['Low'] <= row['Trendline'] and 
            abs(row['Low'] - row['Trendline']) / row['Trendline'] < 0.01):  # Retest within 1%
            return 1
    return 0

def trend_continuation_after_consolidation(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['High']) and pd.notna(row['Low']) and 
            row['Close'] != 0 and row['High'] != row['Low'] and 
            row['Close'] > row['Resistance'] and 
            row['Close'] > row['Close_prev']):  # Bullish confirmation
            return 1
    return 0

def bullish_market_structure_multiple_timeframes(row):
    if (pd.notna(row['High_4H']) and pd.notna(row['High_4H_prev']) and 
            pd.notna(row['Low_4H']) and pd.notna(row['Low_4H_prev']) and 
            pd.notna(row['High_1D']) and pd.notna(row['High_1D_prev']) and 
            pd.notna(row['Low_1D']) and pd.notna(row['Low_1D_prev']) and 
            row['High_4H'] > row['High_4H_prev'] and row['Low_4H'] > row['Low_4H_prev'] and 
            row['High_1D'] > row['High_1D_prev'] and row['Low_1D'] > row['Low_1D_prev']):
            return 1
    return 0

def break_of_previous_swing_high(row):
    if (pd.notna(row['High']) and pd.notna(row['Swing_High']) and 
            pd.notna(row['Close']) and pd.notna(row['Close_prev']) and 
            row['High'] > row['Swing_High'] and row['Close'] > row['Close_prev']):
            return 1
    return 0

def rising_channel_midline_support_bounce(row):
    if (pd.notna(row['Close']) and pd.notna(row['Channel_Midline']) and 
            row['Channel_Midline'] != 0 and 
            abs(row['Close'] - row['Channel_Midline']) / row['Channel_Midline'] < 0.01 and 
            row['Close'] > row['Channel_Midline']):
            return 1
    return 0

def price_reclaiming_previous_range_high(row):
    if (pd.notna(row['Close']) and pd.notna(row['Range_High']) and 
            pd.notna(row['Close_prev']) and 
            row['Close'] > row['Range_High'] and row['Close_prev'] <= row['Range_High']):
            return 1
    return 0

def support_formed_above_previous_resistance(row):
    if (pd.notna(row['Support']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Close']) and row['Support'] != 0 and 
            row['Support'] > row['Resistance'] and 
            abs(row['Close'] - row['Support']) / row['Support'] < 0.01 and  # Price near support
            row['Close'] > row['Support']):
            return 1
    return 0

def low_volatility_range_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['High']) and pd.notna(row['Low']) and 
            row['Close'] != 0 and row['High'] != row['Low'] and 
            row['Close'] > row['Resistance'] and 
            row['Close'] > row['Close_prev']):  # Bullish confirmation
            return 1
    return 0


########################### Candle Stick Pattern Strategies #########################
# 1. Single-Candle Patterns (Bullish Reversal)
def hammer(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and pd.notna(row['Low']) and
            row['Close'] > row['Open'] and 
            row['Low'] < row['Open'] - 2 * abs(row['Close'] - row['Open'])):
            return 1
    return 0

def inverse_hammer(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and pd.notna(row['High']) and
            row['Close'] > row['Open'] and 
            row['High'] > row['Close'] + 2 * abs(row['Close'] - row['Open'])):
            return 1
    return 0

def bullish_marubozu(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['High']) and pd.notna(row['Low']) and
            row['Close'] != 0 and
            row['Close'] > row['Open'] and 
            abs(row['High'] - row['Close']) < 0.005 * row['Close'] and  # Relaxed to 0.5%
            abs(row['Open'] - row['Low']) < 0.005 * row['Close']):
            return 1
    return 0

def dragonfly_doji(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['High']) and pd.notna(row['Low']) and
            row['Close'] != 0 and
            abs(row['Close'] - row['Open']) < 0.005 * row['Close'] and  # Small body
            abs(row['High'] - max(row['Close'], row['Open'])) < 0.005 * row['Close'] and  # No upper shadow
            (row['Open'] - row['Low']) > 2 * abs(row['Close'] - row['Open'])):  # Long lower shadow
            return 1
    return 0

def spinning_top_bullish(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['High']) and pd.notna(row['Low']) and
            row['High'] != row['Low'] and
            abs(row['Close'] - row['Open']) < 0.3 * abs(row['High'] - row['Low']) and 
            row['Close'] > row['Open']):
            return 1
    return 0

def hanging_man_in_uptrend(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Low']) and pd.notna(row['Close_prev']) and 
            pd.notna(row['Open_prev']) and
            row['Close'] > row['Open'] and 
            row['Low'] < row['Open'] - 2 * abs(row['Close'] - row['Open']) and
            row['Close_prev'] > row['Open_prev']):  # Confirm prior uptrend
            return 1
    return 0

# Define two-candle pattern strategies (Bullish Reversal)

def bullish_engulfing(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and
            row['Close'] > row['Open'] and row['Close_prev'] < row['Open_prev'] and 
            row['Close'] > row['Open_prev'] and row['Open'] < row['Close_prev']):
            return 1
    return 0

def piercing_line(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Low_prev']) and
            row['Close_prev'] < row['Open_prev'] and 
            row['Open'] < row['Low_prev'] and  # Open below prior low
            row['Close'] > (row['Open_prev'] + row['Close_prev']) / 2):
            return 1
    return 0

def tweezer_bottom(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and
            row['Low'] != 0 and
            abs(row['Low'] - row['Low_prev']) / row['Low'] < 0.001 and  # Similar lows
            row['Close_prev'] < row['Open_prev'] and  # Bearish first candle
            row['Close'] > row['Open']):  # Bullish second candle
            return 1
    return 0

def kicking_bullish(row):
    if (pd.notna(row['Open']) and pd.notna(row['Close']) and 
            pd.notna(row['Open_prev']) and pd.notna(row['Close_prev']) and
            row['Close_prev'] < row['Open_prev'] and  # Bearish first candle
            row['Close'] > row['Open'] and  # Bullish second candle
            row['Open'] > row['Close_prev']):  # Gap up
            return 1
    return 0

def matching_low(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and
            row['Low'] != 0 and
            abs(row['Low'] - row['Low_prev']) / row['Low'] < 0.001 and  # Similar lows
            row['Close_prev'] < row['Open_prev'] and  # Bearish first candle
            row['Close'] > row['Close_prev']):  # Bullish reversal
            return 1
    return 0

def on_neck_line_bullish(row):
    if (pd.notna(row['Close']) and pd.notna(row['Low']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and
            row['Close_prev'] != 0 and
            row['Close_prev'] < row['Open_prev'] and  # Bearish first candle
            row['Close'] > row['Open'] and  # Bullish second candle
            abs(row['Close'] - row['Close_prev']) / row['Close_prev'] < 0.001):  # Close near prior close
            return 1
    return 0

# Define three-candle pattern strategies (Bullish Reversal)

def morning_star(row):
    if (pd.notna(row['Close']) and pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish first candle
            abs(row['Close_prev'] - row['Open_prev']) < 0.005 * row['Close_prev'] and  # Small second candle
            row['Close'] > row['Open'] and  # Bullish third candle
            row['Close'] > (row['Open_prev2'] + row['Close_prev2']) / 2):  # Above midpoint of first candle
            return 1
    return 0

def morning_doji_star(row):
    if (pd.notna(row['Close']) and pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            row['Close_prev'] != 0 and
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish first candle
            abs(row['Close_prev'] - row['Open_prev']) < 0.005 * row['Close_prev'] and  # Doji second candle
            row['Close'] > row['Open'] and  # Bullish third candle
            row['Close'] > (row['Open_prev2'] + row['Close_prev2']) / 2):  # Above midpoint of first candle
            return 1
    return 0

def abandoned_baby_bullish(row):
    if (pd.notna(row['Close']) and pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Low_prev']) and pd.notna(row['High_prev2']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            pd.notna(row['Low']) and pd.notna(row['High_prev']) and
            row['Close_prev'] != 0 and
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish first candle
            abs(row['Close_prev'] - row['Open_prev']) < 0.005 * row['Close_prev'] and  # Doji second candle
            row['Low_prev'] > row['High_prev2'] and  # Gap between first and second
            row['Low'] > row['High_prev'] and  # Gap between second and third
            row['Close'] > row['Open'] and  # Bullish third candle
            row['Close'] > (row['Open_prev2'] + row['Close_prev2']) / 2):  # Above midpoint
            return 1
    return 0

def three_inside_up(row):
    if (pd.notna(row['Close']) and pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish first candle
            row['Close_prev'] > row['Open_prev'] and  # Bullish second candle
            row['Open_prev'] > row['Close_prev2'] and row['Close_prev'] < row['Open_prev2'] and  # Harami
            row['Close'] > row['Open'] and row['Close'] > row['Close_prev']):  # Bullish third candle
            return 1
    return 0

def three_outside_up(row):
    if (pd.notna(row['Close']) and pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish first candle
            row['Close_prev'] > row['Open_prev'] and  # Bullish second candle
            row['Open_prev'] < row['Close_prev2'] and row['Close_prev'] > row['Open_prev2'] and  # Engulfing
            row['Close'] > row['Open'] and row['Close'] > row['Close_prev']):  # Bullish third candle
            return 1
    return 0

def stick_sandwich_bullish(row):
    if (pd.notna(row['Close']) and pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            pd.notna(row['Open']) and row['Close_prev2'] != 0 and
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish first candle
            row['Close_prev'] > row['Open_prev'] and  # Bullish second candle
            row['Open_prev'] > row['Close_prev2'] and  # Gap up
            row['Close'] < row['Open'] and  # Bearish third candle
            abs(row['Close'] - row['Close_prev2']) / row['Close_prev2'] < 0.001):  # Close near first candle
            return 1
    return 0

def tri_star_bullish(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            row['Close'] != 0 and row['Close_prev'] != 0 and row['Close_prev2'] != 0 and
            abs(row['Close_prev2'] - row['Open_prev2']) < 0.005 * row['Close_prev2'] and  # First doji
            abs(row['Close_prev'] - row['Open_prev']) < 0.005 * row['Close_prev'] and  # Second doji
            abs(row['Close'] - row['Open']) < 0.005 * row['Close'] and  # Third doji
            row['Close_prev'] < row['Close_prev2'] and  # Second doji lower
            row['Close'] > row['Close_prev']):  # Third doji higher (bullish)
            return 1
    return 0

# Define multi-candle pattern strategies (Bullish Continuation/Reversal)

def three_white_soldiers(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            pd.notna(row['High']) and pd.notna(row['High_prev']) and pd.notna(row['High_prev2']) and
            row['Close_prev2'] > row['Open_prev2'] and  # Bullish first candle
            row['Close_prev'] > row['Open_prev'] and  # Bullish second candle
            row['Close'] > row['Open'] and  # Bullish third candle
            row['Close'] > row['Close_prev'] > row['Close_prev2'] and  # Increasing closes
            abs(row['High_prev2'] - row['Close_prev2']) < 0.005 * row['Close_prev2'] and  # Small upper shadows
            abs(row['High_prev'] - row['Close_prev']) < 0.005 * row['Close_prev'] and
            abs(row['High'] - row['Close']) < 0.005 * row['Close']):
            return 1
    return 0

def rising_three_methods(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and 
            pd.notna(row['Close_prev3']) and pd.notna(row['Open_prev3']) and 
            pd.notna(row['Close_prev4']) and pd.notna(row['Open_prev4']) and
            pd.notna(row['Low_prev4']) and
            row['Close_prev4'] > row['Open_prev4'] and  # Bullish first candle
            row['Close_prev3'] < row['Open_prev3'] and  # Bearish second candle
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish third candle
            row['Close_prev'] < row['Open_prev'] and  # Bearish fourth candle
            row['Close_prev3'] > row['Low_prev4'] and  # Within first candle’s range
            row['Close_prev2'] > row['Low_prev4'] and
            row['Close_prev'] > row['Low_prev4'] and
            row['Close'] > row['Open'] and row['Close'] > row['Close_prev4']):  # Bullish fifth candle
            return 1
    return 0

def separating_lines_bullish(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and 
            pd.notna(row['Open_prev']) and
            row['Open_prev2'] != 0 and
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish first candle
            abs(row['Open_prev'] - row['Open_prev2']) / row['Open_prev2'] < 0.001 and  # Similar opens
            row['Close'] > row['Open']):  # Bullish second candle
            return 1
    return 0

def mat_hold(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and 
            pd.notna(row['Close_prev3']) and pd.notna(row['Open_prev3']) and 
            pd.notna(row['Close_prev4']) and pd.notna(row['Open_prev4']) and
            pd.notna(row['Low_prev4']) and
            row['Close_prev4'] > row['Open_prev4'] and  # Bullish first candle
            row['Close_prev3'] < row['Open_prev3'] and  # Bearish second candle
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish third candle
            row['Close_prev'] < row['Open_prev'] and  # Bearish fourth candle
            row['Close_prev3'] > row['Low_prev4'] and  # Within first candle’s range
            row['Close_prev2'] > row['Low_prev4'] and
            row['Close_prev'] > row['Low_prev4'] and
            row['Close'] > row['Open'] and row['Close'] > row['Close_prev4']):  # Bullish fifth candle
            return 1
    return 0

def side_by_side_white_lines(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            row['Open_prev2'] != 0 and
            row['Close_prev2'] > row['Open_prev2'] and  # Bullish first candle or context
            row['Close_prev'] > row['Open_prev'] and  # Bullish second candle
            row['Close'] > row['Open'] and  # Bullish third candle
            abs(row['Open_prev'] - row['Open_prev2']) / row['Open_prev2'] < 0.001 and  # Similar opens
            abs((row['Close_prev'] - row['Open_prev']) - (row['Close'] - row['Open'])) / row['Close_prev'] < 0.1):  # Similar sizes
            return 1
    return 0

def upside_tasuki_gap(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and
            row['Close_prev2'] > row['Open_prev2'] and  # Bullish first candle
            row['Open_prev'] > row['Close_prev2'] and  # Gap up
            row['Close_prev'] > row['Open_prev'] and  # Bullish second candle
            row['Open'] < row['Close_prev'] and  # Third candle fills gap
            row['Close'] < row['Open'] and  # Bearish third candle
            row['Close'] > row['Close_prev2'] and  # Gap partially unfilled
            row['Close'] > row['Open_prev']):  # Close above second candle’s open
            return 1
    return 0

# Define bullish breakout pattern strategies

def bullish_flag_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['High']) and pd.notna(row['High_prev']) and 
            pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            pd.notna(row['Open']) and
            row['Close'] > row['Resistance'] and 
            row['High'] > row['High_prev'] and row['Low'] > row['Low_prev'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def bullish_pennant_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Open']) and pd.notna(row['High']) and pd.notna(row['Low']) and
            row['Close'] != 0 and
            row['Close'] > row['Resistance'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def falling_wedge_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Open']) and
            row['Close'] > row['Resistance'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def symmetrical_triangle_breakout_upward(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Open']) and pd.notna(row['High']) and pd.notna(row['Low']) and
            row['Close'] != 0 and
            row['Close'] > row['Resistance'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def ascending_triangle_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            pd.notna(row['Open']) and
            row['Close'] > row['Resistance'] and 
            row['Low'] > row['Low_prev'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def double_bottom_breakout(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev2']) and 
            pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Open']) and
            row['Low'] != 0 and
            abs(row['Low'] - row['Low_prev2']) / row['Low'] < 0.001 and  # Similar lows
            row['Close'] > row['Resistance'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def triple_bottom_breakout(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev2']) and 
            pd.notna(row['Low_prev3']) and pd.notna(row['Close']) and 
            pd.notna(row['Resistance']) and pd.notna(row['Open']) and
            row['Low'] != 0 and
            abs(row['Low'] - row['Low_prev3']) / row['Low'] < 0.001 and  # Similar lows
            abs(row['Low_prev2'] - row['Low_prev3']) / row['Low_prev3'] < 0.001 and
            row['Close'] > row['Resistance'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def rounding_bottom_breakout(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            pd.notna(row['Low_prev2']) and pd.notna(row['Close']) and 
            pd.notna(row['Resistance']) and pd.notna(row['Open']) and
            row['Low'] > row['Low_prev'] and row['Low_prev'] > row['Low_prev2'] and 
            row['Close'] > row['Resistance'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def cup_and_handle_breakout(row):
    if (pd.notna(row['Close']) and pd.notna(row['Resistance']) and 
            pd.notna(row['Volume']) and pd.notna(row['Volume_20_SMA']) and 
            pd.notna(row['Open']) and
            row['Close'] > row['Resistance'] and 
            row['Volume'] > row['Volume_20_SMA'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

# Define gaps & exhaustion pattern strategies (Bullish)

def breakaway_gap_bullish(row):
    if (pd.notna(row['Open']) and pd.notna(row['High_prev']) and 
            pd.notna(row['Close']) and
            row['Open'] > row['High_prev'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def runaway_gap_bullish(row):
    if (pd.notna(row['Open']) and pd.notna(row['Close_prev']) and 
            pd.notna(row['Close']) and
            row['Open'] > row['Close_prev'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

def island_reversal_bullish(row):
    if (pd.notna(row['Low']) and pd.notna(row['Low_prev']) and 
            pd.notna(row['High_prev']) and pd.notna(row['High_prev2']) and
            pd.notna(row['Close']) and pd.notna(row['Open']) and
            row['Low_prev'] > row['High_prev2'] and  # Gap down
            row['Low'] > row['High_prev'] and  # Gap up
            row['Close'] > row['Open'] and  # Bullish candle
            row['Close'] > row['High_prev']):  # Close above gap
            return 1
    return 0

def exhaustion_gap_bullish(row):
    if (pd.notna(row['Open']) and pd.notna(row['High_prev']) and 
            pd.notna(row['Close']) and pd.notna(row['Open']) and
            row['Open'] > row['High_prev'] and 
            row['Close'] > row['Open']):  # Bullish candle
            return 1
    return 0

# Define rare but high-probability pattern strategies (Bullish)

def bullish_counterattack_line(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and
            row['Open_prev'] != 0 and
            row['Close_prev'] < row['Open_prev'] and  # Bearish first candle
            row['Close'] >= row['Open_prev'] and  # Close at or above prior open
            abs(row['Close'] - row['Open_prev']) / row['Open_prev'] < 0.001 and  # Close near prior open
            row['Close'] > row['Open']):  # Bullish second candle
            return 1
    return 0

def ladder_bottom(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and 
            pd.notna(row['Close_prev3']) and pd.notna(row['Open_prev3']) and 
            pd.notna(row['Close_prev4']) and pd.notna(row['Open_prev4']) and
            row['Close_prev4'] < row['Open_prev4'] and  # Bearish first candle
            row['Close_prev3'] < row['Open_prev3'] and  # Bearish second candle
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish third candle
            row['Close_prev4'] > row['Close_prev3'] > row['Close_prev2'] and  # Decreasing closes
            row['Close_prev'] > row['Open_prev'] and  # Bullish fourth candle
            row['Close'] > row['Open']):  # Bullish fifth candle
            return 1
    return 0

def unique_three_river_bottom(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and 
            pd.notna(row['Low_prev']) and pd.notna(row['Low_prev2']) and pd.notna(row['Low']) and
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish first candle
            row['Close_prev'] < row['Open_prev'] and  # Bearish second candle
            row['Close'] > row['Open'] and  # Bullish third candle
            row['Low_prev'] < row['Low_prev2'] and  # Lower low in second candle
            row['Low'] > row['Low_prev']):  # Higher low in third candle
            return 1
    return 0

def concealing_baby_swallow(row):
    if (pd.notna(row['Close']) and pd.notna(row['Open']) and 
            pd.notna(row['Close_prev']) and pd.notna(row['Open_prev']) and 
            pd.notna(row['Close_prev2']) and pd.notna(row['Open_prev2']) and 
            pd.notna(row['Close_prev3']) and pd.notna(row['Open_prev3']) and
            pd.notna(row['High_prev']) and pd.notna(row['Low_prev']) and
            pd.notna(row['High_prev2']) and pd.notna(row['Low_prev2']) and
            pd.notna(row['High_prev3']) and pd.notna(row['Low_prev3']) and
            row['Close_prev3'] < row['Open_prev3'] and  # Bearish first Marubozu
            abs(row['High_prev3'] - row['Open_prev3']) < 0.005 * row['Open_prev3'] and  # Small upper shadow
            abs(row['Close_prev3'] - row['Low_prev3']) < 0.005 * row['Close_prev3'] and  # Small lower shadow
            row['Close_prev2'] < row['Open_prev2'] and  # Bearish second Marubozu
            abs(row['High_prev2'] - row['Open_prev2']) < 0.005 * row['Open_prev2'] and
            abs(row['Close_prev2'] - row['Low_prev2']) < 0.005 * row['Close_prev2'] and
            row['Close_prev'] < row['Open_prev'] and  # Bearish third candle
            row['Open'] < row['Low_prev'] and  # Gap down
            row['Close'] > row['Open'] and  # Bullish fourth candle
            row['Close'] > row['Open_prev'] and  # Engulfs third candle
            row['Close'] > row['Close_prev2']):  # Closes into second candle
            return 1
    return 0

# Define confirmation pattern strategies (Bullish)

def hammer_rsi_oversold(row):
    if (pd.notna(row['RSI']) and
            hammer(row) == 1 and 
            row['RSI'] < 30):
            return 1
    return 0

def bullish_engulfing_macd_crossover(row):
    if (bullish_engulfing(row) == 1 and 
            macd_bullish_crossover(row) == 1):
            return 1
    return 0

def morning_star_volume_spike(row):
    if (pd.notna(row['Volume']) and pd.notna(row['Volume_20_SMA']) and
            morning_star(row) == 1 and 
            row['Volume'] > 2 * row['Volume_20_SMA']):
            return 1
    return 0

def three_white_soldiers_adx(row):
    if (pd.notna(row['ADX']) and
            three_white_soldiers(row) == 1 and 
            row['ADX'] > 25):
            return 1
    return 0