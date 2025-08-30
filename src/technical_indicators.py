"""
Technical indicators for trading analysis using NumPy
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

def simple_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Simple Moving Average (SMA)
    
    Args:
        prices: Array of prices
        window: Moving average window size
    
    Returns:
        Array of SMA values
    """
    if window > len(prices):
        return np.full_like(prices, np.nan)
    
    sma = np.convolve(prices, np.ones(window)/window, mode='valid')
    # Pad the beginning with NaN values
    padding = np.full(window - 1, np.nan)
    return np.concatenate([padding, sma])

def exponential_moving_average(prices: np.ndarray, window: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average (EMA)
    
    Args:
        prices: Array of prices
        window: EMA window size
    
    Returns:
        Array of EMA values
    """
    alpha = 2 / (window + 1)
    ema = np.zeros_like(prices)
    ema[0] = prices[0]  # First value is the first price
    
    for i in range(1, len(prices)):
        ema[i] = alpha * prices[i] + (1 - alpha) * ema[i-1]
    
    return ema

def relative_strength_index(prices: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        prices: Array of prices
        window: RSI window size (default: 14)
    
    Returns:
        Array of RSI values (0-100)
    """
    if len(prices) < window + 1:
        return np.full_like(prices, np.nan)
    
    # Calculate price changes
    deltas = np.diff(prices)
    
    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)
    
    # Calculate average gains and losses
    avg_gains = simple_moving_average(gains, window)
    avg_losses = simple_moving_average(losses, window)
    
    # Calculate RSI
    rs = avg_gains / (avg_losses + 1e-10)  # Add small value to avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    
    # Pad the beginning with NaN
    padding = np.full(1, np.nan)
    return np.concatenate([padding, rsi])

def bollinger_bands(prices: np.ndarray, window: int = 20, 
                    num_std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate Bollinger Bands
    
    Args:
        prices: Array of prices
        window: Moving average window size
        num_std: Number of standard deviations
    
    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = simple_moving_average(prices, window)
    
    # Calculate rolling standard deviation
    std = np.zeros_like(prices)
    for i in range(window - 1, len(prices)):
        std[i] = np.std(prices[i-window+1:i+1])
    
    upper = middle + (num_std * std)
    lower = middle - (num_std * std)
    
    return upper, middle, lower

def macd(prices: np.ndarray, fast: int = 12, slow: int = 26, 
          signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        prices: Array of prices
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    ema_fast = exponential_moving_average(prices, fast)
    ema_slow = exponential_moving_average(prices, slow)
    
    macd_line = ema_fast - ema_slow
    signal_line = exponential_moving_average(macd_line, signal)
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram

def stochastic_oscillator(high: np.ndarray, low: np.ndarray, 
                         close: np.ndarray, k_window: int = 14, 
                         d_window: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate Stochastic Oscillator
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        k_window: %K window size
        d_window: %D window size
    
    Returns:
        Tuple of (%K, %D)
    """
    if len(high) < k_window:
        return np.full_like(high, np.nan), np.full_like(high, np.nan)
    
    # Calculate %K
    k_percent = np.zeros_like(high)
    for i in range(k_window - 1, len(high)):
        highest_high = np.max(high[i-k_window+1:i+1])
        lowest_low = np.min(low[i-k_window+1:i+1])
        k_percent[i] = 100 * (close[i] - lowest_low) / (highest_high - lowest_low + 1e-10)
    
    # Calculate %D (SMA of %K)
    d_percent = simple_moving_average(k_percent, d_window)
    
    return k_percent, d_percent

def average_true_range(high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Average True Range (ATR)
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        window: ATR window size
    
    Returns:
        Array of ATR values
    """
    if len(high) < 2:
        return np.full_like(high, np.nan)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = np.abs(high - np.roll(close, 1))
    tr3 = np.abs(low - np.roll(close, 1))
    
    # True Range is the maximum of the three
    true_range = np.maximum(tr1, np.maximum(tr2, tr3))
    
    # Calculate ATR using simple moving average
    atr = simple_moving_average(true_range, window)
    
    return atr

def williams_r(high: np.ndarray, low: np.ndarray, 
               close: np.ndarray, window: int = 14) -> np.ndarray:
    """
    Calculate Williams %R
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        window: Williams %R window size
    
    Returns:
        Array of Williams %R values (-100 to 0)
    """
    if len(high) < window:
        return np.full_like(high, np.nan)
    
    williams_r = np.zeros_like(high)
    for i in range(window - 1, len(high)):
        highest_high = np.max(high[i-window+1:i+1])
        lowest_low = np.min(low[i-window+1:i+1])
        williams_r[i] = -100 * (highest_high - close[i]) / (highest_high - lowest_low + 1e-10)
    
    return williams_r

def commodity_channel_index(high: np.ndarray, low: np.ndarray, 
                           close: np.ndarray, window: int = 20) -> np.ndarray:
    """
    Calculate Commodity Channel Index (CCI)
    
    Args:
        high: Array of high prices
        low: Array of low prices
        close: Array of close prices
        window: CCI window size
    
    Returns:
        Array of CCI values
    """
    if len(high) < window:
        return np.full_like(high, np.nan)
    
    # Calculate Typical Price
    typical_price = (high + low + close) / 3
    
    # Calculate Simple Moving Average of Typical Price
    sma_tp = simple_moving_average(typical_price, window)
    
    # Calculate Mean Deviation
    cci = np.zeros_like(high)
    for i in range(window - 1, len(high)):
        mean_deviation = np.mean(np.abs(typical_price[i-window+1:i+1] - sma_tp[i]))
        cci[i] = (typical_price[i] - sma_tp[i]) / (0.015 * mean_deviation + 1e-10)
    
    return cci

def add_technical_indicators(df: pd.DataFrame, 
                           sma_windows: list = [20, 50],
                           ema_windows: list = [12, 26],
                           rsi_window: int = 14,
                           bb_window: int = 20,
                           macd_fast: int = 12,
                           macd_slow: int = 26,
                           macd_signal: int = 9) -> pd.DataFrame:
    """
    Add multiple technical indicators to a DataFrame
    
    Args:
        df: DataFrame with OHLCV data
        sma_windows: List of SMA window sizes
        ema_windows: List of EMA window sizes
        rsi_window: RSI window size
        bb_window: Bollinger Bands window size
        macd_fast: MACD fast period
        macd_slow: MACD slow period
        macd_signal: MACD signal period
    
    Returns:
        DataFrame with added technical indicators
    """
    df_copy = df.copy()
    
    # Add Simple Moving Averages
    for window in sma_windows:
        df_copy[f'SMA_{window}'] = simple_moving_average(df_copy['close'].values, window)
    
    # Add Exponential Moving Averages
    for window in ema_windows:
        df_copy[f'EMA_{window}'] = exponential_moving_average(df_copy['close'].values, window)
    
    # Add RSI
    df_copy['RSI'] = relative_strength_index(df_copy['close'].values, rsi_window)
    
    # Add Bollinger Bands
    bb_upper, bb_middle, bb_lower = bollinger_bands(df_copy['close'].values, bb_window)
    df_copy['BB_Upper'] = bb_upper
    df_copy['BB_Middle'] = bb_middle
    df_copy['BB_Lower'] = bb_lower
    
    # Add MACD
    macd_line, signal_line, histogram = macd(df_copy['close'].values, 
                                            macd_fast, macd_slow, macd_signal)
    df_copy['MACD'] = macd_line
    df_copy['MACD_Signal'] = signal_line
    df_copy['MACD_Histogram'] = histogram
    
    # Add Stochastic Oscillator
    k_percent, d_percent = stochastic_oscillator(df_copy['high'].values, 
                                                df_copy['low'].values, 
                                                df_copy['close'].values)
    df_copy['Stoch_K'] = k_percent
    df_copy['Stoch_D'] = d_percent
    
    # Add Williams %R
    df_copy['Williams_R'] = williams_r(df_copy['high'].values, 
                                      df_copy['low'].values, 
                                      df_copy['close'].values)
    
    # Add ATR
    df_copy['ATR'] = average_true_range(df_copy['high'].values, 
                                       df_copy['low'].values, 
                                       df_copy['close'].values)
    
    # Add CCI
    df_copy['CCI'] = commodity_channel_index(df_copy['high'].values, 
                                           df_copy['low'].values, 
                                           df_copy['close'].values)
    
    return df_copy
