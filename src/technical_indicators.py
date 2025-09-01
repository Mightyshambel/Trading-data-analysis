"""
Technical Indicators Module

Comprehensive technical analysis tools implemented with NumPy for maximum performance.
Includes moving averages, momentum indicators, trend indicators, and oscillators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class IndicatorResult:
    """Container for technical indicator results."""
    values: pd.Series
    signals: Optional[pd.Series] = None
    metadata: Optional[Dict[str, Any]] = None


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Add comprehensive technical indicators to the dataset.
    
    This function adds all major technical indicators to the provided DataFrame,
    making it ready for advanced analysis and strategy development.
    
    Args:
        data: DataFrame with OHLCV data
        
    Returns:
        DataFrame with all technical indicators added
        
    Example:
        >>> from src.technical_indicators import add_technical_indicators
        >>> data_with_indicators = add_technical_indicators(data)
        >>> print(f"Added {len(data_with_indicators.columns)} technical indicators")
    """
    df = data.copy()
    
    # Moving Averages
    df = _add_moving_averages(df)
    
    # Momentum Indicators
    df = _add_momentum_indicators(df)
    
    # Trend Indicators
    df = _add_trend_indicators(df)
    
    # Volatility Indicators
    df = _add_volatility_indicators(df)
    
    # Oscillators
    df = _add_oscillators(df)
    
    # Volume Indicators
    df = _add_volume_indicators(df)
    
    logger.info(f"Added {len(df.columns) - len(data.columns)} technical indicators")
    return df


def _add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Add various moving averages to the dataset."""
    close = df['Close']
    
    # Simple Moving Averages
    df['SMA_10'] = close.rolling(window=10).mean()
    df['SMA_20'] = close.rolling(window=20).mean()
    df['SMA_50'] = close.rolling(window=50).mean()
    df['SMA_100'] = close.rolling(window=100).mean()
    df['SMA_200'] = close.rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = close.ewm(span=12).mean()
    df['EMA_26'] = close.ewm(span=26).mean()
    df['EMA_50'] = close.ewm(span=50).mean()
    df['EMA_200'] = close.ewm(span=200).mean()
    
    # Weighted Moving Average
    df['WMA_20'] = _weighted_moving_average(close, 20)
    
    return df


def _add_momentum_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add momentum indicators to the dataset."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Relative Strength Index (RSI)
    df['RSI'] = _calculate_rsi(close, 14)
    
    # Stochastic Oscillator
    df['Stoch_K'], df['Stoch_D'] = _calculate_stochastic(high, low, close, 14, 3)
    
    # Williams %R
    df['Williams_R'] = _calculate_williams_r(high, low, close, 14)
    
    # Rate of Change (ROC)
    df['ROC'] = _calculate_roc(close, 10)
    
    # Money Flow Index (MFI)
    df['MFI'] = _calculate_money_flow_index(df, 14)
    
    return df


def _add_trend_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add trend indicators to the dataset."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # MACD
    df['MACD'], df['MACD_Signal'], df['MACD_Histogram'] = _calculate_macd(close)
    
    # Bollinger Bands
    df['BB_Upper'], df['BB_Middle'], df['BB_Lower'] = _calculate_bollinger_bands(close, 20, 2)
    df['BB_Width'] = (df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']
    df['BB_Position'] = (close - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
    
    # Parabolic SAR
    df['PSAR'] = _calculate_parabolic_sar(high, low, close)
    
    # Average Directional Index (ADX)
    df['ADX'] = _calculate_adx(high, low, close, 14)
    
    return df


def _add_volatility_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility indicators to the dataset."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Average True Range (ATR)
    df['ATR'] = _calculate_atr(high, low, close, 14)
    
    # Historical Volatility
    df['Hist_Volatility'] = _calculate_historical_volatility(close, 20)
    
    # Chaikin Volatility
    df['Chaikin_Volatility'] = _calculate_chaikin_volatility(df, 10)
    
    return df


def _add_oscillators(df: pd.DataFrame) -> pd.DataFrame:
    """Add oscillator indicators to the dataset."""
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # Commodity Channel Index (CCI)
    df['CCI'] = _calculate_cci(high, low, close, 20)
    
    # Ultimate Oscillator
    df['Ultimate_Oscillator'] = _calculate_ultimate_oscillator(high, low, close, 7, 14, 28)
    
    # Awesome Oscillator
    df['Awesome_Oscillator'] = _calculate_awesome_oscillator(high, low, 5, 34)
    
    return df


def _add_volume_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add volume-based indicators to the dataset."""
    if 'Volume' not in df.columns:
        return df
    
    close = df['Close']
    volume = df['Volume']
    
    # On-Balance Volume (OBV)
    df['OBV'] = _calculate_obv(close, volume)
    
    # Volume Price Trend (VPT)
    df['VPT'] = _calculate_vpt(close, volume)
    
    # Accumulation/Distribution Line
    df['ADL'] = _calculate_adl(df)
    
    # Chaikin Money Flow
    df['CMF'] = _calculate_chaikin_money_flow(df, 20)
    
    return df


# Individual Indicator Functions

def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
    """Calculate Stochastic Oscillator."""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    
    k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    d_percent = k_percent.rolling(window=d_period).mean()
    
    return k_percent, d_percent


def _calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, 
                         period: int = 14) -> pd.Series:
    """Calculate Williams %R."""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    
    williams_r = -100 * ((highest_high - close) / (highest_high - lowest_low))
    return williams_r


def _calculate_roc(prices: pd.Series, period: int = 10) -> pd.Series:
    """Calculate Rate of Change."""
    return ((prices - prices.shift(period)) / prices.shift(period)) * 100


def _calculate_money_flow_index(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Money Flow Index."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0).rolling(window=period).sum()
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0).rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + (positive_flow / negative_flow)))
    return mfi


def _calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (Moving Average Convergence Divergence)."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram


def _calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands."""
    middle = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)
    
    return upper, middle, lower


def _calculate_parabolic_sar(high: pd.Series, low: pd.Series, close: pd.Series, 
                           acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Calculate Parabolic SAR."""
    psar = pd.Series(index=close.index, dtype=float)
    psar.iloc[0] = low.iloc[0]
    
    af = acceleration
    ep = high.iloc[0]
    long = True
    
    for i in range(1, len(close)):
        if long:
            psar.iloc[i] = psar.iloc[i-1] + af * (ep - psar.iloc[i-1])
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
            
            if close.iloc[i] > ep:
                ep = high.iloc[i]
                af = min(af + acceleration, maximum)
                
            if close.iloc[i] < psar.iloc[i]:
                long = False
                psar.iloc[i] = ep
                ep = low.iloc[i]
                af = acceleration
        else:
            psar.iloc[i] = psar.iloc[i-1] - af * (psar.iloc[i-1] - ep)
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
            
            if close.iloc[i] < ep:
                ep = low.iloc[i]
                af = min(af + acceleration, maximum)
                
            if close.iloc[i] > psar.iloc[i]:
                long = True
                psar.iloc[i] = ep
                ep = high.iloc[i]
                af = acceleration
    
    return psar


def _calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = true_range.rolling(window=period).mean()
    
    return atr


def _calculate_historical_volatility(prices: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Historical Volatility."""
    returns = prices.pct_change()
    volatility = returns.rolling(window=period).std() * np.sqrt(252) * 100
    return volatility


def _calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    """Calculate Commodity Channel Index."""
    typical_price = (high + low + close) / 3
    sma = typical_price.rolling(window=period).mean()
    mad = typical_price.rolling(window=period).apply(lambda x: np.mean(np.abs(x - x.mean())))
    
    cci = (typical_price - sma) / (0.015 * mad)
    return cci


def _weighted_moving_average(prices: pd.Series, period: int) -> pd.Series:
    """Calculate Weighted Moving Average."""
    weights = np.arange(1, period + 1)
    wma = prices.rolling(window=period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return wma


def _calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    # Simplified ADX calculation
    tr = _calculate_atr(high, low, close, 1)
    dm_plus = high.diff()
    dm_minus = -low.diff()
    
    dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
    dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
    
    di_plus = 100 * (dm_plus.rolling(window=period).mean() / tr.rolling(window=period).mean())
    di_minus = 100 * (dm_minus.rolling(window=period).mean() / tr.rolling(window=period).mean())
    
    dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
    adx = dx.rolling(window=period).mean()
    
    return adx


def _calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate On-Balance Volume."""
    obv = pd.Series(index=close.index, dtype=float)
    obv.iloc[0] = volume.iloc[0]
    
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif close.iloc[i] < close.iloc[i-1]:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv


def _calculate_vpt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Calculate Volume Price Trend."""
    price_change = close.pct_change()
    vpt = (price_change * volume).cumsum()
    return vpt


def _calculate_adl(df: pd.DataFrame) -> pd.Series:
    """Calculate Accumulation/Distribution Line."""
    clv = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    adl = (clv * df['Volume']).cumsum()
    return adl


def _calculate_chaikin_money_flow(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate Chaikin Money Flow."""
    mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low'])
    mfm = mfm.replace([np.inf, -np.inf], 0)
    mfv = mfm * df['Volume']
    cmf = mfv.rolling(window=period).sum() / df['Volume'].rolling(window=period).sum()
    return cmf


def _calculate_chaikin_volatility(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """Calculate Chaikin Volatility."""
    high_low = df['High'] - df['Low']
    chaikin_vol = (high_low.rolling(window=period).mean() / high_low.rolling(window=period).mean().shift(period)) - 1
    return chaikin_vol


def _calculate_ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, 
                                 period1: int = 7, period2: int = 14, period3: int = 28) -> pd.Series:
    """Calculate Ultimate Oscillator."""
    tr = pd.DataFrame({
        'hl': high - low,
        'hc': abs(high - close.shift(1)),
        'lc': abs(low - close.shift(1))
    }).max(axis=1)
    
    bp = close - low
    tr = tr.where(tr != 0, 1)  # Avoid division by zero
    
    avg7 = bp.rolling(window=period1).sum() / tr.rolling(window=period1).sum()
    avg14 = bp.rolling(window=period2).sum() / tr.rolling(window=period2).sum()
    avg28 = bp.rolling(window=period3).sum() / tr.rolling(window=period3).sum()
    
    uo = 100 * ((4 * avg7) + (2 * avg14) + avg28) / (4 + 2 + 1)
    return uo


def _calculate_awesome_oscillator(high: pd.Series, low: pd.Series, 
                                fast_period: int = 5, slow_period: int = 34) -> pd.Series:
    """Calculate Awesome Oscillator."""
    median_price = (high + low) / 2
    ao = median_price.rolling(window=fast_period).mean() - median_price.rolling(window=slow_period).mean()
    return ao


def generate_trading_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate trading signals based on technical indicators.
    
    Args:
        df: DataFrame with technical indicators
        
    Returns:
        DataFrame with trading signals added
    """
    signals = df.copy()
    
    # RSI signals
    signals['RSI_Signal'] = 0
    signals.loc[signals['RSI'] < 30, 'RSI_Signal'] = 1  # Oversold
    signals.loc[signals['RSI'] > 70, 'RSI_Signal'] = -1  # Overbought
    
    # MACD signals
    signals['MACD_Signal'] = 0
    signals.loc[signals['MACD'] > signals['MACD_Signal'], 'MACD_Signal'] = 1
    signals.loc[signals['MACD'] < signals['MACD_Signal'], 'MACD_Signal'] = -1
    
    # Moving average crossover signals
    signals['MA_Signal'] = 0
    signals.loc[signals['SMA_20'] > signals['SMA_50'], 'MA_Signal'] = 1
    signals.loc[signals['SMA_20'] < signals['SMA_50'], 'MA_Signal'] = -1
    
    # Bollinger Bands signals
    signals['BB_Signal'] = 0
    signals.loc[signals['Close'] < signals['BB_Lower'], 'BB_Signal'] = 1  # Oversold
    signals.loc[signals['Close'] > signals['BB_Upper'], 'BB_Signal'] = -1  # Overbought
    
    # Combined signal
    signals['Combined_Signal'] = (
        signals['RSI_Signal'] + 
        signals['MACD_Signal'] + 
        signals['MA_Signal'] + 
        signals['BB_Signal']
    )
    
    return signals
