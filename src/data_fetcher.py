"""
Data Fetcher Module

High-level interface for fetching and processing financial market data
with built-in data validation and preprocessing capabilities.
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from .yfinance_client import YFinanceClient

logger = logging.getLogger(__name__)


def fetch_market_data(client: YFinanceClient, symbol: str, period: str = '1y', 
                     interval: str = '1d', use_sample_data: bool = False) -> pd.DataFrame:
    """
    Fetch and preprocess market data for analysis.
    
    This is the main function for retrieving financial data. It provides a clean interface
    to the Yahoo Finance API with automatic data validation and preprocessing.
    
    Args:
        client: Initialized YFinanceClient instance
        symbol: Financial instrument symbol (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo')
        use_sample_data: If True, generate sample data instead of fetching from API
        
    Returns:
        Preprocessed DataFrame with OHLCV data and additional calculated fields
        
    Example:
        >>> from src.yfinance_client import YFinanceClient
        >>> from src.data_fetcher import fetch_market_data
        >>> 
        >>> client = YFinanceClient()
        >>> data = fetch_market_data(client, 'AAPL', '1y', '1d')
        >>> print(f"Fetched {len(data)} data points for AAPL")
    """
    try:
        logger.info(f"Fetching market data for {symbol} ({period}, {interval})")
        
        # Fetch raw data
        raw_data = client.fetch_data(symbol, period, interval, use_sample_data)
        
        if raw_data.empty:
            raise ValueError(f"No data received for symbol: {symbol}")
        
        # Preprocess the data
        processed_data = _preprocess_data(raw_data, symbol)
        
        logger.info(f"Successfully processed {len(processed_data)} data points for {symbol}")
        return processed_data
        
    except Exception as e:
        logger.error(f"Failed to fetch market data for {symbol}: {str(e)}")
        raise


def _preprocess_data(data: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Preprocess raw market data for analysis.
    
    Args:
        data: Raw OHLCV DataFrame
        symbol: Financial instrument symbol
        
    Returns:
        Preprocessed DataFrame with additional calculated fields
    """
    # Create a copy to avoid modifying original data
    df = data.copy()
    
    # Ensure we have the required columns
    required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")
    
    # Convert index to datetime if it's not already
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    # Sort by date (ascending)
    df.sort_index(inplace=True)
    
    # Remove any duplicate dates
    df = df[~df.index.duplicated(keep='last')]
    
    # Handle missing values
    df = _handle_missing_values(df)
    
    # Add calculated fields
    df = _add_calculated_fields(df)
    
    # Add metadata
    df.attrs['symbol'] = symbol
    df.attrs['last_updated'] = datetime.now()
    df.attrs['data_points'] = len(df)
    
    return df


def _handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with missing values handled
    """
    # Forward fill for OHLC data (use previous day's values)
    ohlc_columns = ['Open', 'High', 'Low', 'Close']
    df[ohlc_columns] = df[ohlc_columns].fillna(method='ffill')
    
    # For volume, use 0 if missing
    if 'Volume' in df.columns:
        df['Volume'] = df['Volume'].fillna(0)
    
    # Remove any remaining rows with NaN values
    df.dropna(inplace=True)
    
    return df


def _add_calculated_fields(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add calculated fields to the dataset.
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with additional calculated fields
    """
    # Price changes
    df['Price_Change'] = df['Close'].diff()
    df['Price_Change_Pct'] = df['Close'].pct_change() * 100
    
    # High-Low range
    df['High_Low_Range'] = df['High'] - df['Low']
    df['High_Low_Range_Pct'] = (df['High_Low_Range'] / df['Close']) * 100
    
    # True Range (for volatility analysis)
    df['True_Range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df['Close'].shift(1)),
            abs(df['Low'] - df['Close'].shift(1))
        )
    )
    
    # Moving averages (basic ones)
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    
    # Volume moving average
    if 'Volume' in df.columns:
        df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    
    # Volatility (20-day rolling standard deviation)
    df['Volatility_20'] = df['Close'].rolling(window=20).std()
    
    return df


def get_market_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a summary of market data.
    
    Args:
        data: Preprocessed market data DataFrame
        
    Returns:
        Dictionary with market summary statistics
    """
    if data.empty:
        return {}
    
    latest = data.iloc[-1]
    first = data.iloc[0]
    
    summary = {
        'symbol': data.attrs.get('symbol', 'Unknown'),
        'data_points': len(data),
        'date_range': {
            'start': data.index[0].strftime('%Y-%m-%d'),
            'end': data.index[-1].strftime('%Y-%m-%d'),
            'days': (data.index[-1] - data.index[0]).days
        },
        'current_price': round(latest['Close'], 2),
        'price_change': {
            'absolute': round(latest['Close'] - first['Close'], 2),
            'percentage': round(((latest['Close'] - first['Close']) / first['Close']) * 100, 2)
        },
        'price_stats': {
            'highest': round(data['High'].max(), 2),
            'lowest': round(data['Low'].min(), 2),
            'average': round(data['Close'].mean(), 2),
            'volatility': round(data['Close'].std(), 2)
        },
        'volume_stats': {
            'current': latest.get('Volume', 0),
            'average': round(data.get('Volume', pd.Series([0])).mean(), 0),
            'highest': data.get('Volume', pd.Series([0])).max()
        } if 'Volume' in data.columns else {},
        'trend': 'Bullish' if latest['Close'] > first['Close'] else 'Bearish'
    }
    
    return summary


def validate_symbol(symbol: str) -> bool:
    """
    Validate if a symbol format is correct.
    
    Args:
        symbol: Financial instrument symbol
        
    Returns:
        True if symbol format is valid, False otherwise
    """
    if not symbol or not isinstance(symbol, str):
        return False
    
    # Basic validation rules
    symbol = symbol.upper().strip()
    
    # Check for common patterns
    valid_patterns = [
        r'^[A-Z]{1,5}$',  # Basic stock symbols (1-5 letters)
        r'^[A-Z]{1,5}=X$',  # Forex pairs
        r'^[A-Z]{1,5}-USD$',  # Cryptocurrencies
        r'^[A-Z]{1,5}=F$',  # Futures
        r'^\^[A-Z]{1,5}$',  # Indices
    ]
    
    import re
    for pattern in valid_patterns:
        if re.match(pattern, symbol):
            return True
    
    return False


def get_available_periods() -> Dict[str, str]:
    """
    Get available data periods with descriptions.
    
    Returns:
        Dictionary mapping period codes to descriptions
    """
    return {
        '1d': '1 Day',
        '5d': '5 Days',
        '1mo': '1 Month',
        '3mo': '3 Months',
        '6mo': '6 Months',
        '1y': '1 Year',
        '2y': '2 Years',
        '5y': '5 Years',
        '10y': '10 Years',
        'ytd': 'Year to Date',
        'max': 'Maximum Available'
    }


def get_available_intervals() -> Dict[str, str]:
    """
    Get available data intervals with descriptions.
    
    Returns:
        Dictionary mapping interval codes to descriptions
    """
    return {
        '1m': '1 Minute',
        '2m': '2 Minutes',
        '5m': '5 Minutes',
        '15m': '15 Minutes',
        '30m': '30 Minutes',
        '60m': '1 Hour',
        '90m': '1.5 Hours',
        '1h': '1 Hour',
        '1d': '1 Day',
        '5d': '5 Days',
        '1wk': '1 Week',
        '1mo': '1 Month'
    }
