"""
Data fetching utilities for Yahoo Finance data
"""

import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
from yfinance_client import YFinanceClient

def fetch_market_data(client: YFinanceClient, 
                     symbol: str,
                     period: str = "1y",
                     interval: str = "1d",
                     start_time: Optional[Union[str, datetime]] = None,
                     end_time: Optional[Union[str, datetime]] = None,
                     use_sample_data: bool = False) -> pd.DataFrame:
    """
    Fetch market data from Yahoo Finance
    
    Args:
        client: YFinance client instance
        symbol: Stock/ETF/Crypto symbol (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD')
        period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
        interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
        start_time: Start time for data range
        end_time: End time for data range
        use_sample_data: If True, use sample data when API fails
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        if start_time and end_time:
            # Use specific date range
            df = client.get_historical_data(symbol, start=start_time, end=end_time, interval=interval)
        else:
            # Use period
            df = client.get_historical_data(symbol, period=period, interval=interval)
        
        if not df.empty:
            print(f"✅ Successfully fetched {len(df)} data points for {symbol}")
            print(f"Date range: {df.index[0]} to {df.index[-1]}")
            print(f"Columns: {list(df.columns)}")
            return df
        
        # If no data and sample data is requested, generate sample data
        if use_sample_data:
            print(f"⚠️  No data received from API for {symbol}")
            print("Generating sample data for demonstration purposes...")
            df = client.generate_sample_data(symbol, days=100)
            return df
        
        return df
        
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        
        # If sample data is requested, generate it as fallback
        if use_sample_data:
            print("Generating sample data as fallback...")
            df = client.generate_sample_data(symbol, days=100)
            return df
        
        return pd.DataFrame()

def fetch_market_data_range(client: YFinanceClient,
                           symbol: str,
                           start_time: Union[str, datetime],
                           end_time: Union[str, datetime],
                           interval: str = "1d") -> pd.DataFrame:
    """
    Fetch market data for a specific time range
    
    Args:
        client: YFinance client instance
        symbol: Stock/ETF/Crypto symbol
        start_time: Start time
        end_time: End time
        interval: Data interval
    
    Returns:
        DataFrame with OHLCV data
    """
    return fetch_market_data(
        client=client,
        symbol=symbol,
        start_time=start_time,
        end_time=end_time,
        interval=interval
    )

def fetch_recent_data(client: YFinanceClient,
                     symbol: str,
                     period: str = "1mo",
                     interval: str = "1d") -> pd.DataFrame:
    """
    Fetch recent market data
    
    Args:
        client: YFinance client instance
        symbol: Stock/ETF/Crypto symbol
        period: Data period (default: 1 month)
        interval: Data interval
    
    Returns:
        DataFrame with OHLCV data
    """
    return fetch_market_data(
        client=client,
        symbol=symbol,
        period=period,
        interval=interval
    )

def fetch_intraday_data(client: YFinanceClient,
                        symbol: str,
                        period: str = "5d",
                        interval: str = "5m") -> pd.DataFrame:
    """
    Fetch intraday market data (higher frequency)
    
    Args:
        client: YFinance client instance
        symbol: Stock/ETF/Crypto symbol
        period: Data period (default: 5 days)
        interval: Data interval (default: 5 minutes)
    
    Returns:
        DataFrame with OHLCV data
    """
    return fetch_market_data(
        client=client,
        symbol=symbol,
        period=period,
        interval=interval
    )

def get_available_symbols(client: YFinanceClient) -> Dict[str, List[str]]:
    """
    Get list of available symbols by category
    
    Args:
        client: YFinance client instance
    
    Returns:
        Dictionary with symbol categories
    """
    try:
        return client.get_available_symbols()
    except Exception as e:
        print(f"Error fetching available symbols: {e}")
        return {}

def search_symbols(client: YFinanceClient, query: str) -> List[Dict]:
    """
    Search for symbols by name or ticker
    
    Args:
        client: YFinance client instance
        query: Search query
    
    Returns:
        List of matching symbols
    """
    try:
        return client.search_symbols(query)
    except Exception as e:
        print(f"Error searching symbols: {e}")
        return []

def get_market_hours() -> Dict[str, str]:
    """
    Get market hours information
    
    Returns:
        Dictionary with market hours information
    """
    return {
        "us_stocks": "9:30 AM - 4:00 PM EST (Monday-Friday)",
        "forex": "24/5 (Sunday 5 PM EST to Friday 5 PM EST)",
        "crypto": "24/7",
        "commodities": "Various hours depending on exchange",
        "note": "Market hours may vary by exchange and instrument type"
    }

def get_symbol_info(client: YFinanceClient, symbol: str) -> Dict:
    """
    Get detailed information about a symbol
    
    Args:
        client: YFinance client instance
        symbol: Stock/ETF/Crypto symbol
    
    Returns:
        Dictionary with symbol information
    """
    try:
        return client.get_ticker_info(symbol)
    except Exception as e:
        print(f"Error fetching symbol info for {symbol}: {e}")
        return {}

def get_realtime_price(client: YFinanceClient, symbol: str) -> Dict:
    """
    Get real-time price information
    
    Args:
        client: YFinance client instance
        symbol: Stock/ETF/Crypto symbol
    
    Returns:
        Dictionary with current price information
    """
    try:
        return client.get_realtime_price(symbol)
    except Exception as e:
        print(f"Error fetching real-time price for {symbol}: {e}")
        return {}
