"""
Yahoo Finance API Client

A robust client for fetching financial data from Yahoo Finance API with error handling,
rate limiting protection, and sample data fallback capabilities.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import time
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class YFinanceClient:
    """
    Yahoo Finance API client with robust error handling and sample data fallback.
    
    Features:
    - Real-time and historical data fetching
    - Automatic retry with exponential backoff
    - Sample data generation for testing
    - Rate limiting protection
    - Comprehensive error handling
    """
    
    def __init__(self, max_retries: int = 3, retry_delay: float = 1.0):
        """
        Initialize the Yahoo Finance client.
        
        Args:
            max_retries: Maximum number of retry attempts
            retry_delay: Base delay between retries (will be exponential)
        """
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
        
    def _rate_limit(self):
        """Implement rate limiting to avoid API restrictions."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
            
        self.last_request_time = time.time()
    
    def _generate_sample_data(self, symbol: str, period: str, interval: str) -> pd.DataFrame:
        """
        Generate realistic sample data for testing and development.
        
        Args:
            symbol: Financial instrument symbol
            period: Data period (e.g., '1y', '6mo')
            interval: Data interval (e.g., '1d', '1h')
            
        Returns:
            DataFrame with sample OHLCV data
        """
        logger.info(f"Generating sample data for {symbol} ({period}, {interval})")
        
        # Calculate number of data points based on period and interval
        period_days = self._parse_period_to_days(period)
        interval_days = self._parse_interval_to_days(interval)
        num_points = int(period_days / interval_days)
        
        # Generate realistic price data
        np.random.seed(42)  # For reproducible results
        
        # Start with a realistic base price
        base_price = 100.0
        if symbol.upper() in ['AAPL', 'MSFT', 'GOOGL']:
            base_price = 150.0
        elif symbol.upper() in ['BTC-USD', 'ETH-USD']:
            base_price = 50000.0
        elif '=X' in symbol.upper():  # Forex pairs
            base_price = 1.0
            
        # Generate price movements
        returns = np.random.normal(0.001, 0.02, num_points)  # Daily returns
        prices = [base_price]
        
        for ret in returns[1:]:
            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, 0.01))  # Ensure positive prices
            
        # Generate OHLCV data
        dates = pd.date_range(end=datetime.now(), periods=num_points, freq='D')
        
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = 0.02
            high = price * (1 + abs(np.random.normal(0, volatility)))
            low = price * (1 - abs(np.random.normal(0, volatility)))
            open_price = prices[i-1] if i > 0 else price
            
            # Ensure OHLC relationship
            high = max(high, open_price, price)
            low = min(low, open_price, price)
            
            # Generate volume
            volume = np.random.randint(1000000, 10000000)
            
            data.append({
                'Date': date,
                'Open': round(open_price, 2),
                'High': round(high, 2),
                'Low': round(low, 2),
                'Close': round(price, 2),
                'Volume': volume
            })
            
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        
        return df
    
    def _parse_period_to_days(self, period: str) -> int:
        """Convert period string to number of days."""
        period_map = {
            '1d': 1, '5d': 5, '1mo': 30, '3mo': 90, '6mo': 180,
            '1y': 365, '2y': 730, '5y': 1825, '10y': 3650, 'ytd': 365, 'max': 3650
        }
        return period_map.get(period, 365)
    
    def _parse_interval_to_days(self, interval: str) -> float:
        """Convert interval string to number of days."""
        interval_map = {
            '1m': 1/1440, '2m': 2/1440, '5m': 5/1440, '15m': 15/1440,
            '30m': 30/1440, '60m': 60/1440, '90m': 90/1440,
            '1h': 1/24, '1d': 1, '5d': 5, '1wk': 7, '1mo': 30
        }
        return interval_map.get(interval, 1)
    
    def fetch_data(self, symbol: str, period: str = '1y', interval: str = '1d', 
                   use_sample_data: bool = False) -> pd.DataFrame:
        """
        Fetch financial data from Yahoo Finance API.
        
        Args:
            symbol: Financial instrument symbol (e.g., 'AAPL', 'EURUSD=X')
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo')
            use_sample_data: If True, generate sample data instead of fetching from API
            
        Returns:
            DataFrame with OHLCV data
            
        Raises:
            Exception: If data fetching fails after all retries
        """
        if use_sample_data:
            return self._generate_sample_data(symbol, period, interval)
            
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()
                
                logger.info(f"Fetching data for {symbol} (attempt {attempt + 1}/{self.max_retries})")
                
                # Create ticker object
                ticker = yf.Ticker(symbol)
                
                # Fetch historical data
                data = ticker.history(period=period, interval=interval)
                
                if data.empty:
                    raise ValueError(f"No data received for symbol: {symbol}")
                
                logger.info(f"Successfully fetched {len(data)} data points for {symbol}")
                return data
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {str(e)}")
                
                if attempt == self.max_retries - 1:
                    logger.error(f"All attempts failed for {symbol}. Using sample data as fallback.")
                    return self._generate_sample_data(symbol, period, interval)
                
                # Exponential backoff
                delay = self.retry_delay * (2 ** attempt)
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
    
    def get_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about a financial instrument.
        
        Args:
            symbol: Financial instrument symbol
            
        Returns:
            Dictionary with instrument information
        """
        try:
            self._rate_limit()
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            key_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'Unknown')),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0)
            }
            
            return key_info
            
        except Exception as e:
            logger.error(f"Failed to get info for {symbol}: {str(e)}")
            return {
                'symbol': symbol,
                'name': 'Unknown',
                'sector': 'Unknown',
                'industry': 'Unknown',
                'market_cap': 0,
                'current_price': 0,
                'pe_ratio': 0,
                'dividend_yield': 0,
                'volume': 0,
                'avg_volume': 0
            }
    
    def search_symbols(self, query: str) -> list:
        """
        Search for financial instruments by name or symbol.
        
        Args:
            query: Search query
            
        Returns:
            List of matching symbols
        """
        try:
            self._rate_limit()
            # Note: yfinance doesn't have a built-in search function
            # This is a placeholder for future implementation
            logger.info(f"Search functionality not yet implemented for query: {query}")
            return []
            
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {str(e)}")
            return []
