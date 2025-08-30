"""
Yahoo Finance Client for trading data analysis
"""

import yfinance as yf
import pandas as pd
from typing import Dict, List, Optional, Union
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class YFinanceClient:
    """Client for interacting with Yahoo Finance API"""
    
    def __init__(self):
        """Initialize Yahoo Finance client"""
        self.client = yf
        
    def get_ticker_info(self, symbol: str) -> Dict:
        """
        Get basic information about a ticker
        
        Args:
            symbol: Stock/ETF/Crypto symbol (e.g., 'AAPL', 'EURUSD=X', 'BTC-USD')
        
        Returns:
            Dictionary with ticker information
        """
        try:
            ticker = self.client.Ticker(symbol)
            info = ticker.info
            
            # Extract key information
            ticker_info = {
                'symbol': symbol,
                'name': info.get('longName', info.get('shortName', 'N/A')),
                'sector': info.get('sector', 'N/A'),
                'industry': info.get('industry', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'currency': info.get('currency', 'N/A'),
                'exchange': info.get('exchange', 'N/A'),
                'country': info.get('country', 'N/A'),
                'website': info.get('website', 'N/A'),
                'description': info.get('longBusinessSummary', 'N/A')
            }
            
            return ticker_info
            
        except Exception as e:
            print(f"Error fetching ticker info for {symbol}: {e}")
            return {}
    
    def get_historical_data(self, symbol: str, 
                           period: str = "1y",
                           interval: str = "1d",
                           start: Optional[Union[str, datetime]] = None,
                           end: Optional[Union[str, datetime]] = None) -> pd.DataFrame:
        """
        Get historical data for a symbol
        
        Args:
            symbol: Stock/ETF/Crypto symbol
            period: Data period ('1d', '5d', '1mo', '3mo', '6mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            interval: Data interval ('1m', '2m', '5m', '15m', '30m', '60m', '90m', '1h', '1d', '5d', '1wk', '1mo', '3mo')
            start: Start date (string or datetime)
            end: End date (string or datetime)
        
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = self.client.Ticker(symbol)
            
            # Add a small delay to avoid rate limiting
            import time
            time.sleep(0.5)
            
            if start and end:
                # Use specific date range
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                # Use period
                df = ticker.history(period=period, interval=interval)
            
            if df.empty:
                print(f"No data received for {symbol} - this may be due to rate limiting or temporary service issues")
                print("Try again in a few minutes or use a different symbol")
                return pd.DataFrame()
            
            # Rename columns to match our standard format
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    print(f"Warning: Missing column {col} for {symbol}")
            
            return df
            
        except Exception as e:
            print(f"Error fetching historical data for {symbol}: {e}")
            print("This may be due to:")
            print("- Rate limiting (try again in a few minutes)")
            print("- Temporary service issues")
            print("- Network connectivity problems")
            print("- Invalid symbol or timeframe")
            
            # Try one more time with a longer delay
            try:
                print("Retrying with longer delay...")
                time.sleep(2)
                ticker = self.client.Ticker(symbol)
                if start and end:
                    df = ticker.history(start=start, end=end, interval=interval)
                else:
                    df = ticker.history(period=period, interval=interval)
                
                if not df.empty:
                    df.columns = [col.lower() for col in df.columns]
                    print("Retry successful!")
                    return df
            except Exception as retry_e:
                print(f"Retry failed for {symbol}: {retry_e}")
            
            return pd.DataFrame()
    
    def get_realtime_price(self, symbol: str) -> Dict:
        """
        Get real-time price information
        
        Args:
            symbol: Stock/ETF/Crypto symbol
        
        Returns:
            Dictionary with current price information
        """
        try:
            ticker = self.client.Ticker(symbol)
            info = ticker.info
            
            price_info = {
                'symbol': symbol,
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 'N/A')),
                'previous_close': info.get('previousClose', 'N/A'),
                'open': info.get('open', 'N/A'),
                'day_high': info.get('dayHigh', 'N/A'),
                'day_low': info.get('dayLow', 'N/A'),
                'volume': info.get('volume', 'N/A'),
                'market_cap': info.get('marketCap', 'N/A'),
                'pe_ratio': info.get('trailingPE', 'N/A'),
                'dividend_yield': info.get('dividendYield', 'N/A'),
                'timestamp': datetime.now().isoformat()
            }
            
            return price_info
            
        except Exception as e:
            print(f"Error fetching real-time price for {symbol}: {e}")
            return {}
    
    def search_symbols(self, query: str) -> List[Dict]:
        """
        Search for symbols by name or ticker
        
        Args:
            query: Search query (company name, ticker, etc.)
        
        Returns:
            List of matching symbols
        """
        try:
            # This is a simplified search - yfinance doesn't have a built-in search
            # We'll return some common symbols that might match
            common_symbols = {
                'apple': ['AAPL', 'AAPL.US'],
                'microsoft': ['MSFT', 'MSFT.US'],
                'google': ['GOOGL', 'GOOG', 'GOOGL.US'],
                'amazon': ['AMZN', 'AMZN.US'],
                'tesla': ['TSLA', 'TSLA.US'],
                'bitcoin': ['BTC-USD', 'BTC-USD'],
                'ethereum': ['ETH-USD', 'ETH-USD'],
                'euro': ['EURUSD=X', 'EUR=X'],
                'pound': ['GBPUSD=X', 'GBP=X'],
                'yen': ['USDJPY=X', 'JPY=X'],
                'gold': ['GC=F', 'GLD'],
                'silver': ['SI=F', 'SLV'],
                'oil': ['CL=F', 'USO'],
                'sp500': ['^GSPC', 'SPY'],
                'nasdaq': ['^IXIC', 'QQQ'],
                'dow': ['^DJI', 'DIA']
            }
            
            query_lower = query.lower()
            matches = []
            
            for key, symbols in common_symbols.items():
                if query_lower in key or any(query_lower in sym.lower() for sym in symbols):
                    for symbol in symbols:
                        matches.append({
                            'symbol': symbol,
                            'name': key.title(),
                            'type': 'Stock' if '=' not in symbol and '-' not in symbol else 'Forex/Crypto'
                        })
            
            return matches[:10]  # Limit to 10 results
            
        except Exception as e:
            print(f"Error searching symbols: {e}")
            return []
    
    def get_available_symbols(self) -> Dict[str, List[str]]:
        """
        Get list of available symbol categories
        
        Returns:
            Dictionary with symbol categories
        """
        return {
            'stocks': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX'],
            'forex': ['EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'USDCAD=X'],
            'crypto': ['BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SOL-USD', 'DOT-USD'],
            'commodities': ['GC=F', 'SI=F', 'CL=F', 'NG=F', 'ZC=F', 'ZS=F'],
            'indices': ['^GSPC', '^IXIC', '^DJI', '^RUT', '^VIX', '^TNX'],
            'etfs': ['SPY', 'QQQ', 'DIA', 'IWM', 'VTI', 'VEA', 'VWO']
        }
    
    def test_connection(self) -> bool:
        """Test API connection by fetching a simple ticker"""
        try:
            # Try to fetch basic info for a well-known stock
            test_symbol = 'AAPL'
            ticker = self.client.Ticker(test_symbol)
            # Just check if we can create a ticker object
            return ticker is not None
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False
    
    def generate_sample_data(self, symbol: str = "SAMPLE", days: int = 100) -> pd.DataFrame:
        """
        Generate sample data for testing when API is not available
        
        Args:
            symbol: Symbol name for the sample data
            days: Number of days of data to generate
        
        Returns:
            DataFrame with sample OHLCV data
        """
        import numpy as np
        from datetime import datetime, timedelta
        
        # Generate sample dates
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Generate sample price data (random walk)
        np.random.seed(42)  # For reproducible results
        base_price = 100.0
        returns = np.random.normal(0, 0.02, len(dates))  # 2% daily volatility
        
        prices = [base_price]
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        # Generate OHLCV data
        data = []
        for i, (date, price) in enumerate(zip(dates, prices)):
            # Generate realistic OHLC from close price
            volatility = np.random.uniform(0.01, 0.03)
            high = price * (1 + volatility)
            low = price * (1 - volatility)
            open_price = np.random.uniform(low, high)
            
            # Generate volume (higher volume on price changes)
            base_volume = 1000000
            price_change = abs(price - prices[i-1]) if i > 0 else 0
            volume = base_volume + int(price_change * 10000000)
            
            data.append({
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(price, 2),
                'volume': volume
            })
        
        df = pd.DataFrame(data, index=dates)
        print(f"Generated sample data for {symbol}: {len(df)} days")
        print("Note: This is sample data for testing purposes only")
        
        return df
