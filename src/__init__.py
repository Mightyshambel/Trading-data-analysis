"""
Trading Data Analysis System

A comprehensive trading data analysis system built with Python, Yahoo Finance API, and NumPy
for market analysis, backtesting, and strategy development across stocks, forex, crypto, and commodities.

Author: The Almighty
License: MIT
"""

__version__ = "1.0.0"
__author__ = "The Almighty"
__license__ = "MIT"

from .yfinance_client import YFinanceClient
from .data_fetcher import fetch_market_data
from .technical_indicators import add_technical_indicators

__all__ = [
    'YFinanceClient',
    'fetch_market_data', 
    'add_technical_indicators'
]
