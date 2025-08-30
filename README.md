# Trading Data Analysis with Yahoo Finance and Python

A comprehensive trading data analysis system built with Python, Yahoo Finance API, and NumPy for market analysis, backtesting, and strategy development across stocks, forex, crypto, and commodities.

## Features

- **Real-time Data Fetching**: Connect to Yahoo Finance API for live market data
- **Multi-Asset Support**: Stocks, ETFs, forex, cryptocurrencies, commodities, and indices
- **Technical Analysis**: Implement various technical indicators using NumPy
- **Data Visualization**: Create interactive charts with Plotly and Matplotlib
- **Backtesting Framework**: Test trading strategies on historical data
- **Risk Management**: Position sizing and risk calculation tools
- **Jupyter Notebooks**: Interactive analysis and development environment

## Prerequisites

- Python 3.9+
- Internet connection for Yahoo Finance API access
- No API key required!

## Installation

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd Trading-data-analysis
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables (optional):**
   Create a `.env` file in the root directory for custom settings:
   ```
   DEFAULT_INSTRUMENT=AAPL
   DEFAULT_GRANULARITY=1d
   ```

## Project Structure

```
Trading-data-analysis/
├── src/
│   ├── __init__.py
│   ├── yfinance_client.py   # Yahoo Finance API client
│   ├── data_fetcher.py      # Data retrieval functions
│   ├── technical_indicators.py  # Technical analysis tools
│   ├── backtesting.py       # Backtesting framework
│   └── risk_management.py   # Risk calculation tools
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_technical_analysis.ipynb
│   └── 03_strategy_backtesting.ipynb
├── data/                    # Data storage
├── strategies/              # Trading strategies
├── tests/                   # Unit tests
├── requirements.txt
└── README.md
```

## Quick Start

1. **Start Jupyter Lab:**
   ```bash
   jupyter lab
   ```

2. **Open the data exploration notebook:**
   - Navigate to `notebooks/01_data_exploration.ipynb`
   - Run the cells to fetch and analyze OANDA data

3. **Example usage:**
   ```python
   from src.yfinance_client import YFinanceClient
   from src.data_fetcher import fetch_market_data
   
   # Initialize client
   client = YFinanceClient()
   
   # Fetch Apple stock data
   data = fetch_market_data(client, 'AAPL', '1y', '1d')
   ```

## Getting Started with Yahoo Finance

1. **No account required** - Yahoo Finance is completely free!
2. **Wide asset coverage** - Access stocks, ETFs, forex, crypto, commodities, and indices
3. **Multiple timeframes** - From 1-minute intraday to monthly data
4. **Real-time data** - Live prices and market information

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational purposes only. Trading forex involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results.
