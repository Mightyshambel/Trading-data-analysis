# Quick Start Guide - Trading Data Analysis

## 🚀 Get Started in 5 Minutes

### 1. Set Up Your Environment

First, make sure you have your virtual environment activated:
```bash
source venv/bin/activate
```

### 2. Yahoo Finance Setup

**No setup required!** Yahoo Finance is completely free and doesn't require any API keys or accounts.

### 3. Create Your Environment File (Optional)

Copy the example environment file for custom settings:
```bash
cp env_example.txt .env
```

Edit `.env` with your preferred defaults (optional):
```bash
DEFAULT_INSTRUMENT=AAPL
DEFAULT_GRANULARITY=1d
```

### 4. Start Jupyter Lab

```bash
jupyter lab
```

Your browser will open to Jupyter Lab. Navigate to `notebooks/01_data_exploration.ipynb`

### 5. Run Your First Analysis

Open the notebook and run the cells to:
- ✅ Connect to Yahoo Finance API
- 📊 Fetch Apple stock data
- 📈 Add technical indicators (SMA, EMA, RSI, MACD, etc.)
- 📊 Create beautiful charts
- 💾 Save data for further analysis

## 🎯 What You Can Do Now

### Basic Data Fetching
```python
from src.yfinance_client import YFinanceClient
from src.data_fetcher import fetch_market_data

# Initialize client
client = YFinanceClient()

# Fetch Apple stock data
data = fetch_market_data(client, 'AAPL', '1y', '1d')
```

### Technical Analysis
```python
from src.technical_indicators import add_technical_indicators

# Add all indicators
data_with_indicators = add_technical_indicators(data)
```

### Available Instruments
- **Stocks**: AAPL, MSFT, GOOGL, TSLA, META, NVDA
- **Forex**: EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X
- **Crypto**: BTC-USD, ETH-USD, ADA-USD, SOL-USD
- **Commodities**: GC=F (Gold), CL=F (Oil), SI=F (Silver)
- **Indices**: ^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow)
- **ETFs**: SPY, QQQ, DIA, VTI, VEA, VWO

### Timeframes
- **Intraday**: 1m, 2m, 5m, 15m, 30m, 60m, 90m
- **Daily**: 1d, 5d, 1wk, 1mo
- **Long-term**: 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max

## 🔧 Troubleshooting

### Common Issues

1. **Import Error**: Make sure you're in the project root directory
2. **API Connection Failed**: Check your internet connection
3. **No Data Received**: Verify the symbol exists and try a different timeframe

### Need Help?

- Check the main `README.md` for detailed documentation
- Run `python test_setup.py` to verify your installation
- Ensure all packages are installed: `pip install -r requirements.txt`

## 🚀 Next Steps

After mastering the basics:

1. **Explore Different Currency Pairs**
2. **Try Different Timeframes**
3. **Develop Trading Strategies**
4. **Build Backtesting Systems**
5. **Create Automated Trading Bots**

## ⚠️ Important Notes

- **This is for educational purposes only**
- **Always use proper risk management**
- **Test strategies thoroughly before live trading**
- **Keep your API credentials secure**
- **Start with demo accounts**

---

**Happy Trading! 📈💰**
