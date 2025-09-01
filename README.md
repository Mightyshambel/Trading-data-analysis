# 🚀 Trading Data Analysis System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive, production-ready trading data analysis system built with Python, Yahoo Finance API, and NumPy for market analysis, backtesting, and strategy development across stocks, forex, crypto, and commodities.

## 📊 Live Demo

🌐 **View Live Analysis Reports:**
- [1-Year Analysis Report](charts_report.html) - Comprehensive market analysis with technical indicators
- [5-Year Analysis Report](charts_report_5year.html) - Long-term trends and institutional insights

## ✨ Key Features

- 🔌 **Real-time Data Fetching**: Connect to Yahoo Finance API for live market data
- 📈 **Multi-Asset Support**: Stocks, ETFs, forex, cryptocurrencies, commodities, and indices
- 📊 **Technical Analysis**: 20+ technical indicators implemented with NumPy
- 📉 **Data Visualization**: Interactive charts with Plotly and Matplotlib
- 🎯 **Trading Strategies**: Pre-built strategies with backtesting framework
- 🛡️ **Risk Management**: Position sizing and risk calculation tools
- 📓 **Jupyter Notebooks**: Interactive analysis and development environment
- 🧪 **Comprehensive Testing**: 80%+ test coverage with manual testing

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Mightyshambel/Trading-data-analysis.git
cd Trading-data-analysis

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
from src.yfinance_client import YFinanceClient
from src.data_fetcher import fetch_market_data
from src.technical_indicators import add_technical_indicators

# Initialize client
client = YFinanceClient()

# Fetch Apple stock data
data = fetch_market_data(client, 'AAPL', '1y', '1d')

# Add technical indicators
data_with_indicators = add_technical_indicators(data)

# Analyze market conditions
current_price = data_with_indicators['Close'].iloc[-1]
rsi = data_with_indicators['RSI'].iloc[-1]
print(f"Price: ${current_price:.2f}, RSI: {rsi:.2f}")
```

### Advanced Usage with Trading Strategies

```python
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy, StrategyConfig

# Configure strategy
config = StrategyConfig(
    fast_period=20,
    slow_period=50,
    position_size=0.1,  # 10% of capital per trade
    stop_loss=0.05,     # 5% stop loss
    take_profit=0.15    # 15% take profit
)

# Initialize and run strategy
strategy = MovingAverageCrossoverStrategy(config)
signals_df = strategy.calculate_signals(data_with_indicators)

# Get strategy performance
summary = strategy.get_strategy_summary()
print(f"Total Trades: {summary['total_trades']}")
print(f"Win Rate: {summary['win_rate']:.2%}")
```

## 📈 Code Examples

### 1. Data Fetching and Analysis

```python
# Fetch data for multiple instruments
symbols = ['AAPL', 'MSFT', 'GOOGL', 'BTC-USD', 'EURUSD=X']
all_data = {}

for symbol in symbols:
    data = fetch_market_data(client, symbol, '1y', '1d')
    data_with_indicators = add_technical_indicators(data)
    all_data[symbol] = data_with_indicators

# Compare performance across instruments
for symbol, data in all_data.items():
    total_return = (data['Close'].iloc[-1] - data['Close'].iloc[0]) / data['Close'].iloc[0]
    print(f"{symbol}: {total_return:.2%} return")
```

### 2. Technical Analysis

```python
# Generate trading signals
from src.technical_indicators import generate_trading_signals

signals = generate_trading_signals(data_with_indicators)

# Analyze signal strength
strong_buy_signals = signals[signals['Combined_Signal'] >= 2]
strong_sell_signals = signals[signals['Combined_Signal'] <= -2]

print(f"Strong buy signals: {len(strong_buy_signals)}")
print(f"Strong sell signals: {len(strong_sell_signals)}")
```

### 3. Risk Management

```python
# Calculate position size based on risk
def calculate_position_size(capital, risk_per_trade, stop_loss_pct):
    risk_amount = capital * risk_per_trade
    position_size = risk_amount / stop_loss_pct
    return position_size

# Example usage
capital = 10000
risk_per_trade = 0.02  # 2% risk per trade
stop_loss_pct = 0.05   # 5% stop loss

position_size = calculate_position_size(capital, risk_per_trade, stop_loss_pct)
print(f"Position size: ${position_size:.2f}")
```

## 📊 Available Instruments

### Stocks & ETFs
- **US Stocks**: AAPL, MSFT, GOOGL, TSLA, META, NVDA, NFLX
- **ETFs**: SPY, QQQ, DIA, VTI, VEA, VWO, IWM

### Forex
- **Major Pairs**: EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X
- **Commodity Pairs**: AUDUSD=X, USDCAD=X, NZDUSD=X

### Cryptocurrencies
- **Major Coins**: BTC-USD, ETH-USD, BNB-USD, ADA-USD, SOL-USD

### Commodities
- **Precious Metals**: GC=F (Gold), SI=F (Silver)
- **Energy**: CL=F (Oil), NG=F (Natural Gas)
- **Agriculture**: ZC=F (Corn), ZS=F (Soybeans)

### Indices
- **US Indices**: ^GSPC (S&P 500), ^IXIC (NASDAQ), ^DJI (Dow)
- **Volatility**: ^VIX (Fear Index), ^TNX (10-Year Treasury)

## 📈 Technical Indicators

### Moving Averages
- Simple Moving Average (SMA) - 10, 20, 50, 100, 200 periods
- Exponential Moving Average (EMA) - 12, 26, 50, 200 periods
- Weighted Moving Average (WMA) - 20 period

### Momentum Indicators
- Relative Strength Index (RSI) - 14 period
- Stochastic Oscillator - 14 period
- Williams %R - 14 period
- Rate of Change (ROC) - 10 period
- Money Flow Index (MFI) - 14 period

### Trend Indicators
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands - 20 period, 2 standard deviations
- Parabolic SAR
- Average Directional Index (ADX) - 14 period

### Volatility Indicators
- Average True Range (ATR) - 14 period
- Historical Volatility - 20 period
- Chaikin Volatility - 10 period

### Oscillators
- Commodity Channel Index (CCI) - 20 period
- Ultimate Oscillator - 7, 14, 28 periods
- Awesome Oscillator - 5, 34 periods

### Volume Indicators
- On-Balance Volume (OBV)
- Volume Price Trend (VPT)
- Accumulation/Distribution Line (ADL)
- Chaikin Money Flow (CMF) - 20 period

## 🎯 Trading Strategies

### 1. Moving Average Crossover Strategy
```python
# Trend-following strategy using MA crossovers
from strategies.moving_average_crossover import MovingAverageCrossoverStrategy

config = StrategyConfig(fast_period=20, slow_period=50)
strategy = MovingAverageCrossoverStrategy(config)
```

### 2. RSI Mean Reversion Strategy
```python
# Mean reversion strategy using RSI
from strategies.rsi_mean_reversion import RSIMeanReversionStrategy

config = RSIStrategyConfig(
    oversold_threshold=30,
    overbought_threshold=70,
    use_divergence=True
)
strategy = RSIMeanReversionStrategy(config)
```

## 🏗️ Project Structure

```
Trading-data-analysis/
├── src/                          # Core Python modules
│   ├── __init__.py              # Package initialization
│   ├── yfinance_client.py       # Yahoo Finance API client
│   ├── data_fetcher.py          # Data retrieval utilities
│   └── technical_indicators.py  # Technical analysis tools
├── strategies/                   # Trading strategies
│   ├── moving_average_crossover.py  # MA crossover strategy
│   └── rsi_mean_reversion.py    # RSI mean reversion strategy
├── tests/                       # Comprehensive test suite
│   ├── test_yfinance_client.py  # API client tests
│   └── test_technical_indicators.py  # Technical analysis tests
├── notebooks/                    # Jupyter notebooks
├── data/                        # Data storage
├── generated_charts/            # Generated visualizations

├── requirements.txt             # Python dependencies
├── CONTRIBUTING.md              # Contribution guidelines
└── README.md                    # This file
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_yfinance_client.py
```

## 🧪 Testing

The project includes comprehensive testing:

- ✅ **Unit Tests** - Test individual functions and classes
- ✅ **Integration Tests** - Test component interactions
- ✅ **Manual Testing** - Run tests locally with pytest
- ✅ **Code Quality** - Use development tools for quality checks

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Quick Contribution Steps:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following our coding standards
4. **Run tests locally**: `pytest`
5. **Submit a pull request**

## 📚 Documentation

- **[Quick Start Guide](QUICKSTART.md)** - Get started in 5 minutes
- **[Project Summary](PROJECT_SUMMARY.md)** - Comprehensive project overview
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. Trading involves substantial risk and is not suitable for all investors. Past performance does not guarantee future results. Always do your own research and use proper risk management.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free market data
- **NumPy** and **Pandas** for efficient data processing
- **Matplotlib** and **Plotly** for data visualization
- **The open-source community** for inspiration and support


---

**Built by The Almighty**

[![GitHub stars](https://img.shields.io/github/stars/Mightyshambel/Trading-data-analysis?style=social)](https://github.com/Mightyshambel/Trading-data-analysis/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/Mightyshambel/Trading-data-analysis?style=social)](https://github.com/Mightyshambel/Trading-data-analysis/network)
[![GitHub issues](https://img.shields.io/github/issues/Mightyshambel/Trading-data-analysis)](https://github.com/Mightyshambel/Trading-data-analysis/issues)
