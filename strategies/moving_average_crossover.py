"""
Moving Average Crossover Strategy

A classic trend-following strategy that generates buy/sell signals based on
the crossover of two moving averages. This strategy is widely used by both
retail and institutional traders.

Strategy Logic:
- Buy Signal: When the fast moving average crosses above the slow moving average
- Sell Signal: When the fast moving average crosses below the slow moving average
- Exit Signal: When the opposite crossover occurs

Author: The Almighty
License: MIT
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class StrategyConfig:
    """Configuration for the Moving Average Crossover strategy."""
    fast_period: int = 20
    slow_period: int = 50
    position_size: float = 1.0  # Percentage of capital to risk
    stop_loss: Optional[float] = None  # Stop loss percentage
    take_profit: Optional[float] = None  # Take profit percentage
    max_positions: int = 1  # Maximum concurrent positions


@dataclass
class TradeSignal:
    """Container for trade signals."""
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'EXIT'
    price: float
    confidence: float
    reason: str


@dataclass
class Position:
    """Container for trading positions."""
    entry_time: datetime
    entry_price: float
    position_type: str  # 'LONG', 'SHORT'
    size: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


class MovingAverageCrossoverStrategy:
    """
    Moving Average Crossover Strategy Implementation.
    
    This strategy generates trading signals based on the crossover of two moving averages.
    It's a trend-following strategy that works well in trending markets but may generate
    false signals in sideways markets.
    
    Features:
    - Configurable fast and slow moving average periods
    - Built-in risk management with stop loss and take profit
    - Position sizing based on risk percentage
    - Signal confidence scoring
    - Comprehensive trade tracking
    """
    
    def __init__(self, config: StrategyConfig):
        """
        Initialize the strategy with configuration.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.positions: List[Position] = []
        self.trade_history: List[Dict] = []
        self.signals: List[TradeSignal] = []
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trading signals for the given data.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with trading signals added
        """
        df = data.copy()
        
        # Calculate moving averages if not already present
        if f'SMA_{self.config.fast_period}' not in df.columns:
            df[f'SMA_{self.config.fast_period}'] = df['Close'].rolling(window=self.config.fast_period).mean()
        
        if f'SMA_{self.config.slow_period}' not in df.columns:
            df[f'SMA_{self.config.slow_period}'] = df['Close'].rolling(window=self.config.slow_period).mean()
        
        # Generate crossover signals
        df = self._generate_crossover_signals(df)
        
        # Add signal confidence
        df = self._calculate_signal_confidence(df)
        
        # Add position management
        df = self._add_position_management(df)
        
        return df
    
    def _generate_crossover_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate crossover signals based on moving averages."""
        fast_ma = df[f'SMA_{self.config.fast_period}']
        slow_ma = df[f'SMA_{self.config.slow_period}']
        
        # Initialize signal columns
        df['MA_Signal'] = 0
        df['MA_Signal_Type'] = ''
        df['MA_Confidence'] = 0.0
        
        # Detect crossovers
        for i in range(1, len(df)):
            current_fast = fast_ma.iloc[i]
            current_slow = slow_ma.iloc[i]
            prev_fast = fast_ma.iloc[i-1]
            prev_slow = slow_ma.iloc[i-1]
            
            # Bullish crossover (fast MA crosses above slow MA)
            if prev_fast <= prev_slow and current_fast > current_slow:
                df.iloc[i, df.columns.get_loc('MA_Signal')] = 1
                df.iloc[i, df.columns.get_loc('MA_Signal_Type')] = 'BUY'
                
            # Bearish crossover (fast MA crosses below slow MA)
            elif prev_fast >= prev_slow and current_fast < current_slow:
                df.iloc[i, df.columns.get_loc('MA_Signal')] = -1
                df.iloc[i, df.columns.get_loc('MA_Signal_Type')] = 'SELL'
        
        return df
    
    def _calculate_signal_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence level for each signal."""
        for i in range(len(df)):
            if df.iloc[i]['MA_Signal'] != 0:
                confidence = self._calculate_confidence_score(df, i)
                df.iloc[i, df.columns.get_loc('MA_Confidence')] = confidence
        
        return df
    
    def _calculate_confidence_score(self, df: pd.DataFrame, index: int) -> float:
        """Calculate confidence score for a signal based on multiple factors."""
        confidence = 0.5  # Base confidence
        
        # Factor 1: Trend strength (difference between MAs)
        fast_ma = df.iloc[index][f'SMA_{self.config.fast_period}']
        slow_ma = df.iloc[index][f'SMA_{self.config.slow_period}']
        ma_diff = abs(fast_ma - slow_ma) / slow_ma
        
        if ma_diff > 0.05:  # Strong trend
            confidence += 0.2
        elif ma_diff > 0.02:  # Moderate trend
            confidence += 0.1
        
        # Factor 2: Volume confirmation
        if 'Volume' in df.columns:
            current_volume = df.iloc[index]['Volume']
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[index]
            
            if current_volume > avg_volume * 1.5:  # High volume
                confidence += 0.15
            elif current_volume > avg_volume:  # Above average volume
                confidence += 0.1
        
        # Factor 3: Price momentum
        if index > 0:
            price_change = (df.iloc[index]['Close'] - df.iloc[index-1]['Close']) / df.iloc[index-1]['Close']
            if abs(price_change) > 0.02:  # Significant price movement
                confidence += 0.1
        
        # Factor 4: Market volatility
        if 'ATR' in df.columns:
            atr = df.iloc[index]['ATR']
            price = df.iloc[index]['Close']
            volatility = atr / price
            
            if 0.01 < volatility < 0.05:  # Optimal volatility range
                confidence += 0.1
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _add_position_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position management signals."""
        df['Position_Action'] = ''
        df['Position_Size'] = 0.0
        df['Stop_Loss'] = 0.0
        df['Take_Profit'] = 0.0
        
        current_position = None
        
        for i in range(len(df)):
            signal = df.iloc[i]['MA_Signal']
            signal_type = df.iloc[i]['MA_Signal_Type']
            price = df.iloc[i]['Close']
            
            # Handle new signals
            if signal == 1 and signal_type == 'BUY' and current_position is None:
                # Open long position
                current_position = Position(
                    entry_time=df.index[i],
                    entry_price=price,
                    position_type='LONG',
                    size=self.config.position_size
                )
                
                df.iloc[i, df.columns.get_loc('Position_Action')] = 'OPEN_LONG'
                df.iloc[i, df.columns.get_loc('Position_Size')] = self.config.position_size
                
                # Set stop loss and take profit
                if self.config.stop_loss:
                    stop_loss = price * (1 - self.config.stop_loss)
                    df.iloc[i, df.columns.get_loc('Stop_Loss')] = stop_loss
                    current_position.stop_loss = stop_loss
                
                if self.config.take_profit:
                    take_profit = price * (1 + self.config.take_profit)
                    df.iloc[i, df.columns.get_loc('Take_Profit')] = take_profit
                    current_position.take_profit = take_profit
            
            elif signal == -1 and signal_type == 'SELL' and current_position is None:
                # Open short position
                current_position = Position(
                    entry_time=df.index[i],
                    entry_price=price,
                    position_type='SHORT',
                    size=self.config.position_size
                )
                
                df.iloc[i, df.columns.get_loc('Position_Action')] = 'OPEN_SHORT'
                df.iloc[i, df.columns.get_loc('Position_Size')] = self.config.position_size
                
                # Set stop loss and take profit
                if self.config.stop_loss:
                    stop_loss = price * (1 + self.config.stop_loss)
                    df.iloc[i, df.columns.get_loc('Stop_Loss')] = stop_loss
                    current_position.stop_loss = stop_loss
                
                if self.config.take_profit:
                    take_profit = price * (1 - self.config.take_profit)
                    df.iloc[i, df.columns.get_loc('Take_Profit')] = take_profit
                    current_position.take_profit = take_profit
            
            # Handle position exits
            elif current_position is not None:
                # Check for exit signals
                should_exit = False
                exit_reason = ''
                
                # Opposite crossover signal
                if (current_position.position_type == 'LONG' and signal == -1) or \
                   (current_position.position_type == 'SHORT' and signal == 1):
                    should_exit = True
                    exit_reason = 'CROSSOVER_EXIT'
                
                # Stop loss hit
                elif current_position.stop_loss:
                    if (current_position.position_type == 'LONG' and price <= current_position.stop_loss) or \
                       (current_position.position_type == 'SHORT' and price >= current_position.stop_loss):
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                
                # Take profit hit
                elif current_position.take_profit:
                    if (current_position.position_type == 'LONG' and price >= current_position.take_profit) or \
                       (current_position.position_type == 'SHORT' and price <= current_position.take_profit):
                        should_exit = True
                        exit_reason = 'TAKE_PROFIT'
                
                if should_exit:
                    # Record the trade
                    trade_result = {
                        'entry_time': current_position.entry_time,
                        'exit_time': df.index[i],
                        'entry_price': current_position.entry_price,
                        'exit_price': price,
                        'position_type': current_position.position_type,
                        'size': current_position.size,
                        'exit_reason': exit_reason,
                        'pnl': self._calculate_pnl(current_position, price)
                    }
                    self.trade_history.append(trade_result)
                    
                    df.iloc[i, df.columns.get_loc('Position_Action')] = f'CLOSE_{current_position.position_type}'
                    current_position = None
        
        return df
    
    def _calculate_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate profit/loss for a position."""
        if position.position_type == 'LONG':
            return (exit_price - position.entry_price) / position.entry_price
        else:  # SHORT
            return (position.entry_price - exit_price) / position.entry_price
    
    def get_strategy_summary(self) -> Dict:
        """Get a summary of the strategy performance."""
        if not self.trade_history:
            return {'message': 'No trades executed yet'}
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        losing_trades = len([t for t in self.trade_history if t['pnl'] < 0])
        
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        avg_pnl = total_pnl / total_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'best_trade': max(self.trade_history, key=lambda x: x['pnl']) if self.trade_history else None,
            'worst_trade': min(self.trade_history, key=lambda x: x['pnl']) if self.trade_history else None
        }


def run_strategy_example():
    """
    Example of how to use the Moving Average Crossover Strategy.
    
    This function demonstrates how to:
    1. Initialize the strategy with configuration
    2. Load and prepare data
    3. Calculate signals
    4. Analyze results
    """
    from src.yfinance_client import YFinanceClient
    from src.data_fetcher import fetch_market_data
    from src.technical_indicators import add_technical_indicators
    
    # Initialize client and fetch data
    client = YFinanceClient()
    data = fetch_market_data(client, 'AAPL', '1y', '1d', use_sample_data=True)
    
    # Add technical indicators
    data_with_indicators = add_technical_indicators(data)
    
    # Configure strategy
    config = StrategyConfig(
        fast_period=20,
        slow_period=50,
        position_size=0.1,  # 10% of capital per trade
        stop_loss=0.05,     # 5% stop loss
        take_profit=0.15    # 15% take profit
    )
    
    # Initialize strategy
    strategy = MovingAverageCrossoverStrategy(config)
    
    # Calculate signals
    signals_df = strategy.calculate_signals(data_with_indicators)
    
    # Get strategy summary
    summary = strategy.get_strategy_summary()
    
    print("=== Moving Average Crossover Strategy Results ===")
    print(f"Total Trades: {summary.get('total_trades', 0)}")
    print(f"Win Rate: {summary.get('win_rate', 0):.2%}")
    print(f"Total P&L: {summary.get('total_pnl', 0):.2%}")
    print(f"Average P&L per Trade: {summary.get('avg_pnl', 0):.2%}")
    
    return signals_df, summary


if __name__ == "__main__":
    # Run the example
    signals_df, summary = run_strategy_example()
