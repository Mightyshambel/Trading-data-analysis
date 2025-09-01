"""
RSI Mean Reversion Strategy

A mean reversion strategy that generates buy/sell signals based on RSI (Relative Strength Index)
overbought and oversold conditions. This strategy works well in sideways markets and can
capture short-term price reversals.

Strategy Logic:
- Buy Signal: When RSI crosses above oversold threshold (typically 30)
- Sell Signal: When RSI crosses below overbought threshold (typically 70)
- Exit Signal: When RSI returns to neutral levels (around 50)

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
class RSIStrategyConfig:
    """Configuration for the RSI Mean Reversion strategy."""
    rsi_period: int = 14
    oversold_threshold: float = 30.0
    overbought_threshold: float = 70.0
    neutral_level: float = 50.0
    position_size: float = 0.1  # Percentage of capital to risk
    stop_loss: Optional[float] = 0.05  # Stop loss percentage
    take_profit: Optional[float] = 0.10  # Take profit percentage
    max_positions: int = 1
    use_divergence: bool = True  # Use RSI divergence for additional signals


@dataclass
class RSISignal:
    """Container for RSI-based signals."""
    timestamp: datetime
    signal_type: str  # 'BUY', 'SELL', 'EXIT'
    rsi_value: float
    price: float
    confidence: float
    reason: str


class RSIMeanReversionStrategy:
    """
    RSI Mean Reversion Strategy Implementation.
    
    This strategy generates trading signals based on RSI overbought and oversold conditions.
    It's a mean reversion strategy that works well in sideways markets and can capture
    short-term price reversals.
    
    Features:
    - Configurable RSI thresholds for overbought/oversold conditions
    - RSI divergence detection for enhanced signals
    - Built-in risk management with stop loss and take profit
    - Signal confidence scoring based on multiple factors
    - Comprehensive trade tracking and performance analysis
    """
    
    def __init__(self, config: RSIStrategyConfig):
        """
        Initialize the strategy with configuration.
        
        Args:
            config: Strategy configuration parameters
        """
        self.config = config
        self.positions: List[Dict] = []
        self.trade_history: List[Dict] = []
        self.signals: List[RSISignal] = []
        
    def calculate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RSI-based trading signals for the given data.
        
        Args:
            data: DataFrame with OHLCV data and technical indicators
            
        Returns:
            DataFrame with RSI trading signals added
        """
        df = data.copy()
        
        # Calculate RSI if not already present
        if 'RSI' not in df.columns:
            df['RSI'] = self._calculate_rsi(df['Close'], self.config.rsi_period)
        
        # Generate RSI signals
        df = self._generate_rsi_signals(df)
        
        # Add divergence signals if enabled
        if self.config.use_divergence:
            df = self._add_divergence_signals(df)
        
        # Add signal confidence
        df = self._calculate_signal_confidence(df)
        
        # Add position management
        df = self._add_position_management(df)
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _generate_rsi_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based trading signals."""
        rsi = df['RSI']
        
        # Initialize signal columns
        df['RSI_Signal'] = 0
        df['RSI_Signal_Type'] = ''
        df['RSI_Confidence'] = 0.0
        
        # Detect oversold and overbought conditions
        for i in range(1, len(df)):
            current_rsi = rsi.iloc[i]
            prev_rsi = rsi.iloc[i-1]
            
            # Oversold condition (potential buy signal)
            if prev_rsi <= self.config.oversold_threshold and current_rsi > self.config.oversold_threshold:
                df.iloc[i, df.columns.get_loc('RSI_Signal')] = 1
                df.iloc[i, df.columns.get_loc('RSI_Signal_Type')] = 'BUY'
                
            # Overbought condition (potential sell signal)
            elif prev_rsi >= self.config.overbought_threshold and current_rsi < self.config.overbought_threshold:
                df.iloc[i, df.columns.get_loc('RSI_Signal')] = -1
                df.iloc[i, df.columns.get_loc('RSI_Signal_Type')] = 'SELL'
            
            # Exit signals when RSI returns to neutral
            elif (prev_rsi < self.config.neutral_level and current_rsi >= self.config.neutral_level) or \
                 (prev_rsi > self.config.neutral_level and current_rsi <= self.config.neutral_level):
                df.iloc[i, df.columns.get_loc('RSI_Signal')] = 0
                df.iloc[i, df.columns.get_loc('RSI_Signal_Type')] = 'EXIT'
        
        return df
    
    def _add_divergence_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI divergence signals for enhanced trading opportunities."""
        df['RSI_Divergence'] = 0
        df['Divergence_Type'] = ''
        
        # Look for divergence patterns
        for i in range(20, len(df)):  # Need enough data for divergence detection
            # Bullish divergence: Price makes lower low, RSI makes higher low
            if self._detect_bullish_divergence(df, i):
                df.iloc[i, df.columns.get_loc('RSI_Divergence')] = 1
                df.iloc[i, df.columns.get_loc('Divergence_Type')] = 'BULLISH'
                
            # Bearish divergence: Price makes higher high, RSI makes lower high
            elif self._detect_bearish_divergence(df, i):
                df.iloc[i, df.columns.get_loc('RSI_Divergence')] = -1
                df.iloc[i, df.columns.get_loc('Divergence_Type')] = 'BEARISH'
        
        return df
    
    def _detect_bullish_divergence(self, df: pd.DataFrame, index: int) -> bool:
        """Detect bullish divergence pattern."""
        if index < 20:
            return False
        
        # Look for recent price low and RSI low
        price_window = df['Close'].iloc[index-20:index+1]
        rsi_window = df['RSI'].iloc[index-20:index+1]
        
        # Find local minima
        price_min_idx = price_window.idxmin()
        rsi_min_idx = rsi_window.idxmin()
        
        # Check if price made lower low but RSI made higher low
        if price_min_idx > rsi_min_idx:  # Price low is more recent
            price_low = price_window.min()
            rsi_low = rsi_window.min()
            
            # Check previous low
            prev_price_window = df['Close'].iloc[index-40:index-20]
            prev_rsi_window = df['RSI'].iloc[index-40:index-20]
            
            if len(prev_price_window) > 0 and len(prev_rsi_window) > 0:
                prev_price_low = prev_price_window.min()
                prev_rsi_low = prev_rsi_window.min()
                
                # Bullish divergence: price lower low, RSI higher low
                if price_low < prev_price_low and rsi_low > prev_rsi_low:
                    return True
        
        return False
    
    def _detect_bearish_divergence(self, df: pd.DataFrame, index: int) -> bool:
        """Detect bearish divergence pattern."""
        if index < 20:
            return False
        
        # Look for recent price high and RSI high
        price_window = df['Close'].iloc[index-20:index+1]
        rsi_window = df['RSI'].iloc[index-20:index+1]
        
        # Find local maxima
        price_max_idx = price_window.idxmax()
        rsi_max_idx = rsi_window.idxmax()
        
        # Check if price made higher high but RSI made lower high
        if price_max_idx > rsi_max_idx:  # Price high is more recent
            price_high = price_window.max()
            rsi_high = rsi_window.max()
            
            # Check previous high
            prev_price_window = df['Close'].iloc[index-40:index-20]
            prev_rsi_window = df['RSI'].iloc[index-40:index-20]
            
            if len(prev_price_window) > 0 and len(prev_rsi_window) > 0:
                prev_price_high = prev_price_window.max()
                prev_rsi_high = prev_rsi_window.max()
                
                # Bearish divergence: price higher high, RSI lower high
                if price_high > prev_price_high and rsi_high < prev_rsi_high:
                    return True
        
        return False
    
    def _calculate_signal_confidence(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate confidence level for each RSI signal."""
        for i in range(len(df)):
            if df.iloc[i]['RSI_Signal'] != 0:
                confidence = self._calculate_rsi_confidence(df, i)
                df.iloc[i, df.columns.get_loc('RSI_Confidence')] = confidence
        
        return df
    
    def _calculate_rsi_confidence(self, df: pd.DataFrame, index: int) -> float:
        """Calculate confidence score for RSI signals."""
        confidence = 0.5  # Base confidence
        
        rsi_value = df.iloc[i]['RSI']
        signal_type = df.iloc[i]['RSI_Signal_Type']
        
        # Factor 1: RSI extremity
        if signal_type == 'BUY' and rsi_value < 25:  # Very oversold
            confidence += 0.2
        elif signal_type == 'SELL' and rsi_value > 75:  # Very overbought
            confidence += 0.2
        elif signal_type == 'BUY' and rsi_value < 30:  # Oversold
            confidence += 0.1
        elif signal_type == 'SELL' and rsi_value > 70:  # Overbought
            confidence += 0.1
        
        # Factor 2: RSI momentum
        if index > 0:
            rsi_change = df.iloc[i]['RSI'] - df.iloc[i-1]['RSI']
            if abs(rsi_change) > 5:  # Strong RSI movement
                confidence += 0.15
        
        # Factor 3: Volume confirmation
        if 'Volume' in df.columns:
            current_volume = df.iloc[i]['Volume']
            avg_volume = df['Volume'].rolling(window=20).mean().iloc[i]
            
            if current_volume > avg_volume * 1.5:  # High volume
                confidence += 0.15
            elif current_volume > avg_volume:  # Above average volume
                confidence += 0.1
        
        # Factor 4: Divergence confirmation
        if df.iloc[i]['RSI_Divergence'] != 0:
            confidence += 0.2
        
        # Factor 5: Price action confirmation
        if index > 0:
            price_change = (df.iloc[i]['Close'] - df.iloc[i-1]['Close']) / df.iloc[i-1]['Close']
            if signal_type == 'BUY' and price_change > 0.01:  # Price moving up on buy signal
                confidence += 0.1
            elif signal_type == 'SELL' and price_change < -0.01:  # Price moving down on sell signal
                confidence += 0.1
        
        return min(confidence, 1.0)  # Cap at 1.0
    
    def _add_position_management(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add position management for RSI strategy."""
        df['RSI_Position_Action'] = ''
        df['RSI_Position_Size'] = 0.0
        df['RSI_Stop_Loss'] = 0.0
        df['RSI_Take_Profit'] = 0.0
        
        current_position = None
        
        for i in range(len(df)):
            signal = df.iloc[i]['RSI_Signal']
            signal_type = df.iloc[i]['RSI_Signal_Type']
            price = df.iloc[i]['Close']
            rsi_value = df.iloc[i]['RSI']
            
            # Handle new signals
            if signal == 1 and signal_type == 'BUY' and current_position is None:
                # Open long position
                current_position = {
                    'entry_time': df.index[i],
                    'entry_price': price,
                    'position_type': 'LONG',
                    'size': self.config.position_size,
                    'entry_rsi': rsi_value
                }
                
                df.iloc[i, df.columns.get_loc('RSI_Position_Action')] = 'OPEN_LONG'
                df.iloc[i, df.columns.get_loc('RSI_Position_Size')] = self.config.position_size
                
                # Set stop loss and take profit
                if self.config.stop_loss:
                    stop_loss = price * (1 - self.config.stop_loss)
                    df.iloc[i, df.columns.get_loc('RSI_Stop_Loss')] = stop_loss
                    current_position['stop_loss'] = stop_loss
                
                if self.config.take_profit:
                    take_profit = price * (1 + self.config.take_profit)
                    df.iloc[i, df.columns.get_loc('RSI_Take_Profit')] = take_profit
                    current_position['take_profit'] = take_profit
            
            elif signal == -1 and signal_type == 'SELL' and current_position is None:
                # Open short position
                current_position = {
                    'entry_time': df.index[i],
                    'entry_price': price,
                    'position_type': 'SHORT',
                    'size': self.config.position_size,
                    'entry_rsi': rsi_value
                }
                
                df.iloc[i, df.columns.get_loc('RSI_Position_Action')] = 'OPEN_SHORT'
                df.iloc[i, df.columns.get_loc('RSI_Position_Size')] = self.config.position_size
                
                # Set stop loss and take profit
                if self.config.stop_loss:
                    stop_loss = price * (1 + self.config.stop_loss)
                    df.iloc[i, df.columns.get_loc('RSI_Stop_Loss')] = stop_loss
                    current_position['stop_loss'] = stop_loss
                
                if self.config.take_profit:
                    take_profit = price * (1 - self.config.take_profit)
                    df.iloc[i, df.columns.get_loc('RSI_Take_Profit')] = take_profit
                    current_position['take_profit'] = take_profit
            
            # Handle position exits
            elif current_position is not None:
                should_exit = False
                exit_reason = ''
                
                # Exit signal
                if signal_type == 'EXIT':
                    should_exit = True
                    exit_reason = 'RSI_NEUTRAL'
                
                # Stop loss hit
                elif current_position.get('stop_loss'):
                    if (current_position['position_type'] == 'LONG' and price <= current_position['stop_loss']) or \
                       (current_position['position_type'] == 'SHORT' and price >= current_position['stop_loss']):
                        should_exit = True
                        exit_reason = 'STOP_LOSS'
                
                # Take profit hit
                elif current_position.get('take_profit'):
                    if (current_position['position_type'] == 'LONG' and price >= current_position['take_profit']) or \
                       (current_position['position_type'] == 'SHORT' and price <= current_position['take_profit']):
                        should_exit = True
                        exit_reason = 'TAKE_PROFIT'
                
                # RSI extreme exit (optional)
                if current_position['position_type'] == 'LONG' and rsi_value > 80:
                    should_exit = True
                    exit_reason = 'RSI_EXTREME_OVERBOUGHT'
                elif current_position['position_type'] == 'SHORT' and rsi_value < 20:
                    should_exit = True
                    exit_reason = 'RSI_EXTREME_OVERSOLD'
                
                if should_exit:
                    # Record the trade
                    trade_result = {
                        'entry_time': current_position['entry_time'],
                        'exit_time': df.index[i],
                        'entry_price': current_position['entry_price'],
                        'exit_price': price,
                        'entry_rsi': current_position['entry_rsi'],
                        'exit_rsi': rsi_value,
                        'position_type': current_position['position_type'],
                        'size': current_position['size'],
                        'exit_reason': exit_reason,
                        'pnl': self._calculate_rsi_pnl(current_position, price)
                    }
                    self.trade_history.append(trade_result)
                    
                    df.iloc[i, df.columns.get_loc('RSI_Position_Action')] = f'CLOSE_{current_position["position_type"]}'
                    current_position = None
        
        return df
    
    def _calculate_rsi_pnl(self, position: Dict, exit_price: float) -> float:
        """Calculate profit/loss for RSI position."""
        if position['position_type'] == 'LONG':
            return (exit_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            return (position['entry_price'] - exit_price) / position['entry_price']
    
    def get_strategy_summary(self) -> Dict:
        """Get a summary of the RSI strategy performance."""
        if not self.trade_history:
            return {'message': 'No trades executed yet'}
        
        total_trades = len(self.trade_history)
        winning_trades = len([t for t in self.trade_history if t['pnl'] > 0])
        losing_trades = len([t for t in self.trade_history if t['pnl'] < 0])
        
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        avg_pnl = total_pnl / total_trades
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # Analyze exit reasons
        exit_reasons = {}
        for trade in self.trade_history:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl,
            'best_trade': max(self.trade_history, key=lambda x: x['pnl']) if self.trade_history else None,
            'worst_trade': min(self.trade_history, key=lambda x: x['pnl']) if self.trade_history else None,
            'exit_reasons': exit_reasons
        }


def run_rsi_strategy_example():
    """
    Example of how to use the RSI Mean Reversion Strategy.
    
    This function demonstrates how to:
    1. Initialize the RSI strategy with configuration
    2. Load and prepare data
    3. Calculate RSI signals
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
    
    # Configure RSI strategy
    config = RSIStrategyConfig(
        rsi_period=14,
        oversold_threshold=30.0,
        overbought_threshold=70.0,
        neutral_level=50.0,
        position_size=0.1,  # 10% of capital per trade
        stop_loss=0.05,     # 5% stop loss
        take_profit=0.10,   # 10% take profit
        use_divergence=True
    )
    
    # Initialize strategy
    strategy = RSIMeanReversionStrategy(config)
    
    # Calculate signals
    signals_df = strategy.calculate_signals(data_with_indicators)
    
    # Get strategy summary
    summary = strategy.get_strategy_summary()
    
    print("=== RSI Mean Reversion Strategy Results ===")
    print(f"Total Trades: {summary.get('total_trades', 0)}")
    print(f"Win Rate: {summary.get('win_rate', 0):.2%}")
    print(f"Total P&L: {summary.get('total_pnl', 0):.2%}")
    print(f"Average P&L per Trade: {summary.get('avg_pnl', 0):.2%}")
    
    if 'exit_reasons' in summary:
        print("\nExit Reasons:")
        for reason, count in summary['exit_reasons'].items():
            print(f"  {reason}: {count}")
    
    return signals_df, summary


if __name__ == "__main__":
    # Run the example
    signals_df, summary = run_rsi_strategy_example()
