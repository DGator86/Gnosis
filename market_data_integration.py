"""
Gnosis Real Market Data Integration System
Complete real-time market data feeds for live options analysis
"""

import asyncio
import aiohttp
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor
import websocket
import threading
import queue
import time
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSource(Enum):
    """Available data sources for market data"""
    YAHOO_FINANCE = "yahoo"
    POLYGON = "polygon"
    ALPHA_VANTAGE = "alpha_vantage"
    TWELVE_DATA = "twelve_data"
    IEX_CLOUD = "iex_cloud"

@dataclass
class OptionContract:
    """Options contract data structure"""
    symbol: str
    strike: float
    expiry: str
    option_type: str  # 'call' or 'put'
    bid: float = 0.0
    ask: float = 0.0
    last: float = 0.0
    volume: int = 0
    open_interest: int = 0
    implied_volatility: float = 0.0
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    rho: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class MarketData:
    """Real-time market data structure"""
    symbol: str
    price: float
    bid: float
    ask: float
    volume: int
    high: float
    low: float
    open_price: float
    previous_close: float
    change: float
    change_percent: float
    timestamp: datetime = field(default_factory=datetime.now)

class RealTimeMarketData:
    """Real-time market data feed manager"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.api_keys = api_keys or {}
        self.subscribers = {}
        self.data_cache = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def connect(self, symbols: List[str], data_source: DataSource = DataSource.YAHOO_FINANCE):
        """Connect to real-time data feed"""
        self.running = True
        logger.info(f"Connecting to {data_source.value} for symbols: {symbols}")
        
        if data_source == DataSource.YAHOO_FINANCE:
            await self._connect_yahoo_finance(symbols)
        elif data_source == DataSource.POLYGON:
            await self._connect_polygon(symbols)
        else:
            logger.warning(f"Data source {data_source.value} not implemented yet")
            
    async def _connect_yahoo_finance(self, symbols: List[str]):
        """Connect to Yahoo Finance real-time data"""
        while self.running:
            try:
                for symbol in symbols:
                    ticker = yf.Ticker(symbol)
                    info = ticker.history(period="1d", interval="1m").iloc[-1]
                    
                    market_data = MarketData(
                        symbol=symbol,
                        price=info['Close'],
                        bid=info['Close'] * 0.999,  # Approximate bid
                        ask=info['Close'] * 1.001,  # Approximate ask
                        volume=int(info['Volume']),
                        high=info['High'],
                        low=info['Low'],
                        open_price=info['Open'],
                        previous_close=ticker.info.get('previousClose', info['Close']),
                        change=info['Close'] - ticker.info.get('previousClose', info['Close']),
                        change_percent=((info['Close'] / ticker.info.get('previousClose', info['Close'])) - 1) * 100
                    )
                    
                    self.data_cache[symbol] = market_data
                    await self._notify_subscribers(symbol, market_data)
                    
                await asyncio.sleep(1)  # Update every second
                
            except Exception as e:
                logger.error(f"Error in Yahoo Finance feed: {e}")
                await asyncio.sleep(5)
                
    async def _connect_polygon(self, symbols: List[str]):
        """Connect to Polygon.io real-time data"""
        if 'polygon' not in self.api_keys:
            logger.error("Polygon API key not provided")
            return
            
        # Polygon WebSocket implementation would go here
        logger.info("Polygon connection would be implemented here with WebSocket")
        
    async def _notify_subscribers(self, symbol: str, data: MarketData):
        """Notify all subscribers of new data"""
        if symbol in self.subscribers:
            for callback in self.subscribers[symbol]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")
                    
    def subscribe(self, symbol: str, callback):
        """Subscribe to real-time data for a symbol"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        self.subscribers[symbol].append(callback)
        
    def get_latest(self, symbol: str) -> Optional[MarketData]:
        """Get latest market data for a symbol"""
        return self.data_cache.get(symbol)
        
    def disconnect(self):
        """Disconnect from data feed"""
        self.running = False
        logger.info("Disconnected from market data feed")

class OptionsChainManager:
    """Options chain data manager with real-time Greeks"""
    
    def __init__(self, data_feed: RealTimeMarketData):
        self.data_feed = data_feed
        self.options_cache = {}
        
    async def get_options_chain(self, symbol: str, expiry_date: str = None) -> List[OptionContract]:
        """Get complete options chain for a symbol"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get available expiry dates
            if expiry_date is None:
                expiry_dates = ticker.options
                if not expiry_dates:
                    logger.warning(f"No options available for {symbol}")
                    return []
                expiry_date = expiry_dates[0]  # Use nearest expiry
                
            # Get options chain
            opt_chain = ticker.option_chain(expiry_date)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            contracts = []
            
            # Process calls
            for _, call in calls.iterrows():
                contract = OptionContract(
                    symbol=f"{symbol}_{expiry_date}_C_{call['strike']}",
                    strike=call['strike'],
                    expiry=expiry_date,
                    option_type='call',
                    bid=call.get('bid', 0.0),
                    ask=call.get('ask', 0.0),
                    last=call.get('lastPrice', 0.0),
                    volume=call.get('volume', 0),
                    open_interest=call.get('openInterest', 0),
                    implied_volatility=call.get('impliedVolatility', 0.0)
                )
                contracts.append(contract)
                
            # Process puts
            for _, put in puts.iterrows():
                contract = OptionContract(
                    symbol=f"{symbol}_{expiry_date}_P_{put['strike']}",
                    strike=put['strike'],
                    expiry=expiry_date,
                    option_type='put',
                    bid=put.get('bid', 0.0),
                    ask=put.get('ask', 0.0),
                    last=put.get('lastPrice', 0.0),
                    volume=put.get('volume', 0),
                    open_interest=put.get('openInterest', 0),
                    implied_volatility=put.get('impliedVolatility', 0.0)
                )
                contracts.append(contract)
                
            # Calculate Greeks for each contract
            for contract in contracts:
                await self._calculate_greeks(contract, symbol)
                
            self.options_cache[f"{symbol}_{expiry_date}"] = contracts
            return contracts
            
        except Exception as e:
            logger.error(f"Error getting options chain for {symbol}: {e}")
            return []
            
    async def _calculate_greeks(self, contract: OptionContract, underlying_symbol: str):
        """Calculate Greeks for an option contract"""
        try:
            # Get current underlying price
            underlying_data = self.data_feed.get_latest(underlying_symbol)
            if not underlying_data:
                return
                
            S = underlying_data.price  # Current stock price
            K = contract.strike       # Strike price
            T = self._time_to_expiry(contract.expiry)  # Time to expiry
            r = 0.05  # Risk-free rate (approximation)
            sigma = contract.implied_volatility  # Implied volatility
            
            if sigma <= 0 or T <= 0:
                return
                
            # Black-Scholes calculations
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            from scipy.stats import norm
            
            if contract.option_type == 'call':
                contract.delta = norm.cdf(d1)
                contract.theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) - 
                                r*K*np.exp(-r*T)*norm.cdf(d2)) / 365
            else:  # put
                contract.delta = norm.cdf(d1) - 1
                contract.theta = (-S*norm.pdf(d1)*sigma/(2*np.sqrt(T)) + 
                                r*K*np.exp(-r*T)*norm.cdf(-d2)) / 365
                
            # Greeks that are same for calls and puts
            contract.gamma = norm.pdf(d1) / (S*sigma*np.sqrt(T))
            contract.vega = S*norm.pdf(d1)*np.sqrt(T) / 100
            
            if contract.option_type == 'call':
                contract.rho = K*T*np.exp(-r*T)*norm.cdf(d2) / 100
            else:
                contract.rho = -K*T*np.exp(-r*T)*norm.cdf(-d2) / 100
                
        except Exception as e:
            logger.error(f"Error calculating Greeks: {e}")
            
    def _time_to_expiry(self, expiry_str: str) -> float:
        """Calculate time to expiry in years"""
        try:
            expiry = datetime.strptime(expiry_str, '%Y-%m-%d')
            now = datetime.now()
            days_to_expiry = (expiry - now).days
            return max(days_to_expiry / 365.0, 0.0)
        except:
            return 0.0

class TechnicalIndicators:
    """Technical analysis indicators calculator"""
    
    @staticmethod
    def sma(prices: List[float], period: int) -> float:
        """Simple Moving Average"""
        if len(prices) < period:
            return 0.0
        return sum(prices[-period:]) / period
        
    @staticmethod
    def ema(prices: List[float], period: int) -> float:
        """Exponential Moving Average"""
        if len(prices) < period:
            return prices[-1] if prices else 0.0
            
        multiplier = 2 / (period + 1)
        ema = prices[0]
        
        for price in prices[1:]:
            ema = (price * multiplier) + (ema * (1 - multiplier))
            
        return ema
        
    @staticmethod
    def rsi(prices: List[float], period: int = 14) -> float:
        """Relative Strength Index"""
        if len(prices) < period + 1:
            return 50.0
            
        gains = []
        losses = []
        
        for i in range(1, len(prices)):
            change = prices[i] - prices[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
                
        if len(gains) < period:
            return 50.0
            
        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period
        
        if avg_loss == 0:
            return 100.0
            
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
        
    @staticmethod
    def bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2) -> Tuple[float, float, float]:
        """Bollinger Bands (upper, middle, lower)"""
        if len(prices) < period:
            last_price = prices[-1] if prices else 0.0
            return last_price, last_price, last_price
            
        recent_prices = prices[-period:]
        sma = sum(recent_prices) / period
        variance = sum((p - sma) ** 2 for p in recent_prices) / period
        std = variance ** 0.5
        
        upper = sma + (std_dev * std)
        lower = sma - (std_dev * std)
        
        return upper, sma, lower

class MarketHoursManager:
    """Market hours and trading status manager"""
    
    def __init__(self):
        self.market_hours = {
            'NYSE': {'open': '09:30', 'close': '16:00', 'timezone': 'US/Eastern'},
            'NASDAQ': {'open': '09:30', 'close': '16:00', 'timezone': 'US/Eastern'},
            'CME': {'open': '17:00', 'close': '16:00', 'timezone': 'US/Central'},  # Futures
        }
        
    def is_market_open(self, exchange: str = 'NYSE') -> bool:
        """Check if market is currently open"""
        try:
            from pytz import timezone
            import pytz
            
            market_info = self.market_hours.get(exchange, self.market_hours['NYSE'])
            market_tz = timezone(market_info['timezone'])
            
            now = datetime.now(market_tz)
            
            # Check if it's a weekday
            if now.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
                
            # Check time
            open_time = datetime.strptime(market_info['open'], '%H:%M').time()
            close_time = datetime.strptime(market_info['close'], '%H:%M').time()
            
            current_time = now.time()
            
            if exchange == 'CME':  # Futures market (overnight)
                return current_time >= open_time or current_time <= close_time
            else:  # Regular equity markets
                return open_time <= current_time <= close_time
                
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False
            
    def time_to_open(self, exchange: str = 'NYSE') -> timedelta:
        """Time until market opens"""
        try:
            from pytz import timezone
            
            market_info = self.market_hours.get(exchange, self.market_hours['NYSE'])
            market_tz = timezone(market_info['timezone'])
            
            now = datetime.now(market_tz)
            open_time = datetime.strptime(market_info['open'], '%H:%M').time()
            
            # Calculate next market open
            next_open = datetime.combine(now.date(), open_time)
            next_open = market_tz.localize(next_open)
            
            # If market already opened today, get next business day
            if now.time() > open_time or now.weekday() >= 5:
                next_open += timedelta(days=1)
                while next_open.weekday() >= 5:  # Skip weekends
                    next_open += timedelta(days=1)
                    
            return next_open - now
            
        except Exception as e:
            logger.error(f"Error calculating time to open: {e}")
            return timedelta(0)

class MarketDataManager:
    """Central manager for all Gnosis market data needs"""
    
    def __init__(self, api_keys: Dict[str, str] = None):
        self.real_time_feed = RealTimeMarketData(api_keys)
        self.options_manager = OptionsChainManager(self.real_time_feed)
        self.market_hours = MarketHoursManager()
        self.technical_indicators = TechnicalIndicators()
        
        self.active_symbols = set()
        self.price_history = {}
        
    async def initialize(self, symbols: List[str]):
        """Initialize market data for given symbols"""
        logger.info(f"Initializing market data for: {symbols}")
        
        self.active_symbols.update(symbols)
        
        # Start real-time feeds
        await self.real_time_feed.connect(symbols)
        
        # Subscribe to price updates for history tracking
        for symbol in symbols:
            self.real_time_feed.subscribe(symbol, self._update_price_history)
            
        logger.info("Market data initialization complete")
        
    async def _update_price_history(self, market_data: MarketData):
        """Update price history for technical analysis"""
        symbol = market_data.symbol
        
        if symbol not in self.price_history:
            self.price_history[symbol] = []
            
        # Keep last 200 prices for technical analysis
        self.price_history[symbol].append(market_data.price)
        if len(self.price_history[symbol]) > 200:
            self.price_history[symbol].pop(0)
            
    async def get_market_data(self, symbol: str) -> Optional[MarketData]:
        """Get current market data for symbol"""
        return self.real_time_feed.get_latest(symbol)
        
    async def get_options_chain(self, symbol: str, expiry: str = None) -> List[OptionContract]:
        """Get options chain with real-time Greeks"""
        return await self.options_manager.get_options_chain(symbol, expiry)
        
    def get_technical_indicators(self, symbol: str) -> Dict[str, Any]:
        """Get technical indicators for symbol"""
        if symbol not in self.price_history:
            return {}
            
        prices = self.price_history[symbol]
        
        if len(prices) < 20:
            return {'status': 'insufficient_data'}
            
        indicators = {
            'sma_20': self.technical_indicators.sma(prices, 20),
            'sma_50': self.technical_indicators.sma(prices, 50),
            'ema_12': self.technical_indicators.ema(prices, 12),
            'ema_26': self.technical_indicators.ema(prices, 26),
            'rsi': self.technical_indicators.rsi(prices),
            'bollinger_bands': self.technical_indicators.bollinger_bands(prices)
        }
        
        # MACD
        ema_12 = indicators['ema_12']
        ema_26 = indicators['ema_26']
        indicators['macd'] = ema_12 - ema_26
        
        return indicators
        
    def is_market_open(self, exchange: str = 'NYSE') -> bool:
        """Check if market is open"""
        return self.market_hours.is_market_open(exchange)
        
    def get_market_status(self) -> Dict[str, Any]:
        """Get comprehensive market status"""
        return {
            'nyse_open': self.market_hours.is_market_open('NYSE'),
            'nasdaq_open': self.market_hours.is_market_open('NASDAQ'),
            'futures_open': self.market_hours.is_market_open('CME'),
            'time_to_open': str(self.market_hours.time_to_open()),
            'active_symbols': list(self.active_symbols),
            'data_sources': ['yahoo_finance']  # Add more as implemented
        }
        
    async def shutdown(self):
        """Shutdown market data feeds"""
        self.real_time_feed.disconnect()
        logger.info("Market data manager shutdown complete")

# Usage example for Gnosis integration
async def main():
    """Example usage of market data integration"""
    
    # Initialize market data manager
    api_keys = {
        'polygon': 'your_polygon_key',
        'alpha_vantage': 'your_alpha_vantage_key'
    }
    
    manager = MarketDataManager(api_keys)
    
    # SPY and popular options symbols
    symbols = ['SPY', 'QQQ', 'IWM', 'AAPL', 'TSLA', 'NVDA']
    
    try:
        # Initialize data feeds
        await manager.initialize(symbols)
        
        # Wait for some data to accumulate
        await asyncio.sleep(5)
        
        # Get market data
        for symbol in symbols:
            market_data = await manager.get_market_data(symbol)
            if market_data:
                print(f"\n{symbol}: ${market_data.price:.2f} "
                      f"({market_data.change:+.2f}, {market_data.change_percent:+.2f}%)")
                
                # Get technical indicators
                indicators = manager.get_technical_indicators(symbol)
                if indicators and 'status' not in indicators:
                    print(f"  RSI: {indicators['rsi']:.2f}")
                    print(f"  SMA20: {indicators['sma_20']:.2f}")
                    
        # Get options chain for SPY
        print("\n=== SPY Options Chain ===")
        options = await manager.get_options_chain('SPY')
        
        # Show first 5 calls and puts
        calls = [opt for opt in options if opt.option_type == 'call'][:5]
        puts = [opt for opt in options if opt.option_type == 'put'][:5]
        
        print("Calls:")
        for call in calls:
            print(f"  Strike {call.strike}: ${call.last:.2f} "
                  f"Delta={call.delta:.3f} IV={call.implied_volatility:.3f}")
                  
        print("Puts:")
        for put in puts:
            print(f"  Strike {put.strike}: ${put.last:.2f} "
                  f"Delta={put.delta:.3f} IV={put.implied_volatility:.3f}")
                  
        # Market status
        print(f"\n=== Market Status ===")
        status = manager.get_market_status()
        for key, value in status.items():
            print(f"{key}: {value}")
            
    finally:
        await manager.shutdown()

if __name__ == "__main__":
    asyncio.run(main())