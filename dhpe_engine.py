#!/usr/bin/env python3
"""
Gnosis DHPE (Dealer Hedge Pressure Ecosystem) Engine
Real options market microstructure analysis with gamma exposure calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

@dataclass
class OptionsContract:
    """Options contract data structure"""
    symbol: str
    strike: float
    expiry: datetime
    option_type: str  # 'call' or 'put'
    price: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: float
    gamma: float
    theta: float
    vega: float

@dataclass
class DHPEMetrics:
    """DHPE calculation results"""
    symbol: str
    timestamp: datetime
    
    # Gamma exposure metrics
    total_gamma_exposure: float
    call_gamma_exposure: float
    put_gamma_exposure: float
    net_gamma_exposure: float
    
    # Hedge pressure indicators
    dealer_positioning: float  # -1 (short gamma) to +1 (long gamma)
    hedge_pressure_score: float
    
    # Max pain analysis
    max_pain_strike: float
    max_pain_value: float
    
    # Flow analysis
    call_put_ratio: float
    volume_weighted_iv: float
    
    # Pressure levels
    vanna_pressure: float  # Delta sensitivity to volatility
    charm_pressure: float  # Gamma decay over time

class DHPEEngine:
    """
    DHPE Engine: Real options market microstructure analysis
    
    Calculates dealer hedge pressure, gamma exposure, and flow dynamics
    """
    
    def __init__(self):
        """Initialize DHPE engine"""
        self.options_data: Dict[str, List[OptionsContract]] = {}
        self.price_cache: Dict[str, float] = {}
        self.calculations_cache: Dict[str, DHPEMetrics] = {}
        
        # Configuration
        self.min_volume_threshold = 50
        self.min_oi_threshold = 100
        self.gamma_scaling_factor = 100  # Scale gamma for readability
        
        logger.info("DHPE Engine initialized")
    
    def add_options_data(self, symbol: str, options: List[OptionsContract]) -> None:
        """Add options chain data for analysis"""
        
        # Filter for liquid options
        filtered_options = [
            opt for opt in options 
            if opt.volume >= self.min_volume_threshold 
            and opt.open_interest >= self.min_oi_threshold
        ]
        
        self.options_data[symbol] = filtered_options
        logger.info(f"Added {len(filtered_options)} liquid options for {symbol}")
    
    def calculate_gamma_exposure(self, symbol: str, spot_price: float) -> Tuple[float, float, float]:
        """
        Calculate gamma exposure levels
        
        Returns:
            Tuple of (total_gamma_exposure, call_gamma, put_gamma)
        """
        
        if symbol not in self.options_data:
            return 0.0, 0.0, 0.0
        
        options = self.options_data[symbol]
        
        call_gamma = 0.0
        put_gamma = 0.0
        
        for opt in options:
            # Calculate notional gamma exposure
            # Gamma exposure = Gamma * Open Interest * 100 * Spot Price^2
            notional_gamma = opt.gamma * opt.open_interest * 100 * (spot_price ** 2)
            
            if opt.option_type.lower() == 'call':
                # Market makers are short calls (negative gamma for MM)
                call_gamma -= notional_gamma
            else:
                # Market makers are short puts (negative gamma for MM)  
                put_gamma -= notional_gamma
        
        total_gamma = call_gamma + put_gamma
        
        return total_gamma / 1e9, call_gamma / 1e9, put_gamma / 1e9  # Scale to billions
    
    def calculate_max_pain(self, symbol: str) -> Tuple[float, float]:
        """
        Calculate max pain strike (strike with minimum option value)
        
        Returns:
            Tuple of (max_pain_strike, max_pain_value)
        """
        
        if symbol not in self.options_data:
            return 0.0, 0.0
        
        options = self.options_data[symbol]
        
        # Get all unique strikes
        strikes = sorted(list(set(opt.strike for opt in options)))
        
        if not strikes:
            return 0.0, 0.0
        
        min_pain = float('inf')
        max_pain_strike = strikes[0]
        
        # Calculate total option value at each strike
        for strike in strikes:
            total_value = 0.0
            
            for opt in options:
                if opt.option_type.lower() == 'call':
                    # Call value at expiry
                    intrinsic = max(0, strike - opt.strike)
                else:
                    # Put value at expiry
                    intrinsic = max(0, opt.strike - strike)
                
                total_value += intrinsic * opt.open_interest
            
            if total_value < min_pain:
                min_pain = total_value
                max_pain_strike = strike
        
        return max_pain_strike, min_pain / 1e6  # Scale to millions
    
    def calculate_flow_metrics(self, symbol: str) -> Dict[str, float]:
        """Calculate options flow and positioning metrics"""
        
        if symbol not in self.options_data:
            return {}
        
        options = self.options_data[symbol]
        
        # Volume analysis
        call_volume = sum(opt.volume for opt in options if opt.option_type.lower() == 'call')
        put_volume = sum(opt.volume for opt in options if opt.option_type.lower() == 'put')
        total_volume = call_volume + put_volume
        
        # Open interest analysis  
        call_oi = sum(opt.open_interest for opt in options if opt.option_type.lower() == 'call')
        put_oi = sum(opt.open_interest for opt in options if opt.option_type.lower() == 'put')
        
        # IV analysis
        total_vega_weighted_iv = sum(opt.implied_volatility * opt.vega * opt.open_interest for opt in options)
        total_vega_oi = sum(opt.vega * opt.open_interest for opt in options)
        
        return {
            'call_volume': call_volume,
            'put_volume': put_volume,
            'call_put_volume_ratio': call_volume / max(put_volume, 1),
            'call_oi': call_oi,
            'put_oi': put_oi,
            'call_put_oi_ratio': call_oi / max(put_oi, 1),
            'total_volume': total_volume,
            'volume_weighted_iv': total_vega_weighted_iv / max(total_vega_oi, 1)
        }
    
    def calculate_vanna_pressure(self, symbol: str, spot_price: float) -> float:
        """
        Calculate vanna pressure (delta sensitivity to volatility changes)
        """
        
        if symbol not in self.options_data:
            return 0.0
        
        options = self.options_data[symbol]
        
        total_vanna = 0.0
        
        for opt in options:
            # Approximate vanna as gamma * vega / spot_price
            # (More sophisticated calculation would use actual vanna greeks)
            approx_vanna = opt.gamma * opt.vega / spot_price
            
            # Weight by open interest
            vanna_exposure = approx_vanna * opt.open_interest
            
            if opt.option_type.lower() == 'call':
                total_vanna += vanna_exposure
            else:
                total_vanna -= vanna_exposure  # Put vanna has opposite sign
        
        return total_vanna / 1e6  # Scale for readability
    
    def calculate_charm_pressure(self, symbol: str) -> float:
        """
        Calculate charm pressure (gamma decay over time)
        """
        
        if symbol not in self.options_data:
            return 0.0
        
        options = self.options_data[symbol]
        
        total_charm = 0.0
        
        for opt in options:
            # Approximate charm as gamma * theta
            # (Actual charm calculation requires more sophisticated modeling)
            approx_charm = abs(opt.gamma * opt.theta)
            
            # Weight by open interest
            charm_exposure = approx_charm * opt.open_interest
            total_charm += charm_exposure
        
        return total_charm / 1e6  # Scale for readability
    
    def analyze_dhpe(self, symbol: str, spot_price: float) -> DHPEMetrics:
        """
        Complete DHPE analysis for a symbol
        
        Args:
            symbol: Stock symbol
            spot_price: Current stock price
            
        Returns:
            DHPEMetrics with complete analysis
        """
        
        # Cache spot price
        self.price_cache[symbol] = spot_price
        
        # Gamma exposure analysis
        total_gamma, call_gamma, put_gamma = self.calculate_gamma_exposure(symbol, spot_price)
        net_gamma = call_gamma + put_gamma
        
        # Max pain analysis
        max_pain_strike, max_pain_value = self.calculate_max_pain(symbol)
        
        # Flow metrics
        flow_metrics = self.calculate_flow_metrics(symbol)
        
        # Pressure calculations
        vanna_pressure = self.calculate_vanna_pressure(symbol, spot_price)
        charm_pressure = self.calculate_charm_pressure(symbol)
        
        # Dealer positioning score (-1 to +1)
        # Negative gamma = dealers short gamma (need to hedge by buying/selling stock)
        if abs(total_gamma) > 0:
            dealer_positioning = np.tanh(total_gamma / 5.0)  # Normalize to [-1, 1]
        else:
            dealer_positioning = 0.0
        
        # Hedge pressure score (0 to 1, higher = more pressure)
        hedge_pressure_components = [
            abs(total_gamma) / 10.0,  # Gamma magnitude
            abs(spot_price - max_pain_strike) / spot_price,  # Distance from max pain
            min(vanna_pressure / 5.0, 1.0),  # Vanna pressure
            min(charm_pressure / 10.0, 1.0)   # Charm pressure
        ]
        hedge_pressure_score = np.mean(hedge_pressure_components)
        
        # Create metrics object
        metrics = DHPEMetrics(
            symbol=symbol,
            timestamp=datetime.now(),
            total_gamma_exposure=total_gamma,
            call_gamma_exposure=call_gamma,
            put_gamma_exposure=put_gamma,
            net_gamma_exposure=net_gamma,
            dealer_positioning=dealer_positioning,
            hedge_pressure_score=min(hedge_pressure_score, 1.0),
            max_pain_strike=max_pain_strike,
            max_pain_value=max_pain_value,
            call_put_ratio=flow_metrics.get('call_put_volume_ratio', 1.0),
            volume_weighted_iv=flow_metrics.get('volume_weighted_iv', 0.0),
            vanna_pressure=vanna_pressure,
            charm_pressure=charm_pressure
        )
        
        # Cache results
        self.calculations_cache[symbol] = metrics
        
        logger.info(f"DHPE analysis complete for {symbol}: "
                   f"Gamma={total_gamma:.2f}B, MaxPain=${max_pain_strike:.2f}, "
                   f"HedgePressure={hedge_pressure_score:.3f}")
        
        return metrics
    
    def get_summary(self, symbol: str) -> Dict[str, any]:
        """Get human-readable summary of DHPE analysis"""
        
        if symbol not in self.calculations_cache:
            return {"error": f"No analysis found for {symbol}"}
        
        metrics = self.calculations_cache[symbol]
        spot_price = self.price_cache.get(symbol, 0)
        
        # Determine market regime
        if metrics.dealer_positioning < -0.3:
            regime = "Dealer Short Gamma (Bullish Pressure)"
        elif metrics.dealer_positioning > 0.3:
            regime = "Dealer Long Gamma (Bearish Pressure)"  
        else:
            regime = "Neutral Gamma Environment"
        
        # Pressure assessment
        if metrics.hedge_pressure_score > 0.7:
            pressure_level = "EXTREME"
        elif metrics.hedge_pressure_score > 0.5:
            pressure_level = "HIGH"
        elif metrics.hedge_pressure_score > 0.3:
            pressure_level = "MODERATE"
        else:
            pressure_level = "LOW"
        
        return {
            'symbol': symbol,
            'current_price': spot_price,
            'regime': regime,
            'pressure_level': pressure_level,
            'total_gamma_billions': metrics.total_gamma_exposure,
            'max_pain_strike': metrics.max_pain_strike,
            'distance_from_max_pain_pct': abs(spot_price - metrics.max_pain_strike) / spot_price * 100,
            'call_put_ratio': metrics.call_put_ratio,
            'hedge_pressure_score': metrics.hedge_pressure_score,
            'dealer_positioning': metrics.dealer_positioning,
            'vanna_pressure': metrics.vanna_pressure,
            'charm_pressure': metrics.charm_pressure
        }

def create_sample_options_data(symbol: str, spot_price: float) -> List[OptionsContract]:
    """Create sample options data for testing"""
    
    options = []
    strikes = np.arange(spot_price * 0.8, spot_price * 1.2, 5)
    expiry = datetime(2024, 12, 20)  # Sample expiry
    
    for strike in strikes:
        # Call option
        call_delta = max(0.05, min(0.95, 0.5 + (spot_price - strike) / spot_price * 2))
        call_gamma = 0.1 * np.exp(-0.5 * ((spot_price - strike) / spot_price) ** 2)
        
        call = OptionsContract(
            symbol=f"{symbol}241220C{strike:08.0f}",
            strike=strike,
            expiry=expiry,
            option_type='call',
            price=max(0.01, spot_price - strike + 5),
            volume=np.random.randint(100, 2000),
            open_interest=np.random.randint(500, 5000),
            implied_volatility=0.15 + np.random.rand() * 0.1,
            delta=call_delta,
            gamma=call_gamma,
            theta=-0.05 - np.random.rand() * 0.05,
            vega=0.1 + np.random.rand() * 0.1
        )
        options.append(call)
        
        # Put option
        put_delta = call_delta - 1.0
        put = OptionsContract(
            symbol=f"{symbol}241220P{strike:08.0f}",
            strike=strike,
            expiry=expiry,
            option_type='put',
            price=max(0.01, strike - spot_price + 5),
            volume=np.random.randint(100, 2000),
            open_interest=np.random.randint(500, 5000),
            implied_volatility=0.15 + np.random.rand() * 0.1,
            delta=put_delta,
            gamma=call_gamma,  # Same gamma as call
            theta=-0.05 - np.random.rand() * 0.05,
            vega=0.1 + np.random.rand() * 0.1
        )
        options.append(put)
    
    return options

def test_dhpe_engine():
    """Test the DHPE engine"""
    
    print("=== Testing DHPE Engine ===")
    
    # Initialize engine
    dhpe = DHPEEngine()
    
    # Test with SPY
    spy_price = 432.50
    spy_options = create_sample_options_data("SPY", spy_price)
    
    print(f"Created {len(spy_options)} sample options contracts")
    
    # Add data and analyze
    dhpe.add_options_data("SPY", spy_options)
    metrics = dhpe.analyze_dhpe("SPY", spy_price)
    
    print(f"\n=== DHPE Analysis Results ===")
    print(f"Symbol: {metrics.symbol}")
    print(f"Total Gamma Exposure: {metrics.total_gamma_exposure:.2f}B")
    print(f"Call Gamma: {metrics.call_gamma_exposure:.2f}B")
    print(f"Put Gamma: {metrics.put_gamma_exposure:.2f}B")
    print(f"Dealer Positioning: {metrics.dealer_positioning:.3f}")
    print(f"Hedge Pressure Score: {metrics.hedge_pressure_score:.3f}")
    print(f"Max Pain Strike: ${metrics.max_pain_strike:.2f}")
    print(f"Call/Put Ratio: {metrics.call_put_ratio:.2f}")
    print(f"Vanna Pressure: {metrics.vanna_pressure:.2f}")
    print(f"Charm Pressure: {metrics.charm_pressure:.2f}")
    
    # Get summary
    summary = dhpe.get_summary("SPY")
    print(f"\n=== Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    
    print("\n✅ DHPE Engine test completed successfully!")
    
    return dhpe

if __name__ == "__main__":
    test_dhpe_engine()