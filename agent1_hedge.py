#!/usr/bin/env python3
"""
Agent 1: Hedge Agent
Real hedge calculations, position sizing, and risk management for options strategies
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class PositionType(Enum):
    """Position types for hedge calculations"""
    LONG_CALL = "LONG_CALL"
    SHORT_CALL = "SHORT_CALL"
    LONG_PUT = "LONG_PUT"
    SHORT_PUT = "SHORT_PUT"
    LONG_STOCK = "LONG_STOCK"
    SHORT_STOCK = "SHORT_STOCK"

class RiskLevel(Enum):
    """Risk assessment levels"""
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    EXTREME = "EXTREME"

@dataclass
class Position:
    """Individual position in portfolio"""
    symbol: str
    position_type: PositionType
    quantity: int
    entry_price: float
    current_price: float
    strike: Optional[float] = None
    expiry: Optional[datetime] = None
    
    # Greeks (for options)
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    
    # Risk metrics
    unrealized_pnl: float = 0.0
    max_loss: float = 0.0
    break_even: float = 0.0

@dataclass
class PortfolioRisk:
    """Portfolio-level risk metrics"""
    total_delta: float
    total_gamma: float
    total_theta: float
    total_vega: float
    
    net_exposure: float
    max_portfolio_loss: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float
    
    leverage_ratio: float
    concentration_risk: float
    time_decay_per_day: float

@dataclass
class HedgeRecommendation:
    """Hedge recommendation with specifics"""
    action: str  # "BUY", "SELL", "HOLD", "CLOSE"
    instrument: str  # What to trade
    quantity: int
    target_delta: float
    confidence: float
    reasoning: str
    
    # Cost analysis
    estimated_cost: float
    risk_reduction: float
    cost_effectiveness: float

class Agent1HedgeEngine:
    """
    Agent 1: Real Hedge Calculations and Risk Management
    
    Features:
    - Portfolio delta hedging
    - Multi-leg options strategy management
    - Real-time risk assessment
    - Position sizing algorithms
    - Greeks-based hedge recommendations
    """
    
    def __init__(self, account_size: float = 100000):
        """Initialize hedge agent"""
        self.account_size = account_size
        self.positions: Dict[str, Position] = {}
        self.portfolio_history: List[PortfolioRisk] = []
        
        # Risk parameters
        self.max_portfolio_delta = 0.1  # 10% of account
        self.max_position_size = 0.05   # 5% per position
        self.max_leverage = 2.0
        self.var_confidence = 0.95
        
        # Hedging thresholds
        self.delta_hedge_threshold = 0.15  # Hedge when delta > 15%
        self.gamma_risk_threshold = 0.5    # High gamma risk level
        self.theta_decay_threshold = 500   # Daily theta loss limit
        
        logger.info(f"Agent 1 Hedge Engine initialized with ${account_size:,.0f} account")
    
    def add_position(self, position: Position) -> None:
        """Add position to portfolio"""
        
        # Calculate derived metrics
        self._calculate_position_metrics(position)
        
        # Store position
        position_key = f"{position.symbol}_{position.position_type.value}_{position.strike or 'STOCK'}"
        self.positions[position_key] = position
        
        logger.info(f"Added position: {position_key} qty={position.quantity}")
    
    def _calculate_position_metrics(self, position: Position) -> None:
        """Calculate P&L and risk metrics for position"""
        
        if position.position_type in [PositionType.LONG_STOCK, PositionType.SHORT_STOCK]:
            # Stock position
            if position.position_type == PositionType.LONG_STOCK:
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity
                position.delta = position.quantity
                position.max_loss = position.entry_price * position.quantity  # Stock can go to zero
            else:  # Short stock
                position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity
                position.delta = -position.quantity
                position.max_loss = float('inf')  # Unlimited loss potential
                
        else:
            # Options position
            intrinsic_value = self._calculate_intrinsic_value(position)
            
            if position.position_type in [PositionType.LONG_CALL, PositionType.LONG_PUT]:
                # Long options
                position.unrealized_pnl = (position.current_price - position.entry_price) * position.quantity * 100
                position.max_loss = position.entry_price * position.quantity * 100
            else:
                # Short options
                position.unrealized_pnl = (position.entry_price - position.current_price) * position.quantity * 100
                position.max_loss = max(intrinsic_value, position.strike) * position.quantity * 100
    
    def _calculate_intrinsic_value(self, position: Position) -> float:
        """Calculate intrinsic value of option"""
        
        if not position.strike:
            return 0.0
        
        if position.position_type in [PositionType.LONG_CALL, PositionType.SHORT_CALL]:
            return max(0, position.current_price - position.strike)
        else:  # Put options
            return max(0, position.strike - position.current_price)
    
    def calculate_portfolio_risk(self) -> PortfolioRisk:
        """Calculate comprehensive portfolio risk metrics"""
        
        if not self.positions:
            return PortfolioRisk(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
        
        # Aggregate Greeks
        total_delta = sum(pos.delta for pos in self.positions.values())
        total_gamma = sum(pos.gamma * pos.quantity for pos in self.positions.values())
        total_theta = sum(pos.theta * pos.quantity for pos in self.positions.values())
        total_vega = sum(pos.vega * pos.quantity for pos in self.positions.values())
        
        # Net exposure
        net_exposure = sum(abs(pos.delta * pos.current_price) for pos in self.positions.values())
        
        # Maximum loss calculation
        max_portfolio_loss = sum(pos.max_loss for pos in self.positions.values() if pos.max_loss != float('inf'))
        
        # Value at Risk (simplified calculation)
        position_values = [pos.delta * pos.current_price for pos in self.positions.values()]
        if len(position_values) > 1:
            portfolio_std = np.std(position_values) * np.sqrt(252)  # Annualized
            var_95 = np.percentile(position_values, 5) * portfolio_std
            shortfall_values = [v for v in position_values if v <= var_95]
            expected_shortfall = np.mean(shortfall_values) if shortfall_values else 0
        else:
            var_95 = 0
            expected_shortfall = 0
        
        # Leverage and concentration
        leverage_ratio = net_exposure / self.account_size
        
        # Concentration risk (largest position as % of portfolio)
        position_weights = [abs(pos.delta * pos.current_price) / net_exposure 
                          for pos in self.positions.values() if net_exposure > 0]
        concentration_risk = max(position_weights) if position_weights else 0
        
        # Time decay per day
        time_decay_per_day = abs(total_theta)
        
        return PortfolioRisk(
            total_delta=total_delta,
            total_gamma=total_gamma,
            total_theta=total_theta,
            total_vega=total_vega,
            net_exposure=net_exposure,
            max_portfolio_loss=max_portfolio_loss,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            leverage_ratio=leverage_ratio,
            concentration_risk=concentration_risk,
            time_decay_per_day=time_decay_per_day
        )
    
    def assess_risk_level(self, portfolio_risk: PortfolioRisk) -> RiskLevel:
        """Assess overall portfolio risk level"""
        
        risk_factors = []
        
        # Delta risk
        delta_pct = abs(portfolio_risk.total_delta) / (self.account_size / 100)  # Delta as % of account
        if delta_pct > 20:
            risk_factors.append("HIGH_DELTA")
        elif delta_pct > 10:
            risk_factors.append("MODERATE_DELTA")
        
        # Leverage risk
        if portfolio_risk.leverage_ratio > 3:
            risk_factors.append("HIGH_LEVERAGE")
        elif portfolio_risk.leverage_ratio > 1.5:
            risk_factors.append("MODERATE_LEVERAGE")
        
        # Concentration risk
        if portfolio_risk.concentration_risk > 0.5:
            risk_factors.append("HIGH_CONCENTRATION")
        elif portfolio_risk.concentration_risk > 0.3:
            risk_factors.append("MODERATE_CONCENTRATION")
        
        # Time decay risk
        theta_pct = portfolio_risk.time_decay_per_day / self.account_size
        if theta_pct > 0.02:  # 2% per day
            risk_factors.append("HIGH_THETA")
        elif theta_pct > 0.01:  # 1% per day
            risk_factors.append("MODERATE_THETA")
        
        # Assess overall risk
        high_risk_count = len([f for f in risk_factors if "HIGH" in f])
        moderate_risk_count = len([f for f in risk_factors if "MODERATE" in f])
        
        if high_risk_count >= 2:
            return RiskLevel.EXTREME
        elif high_risk_count >= 1:
            return RiskLevel.HIGH
        elif moderate_risk_count >= 2:
            return RiskLevel.MODERATE
        else:
            return RiskLevel.LOW
    
    def generate_hedge_recommendations(self, underlying_price: float, 
                                     target_delta: float = 0.0) -> List[HedgeRecommendation]:
        """
        Generate hedge recommendations to achieve target delta
        
        Args:
            underlying_price: Current price of underlying asset
            target_delta: Desired portfolio delta (default: delta neutral)
        """
        
        portfolio_risk = self.calculate_portfolio_risk()
        current_delta = portfolio_risk.total_delta
        delta_to_hedge = current_delta - target_delta
        
        recommendations = []
        
        # Primary hedge: Stock position
        if abs(delta_to_hedge) > self.delta_hedge_threshold * self.account_size / underlying_price:
            
            stock_quantity = int(-delta_to_hedge)  # Opposite direction to neutralize
            
            if stock_quantity > 0:
                action = "BUY"
                instrument = "STOCK"
            else:
                action = "SELL"
                instrument = "STOCK"
                stock_quantity = abs(stock_quantity)
            
            estimated_cost = stock_quantity * underlying_price
            risk_reduction = abs(delta_to_hedge) / max(abs(current_delta), 1)
            cost_effectiveness = risk_reduction / max(estimated_cost / self.account_size, 0.001)
            
            recommendations.append(HedgeRecommendation(
                action=action,
                instrument=instrument,
                quantity=stock_quantity,
                target_delta=target_delta,
                confidence=0.9,  # High confidence in stock hedging
                reasoning=f"Delta hedge: Current δ={current_delta:.2f}, Target δ={target_delta:.2f}",
                estimated_cost=estimated_cost,
                risk_reduction=risk_reduction,
                cost_effectiveness=cost_effectiveness
            ))
        
        # Secondary hedge: Options-based hedging for gamma/vega
        if abs(portfolio_risk.total_gamma) > self.gamma_risk_threshold:
            
            # Recommend opposite gamma position
            if portfolio_risk.total_gamma > 0:
                # Long gamma - recommend short options
                recommendations.append(HedgeRecommendation(
                    action="SELL",
                    instrument="ATM_CALL_SPREAD",
                    quantity=int(portfolio_risk.total_gamma * 10),  # Rough sizing
                    target_delta=target_delta,
                    confidence=0.7,
                    reasoning=f"Gamma hedge: Portfolio γ={portfolio_risk.total_gamma:.3f} too high",
                    estimated_cost=underlying_price * 0.05 * abs(portfolio_risk.total_gamma * 10),
                    risk_reduction=0.6,
                    cost_effectiveness=0.8
                ))
            else:
                # Short gamma - recommend long options
                recommendations.append(HedgeRecommendation(
                    action="BUY",
                    instrument="ATM_STRADDLE",
                    quantity=int(abs(portfolio_risk.total_gamma) * 10),
                    target_delta=target_delta,
                    confidence=0.7,
                    reasoning=f"Gamma hedge: Portfolio γ={portfolio_risk.total_gamma:.3f} too negative",
                    estimated_cost=underlying_price * 0.08 * abs(portfolio_risk.total_gamma * 10),
                    risk_reduction=0.5,
                    cost_effectiveness=0.6
                ))
        
        # Time decay management
        if portfolio_risk.time_decay_per_day > self.theta_decay_threshold:
            recommendations.append(HedgeRecommendation(
                action="CLOSE",
                instrument="SHORTEST_EXPIRY_OPTIONS",
                quantity=0,  # Strategy-dependent
                target_delta=target_delta,
                confidence=0.8,
                reasoning=f"Theta risk: ${portfolio_risk.time_decay_per_day:.0f}/day decay exceeds limit",
                estimated_cost=0,  # Closing should generate credit
                risk_reduction=0.7,
                cost_effectiveness=1.0
            ))
        
        # Sort by cost-effectiveness
        recommendations.sort(key=lambda x: x.cost_effectiveness, reverse=True)
        
        return recommendations
    
    def calculate_position_size(self, strategy_type: str, underlying_price: float, 
                              risk_per_trade: float = 0.02) -> Dict[str, int]:
        """
        Calculate appropriate position sizes for new trades
        
        Args:
            strategy_type: Type of strategy ("long_call", "short_put", "iron_condor", etc.)
            underlying_price: Current underlying price
            risk_per_trade: Risk per trade as fraction of account (default: 2%)
        """
        
        max_risk_dollars = self.account_size * risk_per_trade
        portfolio_risk = self.calculate_portfolio_risk()
        
        # Adjust for existing risk
        available_risk = max_risk_dollars * (1 - portfolio_risk.leverage_ratio / self.max_leverage)
        
        sizes = {}
        
        if strategy_type == "long_call":
            # Risk = premium paid
            option_price = underlying_price * 0.05  # Estimate 5% for ATM option
            max_contracts = int(available_risk / (option_price * 100))
            sizes["calls"] = min(max_contracts, 10)  # Cap at 10 contracts
            
        elif strategy_type == "short_put":
            # Risk = strike - premium received
            strike = underlying_price * 0.95  # 5% OTM put
            premium = underlying_price * 0.02  # Estimate 2% premium
            risk_per_contract = (strike - premium) * 100
            max_contracts = int(available_risk / risk_per_contract)
            sizes["puts"] = min(max_contracts, 5)  # Conservative cap
            
        elif strategy_type == "iron_condor":
            # Risk = spread width - net credit
            spread_width = underlying_price * 0.05  # 5% wide spreads
            net_credit = underlying_price * 0.01   # Estimate 1% credit
            risk_per_spread = (spread_width - net_credit) * 100
            max_spreads = int(available_risk / risk_per_spread)
            sizes["iron_condors"] = min(max_spreads, 3)  # Conservative
            
        elif strategy_type == "covered_call":
            # Need 100 shares per call
            shares_needed = 100
            share_cost = shares_needed * underlying_price
            max_positions = int(available_risk / share_cost)
            sizes["covered_calls"] = min(max_positions, 2)  # Very conservative
        
        return sizes
    
    def get_portfolio_summary(self) -> Dict[str, any]:
        """Get comprehensive portfolio summary"""
        
        portfolio_risk = self.calculate_portfolio_risk()
        risk_level = self.assess_risk_level(portfolio_risk)
        
        # Position breakdown
        position_summary = {}
        for key, pos in self.positions.items():
            position_summary[key] = {
                'type': pos.position_type.value,
                'quantity': pos.quantity,
                'pnl': pos.unrealized_pnl,
                'delta': pos.delta,
                'max_loss': pos.max_loss if pos.max_loss != float('inf') else 'Unlimited'
            }
        
        return {
            'account_size': self.account_size,
            'total_positions': len(self.positions),
            'portfolio_delta': portfolio_risk.total_delta,
            'portfolio_gamma': portfolio_risk.total_gamma,
            'portfolio_theta': portfolio_risk.total_theta,
            'portfolio_vega': portfolio_risk.total_vega,
            'net_exposure': portfolio_risk.net_exposure,
            'leverage_ratio': portfolio_risk.leverage_ratio,
            'risk_level': risk_level.value,
            'daily_theta_decay': portfolio_risk.time_decay_per_day,
            'var_95': portfolio_risk.var_95,
            'concentration_risk': portfolio_risk.concentration_risk,
            'positions': position_summary
        }

def create_sample_portfolio() -> Agent1HedgeEngine:
    """Create sample portfolio for testing"""
    
    agent1 = Agent1HedgeEngine(account_size=100000)
    
    # Add some sample positions
    
    # Long 500 shares SPY
    spy_stock = Position(
        symbol="SPY",
        position_type=PositionType.LONG_STOCK,
        quantity=500,
        entry_price=430.00,
        current_price=432.50,
        delta=500,
        unrealized_pnl=1250
    )
    agent1.add_position(spy_stock)
    
    # Short 5 SPY calls
    spy_calls = Position(
        symbol="SPY",
        position_type=PositionType.SHORT_CALL,
        quantity=5,
        entry_price=8.50,
        current_price=12.00,
        strike=440.0,
        expiry=datetime(2024, 12, 20),
        delta=-0.65 * 5,
        gamma=0.05 * 5,
        theta=-15 * 5,
        vega=8 * 5
    )
    agent1.add_position(spy_calls)
    
    # Long 3 SPY puts (protective)
    spy_puts = Position(
        symbol="SPY",
        position_type=PositionType.LONG_PUT,
        quantity=3,
        entry_price=6.00,
        current_price=3.50,
        strike=420.0,
        expiry=datetime(2024, 12, 20),
        delta=-0.25 * 3,
        gamma=0.04 * 3,
        theta=-8 * 3,
        vega=6 * 3
    )
    agent1.add_position(spy_puts)
    
    return agent1

def test_agent1_hedge():
    """Test Agent 1 Hedge Engine"""
    
    print("=== Testing Agent 1 Hedge Engine ===")
    
    # Create sample portfolio
    agent1 = create_sample_portfolio()
    
    # Calculate risk
    portfolio_risk = agent1.calculate_portfolio_risk()
    risk_level = agent1.assess_risk_level(portfolio_risk)
    
    print(f"\n📊 Portfolio Risk Analysis:")
    print(f"Total Delta: {portfolio_risk.total_delta:.2f}")
    print(f"Total Gamma: {portfolio_risk.total_gamma:.3f}")
    print(f"Total Theta: {portfolio_risk.total_theta:.0f}")
    print(f"Net Exposure: ${portfolio_risk.net_exposure:,.0f}")
    print(f"Leverage Ratio: {portfolio_risk.leverage_ratio:.2f}x")
    print(f"Risk Level: {risk_level.value}")
    print(f"Daily Theta Decay: ${portfolio_risk.time_decay_per_day:.0f}")
    
    # Generate hedge recommendations
    current_spy_price = 432.50
    recommendations = agent1.generate_hedge_recommendations(current_spy_price, target_delta=0.0)
    
    print(f"\n🎯 Hedge Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec.action} {rec.quantity} {rec.instrument}")
        print(f"   Reasoning: {rec.reasoning}")
        print(f"   Cost: ${rec.estimated_cost:,.0f}")
        print(f"   Risk Reduction: {rec.risk_reduction:.1%}")
        print(f"   Cost Effectiveness: {rec.cost_effectiveness:.2f}")
        print()
    
    # Position sizing example
    print("💰 Position Sizing Examples:")
    sizing_examples = [
        ("long_call", "Long Call"),
        ("short_put", "Short Put"),
        ("iron_condor", "Iron Condor"),
        ("covered_call", "Covered Call")
    ]
    
    for strategy_code, strategy_name in sizing_examples:
        sizes = agent1.calculate_position_size(strategy_code, current_spy_price)
        print(f"{strategy_name}: {sizes}")
    
    # Portfolio summary
    print(f"\n📋 Portfolio Summary:")
    summary = agent1.get_portfolio_summary()
    for key, value in summary.items():
        if key != 'positions':
            print(f"{key}: {value}")
    
    print(f"\n✅ Agent 1 Hedge Engine test completed!")
    
    return agent1

if __name__ == "__main__":
    test_agent1_hedge()