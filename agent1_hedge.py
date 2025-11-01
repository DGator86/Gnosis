"""
Gnosis Agent 1: Advanced Hedge Agent
Complete implementation for position sizing, risk management, and hedge calculations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from scipy.optimize import minimize
from scipy.stats import norm
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class RiskLevel(Enum):
    """Risk tolerance levels"""
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    EXTREME = "extreme"

class HedgeType(Enum):
    """Types of hedge strategies"""
    DELTA_NEUTRAL = "delta_neutral"
    GAMMA_SCALPING = "gamma_scalping"
    VOLATILITY_HEDGE = "volatility_hedge"
    TAIL_HEDGE = "tail_hedge"
    CORRELATION_HEDGE = "correlation_hedge"

@dataclass
class Position:
    """Trading position representation"""
    symbol: str
    quantity: int
    entry_price: float
    current_price: float
    position_type: str  # 'long', 'short', 'call', 'put'
    expiry: Optional[str] = None
    strike: Optional[float] = None
    delta: float = 0.0
    gamma: float = 0.0
    theta: float = 0.0
    vega: float = 0.0
    pnl: float = field(init=False)
    
    def __post_init__(self):
        self.pnl = self.calculate_pnl()
        
    def calculate_pnl(self) -> float:
        """Calculate position P&L"""
        if self.position_type in ['long', 'call']:
            return (self.current_price - self.entry_price) * self.quantity
        else:  # short, put
            return (self.entry_price - self.current_price) * self.quantity

@dataclass
class Portfolio:
    """Portfolio representation with Greeks"""
    positions: List[Position]
    cash: float
    total_delta: float = field(init=False)
    total_gamma: float = field(init=False)
    total_theta: float = field(init=False)
    total_vega: float = field(init=False)
    total_pnl: float = field(init=False)
    net_value: float = field(init=False)
    
    def __post_init__(self):
        self.calculate_portfolio_greeks()
        
    def calculate_portfolio_greeks(self):
        """Calculate portfolio-level Greeks"""
        self.total_delta = sum(pos.delta * pos.quantity for pos in self.positions)
        self.total_gamma = sum(pos.gamma * pos.quantity for pos in self.positions)
        self.total_theta = sum(pos.theta * pos.quantity for pos in self.positions)
        self.total_vega = sum(pos.vega * pos.quantity for pos in self.positions)
        self.total_pnl = sum(pos.pnl for pos in self.positions)
        self.net_value = self.cash + self.total_pnl

@dataclass
class RiskMetrics:
    """Portfolio risk metrics"""
    var_95: float  # Value at Risk (95%)
    var_99: float  # Value at Risk (99%)
    expected_shortfall: float  # Conditional VaR
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    beta: float
    correlation_spy: float
    volatility: float

class PositionSizer:
    """Advanced position sizing with Kelly Criterion and risk parity"""
    
    def __init__(self, risk_tolerance: RiskLevel = RiskLevel.MODERATE):
        self.risk_tolerance = risk_tolerance
        self.risk_params = {
            RiskLevel.CONSERVATIVE: {'max_position': 0.02, 'max_portfolio_risk': 0.10, 'kelly_fraction': 0.25},
            RiskLevel.MODERATE: {'max_position': 0.05, 'max_portfolio_risk': 0.15, 'kelly_fraction': 0.50},
            RiskLevel.AGGRESSIVE: {'max_position': 0.10, 'max_portfolio_risk': 0.25, 'kelly_fraction': 0.75},
            RiskLevel.EXTREME: {'max_position': 0.20, 'max_portfolio_risk': 0.35, 'kelly_fraction': 1.00}
        }
        
    def kelly_criterion(self, win_rate: float, avg_win: float, avg_loss: float) -> float:
        """Calculate Kelly Criterion optimal position size"""
        if avg_loss <= 0:
            return 0.0
            
        b = avg_win / avg_loss  # Ratio of win to loss
        p = win_rate  # Probability of winning
        q = 1 - p  # Probability of losing
        
        kelly_fraction = (b * p - q) / b
        
        # Apply safety margin
        params = self.risk_params[self.risk_tolerance]
        return max(0, min(kelly_fraction * params['kelly_fraction'], params['max_position']))
        
    def volatility_adjusted_size(self, portfolio_value: float, target_volatility: float, 
                                asset_volatility: float, correlation: float = 1.0) -> float:
        """Calculate position size based on volatility targeting"""
        if asset_volatility <= 0:
            return 0.0
            
        params = self.risk_params[self.risk_tolerance]
        max_risk = portfolio_value * params['max_portfolio_risk']
        
        # Adjust for correlation
        adjusted_volatility = asset_volatility * correlation
        
        position_size = (target_volatility * portfolio_value) / adjusted_volatility
        
        # Cap at maximum position limit
        max_position_value = portfolio_value * params['max_position']
        return min(position_size, max_position_value)
        
    def risk_parity_size(self, portfolio: Portfolio, new_asset_volatility: float, 
                        target_risk_contribution: float = 0.1) -> float:
        """Calculate position size for risk parity contribution"""
        if not portfolio.positions:
            return portfolio.net_value * target_risk_contribution / new_asset_volatility
            
        # Calculate current portfolio volatility
        portfolio_vol = self._calculate_portfolio_volatility(portfolio)
        
        if portfolio_vol <= 0:
            return 0.0
            
        # Target risk contribution in dollar terms
        target_risk_dollar = portfolio.net_value * target_risk_contribution
        
        # Position size to achieve target risk contribution
        position_size = target_risk_dollar / new_asset_volatility
        
        params = self.risk_params[self.risk_tolerance]
        max_position_value = portfolio.net_value * params['max_position']
        
        return min(position_size, max_position_value)
        
    def _calculate_portfolio_volatility(self, portfolio: Portfolio) -> float:
        """Estimate portfolio volatility from positions"""
        # Simplified portfolio volatility calculation
        # In practice, would use covariance matrix
        
        total_exposure = sum(abs(pos.current_price * pos.quantity) for pos in portfolio.positions)
        
        if total_exposure == 0:
            return 0.0
            
        # Weight positions by exposure
        weighted_vol = 0.0
        for pos in portfolio.positions:
            weight = abs(pos.current_price * pos.quantity) / total_exposure
            # Estimate volatility based on Greeks (simplified)
            pos_vol = abs(pos.delta) * 0.20 + abs(pos.gamma) * 0.10  # Rough approximation
            weighted_vol += weight * pos_vol
            
        return weighted_vol

class RiskManager:
    """Advanced portfolio risk management"""
    
    def __init__(self, risk_tolerance: RiskLevel = RiskLevel.MODERATE):
        self.risk_tolerance = risk_tolerance
        self.position_limits = {
            RiskLevel.CONSERVATIVE: {'max_delta': 100, 'max_gamma': 50, 'max_vega': 1000},
            RiskLevel.MODERATE: {'max_delta': 250, 'max_gamma': 125, 'max_vega': 2500},
            RiskLevel.AGGRESSIVE: {'max_delta': 500, 'max_gamma': 250, 'max_vega': 5000},
            RiskLevel.EXTREME: {'max_delta': 1000, 'max_gamma': 500, 'max_vega': 10000}
        }
        
    def calculate_var(self, portfolio: Portfolio, confidence: float = 0.95, 
                     holding_period: int = 1) -> float:
        """Calculate Value at Risk using parametric method"""
        if not portfolio.positions:
            return 0.0
            
        # Simplified VaR calculation
        # In production, would use historical simulation or Monte Carlo
        
        # Estimate portfolio volatility
        portfolio_vol = 0.20  # Assume 20% annual volatility
        
        # Convert to daily volatility
        daily_vol = portfolio_vol / np.sqrt(252)
        
        # Scale for holding period
        period_vol = daily_vol * np.sqrt(holding_period)
        
        # VaR calculation
        z_score = norm.ppf(1 - confidence)
        var = portfolio.net_value * period_vol * z_score
        
        return abs(var)  # Return positive value
        
    def calculate_expected_shortfall(self, portfolio: Portfolio, confidence: float = 0.95) -> float:
        """Calculate Expected Shortfall (Conditional VaR)"""
        var = self.calculate_var(portfolio, confidence)
        
        # Expected Shortfall approximation
        z_score = norm.ppf(1 - confidence)
        expected_shortfall = var * norm.pdf(z_score) / (1 - confidence)
        
        return expected_shortfall
        
    def check_position_limits(self, portfolio: Portfolio) -> Dict[str, bool]:
        """Check if portfolio exceeds position limits"""
        limits = self.position_limits[self.risk_tolerance]
        
        checks = {
            'delta_limit': abs(portfolio.total_delta) <= limits['max_delta'],
            'gamma_limit': abs(portfolio.total_gamma) <= limits['max_gamma'],
            'vega_limit': abs(portfolio.total_vega) <= limits['max_vega']
        }
        
        return checks
        
    def calculate_risk_metrics(self, portfolio: Portfolio, price_history: List[float]) -> RiskMetrics:
        """Calculate comprehensive risk metrics"""
        
        # Calculate returns
        if len(price_history) < 30:
            # Insufficient data for meaningful metrics
            return RiskMetrics(0, 0, 0, 0, 0, 0, 1, 0, 0.20)
            
        returns = np.diff(price_history) / price_history[:-1]
        
        # VaR calculations
        var_95 = self.calculate_var(portfolio, 0.95)
        var_99 = self.calculate_var(portfolio, 0.99)
        expected_shortfall = self.calculate_expected_shortfall(portfolio, 0.95)
        
        # Performance metrics
        annual_return = np.mean(returns) * 252
        volatility = np.std(returns) * np.sqrt(252)
        
        sharpe_ratio = annual_return / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else volatility
        sortino_ratio = annual_return / downside_vol if downside_vol > 0 else 0
        
        # Max drawdown
        cumulative = np.cumprod(1 + returns)
        rolling_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(np.min(drawdowns))
        
        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            expected_shortfall=expected_shortfall,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            beta=1.0,  # Simplified
            correlation_spy=0.5,  # Simplified
            volatility=volatility
        )

class HedgeCalculator:
    """Advanced hedging strategies calculator"""
    
    def __init__(self):
        self.hedge_strategies = {}
        
    async def calculate_delta_hedge(self, portfolio: Portfolio, target_delta: float = 0.0) -> Dict[str, Any]:
        """Calculate delta hedge to achieve target delta"""
        current_delta = portfolio.total_delta
        delta_imbalance = current_delta - target_delta
        
        if abs(delta_imbalance) < 1:  # Close enough
            return {
                'hedge_required': False,
                'current_delta': current_delta,
                'target_delta': target_delta,
                'imbalance': delta_imbalance
            }
            
        # Suggest SPY hedge (delta = 1 for long shares)
        spy_shares_needed = -int(delta_imbalance)  # Negative to hedge
        
        hedge_cost = abs(spy_shares_needed) * 450  # Assume SPY at $450
        
        return {
            'hedge_required': True,
            'current_delta': current_delta,
            'target_delta': target_delta,
            'imbalance': delta_imbalance,
            'hedge_instrument': 'SPY',
            'hedge_quantity': spy_shares_needed,
            'estimated_cost': hedge_cost,
            'hedge_action': 'buy' if spy_shares_needed > 0 else 'sell'
        }
        
    async def calculate_gamma_hedge(self, portfolio: Portfolio, target_gamma: float = 0.0) -> Dict[str, Any]:
        """Calculate gamma hedge using options"""
        current_gamma = portfolio.total_gamma
        gamma_imbalance = current_gamma - target_gamma
        
        if abs(gamma_imbalance) < 1:
            return {
                'hedge_required': False,
                'current_gamma': current_gamma,
                'target_gamma': target_gamma,
                'imbalance': gamma_imbalance
            }
            
        # Suggest using at-the-money options for gamma hedge
        # Assume ATM option has gamma of 0.05
        option_gamma = 0.05
        contracts_needed = -int(gamma_imbalance / option_gamma)
        
        option_cost = abs(contracts_needed) * 100 * 5.0  # $5 per contract, 100 shares per contract
        
        return {
            'hedge_required': True,
            'current_gamma': current_gamma,
            'target_gamma': target_gamma,
            'imbalance': gamma_imbalance,
            'hedge_instrument': 'ATM_OPTIONS',
            'contracts_needed': contracts_needed,
            'estimated_cost': option_cost,
            'option_type': 'call' if contracts_needed > 0 else 'put'
        }
        
    async def calculate_volatility_hedge(self, portfolio: Portfolio, target_vega: float = 0.0) -> Dict[str, Any]:
        """Calculate volatility hedge using VIX products"""
        current_vega = portfolio.total_vega
        vega_imbalance = current_vega - target_vega
        
        if abs(vega_imbalance) < 100:
            return {
                'hedge_required': False,
                'current_vega': current_vega,
                'target_vega': target_vega,
                'imbalance': vega_imbalance
            }
            
        # VIX hedge calculation
        # Assume VXX has vega-like exposure of -2 per share (inverse relationship)
        vxx_shares_needed = int(vega_imbalance / 2)
        
        vxx_cost = abs(vxx_shares_needed) * 25  # Assume VXX at $25
        
        return {
            'hedge_required': True,
            'current_vega': current_vega,
            'target_vega': target_vega,
            'imbalance': vega_imbalance,
            'hedge_instrument': 'VXX',
            'hedge_quantity': vxx_shares_needed,
            'estimated_cost': vxx_cost,
            'hedge_rationale': 'VIX product hedge for volatility exposure'
        }
        
    async def calculate_tail_hedge(self, portfolio: Portfolio, hedge_ratio: float = 0.02) -> Dict[str, Any]:
        """Calculate tail risk hedge using protective puts"""
        portfolio_value = portfolio.net_value
        hedge_budget = portfolio_value * hedge_ratio
        
        # Suggest OTM puts for tail hedge
        # Assume 10% OTM puts cost 1% of notional
        put_strike_ratio = 0.90  # 10% out of the money
        put_cost_ratio = 0.01  # 1% of notional
        
        notional_coverage = hedge_budget / put_cost_ratio
        
        return {
            'hedge_type': 'tail_protection',
            'portfolio_value': portfolio_value,
            'hedge_budget': hedge_budget,
            'recommended_coverage': notional_coverage,
            'put_strike_level': f"{put_strike_ratio:.1%} of current price",
            'estimated_cost': hedge_budget,
            'protection_level': '10% downside protection',
            'hedge_instrument': 'OTM_PUTS'
        }
        
    async def optimize_hedge_portfolio(self, portfolio: Portfolio, 
                                     constraints: Dict[str, float]) -> Dict[str, Any]:
        """Optimize hedge portfolio using mean-variance optimization"""
        
        # Define optimization objective
        def objective(weights):
            # Minimize portfolio variance subject to constraints
            # Simplified - in practice would use full covariance matrix
            portfolio_risk = np.sum(weights ** 2) * 0.04  # Assume correlation
            return portfolio_risk
            
        # Constraints
        n_assets = len(portfolio.positions) + 3  # Add hedge instruments
        bounds = [(0, constraints.get('max_weight', 0.3)) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(objective, x0, bounds=bounds, method='SLSQP')
        
        if result.success:
            optimal_weights = result.x
            optimized_risk = result.fun
            
            return {
                'optimization_successful': True,
                'optimal_weights': optimal_weights.tolist(),
                'optimized_risk': optimized_risk,
                'hedge_allocation': {
                    'spy_hedge': optimal_weights[-3],
                    'vix_hedge': optimal_weights[-2], 
                    'bond_hedge': optimal_weights[-1]
                }
            }
        else:
            return {
                'optimization_successful': False,
                'error': result.message
            }

class GnosisHedgeAgent:
    """Main Gnosis Hedge Agent - Agent 1"""
    
    def __init__(self, risk_tolerance: RiskLevel = RiskLevel.MODERATE):
        self.risk_tolerance = risk_tolerance
        self.position_sizer = PositionSizer(risk_tolerance)
        self.risk_manager = RiskManager(risk_tolerance)
        self.hedge_calculator = HedgeCalculator()
        
        self.active_hedges = {}
        self.risk_alerts = []
        
        logger.info(f"Gnosis Hedge Agent initialized with {risk_tolerance.value} risk tolerance")
        
    async def analyze_portfolio_risk(self, portfolio: Portfolio, 
                                   price_history: List[float] = None) -> Dict[str, Any]:
        """Comprehensive portfolio risk analysis"""
        
        # Position limit checks
        limit_checks = self.risk_manager.check_position_limits(portfolio)
        
        # Risk metrics calculation
        if price_history:
            risk_metrics = self.risk_manager.calculate_risk_metrics(portfolio, price_history)
        else:
            risk_metrics = RiskMetrics(0, 0, 0, 0, 0, 0, 1, 0, 0.20)
            
        # Greeks analysis
        greeks_analysis = {
            'total_delta': portfolio.total_delta,
            'total_gamma': portfolio.total_gamma,
            'total_theta': portfolio.total_theta,
            'total_vega': portfolio.total_vega,
            'delta_risk': 'HIGH' if abs(portfolio.total_delta) > 200 else 'LOW',
            'gamma_risk': 'HIGH' if abs(portfolio.total_gamma) > 100 else 'LOW',
            'vega_risk': 'HIGH' if abs(portfolio.total_vega) > 2000 else 'LOW'
        }
        
        return {
            'portfolio_summary': {
                'total_positions': len(portfolio.positions),
                'net_value': portfolio.net_value,
                'cash': portfolio.cash,
                'total_pnl': portfolio.total_pnl
            },
            'risk_metrics': {
                'var_95': risk_metrics.var_95,
                'var_99': risk_metrics.var_99,
                'expected_shortfall': risk_metrics.expected_shortfall,
                'max_drawdown': risk_metrics.max_drawdown,
                'sharpe_ratio': risk_metrics.sharpe_ratio,
                'volatility': risk_metrics.volatility
            },
            'greeks_analysis': greeks_analysis,
            'limit_checks': limit_checks,
            'risk_alerts': self._generate_risk_alerts(portfolio, limit_checks, risk_metrics)
        }
        
    async def calculate_position_size(self, symbol: str, strategy: str, 
                                    portfolio: Portfolio, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal position size for new trade"""
        
        current_price = market_data.get('price', 100)
        volatility = market_data.get('volatility', 0.20)
        win_rate = market_data.get('win_rate', 0.55)
        avg_win = market_data.get('avg_win', 0.10)
        avg_loss = market_data.get('avg_loss', 0.05)
        
        # Kelly Criterion sizing
        kelly_size = self.position_sizer.kelly_criterion(win_rate, avg_win, avg_loss)
        kelly_position_value = portfolio.net_value * kelly_size
        kelly_shares = int(kelly_position_value / current_price)
        
        # Volatility-based sizing
        vol_size = self.position_sizer.volatility_adjusted_size(
            portfolio.net_value, 0.15, volatility
        )
        vol_shares = int(vol_size / current_price)
        
        # Risk parity sizing
        risk_parity_size = self.position_sizer.risk_parity_size(
            portfolio, volatility, 0.10
        )
        risk_parity_shares = int(risk_parity_size / current_price)
        
        # Final recommendation (conservative approach)
        recommended_shares = min(kelly_shares, vol_shares, risk_parity_shares)
        
        return {
            'symbol': symbol,
            'strategy': strategy,
            'current_price': current_price,
            'sizing_methods': {
                'kelly_criterion': {
                    'shares': kelly_shares,
                    'position_value': kelly_position_value,
                    'portfolio_percent': kelly_size * 100
                },
                'volatility_adjusted': {
                    'shares': vol_shares,
                    'position_value': vol_size,
                    'target_volatility': 0.15
                },
                'risk_parity': {
                    'shares': risk_parity_shares,
                    'position_value': risk_parity_size,
                    'risk_contribution': 0.10
                }
            },
            'recommendation': {
                'shares': recommended_shares,
                'position_value': recommended_shares * current_price,
                'portfolio_percent': (recommended_shares * current_price) / portfolio.net_value * 100,
                'rationale': 'Conservative minimum of all sizing methods'
            }
        }
        
    async def generate_hedge_recommendations(self, portfolio: Portfolio) -> Dict[str, Any]:
        """Generate comprehensive hedge recommendations"""
        
        recommendations = {}
        
        # Delta hedge
        delta_hedge = await self.hedge_calculator.calculate_delta_hedge(portfolio)
        if delta_hedge['hedge_required']:
            recommendations['delta_hedge'] = delta_hedge
            
        # Gamma hedge  
        gamma_hedge = await self.hedge_calculator.calculate_gamma_hedge(portfolio)
        if gamma_hedge['hedge_required']:
            recommendations['gamma_hedge'] = gamma_hedge
            
        # Volatility hedge
        vol_hedge = await self.hedge_calculator.calculate_volatility_hedge(portfolio)
        if vol_hedge['hedge_required']:
            recommendations['volatility_hedge'] = vol_hedge
            
        # Tail hedge
        tail_hedge = await self.hedge_calculator.calculate_tail_hedge(portfolio)
        recommendations['tail_hedge'] = tail_hedge
        
        # Portfolio optimization
        optimization = await self.hedge_calculator.optimize_hedge_portfolio(
            portfolio, {'max_weight': 0.25}
        )
        recommendations['portfolio_optimization'] = optimization
        
        return {
            'hedge_analysis_timestamp': datetime.now().isoformat(),
            'portfolio_greeks': {
                'delta': portfolio.total_delta,
                'gamma': portfolio.total_gamma,
                'theta': portfolio.total_theta,
                'vega': portfolio.total_vega
            },
            'hedge_recommendations': recommendations,
            'priority_actions': self._prioritize_hedge_actions(recommendations)
        }
        
    def _generate_risk_alerts(self, portfolio: Portfolio, limit_checks: Dict[str, bool], 
                            risk_metrics: RiskMetrics) -> List[Dict[str, str]]:
        """Generate risk alerts based on portfolio analysis"""
        alerts = []
        
        # Position limit alerts
        if not limit_checks['delta_limit']:
            alerts.append({
                'type': 'POSITION_LIMIT',
                'severity': 'HIGH',
                'message': f'Delta exposure ({portfolio.total_delta:.0f}) exceeds limits',
                'recommendation': 'Reduce delta exposure or implement delta hedge'
            })
            
        if not limit_checks['gamma_limit']:
            alerts.append({
                'type': 'POSITION_LIMIT', 
                'severity': 'MEDIUM',
                'message': f'Gamma exposure ({portfolio.total_gamma:.2f}) exceeds limits',
                'recommendation': 'Consider gamma hedging with options'
            })
            
        # Risk metric alerts
        if risk_metrics.var_95 > portfolio.net_value * 0.05:  # 5% VaR threshold
            alerts.append({
                'type': 'RISK_METRIC',
                'severity': 'HIGH', 
                'message': f'Daily VaR ({risk_metrics.var_95:.0f}) exceeds 5% of portfolio value',
                'recommendation': 'Reduce position sizes or implement protective hedges'
            })
            
        if risk_metrics.sharpe_ratio < 0.5:
            alerts.append({
                'type': 'PERFORMANCE',
                'severity': 'MEDIUM',
                'message': f'Poor risk-adjusted returns (Sharpe: {risk_metrics.sharpe_ratio:.2f})',
                'recommendation': 'Review trading strategy and risk management'
            })
            
        return alerts
        
    def _prioritize_hedge_actions(self, recommendations: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prioritize hedge actions by importance and cost"""
        actions = []
        
        # High priority: Delta hedge if large imbalance
        if 'delta_hedge' in recommendations:
            delta_rec = recommendations['delta_hedge']
            if abs(delta_rec['imbalance']) > 100:
                actions.append({
                    'priority': 'HIGH',
                    'action': 'DELTA_HEDGE',
                    'description': f"Hedge {delta_rec['imbalance']:.0f} delta exposure with {delta_rec['hedge_instrument']}",
                    'estimated_cost': delta_rec['estimated_cost']
                })
                
        # Medium priority: Volatility hedge
        if 'volatility_hedge' in recommendations:
            vol_rec = recommendations['volatility_hedge']
            actions.append({
                'priority': 'MEDIUM',
                'action': 'VOLATILITY_HEDGE',
                'description': f"Hedge vega exposure with {vol_rec['hedge_instrument']}",
                'estimated_cost': vol_rec['estimated_cost']
            })
            
        # Low priority: Tail hedge (insurance)
        if 'tail_hedge' in recommendations:
            tail_rec = recommendations['tail_hedge']
            actions.append({
                'priority': 'LOW',
                'action': 'TAIL_HEDGE',
                'description': f"Implement tail protection with {tail_rec['hedge_instrument']}",
                'estimated_cost': tail_rec['estimated_cost']
            })
            
        return actions
        
    async def run_hedge_analysis(self, portfolio_data: Dict[str, Any], 
                               market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main analysis function for Gnosis system integration"""
        
        # Create portfolio object
        positions = []
        for pos_data in portfolio_data.get('positions', []):
            position = Position(**pos_data)
            positions.append(position)
            
        portfolio = Portfolio(
            positions=positions,
            cash=portfolio_data.get('cash', 0)
        )
        
        # Run comprehensive analysis
        risk_analysis = await self.analyze_portfolio_risk(
            portfolio, market_data.get('price_history', [])
        )
        
        hedge_recommendations = await self.generate_hedge_recommendations(portfolio)
        
        # Generate signal for Gnosis system
        signal_strength = self._calculate_signal_strength(risk_analysis, hedge_recommendations)
        
        return {
            'agent': 'hedge_agent',
            'timestamp': datetime.now().isoformat(),
            'signal_strength': signal_strength,
            'risk_analysis': risk_analysis,
            'hedge_recommendations': hedge_recommendations,
            'portfolio_summary': {
                'net_value': portfolio.net_value,
                'total_pnl': portfolio.total_pnl,
                'cash': portfolio.cash,
                'position_count': len(portfolio.positions)
            },
            'agent_confidence': self._calculate_confidence(risk_analysis)
        }
        
    def _calculate_signal_strength(self, risk_analysis: Dict[str, Any], 
                                 hedge_recommendations: Dict[str, Any]) -> float:
        """Calculate signal strength based on risk and hedge analysis"""
        
        # Base signal strength
        signal = 0.5
        
        # Adjust for risk alerts
        high_risk_alerts = len([alert for alert in risk_analysis['risk_alerts'] 
                               if alert['severity'] == 'HIGH'])
        signal -= high_risk_alerts * 0.2
        
        # Adjust for Greeks exposure
        greeks = risk_analysis['greeks_analysis']
        if greeks['delta_risk'] == 'HIGH':
            signal -= 0.15
        if greeks['gamma_risk'] == 'HIGH':
            signal -= 0.10
            
        # Adjust for hedge requirements
        hedge_recs = hedge_recommendations.get('hedge_recommendations', {})
        required_hedges = len([h for h in hedge_recs.values() 
                             if isinstance(h, dict) and h.get('hedge_required')])
        signal -= required_hedges * 0.1
        
        # Clamp between 0 and 1
        return max(0.0, min(1.0, signal))
        
    def _calculate_confidence(self, risk_analysis: Dict[str, Any]) -> float:
        """Calculate agent confidence in analysis"""
        
        # Base confidence
        confidence = 0.8
        
        # Reduce confidence if insufficient data
        if risk_analysis['risk_metrics']['var_95'] == 0:
            confidence -= 0.3
            
        # Reduce confidence for extreme risk
        high_risk_alerts = len([alert for alert in risk_analysis['risk_alerts'] 
                               if alert['severity'] == 'HIGH'])
        confidence -= high_risk_alerts * 0.1
        
        return max(0.1, min(1.0, confidence))

# Example usage
async def main():
    """Example usage of Gnosis Hedge Agent"""
    
    # Initialize agent
    hedge_agent = GnosisHedgeAgent(RiskLevel.MODERATE)
    
    # Example portfolio data
    portfolio_data = {
        'cash': 50000,
        'positions': [
            {
                'symbol': 'SPY_CALL_450',
                'quantity': 10,
                'entry_price': 5.0,
                'current_price': 6.0,
                'position_type': 'call',
                'delta': 0.6,
                'gamma': 0.05,
                'theta': -0.1,
                'vega': 0.2,
                'strike': 450,
                'expiry': '2024-01-19'
            },
            {
                'symbol': 'QQQ',
                'quantity': -100,
                'entry_price': 380,
                'current_price': 375,
                'position_type': 'short',
                'delta': -1.0,
                'gamma': 0.0,
                'theta': 0.0,
                'vega': 0.0
            }
        ]
    }
    
    # Example market data
    market_data = {
        'price_history': [450 + np.random.randn() * 5 for _ in range(100)],
        'volatility': 0.18,
        'spy_price': 452
    }
    
    # Run analysis
    result = await hedge_agent.run_hedge_analysis(portfolio_data, market_data)
    
    print("=== Gnosis Hedge Agent Analysis ===")
    print(f"Signal Strength: {result['signal_strength']:.2f}")
    print(f"Agent Confidence: {result['agent_confidence']:.2f}")
    
    print("\n=== Risk Analysis ===")
    risk = result['risk_analysis']
    print(f"Portfolio Value: ${risk['portfolio_summary']['net_value']:,.0f}")
    print(f"VaR (95%): ${risk['risk_metrics']['var_95']:,.0f}")
    print(f"Total Delta: {risk['greeks_analysis']['total_delta']:.0f}")
    print(f"Risk Alerts: {len(risk['risk_alerts'])}")
    
    print("\n=== Hedge Recommendations ===")
    hedges = result['hedge_recommendations']['hedge_recommendations']
    for hedge_type, hedge_data in hedges.items():
        if isinstance(hedge_data, dict) and hedge_data.get('hedge_required'):
            print(f"{hedge_type}: {hedge_data.get('hedge_instrument', 'N/A')} "
                  f"({hedge_data.get('hedge_quantity', 0)} units)")

if __name__ == "__main__":
    asyncio.run(main())