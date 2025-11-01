#!/usr/bin/env python3
"""
Agent 4: Signal Mixer
Integrates outputs from all agents into coherent trading signals and decisions
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
import logging

# Import other agents (optional imports for testing)
try:
    from dhpe_engine import DHPEEngine, DHPEMetrics
    from agent2_advanced_liquidity import AdvancedLiquidityAnalyzer  
    from agent3_sentiment import Agent3SentimentInterpreter, MarketRegime
    from agent1_hedge import Agent1HedgeEngine, HedgeRecommendation, RiskLevel
except ImportError:
    # Create mock classes for standalone testing
    class DHPEMetrics:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
    
    class MarketRegime(Enum):
        BULL_EXTREME = "BULL_EXTREME"
        BULL_STRONG = "BULL_STRONG"
        BULL_MODERATE = "BULL_MODERATE"
        NEUTRAL = "NEUTRAL"
        BEAR_MODERATE = "BEAR_MODERATE"
        BEAR_STRONG = "BEAR_STRONG"
        BEAR_EXTREME = "BEAR_EXTREME"
    
    class RiskLevel(Enum):
        LOW = "LOW"
        MODERATE = "MODERATE"
        HIGH = "HIGH"
        EXTREME = "EXTREME"
    
    class HedgeRecommendation:
        pass

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    """Signal strength classifications"""
    VERY_WEAK = "VERY_WEAK"
    WEAK = "WEAK" 
    MODERATE = "MODERATE"
    STRONG = "STRONG"
    VERY_STRONG = "VERY_STRONG"

class TradeDirection(Enum):
    """Trading direction recommendations"""
    STRONG_BEARISH = "STRONG_BEARISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    STRONG_BULLISH = "STRONG_BULLISH"

class StrategyType(Enum):
    """Recommended strategy types"""
    DIRECTIONAL_LONG = "DIRECTIONAL_LONG"
    DIRECTIONAL_SHORT = "DIRECTIONAL_SHORT"
    VOLATILITY_EXPANSION = "VOLATILITY_EXPANSION"
    VOLATILITY_CONTRACTION = "VOLATILITY_CONTRACTION"
    INCOME_GENERATION = "INCOME_GENERATION"
    HEDGE_EXISTING = "HEDGE_EXISTING"
    CASH = "CASH"

@dataclass
class AgentSignal:
    """Individual agent signal contribution"""
    agent_name: str
    signal_strength: float  # -1.0 to +1.0
    confidence: float      # 0.0 to 1.0
    direction: TradeDirection
    reasoning: str
    weight: float          # Agent weight in final decision
    timestamp: datetime

@dataclass
class IntegratedSignal:
    """Final integrated signal from all agents"""
    direction: TradeDirection
    strength: SignalStrength
    confidence: float
    strategy_recommendation: StrategyType
    
    # Agent contributions
    dhpe_signal: AgentSignal
    liquidity_signal: AgentSignal
    sentiment_signal: AgentSignal
    risk_signal: AgentSignal
    
    # Meta information
    signal_agreement: float  # How much agents agree (0-1)
    conviction_score: float  # Overall conviction (0-1)
    risk_adjusted_confidence: float
    
    # Actionable recommendations
    entry_triggers: List[str]
    exit_triggers: List[str]
    position_size_modifier: float  # 0.5 = half size, 2.0 = double size
    
    timestamp: datetime

class Agent4SignalMixer:
    """
    Agent 4: Signal Integration and Decision Making
    
    Combines outputs from:
    - DHPE Engine (gamma exposure, hedge pressure)
    - Agent 2 (liquidity analysis)  
    - Agent 3 (sentiment classification)
    - Agent 1 (risk management, hedging)
    
    Produces integrated trading signals with confidence levels
    """
    
    def __init__(self, base_account_size: float = 100000):
        """Initialize signal mixer"""
        self.account_size = base_account_size
        self.signal_history: List[IntegratedSignal] = []
        
        # Agent weights (must sum to 1.0)
        self.agent_weights = {
            'dhpe': 0.35,      # DHPE gets highest weight (market structure)
            'sentiment': 0.25,  # Sentiment regime important
            'liquidity': 0.25,  # Liquidity flows critical
            'risk': 0.15       # Risk management overlay
        }
        
        # Signal filtering parameters
        self.min_confidence_threshold = 0.4  # Minimum confidence to act
        self.agreement_threshold = 0.6       # Minimum agent agreement
        self.conviction_threshold = 0.5      # Minimum conviction to recommend
        
        # Position sizing parameters
        self.base_position_size = 0.02  # 2% of account per trade
        self.max_position_size = 0.05   # 5% maximum
        
        logger.info("Agent 4 Signal Mixer initialized")
    
    def process_dhpe_signal(self, dhpe_metrics: DHPEMetrics, spot_price: float) -> AgentSignal:
        """Convert DHPE metrics to signal"""
        
        # Dealer positioning analysis
        dealer_pos = dhpe_metrics.dealer_positioning
        hedge_pressure = dhpe_metrics.hedge_pressure_score
        
        # Distance from max pain creates directional bias
        max_pain_distance = (spot_price - dhpe_metrics.max_pain_strike) / spot_price
        
        # Signal strength based on hedge pressure and positioning
        if dealer_pos < -0.5 and hedge_pressure > 0.7:
            # Dealers short gamma + high pressure = strong bullish
            signal_strength = 0.8 + min(0.2, hedge_pressure - 0.7)
            direction = TradeDirection.STRONG_BULLISH
            reasoning = f"Dealers short gamma (δ={dealer_pos:.2f}) with extreme hedge pressure"
        elif dealer_pos < -0.2 and hedge_pressure > 0.5:
            signal_strength = 0.4 + (hedge_pressure - 0.5) * 0.8
            direction = TradeDirection.BULLISH  
            reasoning = f"Moderate dealer short gamma with elevated hedge pressure"
        elif dealer_pos > 0.5 and hedge_pressure > 0.7:
            # Dealers long gamma + high pressure = strong bearish
            signal_strength = -(0.8 + min(0.2, hedge_pressure - 0.7))
            direction = TradeDirection.STRONG_BEARISH
            reasoning = f"Dealers long gamma (δ={dealer_pos:.2f}) with extreme hedge pressure"
        elif dealer_pos > 0.2 and hedge_pressure > 0.5:
            signal_strength = -(0.4 + (hedge_pressure - 0.5) * 0.8)
            direction = TradeDirection.BEARISH
            reasoning = f"Moderate dealer long gamma with elevated hedge pressure"
        else:
            # Neutral conditions
            signal_strength = max_pain_distance * 0.3  # Weak bias toward max pain
            direction = TradeDirection.NEUTRAL
            reasoning = f"Balanced dealer positioning, max pain influence"
        
        # Confidence based on data quality and pressure levels
        confidence = min(0.95, 0.5 + hedge_pressure * 0.5)
        
        return AgentSignal(
            agent_name="DHPE",
            signal_strength=signal_strength,
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            weight=self.agent_weights['dhpe'],
            timestamp=datetime.now()
        )
    
    def process_sentiment_signal(self, regime: MarketRegime, regime_confidence: float, 
                               bias_scores: Dict[str, float]) -> AgentSignal:
        """Convert sentiment analysis to signal"""
        
        # Map regime to signal strength
        regime_mapping = {
            MarketRegime.BULL_EXTREME: (0.9, TradeDirection.STRONG_BULLISH),
            MarketRegime.BULL_STRONG: (0.7, TradeDirection.STRONG_BULLISH),
            MarketRegime.BULL_MODERATE: (0.4, TradeDirection.BULLISH),
            MarketRegime.NEUTRAL: (0.0, TradeDirection.NEUTRAL),
            MarketRegime.BEAR_MODERATE: (-0.4, TradeDirection.BEARISH),
            MarketRegime.BEAR_STRONG: (-0.7, TradeDirection.STRONG_BEARISH),
            MarketRegime.BEAR_EXTREME: (-0.9, TradeDirection.STRONG_BEARISH)
        }
        
        base_strength, direction = regime_mapping[regime]
        
        # Adjust for behavioral biases
        herding_bias = bias_scores.get('herding', 0)
        loss_aversion = bias_scores.get('loss_aversion', 0)
        
        # High herding amplifies signals
        bias_multiplier = 1.0 + (herding_bias * 0.3)
        
        # Loss aversion creates asymmetric bearish bias
        if base_strength < 0:
            bias_multiplier *= (1.0 + loss_aversion * 0.2)
        
        signal_strength = base_strength * bias_multiplier
        
        # Confidence adjusted by regime stability
        confidence = regime_confidence * (1.0 - herding_bias * 0.1)  # Herding reduces confidence
        
        reasoning = f"{regime.value} regime (conf:{regime_confidence:.2f}) with {herding_bias:.2f} herding bias"
        
        return AgentSignal(
            agent_name="Sentiment",
            signal_strength=np.clip(signal_strength, -1.0, 1.0),
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            weight=self.agent_weights['sentiment'],
            timestamp=datetime.now()
        )
    
    def process_liquidity_signal(self, liquidity_metrics: Dict[str, float]) -> AgentSignal:
        """Convert liquidity analysis to signal"""
        
        # Extract key liquidity metrics
        vwap_deviation = liquidity_metrics.get('vwap_deviation', 0)
        volume_profile_strength = liquidity_metrics.get('volume_profile_strength', 0.5)
        aggressive_ratio = liquidity_metrics.get('aggressive_buy_ratio', 0.5)
        
        # VWAP deviation provides directional bias
        vwap_signal = np.tanh(vwap_deviation * 5)  # Normalize to [-1, 1]
        
        # Aggressive buying/selling provides confirmation
        aggression_signal = (aggressive_ratio - 0.5) * 2  # Convert to [-1, 1]
        
        # Combined signal
        signal_strength = (vwap_signal * 0.6) + (aggression_signal * 0.4)
        
        # Direction mapping
        if signal_strength > 0.5:
            direction = TradeDirection.BULLISH
        elif signal_strength > 0.2:
            direction = TradeDirection.BULLISH
        elif signal_strength < -0.5:
            direction = TradeDirection.BEARISH
        elif signal_strength < -0.2:
            direction = TradeDirection.BEARISH
        else:
            direction = TradeDirection.NEUTRAL
        
        # Confidence based on volume profile strength
        confidence = volume_profile_strength
        
        reasoning = f"VWAP dev:{vwap_deviation:.2%}, Aggressive ratio:{aggressive_ratio:.2f}"
        
        return AgentSignal(
            agent_name="Liquidity",
            signal_strength=signal_strength,
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            weight=self.agent_weights['liquidity'],
            timestamp=datetime.now()
        )
    
    def process_risk_signal(self, risk_level: RiskLevel, hedge_recommendations: List[HedgeRecommendation],
                          portfolio_delta: float) -> AgentSignal:
        """Convert risk analysis to signal"""
        
        # Risk level affects position sizing and direction
        risk_mapping = {
            RiskLevel.LOW: 1.0,      # Full signal strength
            RiskLevel.MODERATE: 0.7,  # Reduced strength
            RiskLevel.HIGH: 0.4,      # Much reduced
            RiskLevel.EXTREME: 0.1    # Minimal exposure
        }
        
        risk_multiplier = risk_mapping[risk_level]
        
        # Portfolio delta creates directional bias for hedging
        if abs(portfolio_delta) > 100:  # Significant exposure
            if portfolio_delta > 0:
                # Long delta - bearish signal for hedging
                signal_strength = -min(0.8, abs(portfolio_delta) / 500)
                direction = TradeDirection.BEARISH
                reasoning = f"Long delta hedge: δ={portfolio_delta:.0f}, risk={risk_level.value}"
            else:
                # Short delta - bullish signal for hedging  
                signal_strength = min(0.8, abs(portfolio_delta) / 500)
                direction = TradeDirection.BULLISH
                reasoning = f"Short delta hedge: δ={portfolio_delta:.0f}, risk={risk_level.value}"
        else:
            # Neutral delta
            signal_strength = 0.0
            direction = TradeDirection.NEUTRAL
            reasoning = f"Balanced portfolio: δ={portfolio_delta:.0f}, risk={risk_level.value}"
        
        # Apply risk multiplier
        signal_strength *= risk_multiplier
        
        # Confidence inversely related to risk level
        confidence = 1.0 - (list(RiskLevel).index(risk_level) * 0.2)
        
        return AgentSignal(
            agent_name="Risk",
            signal_strength=signal_strength,
            confidence=confidence,
            direction=direction,
            reasoning=reasoning,
            weight=self.agent_weights['risk'],
            timestamp=datetime.now()
        )
    
    def calculate_signal_agreement(self, signals: List[AgentSignal]) -> float:
        """Calculate agreement between agent signals"""
        
        signal_strengths = [s.signal_strength for s in signals]
        
        # All positive or all negative = high agreement
        all_positive = all(s >= 0 for s in signal_strengths)
        all_negative = all(s <= 0 for s in signal_strengths)
        
        if all_positive or all_negative:
            # Calculate variance - lower variance = higher agreement
            variance = np.var(signal_strengths)
            agreement = max(0, 1.0 - variance)
        else:
            # Mixed signals - calculate how much they cancel out
            positive_sum = sum(s for s in signal_strengths if s > 0)
            negative_sum = abs(sum(s for s in signal_strengths if s < 0))
            total_magnitude = positive_sum + negative_sum
            
            if total_magnitude > 0:
                agreement = 1.0 - (min(positive_sum, negative_sum) / total_magnitude)
            else:
                agreement = 0.5  # Neutral
        
        return np.clip(agreement, 0.0, 1.0)
    
    def integrate_signals(self, dhpe_signal: AgentSignal, sentiment_signal: AgentSignal,
                         liquidity_signal: AgentSignal, risk_signal: AgentSignal) -> IntegratedSignal:
        """Integrate all agent signals into final recommendation"""
        
        signals = [dhpe_signal, sentiment_signal, liquidity_signal, risk_signal]
        
        # Weighted average of signal strengths
        weighted_strength = sum(s.signal_strength * s.weight for s in signals)
        
        # Weighted confidence
        weighted_confidence = sum(s.confidence * s.weight for s in signals)
        
        # Signal agreement
        agreement = self.calculate_signal_agreement(signals)
        
        # Overall conviction (confidence * agreement)
        conviction = weighted_confidence * agreement
        
        # Risk-adjusted confidence (reduced by risk signal)
        risk_adjustment = 1.0 - max(0, -risk_signal.signal_strength) * 0.5
        risk_adjusted_confidence = weighted_confidence * risk_adjustment
        
        # Determine final direction
        if weighted_strength > 0.6:
            direction = TradeDirection.STRONG_BULLISH
            strength = SignalStrength.VERY_STRONG if conviction > 0.8 else SignalStrength.STRONG
        elif weighted_strength > 0.3:
            direction = TradeDirection.BULLISH
            strength = SignalStrength.STRONG if conviction > 0.7 else SignalStrength.MODERATE
        elif weighted_strength < -0.6:
            direction = TradeDirection.STRONG_BEARISH
            strength = SignalStrength.VERY_STRONG if conviction > 0.8 else SignalStrength.STRONG
        elif weighted_strength < -0.3:
            direction = TradeDirection.BEARISH  
            strength = SignalStrength.STRONG if conviction > 0.7 else SignalStrength.MODERATE
        else:
            direction = TradeDirection.NEUTRAL
            strength = SignalStrength.WEAK
        
        # Strategy recommendation
        strategy = self._determine_strategy(direction, strength, dhpe_signal, sentiment_signal)
        
        # Entry/exit triggers
        entry_triggers = self._generate_entry_triggers(direction, signals)
        exit_triggers = self._generate_exit_triggers(direction, conviction)
        
        # Position sizing
        position_size_modifier = self._calculate_position_size(conviction, risk_adjusted_confidence)
        
        return IntegratedSignal(
            direction=direction,
            strength=strength,
            confidence=weighted_confidence,
            strategy_recommendation=strategy,
            dhpe_signal=dhpe_signal,
            liquidity_signal=liquidity_signal,
            sentiment_signal=sentiment_signal,
            risk_signal=risk_signal,
            signal_agreement=agreement,
            conviction_score=conviction,
            risk_adjusted_confidence=risk_adjusted_confidence,
            entry_triggers=entry_triggers,
            exit_triggers=exit_triggers,
            position_size_modifier=position_size_modifier,
            timestamp=datetime.now()
        )
    
    def _determine_strategy(self, direction: TradeDirection, strength: SignalStrength,
                          dhpe_signal: AgentSignal, sentiment_signal: AgentSignal) -> StrategyType:
        """Determine recommended strategy type"""
        
        if direction == TradeDirection.NEUTRAL or strength == SignalStrength.WEAK:
            return StrategyType.CASH
        
        # High conviction directional trades
        if strength in [SignalStrength.STRONG, SignalStrength.VERY_STRONG]:
            if direction in [TradeDirection.BULLISH, TradeDirection.STRONG_BULLISH]:
                return StrategyType.DIRECTIONAL_LONG
            else:
                return StrategyType.DIRECTIONAL_SHORT
        
        # Volatility-based strategies for moderate signals
        if dhpe_signal.signal_strength and abs(dhpe_signal.signal_strength) > 0.5:
            # High hedge pressure suggests volatility
            if direction in [TradeDirection.BULLISH, TradeDirection.STRONG_BULLISH]:
                return StrategyType.VOLATILITY_EXPANSION
            else:
                return StrategyType.VOLATILITY_CONTRACTION
        
        # Income generation for moderate bullish
        if direction == TradeDirection.BULLISH and strength == SignalStrength.MODERATE:
            return StrategyType.INCOME_GENERATION
        
        return StrategyType.HEDGE_EXISTING
    
    def _generate_entry_triggers(self, direction: TradeDirection, signals: List[AgentSignal]) -> List[str]:
        """Generate entry trigger conditions"""
        
        triggers = []
        
        if direction in [TradeDirection.BULLISH, TradeDirection.STRONG_BULLISH]:
            triggers.append("Price breaks above VWAP with volume")
            triggers.append("RSI moves above 50 with momentum")
            if any(s.agent_name == "DHPE" and s.confidence > 0.7 for s in signals):
                triggers.append("Hedge pressure remains elevated")
        
        elif direction in [TradeDirection.BEARISH, TradeDirection.STRONG_BEARISH]:
            triggers.append("Price breaks below VWAP with volume")
            triggers.append("RSI moves below 50 with momentum")
            if any(s.agent_name == "DHPE" and s.confidence > 0.7 for s in signals):
                triggers.append("Dealer positioning shifts")
        
        else:
            triggers.append("Wait for clearer directional signals")
        
        return triggers
    
    def _generate_exit_triggers(self, direction: TradeDirection, conviction: float) -> List[str]:
        """Generate exit trigger conditions"""
        
        triggers = []
        
        # Standard stops
        triggers.append(f"Stop loss: {2.0 / max(conviction, 0.3):.1f}% adverse move")
        triggers.append(f"Profit target: {conviction * 5:.1f}% favorable move")
        
        # Time-based exits
        if conviction < 0.6:
            triggers.append("Time stop: 2-3 days max hold")
        else:
            triggers.append("Time stop: 5-7 days max hold")
        
        # Signal-based exits
        triggers.append("Agent signal reversal (2+ agents flip)")
        triggers.append("DHPE regime change")
        
        return triggers
    
    def _calculate_position_size(self, conviction: float, risk_adjusted_confidence: float) -> float:
        """Calculate position size multiplier"""
        
        # Base size modified by conviction and risk
        base_multiplier = conviction * risk_adjusted_confidence
        
        # Scale to reasonable range
        position_multiplier = 0.5 + (base_multiplier * 1.5)  # 0.5x to 2.0x
        
        # Cap at maximum
        return min(position_multiplier, 2.0)
    
    def get_integrated_analysis(self, dhpe_metrics: DHPEMetrics, spot_price: float,
                              sentiment_regime: MarketRegime, sentiment_confidence: float,
                              bias_scores: Dict[str, float], liquidity_metrics: Dict[str, float],
                              risk_level: RiskLevel, hedge_recommendations: List[HedgeRecommendation],
                              portfolio_delta: float = 0) -> IntegratedSignal:
        """
        Complete integration of all agent outputs
        
        This is the main interface for getting integrated trading signals
        """
        
        # Process each agent signal
        dhpe_signal = self.process_dhpe_signal(dhpe_metrics, spot_price)
        sentiment_signal = self.process_sentiment_signal(sentiment_regime, sentiment_confidence, bias_scores)
        liquidity_signal = self.process_liquidity_signal(liquidity_metrics)
        risk_signal = self.process_risk_signal(risk_level, hedge_recommendations, portfolio_delta)
        
        # Integrate into final signal
        integrated = self.integrate_signals(dhpe_signal, sentiment_signal, liquidity_signal, risk_signal)
        
        # Store in history
        self.signal_history.append(integrated)
        
        # Keep history limited
        if len(self.signal_history) > 100:
            self.signal_history = self.signal_history[-100:]
        
        logger.info(f"Integrated signal: {integrated.direction.value} "
                   f"(strength: {integrated.strength.value}, conviction: {integrated.conviction_score:.3f})")
        
        return integrated

def create_sample_integration() -> Tuple[Agent4SignalMixer, IntegratedSignal]:
    """Create sample integration for testing"""
    
    # Initialize mixer
    mixer = Agent4SignalMixer()
    
    # Sample DHPE metrics
    dhpe_metrics = DHPEMetrics(
        symbol="SPY",
        timestamp=datetime.now(),
        total_gamma_exposure=-150.5,
        call_gamma_exposure=-80.2,
        put_gamma_exposure=-70.3,
        net_gamma_exposure=-150.5,
        dealer_positioning=-0.75,  # Dealers short gamma
        hedge_pressure_score=0.85,  # High pressure
        max_pain_strike=430.0,
        max_pain_value=25.5,
        call_put_ratio=1.2,
        volume_weighted_iv=0.18,
        vanna_pressure=0.05,
        charm_pressure=0.12
    )
    
    # Sample liquidity metrics
    liquidity_metrics = {
        'vwap_deviation': 0.02,  # 2% above VWAP
        'volume_profile_strength': 0.8,
        'aggressive_buy_ratio': 0.65  # More aggressive buying
    }
    
    # Sample bias scores
    bias_scores = {
        'herding': 0.7,
        'anchoring': 0.4,
        'recency': 0.6,
        'confirmation': 0.5,
        'loss_aversion': 0.3
    }
    
    # Get integrated signal
    integrated = mixer.get_integrated_analysis(
        dhpe_metrics=dhpe_metrics,
        spot_price=432.50,
        sentiment_regime=MarketRegime.BULL_MODERATE,
        sentiment_confidence=0.78,
        bias_scores=bias_scores,
        liquidity_metrics=liquidity_metrics,
        risk_level=RiskLevel.MODERATE,
        hedge_recommendations=[],
        portfolio_delta=0
    )
    
    return mixer, integrated

def test_agent4_mixer():
    """Test Agent 4 Signal Mixer"""
    
    print("=== Testing Agent 4 Signal Mixer ===")
    
    mixer, integrated = create_sample_integration()
    
    print(f"\n🎯 Integrated Trading Signal:")
    print(f"Direction: {integrated.direction.value}")
    print(f"Strength: {integrated.strength.value}")
    print(f"Confidence: {integrated.confidence:.3f}")
    print(f"Strategy: {integrated.strategy_recommendation.value}")
    print(f"Signal Agreement: {integrated.signal_agreement:.3f}")
    print(f"Conviction Score: {integrated.conviction_score:.3f}")
    print(f"Risk-Adjusted Confidence: {integrated.risk_adjusted_confidence:.3f}")
    print(f"Position Size Modifier: {integrated.position_size_modifier:.2f}x")
    
    print(f"\n📊 Agent Contributions:")
    for signal in [integrated.dhpe_signal, integrated.sentiment_signal, 
                   integrated.liquidity_signal, integrated.risk_signal]:
        print(f"{signal.agent_name:10} | {signal.direction.value:15} | "
              f"Strength: {signal.signal_strength:+6.3f} | "
              f"Conf: {signal.confidence:.3f} | "
              f"Weight: {signal.weight:.2f}")
        print(f"{' ':10} | Reasoning: {signal.reasoning}")
        print()
    
    print(f"🚦 Entry Triggers:")
    for trigger in integrated.entry_triggers:
        print(f"  • {trigger}")
    
    print(f"\n🛑 Exit Triggers:")  
    for trigger in integrated.exit_triggers:
        print(f"  • {trigger}")
    
    print(f"\n✅ Agent 4 Signal Mixer test completed!")
    
    return mixer, integrated

if __name__ == "__main__":
    test_agent4_mixer()