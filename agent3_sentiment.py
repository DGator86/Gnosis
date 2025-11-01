#!/usr/bin/env python3
"""
Agent 3: Sentiment Interpreter
Real sentiment analysis with seven regime classifications and behavioral bias detection.
"""

import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MarketRegime(Enum):
    """Seven market regime classifications"""
    BULL_EXTREME = "BULL_EXTREME"
    BULL_STRONG = "BULL_STRONG" 
    BULL_MODERATE = "BULL_MODERATE"
    NEUTRAL = "NEUTRAL"
    BEAR_MODERATE = "BEAR_MODERATE"
    BEAR_STRONG = "BEAR_STRONG"
    BEAR_EXTREME = "BEAR_EXTREME"

@dataclass
class RegimeSignal:
    """Market regime signal with confidence and metadata"""
    regime: MarketRegime
    confidence: float
    timestamp: datetime
    contributing_factors: Dict[str, float]
    bias_scores: Dict[str, float]

@dataclass 
class BehavioralBias:
    """Behavioral bias measurement"""
    herding: float        # Crowd following behavior
    anchoring: float      # Price level fixation  
    recency: float        # Recent performance weighting
    confirmation: float   # Directional persistence
    loss_aversion: float  # Asymmetric loss reactions

class Agent3SentimentInterpreter:
    """
    Agent 3: Real Sentiment Analysis
    
    Features:
    - Seven regime classification system
    - Multi-source sentiment aggregation  
    - Behavioral bias detection
    - Signal hysteresis (prevents whipsaws)
    - State persistence across time periods
    """
    
    def __init__(self):
        """Initialize sentiment interpreter with regime tracking"""
        self.current_regime = MarketRegime.NEUTRAL
        self.regime_confidence = 0.0
        self.regime_history: List[RegimeSignal] = []
        
        # Hysteresis parameters (prevent whipsaws)
        self.hysteresis_threshold = 0.15
        self.min_regime_duration = timedelta(minutes=30)
        self.last_regime_change = datetime.now()
        
        # Bias tracking
        self.bias_history: List[BehavioralBias] = []
        self.sentiment_cache = {}
        
        # Technical indicator weights for regime classification
        self.regime_weights = {
            'rsi': 0.25,
            'macd': 0.20, 
            'momentum': 0.15,
            'bb_position': 0.15,
            'volume_trend': 0.10,
            'volatility': 0.15
        }
        
        logger.info("Agent 3 Sentiment Interpreter initialized")

    def classify_regime(self, market_data: pd.Series) -> Tuple[MarketRegime, float]:
        """
        Classify current market regime based on technical indicators
        
        Args:
            market_data: Series with technical indicators (rsi, macd, etc.)
            
        Returns:
            Tuple of (regime, confidence_score)
        """
        
        # Extract indicators with defaults
        rsi = market_data.get('rsi', 50.0)
        macd = market_data.get('macd', 0.0)
        bb_position = market_data.get('bb_position', 0.5)
        volume_ratio = market_data.get('volume_ratio', 1.0)
        volatility = market_data.get('volatility', 0.02)
        
        # Calculate regime scores (-1.0 to +1.0 where negative = bearish)
        scores = {}
        
        # RSI contribution
        if rsi >= 80:
            scores['rsi'] = 1.0  # Extremely overbought
        elif rsi >= 70:
            scores['rsi'] = 0.6  # Overbought
        elif rsi >= 55:
            scores['rsi'] = 0.3  # Bullish
        elif rsi >= 45:
            scores['rsi'] = 0.0  # Neutral
        elif rsi >= 30:
            scores['rsi'] = -0.3  # Bearish
        elif rsi >= 20:
            scores['rsi'] = -0.6  # Oversold
        else:
            scores['rsi'] = -1.0  # Extremely oversold
            
        # MACD contribution
        macd_normalized = np.tanh(macd * 2)  # Normalize to [-1, 1]
        scores['macd'] = macd_normalized
        
        # Bollinger Band position
        bb_score = (bb_position - 0.5) * 2  # Convert [0,1] to [-1,1]
        scores['bb_position'] = bb_score
        
        # Volume trend (higher volume amplifies existing signal direction)
        base_sentiment = (scores['rsi'] + scores['macd'] + scores['bb_position']) / 3
        volume_multiplier = np.tanh((volume_ratio - 1.0) * 1.5)  # -1 to +1 based on volume
        volume_score = base_sentiment * abs(volume_multiplier) * 0.5  # Amplify existing direction
        scores['volume_trend'] = volume_score
        
        # Volatility (high vol = extreme regimes more likely)
        vol_multiplier = 1.0 + min(volatility * 10, 0.5)  # 1.0 to 1.5
        
        # Calculate weighted composite score
        composite_score = sum(
            scores[indicator] * weight 
            for indicator, weight in self.regime_weights.items()
            if indicator in scores
        )
        
        # Apply volatility multiplier to extreme scores
        if abs(composite_score) > 0.6:
            composite_score *= vol_multiplier
            
        # Map composite score to regime
        regime, base_confidence = self._score_to_regime(composite_score)
        
        # Apply hysteresis if regime would change
        if regime != self.current_regime:
            time_since_change = datetime.now() - self.last_regime_change
            
            # Require higher confidence for regime changes
            confidence_threshold = 0.6
            if time_since_change < self.min_regime_duration:
                confidence_threshold = 0.8
                
            if base_confidence < confidence_threshold:
                # Keep current regime but update confidence
                regime = self.current_regime
                base_confidence *= 0.7  # Reduce confidence for unchanged regime
                
        # Update internal state if regime actually changed
        if regime != self.current_regime:
            self.current_regime = regime
            self.regime_confidence = base_confidence
            self.last_regime_change = datetime.now()
            
            # Store regime signal
            signal = RegimeSignal(
                regime=regime,
                confidence=base_confidence,
                timestamp=datetime.now(),
                contributing_factors=scores.copy(),
                bias_scores=self._calculate_bias_scores(market_data, scores)
            )
            self.regime_history.append(signal)
            
            # Keep history limited
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
                
        return regime, base_confidence
    
    def _score_to_regime(self, score: float) -> Tuple[MarketRegime, float]:
        """Convert composite score to regime classification"""
        
        abs_score = abs(score)
        confidence = min(abs_score, 1.0)
        
        if score >= 0.8:
            return MarketRegime.BULL_EXTREME, confidence
        elif score >= 0.5:
            return MarketRegime.BULL_STRONG, confidence
        elif score >= 0.2:
            return MarketRegime.BULL_MODERATE, confidence
        elif score >= -0.2:
            return MarketRegime.NEUTRAL, max(0.3, 1.0 - abs_score)
        elif score >= -0.5:
            return MarketRegime.BEAR_MODERATE, confidence
        elif score >= -0.8:
            return MarketRegime.BEAR_STRONG, confidence
        else:
            return MarketRegime.BEAR_EXTREME, confidence
    
    def _calculate_bias_scores(self, market_data: pd.Series, factor_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate behavioral bias scores based on market conditions"""
        
        volume = market_data.get('volume', 0)
        price = market_data.get('close', 100)
        volatility = market_data.get('volatility', 0.02)
        
        biases = {}
        
        # Herding bias (high volume + extreme moves)
        volume_z = (volume - 2000000) / 1000000  # Normalize around 2M volume
        price_momentum = sum(factor_scores.get(f, 0) for f in ['rsi', 'macd']) / 2
        biases['herding'] = np.tanh(abs(price_momentum) * abs(volume_z) * 2)
        
        # Anchoring bias (resistance at round numbers, previous highs/lows)
        price_remainder = (price % 10) / 10
        anchoring_strength = 1.0 - min(price_remainder, 1.0 - price_remainder) * 2
        biases['anchoring'] = anchoring_strength * 0.7  # Scale to reasonable range
        
        # Recency bias (overweight recent performance)
        recent_strength = abs(factor_scores.get('rsi', 0)) * 0.5
        biases['recency'] = min(recent_strength, 1.0)
        
        # Confirmation bias (persist in direction)
        directional_consistency = abs(price_momentum) * 0.8
        biases['confirmation'] = min(directional_consistency, 1.0)
        
        # Loss aversion (asymmetric reactions to losses)
        loss_signal = min(factor_scores.get('rsi', 0), 0) * -2  # Amplify oversold
        biases['loss_aversion'] = np.tanh(abs(loss_signal))
        
        return biases
    
    def get_sentiment_summary(self) -> Dict[str, any]:
        """Get comprehensive sentiment analysis summary"""
        
        # Calculate regime stability
        recent_signals = [s for s in self.regime_history if 
                         datetime.now() - s.timestamp < timedelta(hours=4)]
        
        regime_stability = 1.0
        if len(recent_signals) > 1:
            regime_changes = sum(1 for i in range(1, len(recent_signals))
                               if recent_signals[i].regime != recent_signals[i-1].regime)
            regime_stability = max(0.0, 1.0 - regime_changes * 0.2)
        
        # Average bias scores
        avg_biases = {}
        if self.regime_history:
            latest_signal = self.regime_history[-1]
            avg_biases = latest_signal.bias_scores
        
        return {
            'current_regime': self.current_regime.value,
            'regime_confidence': self.regime_confidence,
            'regime_stability': regime_stability,
            'time_since_change': (datetime.now() - self.last_regime_change).total_seconds() / 60,
            'bias_scores': avg_biases,
            'signal_history_length': len(self.regime_history)
        }
    
    def analyze_sentiment_divergence(self, price_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze divergence between price action and sentiment indicators
        
        Args:
            price_data: DataFrame with price and volume data
            
        Returns:
            Dict with divergence metrics
        """
        
        if len(price_data) < 20:
            return {'divergence_score': 0.0, 'reliability': 0.0}
        
        # Calculate price momentum (20-period)
        price_momentum = (price_data['close'].iloc[-1] / price_data['close'].iloc[-20] - 1) * 100
        
        # Get regime momentum based on recent classifications
        regime_scores = []
        for _, row in price_data.tail(20).iterrows():
            regime, conf = self.classify_regime(row)
            # Convert regime to numeric score
            regime_numeric = {
                MarketRegime.BULL_EXTREME: 3,
                MarketRegime.BULL_STRONG: 2,
                MarketRegime.BULL_MODERATE: 1,
                MarketRegime.NEUTRAL: 0,
                MarketRegime.BEAR_MODERATE: -1,
                MarketRegime.BEAR_STRONG: -2,
                MarketRegime.BEAR_EXTREME: -3
            }[regime]
            regime_scores.append(regime_numeric * conf)
        
        sentiment_momentum = np.mean(regime_scores[-10:]) - np.mean(regime_scores[:10])
        
        # Calculate divergence (price up but sentiment down, or vice versa)
        divergence_score = 0.0
        if price_momentum > 2 and sentiment_momentum < -0.5:
            divergence_score = min(abs(price_momentum + sentiment_momentum * 10), 100) / 100
        elif price_momentum < -2 and sentiment_momentum > 0.5:
            divergence_score = min(abs(price_momentum - sentiment_momentum * 10), 100) / 100
        
        reliability = min(len(price_data) / 50, 1.0)  # More data = more reliable
        
        return {
            'divergence_score': divergence_score,
            'price_momentum': price_momentum,
            'sentiment_momentum': sentiment_momentum,
            'reliability': reliability
        }

def test_agent3():
    """Test function for Agent 3"""
    
    print("=== Testing Agent 3 Sentiment Interpreter ===")
    
    # Initialize
    agent3 = Agent3SentimentInterpreter()
    print(f"Initialized: Current regime = {agent3.current_regime.value}")
    
    # Test with various market conditions
    test_cases = [
        {'name': 'Bull Market', 'rsi': 75, 'macd': 1.5, 'bb_position': 0.8, 'volume_ratio': 1.3},
        {'name': 'Bear Market', 'rsi': 25, 'macd': -1.2, 'bb_position': 0.2, 'volume_ratio': 1.5},
        {'name': 'Neutral', 'rsi': 50, 'macd': 0.1, 'bb_position': 0.5, 'volume_ratio': 1.0},
        {'name': 'Extreme Bull', 'rsi': 85, 'macd': 2.5, 'bb_position': 0.95, 'volume_ratio': 2.0},
        {'name': 'Extreme Bear', 'rsi': 15, 'macd': -2.0, 'bb_position': 0.05, 'volume_ratio': 2.5}
    ]
    
    print("\n=== Regime Classification Tests ===")
    for test in test_cases:
        data = pd.Series(test)
        regime, confidence = agent3.classify_regime(data)
        print(f"{test['name']:12} -> {regime.value:15} (confidence: {confidence:.3f})")
    
    # Test summary
    summary = agent3.get_sentiment_summary()
    print(f"\n=== Final Summary ===")
    print(f"Current Regime: {summary['current_regime']}")
    print(f"Confidence: {summary['regime_confidence']:.3f}")
    print(f"Stability: {summary['regime_stability']:.3f}")
    
    return agent3

if __name__ == "__main__":
    test_agent3()