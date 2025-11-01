"""
Gnosis Agent 4: Advanced Signal Mixer
Complete implementation for multi-agent signal integration with conviction scoring
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
import json
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Types of trading signals"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    HEDGE = "hedge"
    REDUCE = "reduce"
    INCREASE = "increase"

class ConfidenceLevel(Enum):
    """Signal confidence levels"""
    VERY_LOW = "very_low"
    LOW = "low" 
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

@dataclass
class AgentSignal:
    """Individual agent signal structure"""
    agent_id: str
    agent_name: str
    signal_type: SignalType
    strength: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    symbol: str
    reasoning: str
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0  # Agent weight in ensemble
    
    def __post_init__(self):
        # Validate ranges
        self.strength = max(0.0, min(1.0, self.strength))
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.weight = max(0.0, min(1.0, self.weight))

@dataclass
class MixedSignal:
    """Final mixed signal output"""
    symbol: str
    signal_type: SignalType
    strength: float
    confidence: ConfidenceLevel
    conviction_score: float
    contributing_agents: List[str]
    consensus_level: float
    risk_reward_ratio: float
    timestamp: datetime = field(default_factory=datetime.now)
    execution_priority: str = "medium"
    supporting_analysis: Dict[str, Any] = field(default_factory=dict)

class SignalWeight:
    """Dynamic agent weight calculation"""
    
    def __init__(self):
        self.performance_history = {}
        self.recent_accuracy = {}
        self.correlation_matrix = {}
        
    def calculate_performance_weight(self, agent_id: str, recent_signals: List[AgentSignal], 
                                   market_outcomes: List[float]) -> float:
        """Calculate weight based on historical performance"""
        
        if len(recent_signals) != len(market_outcomes) or len(recent_signals) == 0:
            return 0.5  # Default neutral weight
            
        # Calculate accuracy score
        correct_predictions = 0
        total_predictions = len(recent_signals)
        
        for signal, outcome in zip(recent_signals, market_outcomes):
            # Simplified accuracy calculation
            if signal.signal_type == SignalType.BUY and outcome > 0:
                correct_predictions += 1
            elif signal.signal_type == SignalType.SELL and outcome < 0:
                correct_predictions += 1
            elif signal.signal_type == SignalType.HOLD and abs(outcome) < 0.01:
                correct_predictions += 1
                
        accuracy = correct_predictions / total_predictions
        
        # Weight based on accuracy with some randomness to prevent overfitting
        base_weight = accuracy
        confidence_adj = np.mean([s.confidence for s in recent_signals])
        
        final_weight = (base_weight * 0.7) + (confidence_adj * 0.3)
        
        self.recent_accuracy[agent_id] = accuracy
        return max(0.1, min(0.9, final_weight))  # Keep within reasonable bounds
        
    def calculate_diversification_weight(self, agent_id: str, all_signals: List[AgentSignal]) -> float:
        """Calculate weight based on signal diversification"""
        
        # Group signals by agent
        agent_signals = {}
        for signal in all_signals:
            if signal.agent_id not in agent_signals:
                agent_signals[signal.agent_id] = []
            agent_signals[signal.agent_id].append(signal)
            
        if agent_id not in agent_signals or len(agent_signals[agent_id]) == 0:
            return 0.5
            
        # Calculate correlation with other agents
        target_strengths = [s.strength for s in agent_signals[agent_id]]
        
        correlations = []
        for other_agent, other_signals in agent_signals.items():
            if other_agent != agent_id and len(other_signals) > 0:
                other_strengths = [s.strength for s in other_signals[:len(target_strengths)]]
                
                if len(other_strengths) == len(target_strengths) and len(target_strengths) > 1:
                    corr, _ = stats.pearsonr(target_strengths, other_strengths)
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                        
        # Lower correlation = higher diversification value
        avg_correlation = np.mean(correlations) if correlations else 0.5
        diversification_weight = 1.0 - avg_correlation
        
        return max(0.2, min(0.8, diversification_weight))

class ConvictionCalculator:
    """Calculate conviction scores for mixed signals"""
    
    def __init__(self):
        self.conviction_factors = {
            'agreement': 0.3,      # How much agents agree
            'confidence': 0.25,    # Average confidence level
            'strength': 0.2,       # Signal strength
            'diversity': 0.15,     # Source diversity
            'timing': 0.1          # Timing consistency
        }
        
    def calculate_conviction(self, signals: List[AgentSignal], mixed_signal: MixedSignal) -> float:
        """Calculate overall conviction score"""
        
        if not signals:
            return 0.0
            
        # Agreement factor - how many agents agree on direction
        same_direction = len([s for s in signals if s.signal_type == mixed_signal.signal_type])
        agreement_score = same_direction / len(signals)
        
        # Confidence factor - average confidence
        confidence_score = np.mean([s.confidence for s in signals])
        
        # Strength factor - weighted average strength
        weights = [s.weight for s in signals]
        strengths = [s.strength for s in signals]
        strength_score = np.average(strengths, weights=weights)
        
        # Diversity factor - how many different agents
        unique_agents = len(set(s.agent_id for s in signals))
        max_possible_agents = 4  # DHPE, Liquidity, Sentiment, Hedge
        diversity_score = min(1.0, unique_agents / max_possible_agents)
        
        # Timing factor - how recent and synchronized the signals are
        now = datetime.now()
        ages = [(now - s.timestamp).total_seconds() / 3600 for s in signals]  # Hours
        timing_score = max(0.0, 1.0 - (np.mean(ages) / 24))  # Decay over 24 hours
        
        # Combine all factors
        conviction = (
            agreement_score * self.conviction_factors['agreement'] +
            confidence_score * self.conviction_factors['confidence'] +
            strength_score * self.conviction_factors['strength'] +
            diversity_score * self.conviction_factors['diversity'] +
            timing_score * self.conviction_factors['timing']
        )
        
        return max(0.0, min(1.0, conviction))
        
    def calculate_risk_reward_ratio(self, signals: List[AgentSignal], 
                                  market_data: Dict[str, Any]) -> float:
        """Calculate estimated risk/reward ratio"""
        
        # Extract risk metrics from supporting data
        risk_signals = []
        reward_signals = []
        
        for signal in signals:
            data = signal.supporting_data
            
            # Look for risk indicators
            if 'var' in data or 'risk' in data:
                risk_val = data.get('var', data.get('risk', 0))
                if risk_val > 0:
                    risk_signals.append(risk_val)
                    
            # Look for reward indicators  
            if 'expected_return' in data or 'target' in data:
                reward_val = data.get('expected_return', data.get('target', 0))
                if reward_val != 0:
                    reward_signals.append(abs(reward_val))
                    
        # Fallback to volatility-based estimate
        current_price = market_data.get('price', 100)
        volatility = market_data.get('volatility', 0.20)
        
        estimated_risk = current_price * volatility * 0.1  # 10% of daily vol
        estimated_reward = current_price * volatility * 0.15  # 15% of daily vol
        
        if risk_signals:
            estimated_risk = np.mean(risk_signals)
        if reward_signals:
            estimated_reward = np.mean(reward_signals)
            
        if estimated_risk > 0:
            return estimated_reward / estimated_risk
        else:
            return 1.0  # Neutral if no risk

class EnsembleMethod:
    """Ensemble methods for signal mixing"""
    
    def __init__(self):
        self.method_weights = {
            'weighted_average': 0.4,
            'majority_vote': 0.3,
            'stacking': 0.2,
            'dynamic_selection': 0.1
        }
        
    def weighted_average_mixing(self, signals: List[AgentSignal]) -> Tuple[SignalType, float]:
        """Weighted average of signal strengths"""
        
        if not signals:
            return SignalType.HOLD, 0.0
            
        # Group by signal type
        signal_groups = {}
        for signal in signals:
            if signal.signal_type not in signal_groups:
                signal_groups[signal.signal_type] = []
            signal_groups[signal.signal_type].append(signal)
            
        # Calculate weighted average for each signal type
        signal_scores = {}
        for sig_type, sig_list in signal_groups.items():
            weights = [s.weight * s.confidence for s in sig_list]
            strengths = [s.strength for s in sig_list]
            
            if sum(weights) > 0:
                weighted_strength = np.average(strengths, weights=weights)
                signal_scores[sig_type] = weighted_strength * (len(sig_list) / len(signals))
            else:
                signal_scores[sig_type] = 0.0
                
        # Return strongest signal
        if signal_scores:
            best_signal = max(signal_scores.items(), key=lambda x: x[1])
            return best_signal[0], best_signal[1]
        else:
            return SignalType.HOLD, 0.0
            
    def majority_vote_mixing(self, signals: List[AgentSignal]) -> Tuple[SignalType, float]:
        """Majority vote with confidence weighting"""
        
        if not signals:
            return SignalType.HOLD, 0.0
            
        # Count votes weighted by confidence
        vote_weights = {}
        for signal in signals:
            vote_weight = signal.confidence * signal.weight
            
            if signal.signal_type not in vote_weights:
                vote_weights[signal.signal_type] = 0
            vote_weights[signal.signal_type] += vote_weight
            
        if vote_weights:
            total_weight = sum(vote_weights.values())
            best_signal = max(vote_weights.items(), key=lambda x: x[1])
            
            return best_signal[0], best_signal[1] / total_weight if total_weight > 0 else 0.0
        else:
            return SignalType.HOLD, 0.0
            
    def stacking_mixing(self, signals: List[AgentSignal], 
                       historical_data: Dict[str, Any] = None) -> Tuple[SignalType, float]:
        """Meta-learning approach to signal mixing"""
        
        # Simplified stacking - would use trained model in production
        if not signals:
            return SignalType.HOLD, 0.0
            
        # Feature engineering
        features = []
        for signal in signals:
            signal_features = [
                signal.strength,
                signal.confidence, 
                signal.weight,
                1.0 if signal.signal_type == SignalType.BUY else 0.0,
                1.0 if signal.signal_type == SignalType.SELL else 0.0,
                len(signal.supporting_data)
            ]
            features.extend(signal_features)
            
        # Pad features to fixed length
        while len(features) < 24:  # 4 agents * 6 features
            features.append(0.0)
        features = features[:24]  # Truncate if too long
        
        # Simple meta-model (in production, would be trained ML model)
        feature_array = np.array(features).reshape(1, -1)
        
        # Simplified decision logic
        buy_score = sum(features[i] * features[i+1] * features[i+3] 
                       for i in range(0, len(features), 6))
        sell_score = sum(features[i] * features[i+1] * features[i+4] 
                        for i in range(0, len(features), 6))
        
        if buy_score > sell_score and buy_score > 0.3:
            return SignalType.BUY, min(1.0, buy_score)
        elif sell_score > buy_score and sell_score > 0.3:
            return SignalType.SELL, min(1.0, sell_score)
        else:
            return SignalType.HOLD, max(buy_score, sell_score)
            
    def dynamic_selection_mixing(self, signals: List[AgentSignal], 
                               market_regime: str = "normal") -> Tuple[SignalType, float]:
        """Dynamically select best agents based on market conditions"""
        
        if not signals:
            return SignalType.HOLD, 0.0
            
        # Agent preferences by market regime
        regime_preferences = {
            "bull": {"dhpe": 0.8, "liquidity": 1.0, "sentiment": 1.2, "hedge": 0.6},
            "bear": {"dhpe": 1.2, "liquidity": 0.8, "sentiment": 0.6, "hedge": 1.4},
            "sideways": {"dhpe": 1.0, "liquidity": 1.2, "sentiment": 0.8, "hedge": 1.0},
            "volatile": {"dhpe": 1.4, "liquidity": 0.6, "sentiment": 0.8, "hedge": 1.2},
            "normal": {"dhpe": 1.0, "liquidity": 1.0, "sentiment": 1.0, "hedge": 1.0}
        }
        
        preferences = regime_preferences.get(market_regime, regime_preferences["normal"])
        
        # Adjust signal weights based on regime
        adjusted_signals = []
        for signal in signals:
            # Map agent names to preferences
            agent_key = signal.agent_name.lower()
            for pref_key in preferences:
                if pref_key in agent_key:
                    regime_multiplier = preferences[pref_key]
                    adjusted_weight = signal.weight * regime_multiplier
                    
                    # Create new signal with adjusted weight
                    adjusted_signal = AgentSignal(
                        agent_id=signal.agent_id,
                        agent_name=signal.agent_name,
                        signal_type=signal.signal_type,
                        strength=signal.strength,
                        confidence=signal.confidence,
                        timestamp=signal.timestamp,
                        symbol=signal.symbol,
                        reasoning=signal.reasoning,
                        supporting_data=signal.supporting_data,
                        weight=min(1.0, adjusted_weight)
                    )
                    adjusted_signals.append(adjusted_signal)
                    break
            else:
                adjusted_signals.append(signal)  # Keep original if no match
                
        # Use weighted average on adjusted signals
        return self.weighted_average_mixing(adjusted_signals)
        
    def mix_signals(self, signals: List[AgentSignal], 
                   market_data: Dict[str, Any] = None) -> Tuple[SignalType, float]:
        """Combine all ensemble methods"""
        
        if not signals:
            return SignalType.HOLD, 0.0
            
        # Get results from each method
        wa_signal, wa_strength = self.weighted_average_mixing(signals)
        mv_signal, mv_strength = self.majority_vote_mixing(signals)
        st_signal, st_strength = self.stacking_mixing(signals)
        
        # Determine market regime for dynamic selection
        market_regime = "normal"
        if market_data:
            volatility = market_data.get('volatility', 0.20)
            if volatility > 0.30:
                market_regime = "volatile"
            elif volatility < 0.10:
                market_regime = "sideways"
                
        ds_signal, ds_strength = self.dynamic_selection_mixing(signals, market_regime)
        
        # Combine method results
        method_results = [
            (wa_signal, wa_strength, self.method_weights['weighted_average']),
            (mv_signal, mv_strength, self.method_weights['majority_vote']),
            (st_signal, st_strength, self.method_weights['stacking']),
            (ds_signal, ds_strength, self.method_weights['dynamic_selection'])
        ]
        
        # Group by signal type and weight
        signal_scores = {}
        for signal_type, strength, method_weight in method_results:
            if signal_type not in signal_scores:
                signal_scores[signal_type] = 0.0
            signal_scores[signal_type] += strength * method_weight
            
        # Return best signal
        if signal_scores:
            best_signal = max(signal_scores.items(), key=lambda x: x[1])
            return best_signal[0], min(1.0, best_signal[1])
        else:
            return SignalType.HOLD, 0.0

class SignalQualityFilter:
    """Filter and validate signal quality"""
    
    def __init__(self):
        self.min_confidence = 0.1
        self.max_age_hours = 24
        self.min_strength = 0.05
        
    def filter_signals(self, signals: List[AgentSignal]) -> List[AgentSignal]:
        """Filter signals based on quality criteria"""
        
        filtered_signals = []
        now = datetime.now()
        
        for signal in signals:
            # Age check
            age_hours = (now - signal.timestamp).total_seconds() / 3600
            if age_hours > self.max_age_hours:
                logger.debug(f"Filtered out aged signal from {signal.agent_name}")
                continue
                
            # Confidence check
            if signal.confidence < self.min_confidence:
                logger.debug(f"Filtered out low confidence signal from {signal.agent_name}")
                continue
                
            # Strength check
            if signal.strength < self.min_strength:
                logger.debug(f"Filtered out weak signal from {signal.agent_name}")
                continue
                
            # Data integrity check
            if not signal.reasoning or len(signal.reasoning.strip()) == 0:
                logger.debug(f"Filtered out signal with no reasoning from {signal.agent_name}")
                continue
                
            filtered_signals.append(signal)
            
        return filtered_signals
        
    def detect_anomalous_signals(self, signals: List[AgentSignal]) -> List[AgentSignal]:
        """Detect and flag anomalous signals"""
        
        if len(signals) < 2:
            return signals
            
        # Calculate z-scores for strength and confidence
        strengths = [s.strength for s in signals]
        confidences = [s.confidence for s in signals]
        
        strength_mean = np.mean(strengths)
        strength_std = np.std(strengths)
        confidence_mean = np.mean(confidences)
        confidence_std = np.std(confidences)
        
        non_anomalous = []
        
        for signal in signals:
            # Z-score anomaly detection
            if strength_std > 0:
                strength_z = abs(signal.strength - strength_mean) / strength_std
            else:
                strength_z = 0
                
            if confidence_std > 0:
                confidence_z = abs(signal.confidence - confidence_mean) / confidence_std
            else:
                confidence_z = 0
                
            # Flag if z-score > 2.5 (very unusual)
            if strength_z > 2.5 or confidence_z > 2.5:
                logger.warning(f"Anomalous signal detected from {signal.agent_name}: "
                             f"strength_z={strength_z:.2f}, confidence_z={confidence_z:.2f}")
                # Reduce weight but don't eliminate
                signal.weight *= 0.5
                
            non_anomalous.append(signal)
            
        return non_anomalous

class GnosisSignalMixer:
    """Main Gnosis Signal Mixer - Agent 4"""
    
    def __init__(self):
        self.signal_weights = SignalWeight()
        self.conviction_calculator = ConvictionCalculator()
        self.ensemble_method = EnsembleMethod()
        self.quality_filter = SignalQualityFilter()
        
        self.signal_history = {}
        self.performance_tracking = {}
        
        logger.info("Gnosis Signal Mixer initialized")
        
    async def mix_agent_signals(self, agent_signals: List[Dict[str, Any]], 
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main signal mixing function"""
        
        # Convert dictionaries to AgentSignal objects
        signals = []
        for signal_data in agent_signals:
            try:
                signal = AgentSignal(
                    agent_id=signal_data.get('agent_id', ''),
                    agent_name=signal_data.get('agent_name', ''),
                    signal_type=SignalType(signal_data.get('signal_type', 'hold')),
                    strength=signal_data.get('strength', 0.0),
                    confidence=signal_data.get('confidence', 0.0),
                    timestamp=datetime.fromisoformat(signal_data.get('timestamp', datetime.now().isoformat())),
                    symbol=signal_data.get('symbol', ''),
                    reasoning=signal_data.get('reasoning', ''),
                    supporting_data=signal_data.get('supporting_data', {})
                )
                signals.append(signal)
            except Exception as e:
                logger.error(f"Error parsing signal: {e}")
                continue
                
        return await self._process_signals(signals, market_data)
        
    async def _process_signals(self, signals: List[AgentSignal], 
                             market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and mix the signals"""
        
        if not signals:
            return self._create_hold_signal(market_data.get('symbol', 'UNKNOWN'))
            
        # Filter signal quality
        filtered_signals = self.quality_filter.filter_signals(signals)
        filtered_signals = self.quality_filter.detect_anomalous_signals(filtered_signals)
        
        if not filtered_signals:
            return self._create_hold_signal(signals[0].symbol)
            
        # Update signal weights based on performance
        await self._update_signal_weights(filtered_signals)
        
        # Mix signals using ensemble methods
        mixed_signal_type, signal_strength = self.ensemble_method.mix_signals(
            filtered_signals, market_data
        )
        
        # Calculate consensus level
        consensus_level = self._calculate_consensus(filtered_signals, mixed_signal_type)
        
        # Create mixed signal
        mixed_signal = MixedSignal(
            symbol=filtered_signals[0].symbol,
            signal_type=mixed_signal_type,
            strength=signal_strength,
            confidence=self._determine_confidence_level(signal_strength, consensus_level),
            conviction_score=0.0,  # Will be calculated next
            contributing_agents=[s.agent_name for s in filtered_signals],
            consensus_level=consensus_level,
            risk_reward_ratio=0.0  # Will be calculated next
        )
        
        # Calculate conviction and risk/reward
        mixed_signal.conviction_score = self.conviction_calculator.calculate_conviction(
            filtered_signals, mixed_signal
        )
        mixed_signal.risk_reward_ratio = self.conviction_calculator.calculate_risk_reward_ratio(
            filtered_signals, market_data
        )
        
        # Determine execution priority
        mixed_signal.execution_priority = self._determine_execution_priority(
            mixed_signal, filtered_signals
        )
        
        # Store supporting analysis
        mixed_signal.supporting_analysis = self._compile_supporting_analysis(
            filtered_signals, mixed_signal, market_data
        )
        
        # Track signal for performance monitoring
        self._track_signal(mixed_signal, filtered_signals)
        
        return self._format_output(mixed_signal, filtered_signals)
        
    async def _update_signal_weights(self, signals: List[AgentSignal]):
        """Update agent weights based on recent performance"""
        
        for signal in signals:
            agent_id = signal.agent_id
            
            # Get recent performance data (would be from database in production)
            recent_signals = self.signal_history.get(agent_id, [])
            
            # Mock outcomes for example (in production, would track actual results)
            mock_outcomes = [np.random.normal(0, 0.05) for _ in recent_signals]
            
            # Update weight
            if recent_signals and len(recent_signals) >= 5:  # Minimum history needed
                performance_weight = self.signal_weights.calculate_performance_weight(
                    agent_id, recent_signals, mock_outcomes
                )
                diversification_weight = self.signal_weights.calculate_diversification_weight(
                    agent_id, signals
                )
                
                # Combine weights
                final_weight = (performance_weight * 0.7) + (diversification_weight * 0.3)
                signal.weight = final_weight
                
                logger.debug(f"Updated weight for {signal.agent_name}: {final_weight:.3f}")
                
    def _calculate_consensus(self, signals: List[AgentSignal], 
                           final_signal_type: SignalType) -> float:
        """Calculate consensus level among agents"""
        
        if not signals:
            return 0.0
            
        # Count agreements weighted by agent importance
        agreement_weight = 0.0
        total_weight = 0.0
        
        for signal in signals:
            if signal.signal_type == final_signal_type:
                agreement_weight += signal.weight * signal.confidence
            total_weight += signal.weight
            
        if total_weight > 0:
            return agreement_weight / total_weight
        else:
            return 0.0
            
    def _determine_confidence_level(self, strength: float, consensus: float) -> ConfidenceLevel:
        """Determine confidence level based on strength and consensus"""
        
        combined_score = (strength * 0.6) + (consensus * 0.4)
        
        if combined_score >= 0.80:
            return ConfidenceLevel.VERY_HIGH
        elif combined_score >= 0.65:
            return ConfidenceLevel.HIGH
        elif combined_score >= 0.45:
            return ConfidenceLevel.MEDIUM
        elif combined_score >= 0.25:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.VERY_LOW
            
    def _determine_execution_priority(self, mixed_signal: MixedSignal, 
                                    signals: List[AgentSignal]) -> str:
        """Determine execution priority for the mixed signal"""
        
        # High priority conditions
        high_priority_conditions = [
            mixed_signal.conviction_score > 0.75,
            mixed_signal.confidence in [ConfidenceLevel.HIGH, ConfidenceLevel.VERY_HIGH],
            mixed_signal.consensus_level > 0.80,
            len(signals) >= 3,  # Multiple agents agree
            mixed_signal.risk_reward_ratio > 2.0
        ]
        
        # Medium priority conditions
        medium_priority_conditions = [
            mixed_signal.conviction_score > 0.50,
            mixed_signal.confidence == ConfidenceLevel.MEDIUM,
            mixed_signal.consensus_level > 0.60,
            mixed_signal.risk_reward_ratio > 1.5
        ]
        
        if sum(high_priority_conditions) >= 3:
            return "high"
        elif sum(medium_priority_conditions) >= 2:
            return "medium"
        else:
            return "low"
            
    def _compile_supporting_analysis(self, signals: List[AgentSignal], 
                                   mixed_signal: MixedSignal,
                                   market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compile supporting analysis from all agents"""
        
        analysis = {
            'agent_contributions': {},
            'signal_distribution': {},
            'key_factors': [],
            'risk_assessment': {},
            'market_context': {}
        }
        
        # Agent contributions
        for signal in signals:
            analysis['agent_contributions'][signal.agent_name] = {
                'signal_type': signal.signal_type.value,
                'strength': signal.strength,
                'confidence': signal.confidence,
                'weight': signal.weight,
                'reasoning': signal.reasoning,
                'key_data': signal.supporting_data
            }
            
        # Signal distribution
        signal_types = [s.signal_type for s in signals]
        for sig_type in set(signal_types):
            count = signal_types.count(sig_type)
            analysis['signal_distribution'][sig_type.value] = {
                'count': count,
                'percentage': count / len(signals) * 100
            }
            
        # Key factors from all agents
        all_factors = []
        for signal in signals:
            if 'key_factors' in signal.supporting_data:
                all_factors.extend(signal.supporting_data['key_factors'])
        analysis['key_factors'] = list(set(all_factors))  # Remove duplicates
        
        # Risk assessment
        analysis['risk_assessment'] = {
            'conviction_score': mixed_signal.conviction_score,
            'consensus_level': mixed_signal.consensus_level,
            'risk_reward_ratio': mixed_signal.risk_reward_ratio,
            'execution_priority': mixed_signal.execution_priority
        }
        
        # Market context
        analysis['market_context'] = market_data
        
        return analysis
        
    def _track_signal(self, mixed_signal: MixedSignal, contributing_signals: List[AgentSignal]):
        """Track signals for performance monitoring"""
        
        # Store in history for future weight updates
        for signal in contributing_signals:
            agent_id = signal.agent_id
            if agent_id not in self.signal_history:
                self.signal_history[agent_id] = []
                
            self.signal_history[agent_id].append(signal)
            
            # Keep only recent history (last 50 signals)
            if len(self.signal_history[agent_id]) > 50:
                self.signal_history[agent_id] = self.signal_history[agent_id][-50:]
                
    def _create_hold_signal(self, symbol: str) -> Dict[str, Any]:
        """Create default HOLD signal when no valid signals available"""
        
        hold_signal = MixedSignal(
            symbol=symbol,
            signal_type=SignalType.HOLD,
            strength=0.0,
            confidence=ConfidenceLevel.LOW,
            conviction_score=0.0,
            contributing_agents=[],
            consensus_level=0.0,
            risk_reward_ratio=1.0,
            execution_priority="low"
        )
        
        return self._format_output(hold_signal, [])
        
    def _format_output(self, mixed_signal: MixedSignal, 
                      contributing_signals: List[AgentSignal]) -> Dict[str, Any]:
        """Format final output for Gnosis system"""
        
        return {
            'agent': 'signal_mixer',
            'timestamp': mixed_signal.timestamp.isoformat(),
            'symbol': mixed_signal.symbol,
            'mixed_signal': {
                'type': mixed_signal.signal_type.value,
                'strength': round(mixed_signal.strength, 4),
                'confidence': mixed_signal.confidence.value,
                'conviction_score': round(mixed_signal.conviction_score, 4),
                'consensus_level': round(mixed_signal.consensus_level, 4),
                'risk_reward_ratio': round(mixed_signal.risk_reward_ratio, 4),
                'execution_priority': mixed_signal.execution_priority
            },
            'contributing_agents': mixed_signal.contributing_agents,
            'signal_count': len(contributing_signals),
            'supporting_analysis': mixed_signal.supporting_analysis,
            'performance_metadata': {
                'processing_time': 0.0,  # Would be actual processing time
                'signals_filtered': 0,
                'anomalies_detected': 0
            }
        }
        
    async def run_signal_mixing(self, gnosis_agent_outputs: List[Dict[str, Any]], 
                              market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point for Gnosis system integration"""
        
        start_time = datetime.now()
        
        # Extract signals from agent outputs
        agent_signals = []
        for agent_output in gnosis_agent_outputs:
            try:
                # Convert agent output to signal format
                signal_data = {
                    'agent_id': agent_output.get('agent', 'unknown'),
                    'agent_name': agent_output.get('agent', 'Unknown Agent'),
                    'signal_type': self._extract_signal_type(agent_output),
                    'strength': agent_output.get('signal_strength', 0.0),
                    'confidence': agent_output.get('confidence', 0.0),
                    'timestamp': agent_output.get('timestamp', datetime.now().isoformat()),
                    'symbol': market_data.get('symbol', 'UNKNOWN'),
                    'reasoning': agent_output.get('reasoning', ''),
                    'supporting_data': agent_output.get('analysis', {})
                }
                agent_signals.append(signal_data)
                
            except Exception as e:
                logger.error(f"Error processing agent output: {e}")
                continue
                
        # Mix the signals
        result = await self.mix_agent_signals(agent_signals, market_data)
        
        # Add performance metadata
        processing_time = (datetime.now() - start_time).total_seconds()
        result['performance_metadata']['processing_time'] = processing_time
        
        logger.info(f"Signal mixing completed in {processing_time:.3f}s: "
                   f"{result['mixed_signal']['type']} signal with "
                   f"{result['mixed_signal']['strength']:.2f} strength")
        
        return result
        
    def _extract_signal_type(self, agent_output: Dict[str, Any]) -> str:
        """Extract signal type from agent output"""
        
        # Look for explicit signal type
        if 'signal_type' in agent_output:
            return agent_output['signal_type']
            
        # Infer from signal strength and agent type
        strength = agent_output.get('signal_strength', 0.5)
        agent_name = agent_output.get('agent', '').lower()
        
        if 'hedge' in agent_name:
            return 'hedge'  # Hedge agent primarily provides risk signals
        elif strength > 0.6:
            return 'buy'
        elif strength < 0.4:
            return 'sell' 
        else:
            return 'hold'

# Example usage
async def main():
    """Example usage of Gnosis Signal Mixer"""
    
    # Initialize signal mixer
    mixer = GnosisSignalMixer()
    
    # Example agent outputs (from DHPE, Liquidity, Sentiment, Hedge agents)
    gnosis_agent_outputs = [
        {
            'agent': 'dhpe_engine',
            'signal_strength': 0.75,
            'confidence': 0.80,
            'timestamp': datetime.now().isoformat(),
            'reasoning': 'Strong call gamma suggests upward pressure',
            'analysis': {
                'gamma_exposure': 1500,
                'call_put_ratio': 1.8,
                'key_factors': ['high_gamma', 'bullish_flow']
            }
        },
        {
            'agent': 'liquidity_agent',
            'signal_strength': 0.65,
            'confidence': 0.70,
            'timestamp': datetime.now().isoformat(),
            'reasoning': 'Increasing bid volume and tightening spreads',
            'analysis': {
                'volume_profile': 'bullish',
                'spread_trend': 'tightening',
                'key_factors': ['volume_surge', 'tight_spreads']
            }
        },
        {
            'agent': 'sentiment_agent',
            'signal_strength': 0.45,
            'confidence': 0.60,
            'timestamp': datetime.now().isoformat(),
            'reasoning': 'Mixed sentiment with slight bearish bias',
            'analysis': {
                'regime': 'uncertainty',
                'bias_score': -0.2,
                'key_factors': ['mixed_signals', 'uncertainty']
            }
        },
        {
            'agent': 'hedge_agent',
            'signal_strength': 0.30,
            'confidence': 0.85,
            'timestamp': datetime.now().isoformat(),
            'reasoning': 'High risk metrics suggest caution',
            'analysis': {
                'var_95': 5000,
                'risk_alerts': 2,
                'key_factors': ['high_risk', 'defensive_positioning']
            }
        }
    ]
    
    # Market data
    market_data = {
        'symbol': 'SPY',
        'price': 452.50,
        'volatility': 0.18,
        'volume': 85000000
    }
    
    # Run signal mixing
    result = await mixer.run_signal_mixing(gnosis_agent_outputs, market_data)
    
    print("=== Gnosis Signal Mixer Results ===")
    print(f"Symbol: {result['symbol']}")
    print(f"Mixed Signal: {result['mixed_signal']['type'].upper()}")
    print(f"Strength: {result['mixed_signal']['strength']:.3f}")
    print(f"Confidence: {result['mixed_signal']['confidence']}")
    print(f"Conviction Score: {result['mixed_signal']['conviction_score']:.3f}")
    print(f"Consensus Level: {result['mixed_signal']['consensus_level']:.3f}")
    print(f"Risk/Reward Ratio: {result['mixed_signal']['risk_reward_ratio']:.2f}")
    print(f"Execution Priority: {result['mixed_signal']['execution_priority'].upper()}")
    
    print(f"\nContributing Agents: {', '.join(result['contributing_agents'])}")
    print(f"Processing Time: {result['performance_metadata']['processing_time']:.3f}s")
    
    print("\n=== Agent Contributions ===")
    for agent, contrib in result['supporting_analysis']['agent_contributions'].items():
        print(f"{agent}: {contrib['signal_type'].upper()} "
              f"(strength={contrib['strength']:.2f}, "
              f"confidence={contrib['confidence']:.2f})")

if __name__ == "__main__":
    asyncio.run(main())