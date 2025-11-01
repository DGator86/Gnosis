"""
Gnosis Agent 4: Complete Signal Mixer Agent
Advanced multi-agent signal integration with conviction scoring
Production-ready implementation with full integration
"""

import numpy as np
import pandas as pd
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SignalType(Enum):
    """Signal type classifications"""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"
    VOLATILITY_UP = "volatility_up"
    VOLATILITY_DOWN = "volatility_down"
    MOMENTUM_UP = "momentum_up"
    MOMENTUM_DOWN = "momentum_down"
    REVERSAL_UP = "reversal_up"
    REVERSAL_DOWN = "reversal_down"

class SignalStrength(Enum):
    """Signal strength levels"""
    WEAK = 1
    MODERATE = 2
    STRONG = 3
    VERY_STRONG = 4
    EXTREME = 5

class AgentType(Enum):
    """Gnosis agent types"""
    HEDGE_AGENT = "hedge"
    LIQUIDITY_AGENT = "liquidity" 
    SENTIMENT_AGENT = "sentiment"
    DHPE_ENGINE = "dhpe"
    EXTERNAL_SIGNAL = "external"

@dataclass
class Signal:
    """Individual signal data structure"""
    agent_id: str
    agent_type: AgentType
    signal_type: SignalType
    strength: SignalStrength
    confidence: float  # 0.0 to 1.0
    symbol: str
    timestamp: datetime
    expiry: Optional[datetime] = None
    data: Dict[str, Any] = field(default_factory=dict)
    source_details: str = ""
    
    def is_expired(self) -> bool:
        """Check if signal has expired"""
        if self.expiry is None:
            return False
        return datetime.now() > self.expiry
        
    def age_hours(self) -> float:
        """Get signal age in hours"""
        return (datetime.now() - self.timestamp).total_seconds() / 3600

@dataclass
class ConvictionScore:
    """Conviction scoring for combined signals"""
    symbol: str
    overall_conviction: float  # -1.0 to 1.0 (bearish to bullish)
    conviction_strength: float  # 0.0 to 1.0 (weak to strong)
    directional_signals: Dict[SignalType, float] = field(default_factory=dict)
    agent_contributions: Dict[str, float] = field(default_factory=dict)
    volatility_expectation: float = 0.0
    momentum_score: float = 0.0
    reversal_probability: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    last_update: datetime = field(default_factory=datetime.now)

@dataclass
class SignalCluster:
    """Grouped signals for analysis"""
    cluster_id: str
    symbol: str
    signals: List[Signal] = field(default_factory=list)
    cluster_type: SignalType = SignalType.NEUTRAL
    cluster_strength: float = 0.0
    cluster_confidence: float = 0.0
    formation_time: datetime = field(default_factory=datetime.now)
    
    def add_signal(self, signal: Signal):
        """Add signal to cluster"""
        if signal.symbol == self.symbol:
            self.signals.append(signal)
            self._update_cluster_metrics()
            
    def _update_cluster_metrics(self):
        """Update cluster-level metrics"""
        if not self.signals:
            return
            
        # Calculate weighted average strength
        total_weight = sum(s.confidence * s.strength.value for s in self.signals)
        total_confidence = sum(s.confidence for s in self.signals)
        
        if total_confidence > 0:
            self.cluster_strength = total_weight / total_confidence
            self.cluster_confidence = total_confidence / len(self.signals)

class SignalProcessor:
    """Advanced signal processing and filtering engine"""
    
    def __init__(self):
        self.signal_history = defaultdict(deque)  # Rolling history per symbol
        self.agent_weights = {
            AgentType.DHPE_ENGINE: 0.30,      # Highest weight for options flow
            AgentType.LIQUIDITY_AGENT: 0.25,   # High weight for liquidity analysis
            AgentType.SENTIMENT_AGENT: 0.20,   # Medium weight for sentiment
            AgentType.HEDGE_AGENT: 0.15,       # Medium weight for risk signals
            AgentType.EXTERNAL_SIGNAL: 0.10    # Lower weight for external signals
        }
        self.signal_decay_hours = 24  # Signals decay over 24 hours
        
    def process_signal(self, signal: Signal) -> Dict[str, Any]:
        """Process and validate incoming signal"""
        try:
            # Validate signal
            validation_result = self._validate_signal(signal)
            if not validation_result['valid']:
                return validation_result
                
            # Apply time decay
            decayed_strength = self._apply_time_decay(signal)
            
            # Check for signal conflicts
            conflicts = self._check_signal_conflicts(signal)
            
            # Store in history
            self.signal_history[signal.symbol].append(signal)
            
            # Maintain rolling window
            if len(self.signal_history[signal.symbol]) > 100:
                self.signal_history[signal.symbol].popleft()
                
            return {
                'valid': True,
                'processed_strength': decayed_strength,
                'conflicts': conflicts,
                'historical_context': self._get_historical_context(signal)
            }
            
        except Exception as e:
            logger.error(f"Error processing signal: {e}")
            return {'valid': False, 'error': str(e)}
            
    def _validate_signal(self, signal: Signal) -> Dict[str, Any]:
        """Validate signal integrity"""
        errors = []
        
        # Check required fields
        if not signal.symbol:
            errors.append("Missing symbol")
        if not signal.agent_id:
            errors.append("Missing agent_id")
        if signal.confidence < 0 or signal.confidence > 1:
            errors.append("Invalid confidence range")
            
        # Check signal age
        if signal.age_hours() > 48:  # Reject signals older than 48 hours
            errors.append("Signal too old")
            
        # Check if expired
        if signal.is_expired():
            errors.append("Signal expired")
            
        return {
            'valid': len(errors) == 0,
            'errors': errors
        }
        
    def _apply_time_decay(self, signal: Signal) -> float:
        """Apply time decay to signal strength"""
        age_hours = signal.age_hours()
        
        # Exponential decay: strength * e^(-age/decay_constant)
        decay_constant = self.signal_decay_hours / 2  # Half-life at 12 hours
        decay_factor = math.exp(-age_hours / decay_constant)
        
        return signal.strength.value * decay_factor
        
    def _check_signal_conflicts(self, signal: Signal) -> List[Dict[str, Any]]:
        """Check for conflicting signals"""
        conflicts = []
        recent_signals = [s for s in self.signal_history[signal.symbol] 
                         if s.age_hours() < 6 and not s.is_expired()]
                         
        for existing_signal in recent_signals:
            # Check for directional conflicts
            if self._signals_conflict(signal, existing_signal):
                conflicts.append({
                    'conflicting_signal': existing_signal.agent_id,
                    'conflict_type': 'directional',
                    'severity': self._calculate_conflict_severity(signal, existing_signal)
                })
                
        return conflicts
        
    def _signals_conflict(self, signal1: Signal, signal2: Signal) -> bool:
        """Check if two signals are conflicting"""
        conflicting_pairs = [
            (SignalType.BULLISH, SignalType.BEARISH),
            (SignalType.MOMENTUM_UP, SignalType.MOMENTUM_DOWN),
            (SignalType.REVERSAL_UP, SignalType.REVERSAL_DOWN),
            (SignalType.VOLATILITY_UP, SignalType.VOLATILITY_DOWN)
        ]
        
        for pair in conflicting_pairs:
            if (signal1.signal_type in pair and signal2.signal_type in pair and
                signal1.signal_type != signal2.signal_type):
                return True
                
        return False
        
    def _calculate_conflict_severity(self, signal1: Signal, signal2: Signal) -> float:
        """Calculate severity of signal conflict"""
        strength_diff = abs(signal1.strength.value - signal2.strength.value)
        confidence_product = signal1.confidence * signal2.confidence
        
        return (strength_diff / 5.0) * confidence_product
        
    def _get_historical_context(self, signal: Signal) -> Dict[str, Any]:
        """Get historical context for signal"""
        symbol_history = list(self.signal_history[signal.symbol])
        
        if len(symbol_history) < 2:
            return {'context': 'insufficient_history'}
            
        # Analyze recent signal patterns
        recent_signals = [s for s in symbol_history[-20:] if s.age_hours() < 24]
        
        signal_types = [s.signal_type for s in recent_signals]
        type_counts = {st: signal_types.count(st) for st in set(signal_types)}
        
        return {
            'context': 'normal',
            'recent_signal_count': len(recent_signals),
            'dominant_signal_type': max(type_counts, key=type_counts.get) if type_counts else None,
            'signal_frequency': len(recent_signals) / 24.0  # Signals per hour
        }

class ConvictionCalculator:
    """Advanced conviction scoring algorithm"""
    
    def __init__(self, signal_processor: SignalProcessor):
        self.signal_processor = signal_processor
        self.conviction_cache = {}
        self.cache_ttl = 300  # 5 minutes cache TTL
        
    async def calculate_conviction(self, symbol: str, 
                                 active_signals: List[Signal]) -> ConvictionScore:
        """Calculate comprehensive conviction score"""
        try:
            # Check cache first
            cache_key = f"{symbol}_{len(active_signals)}"
            if cache_key in self.conviction_cache:
                cached_time, cached_score = self.conviction_cache[cache_key]
                if (datetime.now() - cached_time).seconds < self.cache_ttl:
                    return cached_score
                    
            # Filter valid, non-expired signals
            valid_signals = [s for s in active_signals 
                           if not s.is_expired() and s.symbol == symbol]
                           
            if not valid_signals:
                return ConvictionScore(symbol=symbol, overall_conviction=0.0, conviction_strength=0.0)
                
            # Calculate directional conviction
            directional_scores = self._calculate_directional_scores(valid_signals)
            
            # Calculate agent contributions
            agent_contributions = self._calculate_agent_contributions(valid_signals)
            
            # Calculate overall conviction
            overall_conviction = self._calculate_overall_conviction(directional_scores, valid_signals)
            
            # Calculate conviction strength
            conviction_strength = self._calculate_conviction_strength(valid_signals)
            
            # Calculate additional metrics
            volatility_expectation = self._calculate_volatility_expectation(valid_signals)
            momentum_score = self._calculate_momentum_score(valid_signals)
            reversal_probability = self._calculate_reversal_probability(valid_signals)
            
            # Calculate confidence interval
            confidence_interval = self._calculate_confidence_interval(overall_conviction, valid_signals)
            
            conviction_score = ConvictionScore(
                symbol=symbol,
                overall_conviction=overall_conviction,
                conviction_strength=conviction_strength,
                directional_signals=directional_scores,
                agent_contributions=agent_contributions,
                volatility_expectation=volatility_expectation,
                momentum_score=momentum_score,
                reversal_probability=reversal_probability,
                confidence_interval=confidence_interval
            )
            
            # Cache result
            self.conviction_cache[cache_key] = (datetime.now(), conviction_score)
            
            return conviction_score
            
        except Exception as e:
            logger.error(f"Error calculating conviction: {e}")
            return ConvictionScore(symbol=symbol, overall_conviction=0.0, conviction_strength=0.0)
            
    def _calculate_directional_scores(self, signals: List[Signal]) -> Dict[SignalType, float]:
        """Calculate directional signal scores"""
        directional_scores = defaultdict(float)
        
        for signal in signals:
            # Get agent weight
            agent_weight = self.signal_processor.agent_weights.get(
                signal.agent_type, 0.1
            )
            
            # Apply time decay
            time_decay = self.signal_processor._apply_time_decay(signal)
            
            # Calculate weighted score
            weighted_score = (signal.confidence * time_decay * agent_weight)
            directional_scores[signal.signal_type] += weighted_score
            
        return dict(directional_scores)
        
    def _calculate_agent_contributions(self, signals: List[Signal]) -> Dict[str, float]:
        """Calculate individual agent contributions"""
        agent_contributions = defaultdict(float)
        
        for signal in signals:
            agent_weight = self.signal_processor.agent_weights.get(
                signal.agent_type, 0.1
            )
            time_decay = self.signal_processor._apply_time_decay(signal)
            
            contribution = signal.confidence * time_decay * agent_weight
            agent_contributions[signal.agent_id] += contribution
            
        # Normalize contributions
        total_contribution = sum(agent_contributions.values())
        if total_contribution > 0:
            agent_contributions = {k: v/total_contribution 
                                 for k, v in agent_contributions.items()}
                                 
        return dict(agent_contributions)
        
    def _calculate_overall_conviction(self, directional_scores: Dict[SignalType, float],
                                    signals: List[Signal]) -> float:
        """Calculate overall directional conviction (-1 to 1)"""
        
        # Map signal types to directional values
        bullish_signals = [
            SignalType.BULLISH, SignalType.MOMENTUM_UP, SignalType.REVERSAL_UP
        ]
        bearish_signals = [
            SignalType.BEARISH, SignalType.MOMENTUM_DOWN, SignalType.REVERSAL_DOWN
        ]
        
        bullish_score = sum(directional_scores.get(st, 0) for st in bullish_signals)
        bearish_score = sum(directional_scores.get(st, 0) for st in bearish_signals)
        
        total_score = bullish_score + bearish_score
        
        if total_score == 0:
            return 0.0
            
        # Calculate net conviction
        net_conviction = (bullish_score - bearish_score) / total_score
        
        # Apply conviction dampening for conflicting signals
        if len(set(s.signal_type for s in signals)) > 3:  # Many different signal types
            net_conviction *= 0.8  # Reduce conviction by 20%
            
        return max(-1.0, min(1.0, net_conviction))
        
    def _calculate_conviction_strength(self, signals: List[Signal]) -> float:
        """Calculate overall strength of conviction (0 to 1)"""
        if not signals:
            return 0.0
            
        # Calculate weighted average confidence
        total_weighted_confidence = 0
        total_weights = 0
        
        for signal in signals:
            agent_weight = self.signal_processor.agent_weights.get(
                signal.agent_type, 0.1
            )
            time_decay = self.signal_processor._apply_time_decay(signal)
            weight = agent_weight * time_decay
            
            total_weighted_confidence += signal.confidence * weight
            total_weights += weight
            
        if total_weights == 0:
            return 0.0
            
        base_strength = total_weighted_confidence / total_weights
        
        # Boost strength for signal consensus
        unique_agents = len(set(s.agent_id for s in signals))
        consensus_boost = min(unique_agents / 4.0, 1.0)  # Max boost at 4 agents
        
        final_strength = base_strength * (0.7 + 0.3 * consensus_boost)
        
        return min(1.0, final_strength)
        
    def _calculate_volatility_expectation(self, signals: List[Signal]) -> float:
        """Calculate expected volatility change"""
        vol_up_score = sum(
            self.signal_processor._apply_time_decay(s) * s.confidence
            for s in signals if s.signal_type == SignalType.VOLATILITY_UP
        )
        
        vol_down_score = sum(
            self.signal_processor._apply_time_decay(s) * s.confidence
            for s in signals if s.signal_type == SignalType.VOLATILITY_DOWN
        )
        
        total_vol_signals = vol_up_score + vol_down_score
        
        if total_vol_signals == 0:
            return 0.0
            
        return (vol_up_score - vol_down_score) / total_vol_signals
        
    def _calculate_momentum_score(self, signals: List[Signal]) -> float:
        """Calculate momentum score"""
        momentum_up = sum(
            self.signal_processor._apply_time_decay(s) * s.confidence
            for s in signals if s.signal_type == SignalType.MOMENTUM_UP
        )
        
        momentum_down = sum(
            self.signal_processor._apply_time_decay(s) * s.confidence
            for s in signals if s.signal_type == SignalType.MOMENTUM_DOWN
        )
        
        total_momentum = momentum_up + momentum_down
        
        if total_momentum == 0:
            return 0.0
            
        return (momentum_up - momentum_down) / total_momentum
        
    def _calculate_reversal_probability(self, signals: List[Signal]) -> float:
        """Calculate probability of trend reversal"""
        reversal_signals = [
            s for s in signals 
            if s.signal_type in [SignalType.REVERSAL_UP, SignalType.REVERSAL_DOWN]
        ]
        
        if not reversal_signals:
            return 0.0
            
        total_reversal_strength = sum(
            self.signal_processor._apply_time_decay(s) * s.confidence * s.strength.value
            for s in reversal_signals
        )
        
        # Normalize to 0-1 scale
        max_possible_strength = len(reversal_signals) * 5.0  # Max strength = 5
        
        return min(1.0, total_reversal_strength / max_possible_strength)
        
    def _calculate_confidence_interval(self, conviction: float, 
                                     signals: List[Signal]) -> Tuple[float, float]:
        """Calculate confidence interval for conviction score"""
        if not signals:
            return (0.0, 0.0)
            
        # Calculate standard error based on signal dispersion
        confidences = [s.confidence for s in signals]
        mean_confidence = np.mean(confidences)
        std_confidence = np.std(confidences) if len(confidences) > 1 else 0.1
        
        # Confidence interval width based on signal agreement
        interval_width = std_confidence * 1.96  # 95% confidence interval
        
        lower_bound = max(-1.0, conviction - interval_width)
        upper_bound = min(1.0, conviction + interval_width)
        
        return (lower_bound, upper_bound)

class SignalClusterAnalyzer:
    """Advanced signal clustering and pattern analysis"""
    
    def __init__(self):
        self.clusters = {}
        self.cluster_patterns = defaultdict(list)
        
    async def analyze_signal_clusters(self, signals: List[Signal]) -> List[SignalCluster]:
        """Analyze and form signal clusters"""
        try:
            symbol_groups = defaultdict(list)
            
            # Group signals by symbol
            for signal in signals:
                if not signal.is_expired():
                    symbol_groups[signal.symbol].append(signal)
                    
            clusters = []
            
            for symbol, symbol_signals in symbol_groups.items():
                # Create time-based clusters (signals within 1 hour)
                time_clusters = self._create_time_clusters(symbol_signals)
                
                # Create type-based clusters (similar signal types)
                type_clusters = self._create_type_clusters(symbol_signals)
                
                # Merge overlapping clusters
                merged_clusters = self._merge_clusters(time_clusters + type_clusters)
                
                clusters.extend(merged_clusters)
                
            return clusters
            
        except Exception as e:
            logger.error(f"Error analyzing signal clusters: {e}")
            return []
            
    def _create_time_clusters(self, signals: List[Signal]) -> List[SignalCluster]:
        """Create clusters based on time proximity"""
        clusters = []
        signals_by_time = sorted(signals, key=lambda s: s.timestamp)
        
        current_cluster = None
        
        for signal in signals_by_time:
            if (current_cluster is None or 
                (signal.timestamp - current_cluster.formation_time).seconds > 3600):  # 1 hour
                
                # Start new cluster
                cluster_id = f"time_{signal.symbol}_{signal.timestamp.strftime('%Y%m%d_%H%M')}"
                current_cluster = SignalCluster(
                    cluster_id=cluster_id,
                    symbol=signal.symbol,
                    formation_time=signal.timestamp
                )
                clusters.append(current_cluster)
                
            current_cluster.add_signal(signal)
            
        return clusters
        
    def _create_type_clusters(self, signals: List[Signal]) -> List[SignalCluster]:
        """Create clusters based on signal type similarity"""
        clusters = []
        type_groups = defaultdict(list)
        
        # Group by signal type
        for signal in signals:
            type_groups[signal.signal_type].append(signal)
            
        # Create clusters for each type with multiple signals
        for signal_type, type_signals in type_groups.items():
            if len(type_signals) > 1:
                cluster_id = f"type_{type_signals[0].symbol}_{signal_type.value}"
                cluster = SignalCluster(
                    cluster_id=cluster_id,
                    symbol=type_signals[0].symbol,
                    cluster_type=signal_type,
                    formation_time=min(s.timestamp for s in type_signals)
                )
                
                for signal in type_signals:
                    cluster.add_signal(signal)
                    
                clusters.append(cluster)
                
        return clusters
        
    def _merge_clusters(self, clusters: List[SignalCluster]) -> List[SignalCluster]:
        """Merge overlapping clusters"""
        if len(clusters) <= 1:
            return clusters
            
        merged = []
        used_indices = set()
        
        for i, cluster1 in enumerate(clusters):
            if i in used_indices:
                continue
                
            merged_cluster = SignalCluster(
                cluster_id=f"merged_{cluster1.symbol}_{datetime.now().strftime('%H%M%S')}",
                symbol=cluster1.symbol,
                formation_time=cluster1.formation_time
            )
            
            # Add all signals from cluster1
            for signal in cluster1.signals:
                merged_cluster.add_signal(signal)
                
            used_indices.add(i)
            
            # Check for overlapping clusters
            for j, cluster2 in enumerate(clusters[i+1:], i+1):
                if j in used_indices:
                    continue
                    
                if self._clusters_overlap(cluster1, cluster2):
                    # Merge cluster2 into merged_cluster
                    for signal in cluster2.signals:
                        merged_cluster.add_signal(signal)
                    used_indices.add(j)
                    
            merged.append(merged_cluster)
            
        return merged
        
    def _clusters_overlap(self, cluster1: SignalCluster, cluster2: SignalCluster) -> bool:
        """Check if two clusters overlap"""
        if cluster1.symbol != cluster2.symbol:
            return False
            
        # Check time overlap (within 2 hours)
        time_diff = abs((cluster1.formation_time - cluster2.formation_time).seconds)
        if time_diff < 7200:  # 2 hours
            return True
            
        # Check signal overlap
        signals1 = set(s.agent_id for s in cluster1.signals)
        signals2 = set(s.agent_id for s in cluster2.signals)
        
        overlap_ratio = len(signals1 & signals2) / len(signals1 | signals2)
        
        return overlap_ratio > 0.3  # 30% overlap threshold

class GnosisSignalMixer:
    """Main Gnosis Signal Mixer Agent - Complete Implementation"""
    
    def __init__(self, config: Dict[str, Any] = None):
        config = config or {}
        
        # Initialize components
        self.signal_processor = SignalProcessor()
        self.conviction_calculator = ConvictionCalculator(self.signal_processor)
        self.cluster_analyzer = SignalClusterAnalyzer()
        
        # Configuration
        self.max_signals_per_symbol = config.get('max_signals_per_symbol', 50)
        self.conviction_threshold = config.get('conviction_threshold', 0.3)
        self.update_interval = config.get('update_interval', 60)  # seconds
        
        # Data storage
        self.active_signals = defaultdict(list)
        self.conviction_scores = {}
        self.signal_clusters = {}
        
        # Performance tracking
        self.processing_stats = {
            'signals_processed': 0,
            'convictions_calculated': 0,
            'clusters_formed': 0,
            'start_time': datetime.now()
        }
        
        logger.info("Gnosis Signal Mixer Agent initialized")
        
    async def add_signal(self, signal: Signal) -> Dict[str, Any]:
        """Add a new signal to the mixer"""
        try:
            # Process signal
            processing_result = self.signal_processor.process_signal(signal)
            
            if not processing_result['valid']:
                return {
                    'status': 'rejected',
                    'reason': processing_result.get('error', 'Invalid signal'),
                    'signal_id': signal.agent_id
                }
                
            # Add to active signals
            self.active_signals[signal.symbol].append(signal)
            
            # Maintain signal limits
            if len(self.active_signals[signal.symbol]) > self.max_signals_per_symbol:
                # Remove oldest signal
                self.active_signals[signal.symbol].pop(0)
                
            # Update processing stats
            self.processing_stats['signals_processed'] += 1
            
            logger.info(f"Signal added: {signal.agent_id} -> {signal.symbol} "
                       f"({signal.signal_type.value}, strength={signal.strength.value})")
                       
            return {
                'status': 'accepted',
                'signal_id': signal.agent_id,
                'processing_result': processing_result
            }
            
        except Exception as e:
            logger.error(f"Error adding signal: {e}")
            return {
                'status': 'error',
                'reason': str(e),
                'signal_id': getattr(signal, 'agent_id', 'unknown')
            }
            
    async def get_conviction_score(self, symbol: str) -> Optional[ConvictionScore]:
        """Get current conviction score for a symbol"""
        try:
            if symbol not in self.active_signals:
                return None
                
            # Calculate fresh conviction score
            conviction_score = await self.conviction_calculator.calculate_conviction(
                symbol, self.active_signals[symbol]
            )
            
            # Store in cache
            self.conviction_scores[symbol] = conviction_score
            
            # Update stats
            self.processing_stats['convictions_calculated'] += 1
            
            return conviction_score
            
        except Exception as e:
            logger.error(f"Error getting conviction score for {symbol}: {e}")
            return None
            
    async def get_all_convictions(self) -> Dict[str, ConvictionScore]:
        """Get conviction scores for all active symbols"""
        try:
            convictions = {}
            
            for symbol in self.active_signals.keys():
                conviction = await self.get_conviction_score(symbol)
                if conviction:
                    convictions[symbol] = conviction
                    
            return convictions
            
        except Exception as e:
            logger.error(f"Error getting all convictions: {e}")
            return {}
            
    async def get_signal_clusters(self, symbol: str = None) -> List[SignalCluster]:
        """Get signal clusters for analysis"""
        try:
            if symbol:
                signals = self.active_signals.get(symbol, [])
            else:
                signals = []
                for symbol_signals in self.active_signals.values():
                    signals.extend(symbol_signals)
                    
            clusters = await self.cluster_analyzer.analyze_signal_clusters(signals)
            
            # Update stats
            self.processing_stats['clusters_formed'] += len(clusters)
            
            return clusters
            
        except Exception as e:
            logger.error(f"Error getting signal clusters: {e}")
            return []
            
    async def cleanup_expired_signals(self) -> int:
        """Remove expired signals and return count of removed signals"""
        try:
            removed_count = 0
            
            for symbol in list(self.active_signals.keys()):
                original_count = len(self.active_signals[symbol])
                
                # Filter out expired signals
                self.active_signals[symbol] = [
                    signal for signal in self.active_signals[symbol]
                    if not signal.is_expired()
                ]
                
                # Remove empty symbol entries
                if not self.active_signals[symbol]:
                    del self.active_signals[symbol]
                    
                removed_count += original_count - len(self.active_signals.get(symbol, []))
                
            if removed_count > 0:
                logger.info(f"Cleaned up {removed_count} expired signals")
                
            return removed_count
            
        except Exception as e:
            logger.error(f"Error cleaning up signals: {e}")
            return 0
            
    async def get_top_convictions(self, limit: int = 10, 
                                min_strength: float = 0.3) -> List[Tuple[str, ConvictionScore]]:
        """Get top conviction scores across all symbols"""
        try:
            all_convictions = await self.get_all_convictions()
            
            # Filter by minimum strength
            filtered_convictions = [
                (symbol, conviction) for symbol, conviction in all_convictions.items()
                if conviction.conviction_strength >= min_strength
            ]
            
            # Sort by absolute conviction and strength
            filtered_convictions.sort(
                key=lambda x: (abs(x[1].overall_conviction), x[1].conviction_strength),
                reverse=True
            )
            
            return filtered_convictions[:limit]
            
        except Exception as e:
            logger.error(f"Error getting top convictions: {e}")
            return []
            
    async def generate_trading_recommendations(self, 
                                            min_conviction: float = 0.4,
                                            min_strength: float = 0.5) -> List[Dict[str, Any]]:
        """Generate trading recommendations based on conviction scores"""
        try:
            recommendations = []
            
            # Get strong convictions
            top_convictions = await self.get_top_convictions(limit=20)
            
            for symbol, conviction in top_convictions:
                if (abs(conviction.overall_conviction) >= min_conviction and
                    conviction.conviction_strength >= min_strength):
                    
                    # Determine action
                    if conviction.overall_conviction > 0:
                        action = "BUY"
                        direction = "BULLISH"
                    else:
                        action = "SELL"
                        direction = "BEARISH"
                        
                    # Calculate position size recommendation
                    position_size = self._calculate_position_size(conviction)
                    
                    # Get supporting clusters
                    clusters = await self.get_signal_clusters(symbol)
                    supporting_clusters = [c for c in clusters if c.cluster_confidence > 0.6]
                    
                    recommendation = {
                        'symbol': symbol,
                        'action': action,
                        'direction': direction,
                        'conviction': conviction.overall_conviction,
                        'strength': conviction.conviction_strength,
                        'confidence_interval': conviction.confidence_interval,
                        'position_size_pct': position_size,
                        'volatility_expectation': conviction.volatility_expectation,
                        'momentum_score': conviction.momentum_score,
                        'reversal_probability': conviction.reversal_probability,
                        'supporting_agents': list(conviction.agent_contributions.keys()),
                        'signal_clusters': len(supporting_clusters),
                        'recommendation_time': datetime.now().isoformat()
                    }
                    
                    recommendations.append(recommendation)
                    
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
            
    def _calculate_position_size(self, conviction: ConvictionScore) -> float:
        """Calculate recommended position size as percentage of portfolio"""
        
        base_size = abs(conviction.overall_conviction) * conviction.conviction_strength
        
        # Adjust for volatility expectation
        vol_adjustment = 1.0 - abs(conviction.volatility_expectation) * 0.3
        
        # Adjust for reversal probability
        reversal_adjustment = 1.0 - conviction.reversal_probability * 0.5
        
        # Apply adjustments
        adjusted_size = base_size * vol_adjustment * reversal_adjustment
        
        # Cap at reasonable limits
        return min(0.20, max(0.01, adjusted_size))  # 1% to 20% of portfolio
        
    def get_system_statistics(self) -> Dict[str, Any]:
        """Get comprehensive system statistics"""
        uptime = datetime.now() - self.processing_stats['start_time']
        
        active_symbol_count = len(self.active_signals)
        total_signals = sum(len(signals) for signals in self.active_signals.values())
        
        return {
            'uptime_hours': uptime.total_seconds() / 3600,
            'signals_processed': self.processing_stats['signals_processed'],
            'convictions_calculated': self.processing_stats['convictions_calculated'],
            'clusters_formed': self.processing_stats['clusters_formed'],
            'active_symbols': active_symbol_count,
            'total_active_signals': total_signals,
            'average_signals_per_symbol': total_signals / max(active_symbol_count, 1),
            'processing_rate': self.processing_stats['signals_processed'] / max(uptime.total_seconds() / 3600, 0.1),
            'agent_weights': dict(self.signal_processor.agent_weights)
        }
        
    async def run_continuous_processing(self, interval_seconds: int = 60):
        """Run continuous signal processing loop"""
        logger.info(f"Starting continuous processing (interval: {interval_seconds}s)")
        
        while True:
            try:
                # Cleanup expired signals
                await self.cleanup_expired_signals()
                
                # Update all conviction scores
                await self.get_all_convictions()
                
                # Generate recommendations
                recommendations = await self.generate_trading_recommendations()
                
                if recommendations:
                    logger.info(f"Generated {len(recommendations)} trading recommendations")
                    
                await asyncio.sleep(interval_seconds)
                
            except KeyboardInterrupt:
                logger.info("Stopping continuous processing")
                break
            except Exception as e:
                logger.error(f"Error in continuous processing: {e}")
                await asyncio.sleep(interval_seconds)

# Integration interface for Gnosis system
async def create_signal_mixer(config: Dict[str, Any] = None) -> GnosisSignalMixer:
    """Factory function to create Gnosis Signal Mixer"""
    mixer = GnosisSignalMixer(config)
    logger.info("Gnosis Signal Mixer created successfully")
    return mixer

# Example usage
async def main():
    """Example usage of Gnosis Signal Mixer"""
    
    # Create signal mixer
    mixer = await create_signal_mixer({
        'max_signals_per_symbol': 100,
        'conviction_threshold': 0.3,
        'update_interval': 30
    })
    
    # Create sample signals
    signals = [
        Signal(
            agent_id="dhpe_engine_001",
            agent_type=AgentType.DHPE_ENGINE,
            signal_type=SignalType.BULLISH,
            strength=SignalStrength.STRONG,
            confidence=0.85,
            symbol="SPY",
            timestamp=datetime.now(),
            data={'gamma_exposure': 1500, 'call_put_ratio': 1.8}
        ),
        Signal(
            agent_id="liquidity_agent_001", 
            agent_type=AgentType.LIQUIDITY_AGENT,
            signal_type=SignalType.MOMENTUM_UP,
            strength=SignalStrength.MODERATE,
            confidence=0.72,
            symbol="SPY",
            timestamp=datetime.now(),
            data={'volume_surge': 0.35, 'bid_ask_tightening': True}
        ),
        Signal(
            agent_id="sentiment_agent_001",
            agent_type=AgentType.SENTIMENT_AGENT, 
            signal_type=SignalType.VOLATILITY_UP,
            strength=SignalStrength.WEAK,
            confidence=0.65,
            symbol="SPY",
            timestamp=datetime.now(),
            data={'regime': 'momentum', 'bias_score': 0.3}
        ),
        Signal(
            agent_id="hedge_agent_001",
            agent_type=AgentType.HEDGE_AGENT,
            signal_type=SignalType.BULLISH,
            strength=SignalStrength.MODERATE,
            confidence=0.78,
            symbol="SPY", 
            timestamp=datetime.now(),
            data={'delta_exposure': 800, 'risk_score': 0.4}
        )
    ]
    
    # Add signals to mixer
    print("=== Adding Signals ===")
    for signal in signals:
        result = await mixer.add_signal(signal)
        print(f"{signal.agent_id}: {result['status']}")
        
    # Get conviction score
    print("\n=== Conviction Score ===")
    conviction = await mixer.get_conviction_score("SPY")
    if conviction:
        print(f"Symbol: {conviction.symbol}")
        print(f"Overall Conviction: {conviction.overall_conviction:.3f}")
        print(f"Strength: {conviction.conviction_strength:.3f}")
        print(f"Confidence Interval: {conviction.confidence_interval}")
        print(f"Volatility Expectation: {conviction.volatility_expectation:.3f}")
        print(f"Agent Contributions: {conviction.agent_contributions}")
        
    # Get signal clusters
    print(f"\n=== Signal Clusters ===")
    clusters = await mixer.get_signal_clusters("SPY")
    for i, cluster in enumerate(clusters):
        print(f"Cluster {i+1}: {len(cluster.signals)} signals, "
              f"strength={cluster.cluster_strength:.2f}")
              
    # Generate recommendations
    print(f"\n=== Trading Recommendations ===")
    recommendations = await mixer.generate_trading_recommendations(
        min_conviction=0.3, min_strength=0.5
    )
    
    for rec in recommendations:
        print(f"📊 {rec['symbol']}: {rec['action']} "
              f"(conviction={rec['conviction']:.3f}, strength={rec['strength']:.3f})")
        print(f"   Position Size: {rec['position_size_pct']:.1%}")
        print(f"   Supporting Agents: {len(rec['supporting_agents'])}")
        
    # System statistics
    print(f"\n=== System Statistics ===")
    stats = mixer.get_system_statistics()
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.2f}")
        else:
            print(f"{key}: {value}")

if __name__ == "__main__":
    asyncio.run(main())