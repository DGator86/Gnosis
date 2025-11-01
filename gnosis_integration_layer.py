"""
Gnosis Complete Integration Layer
Production-grade orchestration of all Gnosis components with real market data
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Import all Gnosis components
from dhpe_engine import DHPEEngine
from agent2_advanced_liquidity import AdvancedLiquidityAnalyzer
from agent3_sentiment import Agent3SentimentInterpreter  
from agent1_hedge_complete import GnosisHedgeAgent, RiskLevel
from agent4_signal_mixer_complete import GnosisSignalMixer
from market_data_integration import MarketDataManager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class GnosisConfig:
    """Gnosis system configuration"""
    # Symbols to analyze
    symbols: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'IWM'])
    
    # Agent configurations
    risk_tolerance: RiskLevel = RiskLevel.MODERATE
    analysis_interval: int = 60  # seconds
    
    # Market data settings
    api_keys: Dict[str, str] = field(default_factory=dict)
    data_sources: List[str] = field(default_factory=lambda: ['yahoo_finance'])
    
    # Performance settings
    max_concurrent_analysis: int = 3
    cache_duration: int = 300  # seconds
    
    # Alert thresholds
    min_signal_strength: float = 0.3
    min_conviction_score: float = 0.5
    max_risk_tolerance: float = 0.15

@dataclass
class GnosisResult:
    """Complete Gnosis analysis result"""
    timestamp: datetime
    symbol: str
    
    # Individual agent results
    dhpe_result: Dict[str, Any]
    liquidity_result: Dict[str, Any] 
    sentiment_result: Dict[str, Any]
    hedge_result: Dict[str, Any]
    mixed_signal: Dict[str, Any]
    
    # Integrated analysis
    final_recommendation: str
    confidence_score: float
    risk_assessment: Dict[str, Any]
    execution_priority: str
    
    # Supporting data
    market_data: Dict[str, Any]
    performance_metrics: Dict[str, Any]
    alerts: List[Dict[str, str]] = field(default_factory=list)

class GnosisOrchestrator:
    """Main Gnosis system orchestrator"""
    
    def __init__(self, config: GnosisConfig):
        self.config = config
        
        # Initialize market data manager
        self.market_data = MarketDataManager(config.api_keys)
        
        # Initialize agents
        self.dhpe_engine = DHPEEngine()
        self.liquidity_agent = AdvancedLiquidityAnalyzer()
        self.sentiment_agent = Agent3SentimentInterpreter()
        self.hedge_agent = GnosisHedgeAgent(config.risk_tolerance)
        self.signal_mixer = GnosisSignalMixer()
        
        # System state
        self.is_running = False
        self.analysis_cache = {}
        self.performance_history = {}
        
        # Thread pool for concurrent analysis
        self.executor = ThreadPoolExecutor(max_workers=config.max_concurrent_analysis)
        
        logger.info(f"Gnosis system initialized for symbols: {config.symbols}")
        
    async def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            logger.info("Initializing Gnosis system components...")
            
            # Initialize market data feeds
            await self.market_data.initialize(self.config.symbols)
            
            # Verify all agents are ready
            agent_status = await self._check_agent_health()
            
            if not all(agent_status.values()):
                failed_agents = [agent for agent, status in agent_status.items() if not status]
                logger.error(f"Failed to initialize agents: {failed_agents}")
                return False
                
            logger.info("Gnosis system initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"System initialization failed: {e}")
            return False
            
    async def _check_agent_health(self) -> Dict[str, bool]:
        """Check health status of all agents"""
        agent_health = {}
        
        try:
            # Test DHPE engine
            test_data = {'spy_price': 450, 'options_data': []}
            dhpe_test = await self.dhpe_engine.analyze_options_flow(test_data)
            agent_health['dhpe'] = 'pressure_analysis' in dhpe_test
            
            # Test liquidity agent
            liquidity_test = await self.liquidity_agent.analyze_liquidity_flow({'symbol': 'SPY'})
            agent_health['liquidity'] = 'volume_analysis' in liquidity_test
            
            # Test sentiment agent  
            sentiment_test = await self.sentiment_agent.analyze_market_sentiment({'symbol': 'SPY'})
            agent_health['sentiment'] = 'regime_analysis' in sentiment_test
            
            # Test hedge agent
            portfolio_data = {'cash': 10000, 'positions': []}
            hedge_test = await self.hedge_agent.run_hedge_analysis(portfolio_data, {})
            agent_health['hedge'] = 'risk_analysis' in hedge_test
            
            # Test signal mixer
            agent_outputs = [dhpe_test, liquidity_test, sentiment_test, hedge_test]
            mixer_test = await self.signal_mixer.run_signal_mixing(agent_outputs, {'symbol': 'SPY'})
            agent_health['mixer'] = 'mixed_signal' in mixer_test
            
        except Exception as e:
            logger.error(f"Agent health check failed: {e}")
            # Set all to False on error
            for agent in ['dhpe', 'liquidity', 'sentiment', 'hedge', 'mixer']:
                agent_health[agent] = False
                
        return agent_health
        
    async def run_complete_analysis(self, symbol: str) -> GnosisResult:
        """Run complete Gnosis analysis for a symbol"""
        
        start_time = datetime.now()
        
        try:
            logger.info(f"Starting complete analysis for {symbol}")
            
            # Get current market data
            current_market_data = await self._get_market_data(symbol)
            
            if not current_market_data:
                raise Exception(f"Failed to get market data for {symbol}")
                
            # Run all agents concurrently
            agent_tasks = [
                self._run_dhpe_analysis(symbol, current_market_data),
                self._run_liquidity_analysis(symbol, current_market_data),
                self._run_sentiment_analysis(symbol, current_market_data),
                self._run_hedge_analysis(symbol, current_market_data)
            ]
            
            # Wait for all agent results
            agent_results = await asyncio.gather(*agent_tasks, return_exceptions=True)
            
            # Check for exceptions
            dhpe_result, liquidity_result, sentiment_result, hedge_result = agent_results
            
            for i, result in enumerate(agent_results):
                if isinstance(result, Exception):
                    agent_name = ['DHPE', 'Liquidity', 'Sentiment', 'Hedge'][i]
                    logger.error(f"{agent_name} agent failed: {result}")
                    # Create fallback result
                    agent_results[i] = self._create_fallback_result(agent_name.lower())
                    
            dhpe_result, liquidity_result, sentiment_result, hedge_result = agent_results
            
            # Mix signals using Agent 4
            all_agent_outputs = [dhpe_result, liquidity_result, sentiment_result, hedge_result]
            mixed_signal = await self.signal_mixer.run_signal_mixing(all_agent_outputs, current_market_data)
            
            # Generate integrated analysis
            final_recommendation = self._generate_final_recommendation(mixed_signal, all_agent_outputs)
            confidence_score = self._calculate_system_confidence(mixed_signal, all_agent_outputs)
            risk_assessment = self._assess_overall_risk(all_agent_outputs, current_market_data)
            execution_priority = mixed_signal.get('mixed_signal', {}).get('execution_priority', 'medium')
            
            # Calculate performance metrics
            processing_time = (datetime.now() - start_time).total_seconds()
            performance_metrics = {
                'processing_time_seconds': processing_time,
                'agents_successful': sum(1 for r in agent_results if not isinstance(r, Exception)),
                'data_freshness_seconds': self._calculate_data_freshness(current_market_data),
                'system_load': self._get_system_load()
            }
            
            # Generate alerts
            alerts = self._generate_system_alerts(mixed_signal, risk_assessment, performance_metrics)
            
            # Create final result
            result = GnosisResult(
                timestamp=datetime.now(),
                symbol=symbol,
                dhpe_result=dhpe_result,
                liquidity_result=liquidity_result,
                sentiment_result=sentiment_result,
                hedge_result=hedge_result,
                mixed_signal=mixed_signal,
                final_recommendation=final_recommendation,
                confidence_score=confidence_score,
                risk_assessment=risk_assessment,
                execution_priority=execution_priority,
                market_data=current_market_data,
                performance_metrics=performance_metrics,
                alerts=alerts
            )
            
            # Cache result
            self._cache_result(symbol, result)
            
            # Update performance history
            self._update_performance_history(symbol, result)
            
            logger.info(f"Analysis completed for {symbol} in {processing_time:.2f}s: "
                       f"{final_recommendation} (confidence: {confidence_score:.2f})")
            
            return result
            
        except Exception as e:
            logger.error(f"Complete analysis failed for {symbol}: {e}")
            logger.error(traceback.format_exc())
            return self._create_error_result(symbol, str(e))
            
    async def _get_market_data(self, symbol: str) -> Dict[str, Any]:
        """Get comprehensive market data for analysis"""
        
        # Get real-time market data
        market_data = await self.market_data.get_market_data(symbol)
        
        if not market_data:
            return {}
            
        # Get options chain
        options_chain = await self.market_data.get_options_chain(symbol)
        
        # Get technical indicators
        technical_indicators = self.market_data.get_technical_indicators(symbol)
        
        # Compile comprehensive market data
        comprehensive_data = {
            'symbol': symbol,
            'price': market_data.price,
            'bid': market_data.bid,
            'ask': market_data.ask,
            'volume': market_data.volume,
            'change': market_data.change,
            'change_percent': market_data.change_percent,
            'options_chain': [
                {
                    'symbol': opt.symbol,
                    'strike': opt.strike,
                    'expiry': opt.expiry,
                    'type': opt.option_type,
                    'bid': opt.bid,
                    'ask': opt.ask,
                    'volume': opt.volume,
                    'open_interest': opt.open_interest,
                    'iv': opt.implied_volatility,
                    'delta': opt.delta,
                    'gamma': opt.gamma,
                    'theta': opt.theta,
                    'vega': opt.vega
                } for opt in options_chain
            ],
            'technical_indicators': technical_indicators,
            'market_status': self.market_data.get_market_status(),
            'timestamp': market_data.timestamp.isoformat()
        }
        
        return comprehensive_data
        
    async def _run_dhpe_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run DHPE engine analysis"""
        try:
            # Format data for DHPE engine
            dhpe_input = {
                'spy_price': market_data.get('price', 450),
                'options_data': market_data.get('options_chain', []),
                'market_data': market_data
            }
            
            result = await self.dhpe_engine.analyze_options_flow(dhpe_input)
            
            # Add agent metadata
            result.update({
                'agent': 'dhpe_engine',
                'timestamp': datetime.now().isoformat(),
                'signal_strength': result.get('pressure_analysis', {}).get('net_pressure_score', 0.5),
                'confidence': result.get('confidence', 0.7)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"DHPE analysis failed: {e}")
            raise
            
    async def _run_liquidity_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run liquidity agent analysis"""
        try:
            # Run in executor since it's sync
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.liquidity_agent.analyze_underlying_liquidity,
                market_data.get('symbol', 'SPY')
            )
            
            # Add agent metadata
            result.update({
                'agent': 'liquidity_agent',
                'timestamp': datetime.now().isoformat(),
                'signal_strength': result.get('volume_analysis', {}).get('bullish_flow_ratio', 0.5),
                'confidence': result.get('confidence', 0.7)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Liquidity analysis failed: {e}")
            raise
            
    async def _run_sentiment_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run sentiment agent analysis"""
        try:
            # Run in executor since it's sync
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                self.executor,
                self.sentiment_agent.get_sentiment_summary
            )
            
            # Add agent metadata  
            result.update({
                'agent': 'sentiment_agent',
                'timestamp': datetime.now().isoformat(),
                'signal_strength': result.get('regime_analysis', {}).get('bullish_probability', 0.5),
                'confidence': result.get('confidence', 0.7)
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            raise
            
    async def _run_hedge_analysis(self, symbol: str, market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Run hedge agent analysis"""
        try:
            # Create sample portfolio for analysis
            portfolio_data = {
                'cash': 100000,
                'positions': [
                    {
                        'symbol': symbol,
                        'quantity': 100,
                        'entry_price': market_data.get('price', 450) * 0.98,
                        'current_price': market_data.get('price', 450),
                        'position_type': 'long',
                        'delta': 1.0,
                        'gamma': 0.0,
                        'theta': 0.0,
                        'vega': 0.0
                    }
                ]
            }
            
            result = await self.hedge_agent.run_hedge_analysis(portfolio_data, market_data)
            
            return result
            
        except Exception as e:
            logger.error(f"Hedge analysis failed: {e}")
            raise
            
    def _create_fallback_result(self, agent_name: str) -> Dict[str, Any]:
        """Create fallback result for failed agent"""
        return {
            'agent': agent_name,
            'timestamp': datetime.now().isoformat(),
            'signal_strength': 0.5,
            'confidence': 0.1,
            'status': 'failed',
            'error': 'Agent analysis failed'
        }
        
    def _generate_final_recommendation(self, mixed_signal: Dict[str, Any], 
                                     agent_outputs: List[Dict[str, Any]]) -> str:
        """Generate final trading recommendation"""
        
        signal_type = mixed_signal.get('mixed_signal', {}).get('type', 'hold')
        strength = mixed_signal.get('mixed_signal', {}).get('strength', 0.5)
        conviction = mixed_signal.get('mixed_signal', {}).get('conviction_score', 0.5)
        
        # Apply filters based on system configuration
        if strength < self.config.min_signal_strength:
            return 'HOLD - Signal too weak'
        
        if conviction < self.config.min_conviction_score:
            return 'HOLD - Insufficient conviction'
            
        # Generate recommendation based on signal
        if signal_type == 'buy' and strength > 0.6:
            if conviction > 0.8:
                return 'STRONG BUY'
            else:
                return 'BUY'
        elif signal_type == 'sell' and strength > 0.6:
            if conviction > 0.8:
                return 'STRONG SELL'
            else:
                return 'SELL'
        elif signal_type == 'hedge':
            return 'HEDGE RECOMMENDED'
        else:
            return 'HOLD'
            
    def _calculate_system_confidence(self, mixed_signal: Dict[str, Any], 
                                   agent_outputs: List[Dict[str, Any]]) -> float:
        """Calculate overall system confidence"""
        
        # Base confidence from signal mixer
        base_confidence = mixed_signal.get('mixed_signal', {}).get('conviction_score', 0.5)
        
        # Agent agreement factor
        successful_agents = [output for output in agent_outputs 
                           if not isinstance(output, Exception) and output.get('status') != 'failed']
        
        if not successful_agents:
            return 0.1
            
        agreement_factor = len(successful_agents) / 4.0  # 4 total agents
        
        # Data quality factor
        data_quality = 1.0  # Simplified - would check data freshness, completeness
        
        # Final confidence
        system_confidence = (base_confidence * 0.6) + (agreement_factor * 0.3) + (data_quality * 0.1)
        
        return max(0.0, min(1.0, system_confidence))
        
    def _assess_overall_risk(self, agent_outputs: List[Dict[str, Any]], 
                           market_data: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall system risk"""
        
        risk_assessment = {
            'overall_risk_level': 'MEDIUM',
            'risk_factors': [],
            'volatility_risk': 'MEDIUM',
            'liquidity_risk': 'LOW',
            'sentiment_risk': 'MEDIUM',
            'hedge_requirements': []
        }
        
        # Check hedge agent output for risk alerts
        hedge_result = next((output for output in agent_outputs 
                           if output.get('agent') == 'hedge_agent'), {})
        
        if hedge_result:
            risk_analysis = hedge_result.get('risk_analysis', {})
            risk_alerts = risk_analysis.get('risk_alerts', [])
            
            if risk_alerts:
                high_risk_alerts = [alert for alert in risk_alerts 
                                  if alert.get('severity') == 'HIGH']
                if high_risk_alerts:
                    risk_assessment['overall_risk_level'] = 'HIGH'
                    risk_assessment['risk_factors'].extend([alert['message'] for alert in high_risk_alerts])
                    
        # Check market volatility
        volatility = market_data.get('technical_indicators', {}).get('volatility', 0.20)
        if volatility > 0.30:
            risk_assessment['volatility_risk'] = 'HIGH'
            risk_assessment['risk_factors'].append(f'High volatility: {volatility:.2%}')
        elif volatility > 0.25:
            risk_assessment['volatility_risk'] = 'MEDIUM'
            
        return risk_assessment
        
    def _calculate_data_freshness(self, market_data: Dict[str, Any]) -> float:
        """Calculate data freshness in seconds"""
        
        timestamp_str = market_data.get('timestamp')
        if not timestamp_str:
            return 999.0  # Very stale
            
        try:
            data_time = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            freshness = (datetime.now() - data_time.replace(tzinfo=None)).total_seconds()
            return freshness
        except:
            return 999.0
            
    def _get_system_load(self) -> float:
        """Get current system load (simplified)"""
        # In production, would check CPU, memory, etc.
        return 0.5  # Placeholder
        
    def _generate_system_alerts(self, mixed_signal: Dict[str, Any], 
                              risk_assessment: Dict[str, Any],
                              performance_metrics: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate system-level alerts"""
        
        alerts = []
        
        # High risk alerts
        if risk_assessment['overall_risk_level'] == 'HIGH':
            alerts.append({
                'type': 'RISK_WARNING',
                'severity': 'HIGH',
                'message': 'High risk conditions detected',
                'recommendation': 'Consider reducing position sizes or implementing hedges'
            })
            
        # Performance alerts
        processing_time = performance_metrics.get('processing_time_seconds', 0)
        if processing_time > 30:  # Slow processing
            alerts.append({
                'type': 'PERFORMANCE',
                'severity': 'MEDIUM', 
                'message': f'Slow processing time: {processing_time:.1f}s',
                'recommendation': 'Check system resources'
            })
            
        # Data freshness alerts
        data_age = performance_metrics.get('data_freshness_seconds', 0)
        if data_age > 300:  # 5 minutes old
            alerts.append({
                'type': 'DATA_QUALITY',
                'severity': 'MEDIUM',
                'message': f'Stale market data: {data_age:.0f}s old',
                'recommendation': 'Check market data connections'
            })
            
        return alerts
        
    def _cache_result(self, symbol: str, result: GnosisResult):
        """Cache analysis result"""
        cache_key = f"{symbol}_{result.timestamp.strftime('%Y%m%d_%H%M')}"
        self.analysis_cache[cache_key] = result
        
        # Cleanup old cache entries
        cutoff_time = datetime.now() - timedelta(seconds=self.config.cache_duration * 5)
        self.analysis_cache = {
            k: v for k, v in self.analysis_cache.items()
            if v.timestamp > cutoff_time
        }
        
    def _update_performance_history(self, symbol: str, result: GnosisResult):
        """Update performance tracking"""
        if symbol not in self.performance_history:
            self.performance_history[symbol] = []
            
        performance_record = {
            'timestamp': result.timestamp,
            'processing_time': result.performance_metrics['processing_time_seconds'],
            'confidence': result.confidence_score,
            'recommendation': result.final_recommendation,
            'agents_successful': result.performance_metrics['agents_successful']
        }
        
        self.performance_history[symbol].append(performance_record)
        
        # Keep only recent history
        cutoff_time = datetime.now() - timedelta(hours=24)
        self.performance_history[symbol] = [
            record for record in self.performance_history[symbol]
            if record['timestamp'] > cutoff_time
        ]
        
    def _create_error_result(self, symbol: str, error_message: str) -> GnosisResult:
        """Create error result when analysis fails"""
        return GnosisResult(
            timestamp=datetime.now(),
            symbol=symbol,
            dhpe_result={'error': error_message},
            liquidity_result={'error': error_message},
            sentiment_result={'error': error_message},
            hedge_result={'error': error_message},
            mixed_signal={'error': error_message},
            final_recommendation='ERROR',
            confidence_score=0.0,
            risk_assessment={'overall_risk_level': 'UNKNOWN'},
            execution_priority='low',
            market_data={},
            performance_metrics={'processing_time_seconds': 0},
            alerts=[{
                'type': 'SYSTEM_ERROR',
                'severity': 'HIGH',
                'message': f'Analysis failed: {error_message}',
                'recommendation': 'Check system logs and restart if necessary'
            }]
        )
        
    async def run_continuous_analysis(self, symbols: List[str] = None):
        """Run continuous analysis loop"""
        
        if symbols is None:
            symbols = self.config.symbols
            
        self.is_running = True
        logger.info(f"Starting continuous analysis for {symbols}")
        
        try:
            while self.is_running:
                # Run analysis for all symbols
                tasks = [self.run_complete_analysis(symbol) for symbol in symbols]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Log results
                for symbol, result in zip(symbols, results):
                    if isinstance(result, Exception):
                        logger.error(f"Analysis failed for {symbol}: {result}")
                    else:
                        logger.info(f"{symbol}: {result.final_recommendation} "
                                   f"(confidence: {result.confidence_score:.2f})")
                        
                # Wait for next analysis cycle
                await asyncio.sleep(self.config.analysis_interval)
                
        except Exception as e:
            logger.error(f"Continuous analysis loop failed: {e}")
        finally:
            self.is_running = False
            logger.info("Continuous analysis stopped")
            
    def stop_continuous_analysis(self):
        """Stop continuous analysis"""
        self.is_running = False
        
    async def shutdown(self):
        """Shutdown Gnosis system"""
        logger.info("Shutting down Gnosis system...")
        
        self.stop_continuous_analysis()
        await self.market_data.shutdown()
        self.executor.shutdown(wait=True)
        
        logger.info("Gnosis system shutdown complete")

# Usage example and system startup
async def main():
    """Example usage of complete Gnosis system"""
    
    # Configuration
    config = GnosisConfig(
        symbols=['SPY', 'QQQ', 'AAPL'],
        risk_tolerance=RiskLevel.MODERATE,
        analysis_interval=30,  # 30 seconds
        api_keys={
            'polygon': 'your_polygon_key_here',
            'alpha_vantage': 'your_alpha_vantage_key_here'
        }
    )
    
    # Initialize Gnosis system
    gnosis = GnosisOrchestrator(config)
    
    try:
        # Initialize system
        if not await gnosis.initialize():
            logger.error("Failed to initialize Gnosis system")
            return
            
        # Run single analysis
        print("=== Single Analysis Example ===")
        result = await gnosis.run_complete_analysis('SPY')
        
        print(f"\nSymbol: {result.symbol}")
        print(f"Final Recommendation: {result.final_recommendation}")
        print(f"Confidence Score: {result.confidence_score:.3f}")
        print(f"Execution Priority: {result.execution_priority}")
        print(f"Processing Time: {result.performance_metrics['processing_time_seconds']:.2f}s")
        
        # Show mixed signal details
        mixed = result.mixed_signal.get('mixed_signal', {})
        print(f"\nMixed Signal: {mixed.get('type', 'N/A').upper()}")
        print(f"Signal Strength: {mixed.get('strength', 0):.3f}")
        print(f"Conviction Score: {mixed.get('conviction_score', 0):.3f}")
        print(f"Risk/Reward Ratio: {mixed.get('risk_reward_ratio', 0):.2f}")
        
        # Show alerts
        if result.alerts:
            print(f"\nAlerts ({len(result.alerts)}):")
            for alert in result.alerts:
                print(f"  {alert['severity']}: {alert['message']}")
                
        # Show agent contributions
        print(f"\nAgent Results:")
        agents = ['DHPE', 'Liquidity', 'Sentiment', 'Hedge']
        agent_results = [result.dhpe_result, result.liquidity_result, 
                        result.sentiment_result, result.hedge_result]
        
        for agent_name, agent_result in zip(agents, agent_results):
            strength = agent_result.get('signal_strength', 0)
            confidence = agent_result.get('confidence', 0)
            print(f"  {agent_name}: strength={strength:.2f}, confidence={confidence:.2f}")
            
        # Optional: Run continuous analysis for a short time
        print(f"\n=== Starting Continuous Analysis (30 seconds) ===")
        
        # Start continuous analysis in background
        continuous_task = asyncio.create_task(gnosis.run_continuous_analysis(['SPY']))
        
        # Let it run for 30 seconds
        await asyncio.sleep(30)
        
        # Stop continuous analysis
        gnosis.stop_continuous_analysis()
        
        # Wait for task to complete
        try:
            await continuous_task
        except:
            pass  # Task was cancelled, that's expected
            
    finally:
        await gnosis.shutdown()

if __name__ == "__main__":
    asyncio.run(main())