#!/usr/bin/env python3
"""
Simple Gnosis System Integration Test
Tests all 4 agents working together
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Import all agents
from dhpe_engine import DHPEEngine, create_sample_options_data
from agent2_advanced_liquidity import AdvancedLiquidityAnalyzer
from agent3_sentiment import Agent3SentimentInterpreter, MarketRegime
from agent1_hedge import Agent1HedgeEngine, create_sample_portfolio
from agent4_signal_mixer import Agent4SignalMixer

def test_integrated_system():
    """Test complete integrated system"""
    
    print("🚀 === GNOSIS INTEGRATED SYSTEM TEST ===")
    
    # Market scenario
    symbol = "SPY"
    current_price = 432.50
    
    # 1. Initialize all agents
    print("1️⃣ Initializing all agents...")
    dhpe_engine = DHPEEngine()
    liquidity_analyzer = AdvancedLiquidityAnalyzer()
    sentiment_interpreter = Agent3SentimentInterpreter()
    hedge_engine = create_sample_portfolio()
    signal_mixer = Agent4SignalMixer()
    print("✅ All agents ready")
    
    # 2. DHPE Analysis
    print("\n2️⃣ DHPE Analysis...")
    options_data = create_sample_options_data(symbol, current_price)
    dhpe_engine.add_options_data(symbol, options_data)
    dhpe_metrics = dhpe_engine.analyze_dhpe(symbol, current_price)
    print(f"   Gamma: {dhpe_metrics.total_gamma_exposure:.1f}B")
    print(f"   Max Pain: ${dhpe_metrics.max_pain_strike:.2f}")
    print(f"   Hedge Pressure: {dhpe_metrics.hedge_pressure_score:.3f}")
    
    # 3. Liquidity Analysis  
    print("\n3️⃣ Liquidity Analysis...")
    underlying_liq = liquidity_analyzer.analyze_underlying_liquidity(symbol)
    options_liq = liquidity_analyzer.analyze_options_chain_liquidity(symbol)
    print(f"   Liquidity Tier: {underlying_liq.get('liquidity_tier', 'MODERATE')}")
    print(f"   MM Flow Score: {options_liq.get('market_maker_flow_score', 0.5):.3f}")
    
    # 4. Sentiment Analysis
    print("\n4️⃣ Sentiment Analysis...")
    indicators = pd.Series({
        'rsi': 65.0,
        'macd': 1.2,
        'bb_position': 0.75,
        'volume_ratio': 1.5,
        'volatility': 0.025,
        'close': current_price,
        'volume': 3000000
    })
    
    regime, confidence = sentiment_interpreter.classify_regime(indicators)
    summary = sentiment_interpreter.get_sentiment_summary()
    print(f"   Regime: {regime.name}")
    print(f"   Confidence: {confidence:.3f}")
    
    # 5. Risk Analysis
    print("\n5️⃣ Risk Analysis...")
    portfolio_risk = hedge_engine.calculate_portfolio_risk()
    risk_level = hedge_engine.assess_risk_level(portfolio_risk)
    hedge_recs = hedge_engine.generate_hedge_recommendations(current_price)
    print(f"   Portfolio Delta: {portfolio_risk.total_delta:.0f}")
    print(f"   Risk Level: {risk_level.value}")
    
    # 6. Signal Integration
    print("\n6️⃣ Signal Integration...")
    
    liquidity_metrics = {
        'vwap_deviation': 0.015,
        'volume_profile_strength': 0.7,
        'aggressive_buy_ratio': 0.6
    }
    
    integrated = signal_mixer.get_integrated_analysis(
        dhpe_metrics=dhpe_metrics,
        spot_price=current_price,
        sentiment_regime=regime,
        sentiment_confidence=confidence,
        bias_scores=summary.get('bias_scores', {}),
        liquidity_metrics=liquidity_metrics,
        risk_level=risk_level,
        hedge_recommendations=hedge_recs,
        portfolio_delta=portfolio_risk.total_delta
    )
    
    print(f"   Direction: {integrated.direction.value}")
    print(f"   Strength: {integrated.strength.value}")
    print(f"   Strategy: {integrated.strategy_recommendation.value}")
    print(f"   Conviction: {integrated.conviction_score:.3f}")
    
    # 7. Final Summary
    print(f"\n7️⃣ Trading Decision:")
    if integrated.conviction_score > 0.6:
        decision = "🟢 HIGH CONVICTION TRADE"
    elif integrated.conviction_score > 0.4:
        decision = "🟡 MODERATE CONVICTION"
    else:
        decision = "🔴 LOW CONVICTION - WAIT"
    
    print(f"   {decision}")
    print(f"   Action: {integrated.direction.value}")
    print(f"   Size: {integrated.position_size_modifier:.1f}x base")
    print(f"   Agreement: {integrated.signal_agreement:.1%}")
    
    print(f"\n✅ All 4 agents working together successfully!")
    
    return integrated

if __name__ == "__main__":
    result = test_integrated_system()
    print(f"\n🎯 Final Signal: {result.direction.value} with {result.conviction_score:.1%} conviction")