#!/usr/bin/env python3
"""
Gnosis System Integration Demo
Demonstrates how the core agents work together
"""

import pandas as pd
import numpy as np
from datetime import datetime
from agent3_sentiment import Agent3SentimentInterpreter, MarketRegime
from dhpe_engine import DHPEEngine, create_sample_options_data

def demo_integrated_system():
    """Demonstrate integrated Gnosis system"""
    
    print("🚀 === Gnosis Integrated System Demo ===")
    print("Demonstrating Agent 3 + DHPE Engine integration\n")
    
    # Initialize components
    agent3 = Agent3SentimentInterpreter()
    dhpe = DHPEEngine()
    
    # Market scenario
    symbol = "SPY"
    current_price = 432.50
    
    print(f"📊 Analyzing {symbol} at ${current_price}")
    print("-" * 50)
    
    # DHPE Analysis
    print("1️⃣ DHPE (Dealer Hedge Pressure) Analysis:")
    options_data = create_sample_options_data(symbol, current_price)
    dhpe.add_options_data(symbol, options_data)
    dhpe_metrics = dhpe.analyze_dhpe(symbol, current_price)
    dhpe_summary = dhpe.get_summary(symbol)
    
    print(f"   Gamma Exposure: {dhpe_metrics.total_gamma_exposure:.1f}B")
    print(f"   Max Pain: ${dhpe_metrics.max_pain_strike:.2f}")
    print(f"   Regime: {dhpe_summary['regime']}")
    print(f"   Pressure Level: {dhpe_summary['pressure_level']}")
    
    # Agent 3 Analysis
    print("\n2️⃣ Agent 3 (Sentiment) Analysis:")
    
    # Create market indicators based on DHPE results
    market_indicators = pd.Series({
        'rsi': 65.0 + (dhpe_metrics.dealer_positioning * 15),  # DHPE influences RSI
        'macd': dhpe_metrics.hedge_pressure_score * 2.0 - 1.0,  # Convert to MACD signal
        'bb_position': 0.5 + (dhpe_metrics.dealer_positioning * 0.3),
        'volume_ratio': 1.0 + abs(dhpe_metrics.hedge_pressure_score),
        'volatility': dhpe_summary['distance_from_max_pain_pct'] / 100,
        'close': current_price,
        'volume': 3000000
    })
    
    regime, confidence = agent3.classify_regime(market_indicators)
    sentiment_summary = agent3.get_sentiment_summary()
    
    print(f"   Market Regime: {regime.name}")
    print(f"   Confidence: {confidence:.3f}")
    print(f"   Regime Stability: {sentiment_summary['regime_stability']:.3f}")
    
    # Integration Analysis
    print("\n3️⃣ Integrated Analysis:")
    
    # Combine DHPE and sentiment signals
    dhpe_signal_strength = abs(dhpe_metrics.dealer_positioning)
    sentiment_signal_strength = confidence
    
    # Agreement analysis
    dhpe_bullish = dhpe_metrics.dealer_positioning < -0.2  # Dealers short gamma = bullish
    sentiment_bullish = regime.name in ['BULL_MODERATE', 'BULL_STRONG', 'BULL_EXTREME']
    
    signals_agree = dhpe_bullish == sentiment_bullish
    
    print(f"   DHPE Signal: {'BULLISH' if dhpe_bullish else 'BEARISH'} (strength: {dhpe_signal_strength:.3f})")
    print(f"   Sentiment Signal: {'BULLISH' if sentiment_bullish else 'BEARISH'} (strength: {sentiment_signal_strength:.3f})")
    print(f"   Signals Agreement: {'✅ AGREE' if signals_agree else '❌ DIVERGE'}")
    
    # Combined confidence
    if signals_agree:
        combined_confidence = (dhpe_signal_strength + sentiment_signal_strength) / 2
        signal_quality = "HIGH" if combined_confidence > 0.6 else "MODERATE"
    else:
        combined_confidence = abs(dhpe_signal_strength - sentiment_signal_strength) / 2
        signal_quality = "LOW (DIVERGENCE)"
    
    print(f"   Combined Confidence: {combined_confidence:.3f}")
    print(f"   Signal Quality: {signal_quality}")
    
    # Trading implications
    print("\n4️⃣ Trading Implications:")
    
    if signals_agree and combined_confidence > 0.5:
        direction = "BULLISH" if dhpe_bullish else "BEARISH"
        conviction = "HIGH" if combined_confidence > 0.7 else "MODERATE"
        print(f"   🎯 Direction: {direction}")
        print(f"   💪 Conviction: {conviction}")
        
        if dhpe_bullish:
            print(f"   📈 Strategy: Consider calls near max pain ${dhpe_metrics.max_pain_strike:.2f}")
        else:
            print(f"   📉 Strategy: Consider puts near max pain ${dhpe_metrics.max_pain_strike:.2f}")
            
    else:
        print(f"   ⚠️  Conflicting signals - wait for clarity")
        print(f"   🔄 Monitor for regime change")
    
    # System health
    print("\n5️⃣ System Status:")
    print(f"   🔧 DHPE Engine: ✅ Active")
    print(f"   🧠 Agent 3 Sentiment: ✅ Active")
    print(f"   📊 Data Quality: ✅ Good")
    print(f"   🔄 Last Update: {datetime.now().strftime('%H:%M:%S')}")
    
    return {
        'dhpe_metrics': dhpe_metrics,
        'sentiment_regime': regime,
        'combined_confidence': combined_confidence,
        'signals_agree': signals_agree,
        'trading_direction': direction if 'direction' in locals() else 'NEUTRAL'
    }

if __name__ == "__main__":
    results = demo_integrated_system()
    print(f"\n✅ Demo completed successfully!")
    print(f"🎉 Gnosis core system operational with {len(results)} integrated signals")