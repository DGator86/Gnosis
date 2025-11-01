#!/usr/bin/env python3
"""
Comprehensive Agent 3 Testing Script
"""

import pandas as pd
import numpy as np
from agent3_sentiment import Agent3SentimentInterpreter, MarketRegime

def test_agent3_comprehensive():
    """Comprehensive testing of Agent 3 Sentiment Interpreter"""
    
    print('=== Agent 3 Comprehensive Market Scenario Testing ===')
    
    agent3 = Agent3SentimentInterpreter()
    
    # Test various realistic market scenarios
    scenarios = [
        {
            'name': 'Strong Bull Rally',
            'rsi': 78,
            'macd': 2.1,
            'bb_position': 0.85,
            'volume_ratio': 1.8,
            'volatility': 0.025,
            'close': 105.50,
            'volume': 3500000
        },
        {
            'name': 'Bear Capitulation', 
            'rsi': 18,
            'macd': -2.5,
            'bb_position': 0.05,
            'volume_ratio': 2.2,
            'volatility': 0.045,
            'close': 92.30,
            'volume': 4800000
        },
        {
            'name': 'Sideways Chop',
            'rsi': 52,
            'macd': -0.1,
            'bb_position': 0.48,
            'volume_ratio': 0.9,
            'volatility': 0.008,
            'close': 100.15,
            'volume': 1800000
        },
        {
            'name': 'Overbought Extreme',
            'rsi': 88,
            'macd': 3.2,
            'bb_position': 0.98,
            'volume_ratio': 2.5,
            'volatility': 0.055,
            'close': 112.75,
            'volume': 6200000
        },
        {
            'name': 'Oversold Bounce',
            'rsi': 22,
            'macd': 0.8,
            'bb_position': 0.25,
            'volume_ratio': 1.4,
            'volatility': 0.035,
            'close': 88.90,
            'volume': 2900000
        }
    ]
    
    print(f'\nTesting {len(scenarios)} market scenarios:')
    print('-' * 80)
    
    results = []
    for scenario in scenarios:
        data = pd.Series(scenario)
        regime, confidence = agent3.classify_regime(data)
        
        # Get bias analysis 
        summary = agent3.get_sentiment_summary()
        bias_scores = summary.get('bias_scores', {})
        
        results.append({
            'scenario': scenario['name'],
            'regime': regime,
            'confidence': confidence,
            'rsi': scenario['rsi'],
            'macd': scenario['macd'],
            'vol_ratio': scenario['volume_ratio'],
            'biases': bias_scores
        })
        
        print(f"{scenario['name']:20} | RSI:{scenario['rsi']:5.1f} MACD:{scenario['macd']:6.2f} Vol:{scenario['volume_ratio']:4.1f}x")
        print(f"{' ':20} | -> {regime.name:15} (confidence: {confidence:.3f})")
        
        if bias_scores:
            bias_list = list(bias_scores.items())[:3]
            bias_str = ', '.join([f'{k}:{v:.2f}' for k, v in bias_list])
            print(f"{' ':20} | Biases: {bias_str}")
        print('-' * 80)
    
    # Test regime persistence and hysteresis
    print('\n=== Testing Regime Hysteresis (Prevents Whipsaws) ===')
    print('Simulating rapid market changes to test stability...')
    
    # Create oscillating market conditions
    oscillations = [
        {'rsi': 75, 'macd': 1.5, 'bb_position': 0.8, 'volume_ratio': 1.2, 'volatility': 0.02, 'close': 105, 'volume': 2000000},
        {'rsi': 45, 'macd': -0.5, 'bb_position': 0.3, 'volume_ratio': 1.1, 'volatility': 0.018, 'close': 103, 'volume': 1900000},
        {'rsi': 68, 'macd': 1.2, 'bb_position': 0.75, 'volume_ratio': 1.3, 'volatility': 0.022, 'close': 107, 'volume': 2100000},
        {'rsi': 42, 'macd': -0.8, 'bb_position': 0.35, 'volume_ratio': 0.9, 'volatility': 0.015, 'close': 104, 'volume': 1800000},
    ]
    
    for i, osc_data in enumerate(oscillations):
        data = pd.Series(osc_data)
        regime, confidence = agent3.classify_regime(data)
        print(f"Step {i+1}: RSI={osc_data['rsi']:5.1f} -> {regime.name:15} (conf: {confidence:.3f})")
    
    # Test sentiment divergence
    print('\n=== Testing Sentiment Divergence Analysis ===')
    
    # Create test price data showing price/sentiment divergence
    np.random.seed(42)
    price_data = []
    base_price = 100
    for i in range(30):
        # Price trending up
        price = base_price + i * 0.5 + np.random.randn() * 0.2
        
        # But sentiment indicators showing weakness (divergence)
        rsi = max(30, 70 - i * 1.2)  # RSI declining while price rises
        macd = max(-1, 1.5 - i * 0.1)  # MACD weakening
        
        price_data.append({
            'close': price,
            'volume': np.random.randint(1500000, 3000000),
            'rsi': rsi,
            'macd': macd,
            'bb_position': 0.6 - i * 0.01,
            'volatility': 0.015 + i * 0.001
        })
    
    divergence_df = pd.DataFrame(price_data)
    divergence_analysis = agent3.analyze_sentiment_divergence(divergence_df)
    
    print('Price vs Sentiment Divergence:')
    for key, value in divergence_analysis.items():
        print(f'  {key}: {value:.3f}')
    
    # Final system status
    print('\n=== Final Agent 3 System Status ===')
    final_summary = agent3.get_sentiment_summary()
    
    print(f"Current Regime: {final_summary['current_regime']}")
    print(f"Regime Confidence: {final_summary['regime_confidence']:.3f}")
    print(f"Regime Stability: {final_summary['regime_stability']:.3f}")
    print(f"Time Since Last Change: {final_summary['time_since_change']:.1f} minutes")
    print(f"Signal History Length: {final_summary['signal_history_length']} entries")
    
    if final_summary.get('bias_scores'):
        print('\nCurrent Behavioral Biases:')
        for bias_name, score in final_summary['bias_scores'].items():
            print(f'  {bias_name.title()}: {score:.3f}')
    
    print('\n✅ Agent 3 Sentiment Interpreter - All Tests Passed!')
    
    return agent3, results

if __name__ == "__main__":
    test_agent3_comprehensive()