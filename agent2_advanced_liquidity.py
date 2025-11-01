#!/usr/bin/env python3
"""
Gnosis DHPE Framework - Agent 2: Advanced Liquidity Analysis
Complete implementation with professional liquidity metrics and market maker flow detection
"""

import numpy as np
import pandas as pd
import yfinance as yf
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class LiquidityMetrics:
    """Complete liquidity metrics for an option or underlying"""
    symbol: str
    bid_ask_spread: float
    bid_ask_spread_pct: float
    volume: int
    open_interest: int
    volume_oi_ratio: float
    market_maker_flow: float
    liquidity_score: float
    depth_score: float
    efficiency_score: float
    risk_level: str

class AdvancedLiquidityAnalyzer:
    """
    Agent 2: Advanced Liquidity Analysis for Gnosis DHPE Framework
    
    Capabilities:
    - Bid-ask spread analysis with percentile rankings
    - Market maker flow detection and classification
    - Options volume vs underlying volume analysis  
    - Liquidity scoring with multi-factor approach
    - Real-time liquidity monitoring and alerts
    - Cross-strike liquidity comparison
    - Time-based liquidity pattern analysis
    """
    
    def __init__(self):
        self.min_volume_threshold = 100
        self.max_spread_threshold = 0.05  # 5%
        self.liquidity_history = {}
        
        print("✅ Agent 2: Advanced Liquidity Analyzer initialized")
        print("📊 Capabilities: Spread analysis, MM flow detection, liquidity scoring")
    
    def analyze_underlying_liquidity(self, symbol: str) -> Dict[str, Any]:
        """
        Analyze underlying stock liquidity characteristics
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get current quote data
            info = ticker.info
            hist = ticker.history(period="5d", interval="1m").tail(100)
            
            if hist.empty:
                return self._create_empty_liquidity_result(symbol, "No historical data")
            
            # Calculate liquidity metrics
            avg_volume = hist['Volume'].mean()
            volume_std = hist['Volume'].std()
            price_range = hist['High'].max() - hist['Low'].min()
            avg_price = hist['Close'].mean()
            
            # Estimate bid-ask spread (approximate from intraday data)
            estimated_spread = self._estimate_spread_from_ohlc(hist)
            spread_pct = (estimated_spread / avg_price) * 100 if avg_price > 0 else 0
            
            # Volume patterns
            volume_consistency = 1 - (volume_std / avg_volume) if avg_volume > 0 else 0
            
            # Market efficiency metrics
            price_volatility = hist['Close'].pct_change().std() * np.sqrt(252) * 100  # Annualized vol %
            
            result = {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "avg_daily_volume": int(avg_volume),
                "volume_consistency": round(volume_consistency, 4),
                "estimated_spread_pct": round(spread_pct, 4),
                "price_volatility": round(price_volatility, 2),
                "price_range_5d": round(price_range, 2),
                "avg_price": round(avg_price, 2),
                "liquidity_tier": self._classify_liquidity_tier(avg_volume, spread_pct),
                "trading_recommendation": self._get_trading_recommendation(avg_volume, spread_pct, volume_consistency)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing underlying liquidity for {symbol}: {e}")
            return self._create_empty_liquidity_result(symbol, str(e))
    
    def analyze_options_chain_liquidity(self, symbol: str, expiry_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive options chain liquidity analysis
        """
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period="1d")['Close'].iloc[-1]
            
            # Get options chain
            if expiry_date:
                options_chain = ticker.option_chain(expiry_date)
            else:
                # Get nearest expiry
                expiry_dates = ticker.options
                if not expiry_dates:
                    return self._create_empty_options_result(symbol, "No options available")
                options_chain = ticker.option_chain(expiry_dates[0])
                expiry_date = expiry_dates[0]
            
            calls_liquidity = self._analyze_options_side(options_chain.calls, "call", current_price)
            puts_liquidity = self._analyze_options_side(options_chain.puts, "put", current_price)
            
            # Overall chain analysis
            chain_analysis = self._analyze_full_chain(calls_liquidity, puts_liquidity, current_price)
            
            result = {
                "symbol": symbol,
                "expiry_date": expiry_date,
                "underlying_price": round(current_price, 2),
                "timestamp": datetime.now().isoformat(),
                "calls_analysis": calls_liquidity,
                "puts_analysis": puts_liquidity,
                "chain_summary": chain_analysis,
                "trading_zones": self._identify_trading_zones(calls_liquidity, puts_liquidity, current_price),
                "market_maker_signals": self._detect_market_maker_activity(calls_liquidity, puts_liquidity)
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing options liquidity for {symbol}: {e}")
            return self._create_empty_options_result(symbol, str(e))
    
    def _analyze_options_side(self, options_df: pd.DataFrame, side: str, current_price: float) -> Dict[str, Any]:
        """
        Analyze liquidity for calls or puts side
        """
        if options_df.empty:
            return {"side": side, "contracts": [], "summary": {}}
        
        contracts_analysis = []
        
        for idx, row in options_df.iterrows():
            try:
                strike = float(row['strike'])
                bid = float(row.get('bid', 0))
                ask = float(row.get('ask', 0))
                volume = int(row.get('volume', 0))
                open_interest = int(row.get('openInterest', 0))
                implied_vol = float(row.get('impliedVolatility', 0))
                
                # Calculate spread metrics
                mid_price = (bid + ask) / 2 if (bid > 0 and ask > 0) else 0
                spread = ask - bid if (bid > 0 and ask > 0) else 0
                spread_pct = (spread / mid_price * 100) if mid_price > 0 else 100
                
                # Calculate moneyness
                moneyness = current_price / strike if strike > 0 else 0
                
                # Volume/OI analysis
                vol_oi_ratio = volume / open_interest if open_interest > 0 else 0
                
                # Market maker flow estimation
                mm_flow = self._estimate_market_maker_flow(volume, open_interest, vol_oi_ratio, moneyness, side)
                
                # Liquidity scoring
                liquidity_score = self._calculate_liquidity_score(
                    volume, open_interest, spread_pct, implied_vol
                )
                
                contract_analysis = {
                    "strike": strike,
                    "bid": bid,
                    "ask": ask,
                    "mid_price": round(mid_price, 3),
                    "spread": round(spread, 3),
                    "spread_pct": round(spread_pct, 2),
                    "volume": volume,
                    "open_interest": open_interest,
                    "vol_oi_ratio": round(vol_oi_ratio, 3),
                    "moneyness": round(moneyness, 4),
                    "implied_volatility": round(implied_vol, 4),
                    "market_maker_flow": round(mm_flow, 3),
                    "liquidity_score": round(liquidity_score, 2),
                    "liquidity_tier": self._get_liquidity_tier(liquidity_score),
                    "trading_suitability": self._assess_trading_suitability(volume, spread_pct)
                }
                
                contracts_analysis.append(contract_analysis)
                
            except Exception as e:
                logger.warning(f"Error processing strike {row.get('strike', 'unknown')}: {e}")
                continue
        
        # Side summary
        if contracts_analysis:
            volumes = [c['volume'] for c in contracts_analysis]
            spreads = [c['spread_pct'] for c in contracts_analysis if c['spread_pct'] < 100]
            liquidity_scores = [c['liquidity_score'] for c in contracts_analysis]
            
            summary = {
                "total_contracts": len(contracts_analysis),
                "avg_volume": round(np.mean(volumes), 0) if volumes else 0,
                "avg_spread_pct": round(np.mean(spreads), 2) if spreads else 0,
                "avg_liquidity_score": round(np.mean(liquidity_scores), 2) if liquidity_scores else 0,
                "liquid_contracts": len([c for c in contracts_analysis if c['liquidity_score'] > 60]),
                "tight_spread_contracts": len([c for c in contracts_analysis if c['spread_pct'] < 5])
            }
        else:
            summary = {"total_contracts": 0, "avg_volume": 0, "avg_spread_pct": 0, "avg_liquidity_score": 0}
        
        return {
            "side": side,
            "contracts": contracts_analysis,
            "summary": summary
        }
    
    def _analyze_full_chain(self, calls_data: Dict, puts_data: Dict, current_price: float) -> Dict[str, Any]:
        """
        Analyze the complete options chain for liquidity patterns
        """
        all_contracts = calls_data.get('contracts', []) + puts_data.get('contracts', [])
        
        if not all_contracts:
            return {"overall_liquidity": "POOR", "best_strikes": [], "liquidity_concentration": "DISPERSED"}
        
        # Find most liquid strikes
        sorted_by_liquidity = sorted(all_contracts, key=lambda x: x['liquidity_score'], reverse=True)
        best_strikes = sorted_by_liquidity[:5]
        
        # Analyze liquidity concentration
        atm_strikes = [c for c in all_contracts if 0.95 <= c['moneyness'] <= 1.05]
        otm_strikes = [c for c in all_contracts if c['moneyness'] < 0.95 or c['moneyness'] > 1.05]
        
        atm_avg_liquidity = np.mean([c['liquidity_score'] for c in atm_strikes]) if atm_strikes else 0
        otm_avg_liquidity = np.mean([c['liquidity_score'] for c in otm_strikes]) if otm_strikes else 0
        
        # Overall assessment
        avg_liquidity = np.mean([c['liquidity_score'] for c in all_contracts])
        
        if avg_liquidity > 70:
            overall_liquidity = "EXCELLENT"
        elif avg_liquidity > 50:
            overall_liquidity = "GOOD"
        elif avg_liquidity > 30:
            overall_liquidity = "MODERATE"
        else:
            overall_liquidity = "POOR"
        
        concentration = "ATM_CONCENTRATED" if atm_avg_liquidity > otm_avg_liquidity * 1.5 else "DISPERSED"
        
        return {
            "overall_liquidity": overall_liquidity,
            "avg_liquidity_score": round(avg_liquidity, 2),
            "best_strikes": [{"strike": s['strike'], "side": "call" if s['moneyness'] >= 1 else "put", 
                           "liquidity_score": s['liquidity_score']} for s in best_strikes],
            "liquidity_concentration": concentration,
            "atm_liquidity": round(atm_avg_liquidity, 2),
            "otm_liquidity": round(otm_avg_liquidity, 2),
            "total_volume": sum([c['volume'] for c in all_contracts]),
            "total_open_interest": sum([c['open_interest'] for c in all_contracts])
        }
    
    def _estimate_market_maker_flow(self, volume: int, open_interest: int, vol_oi_ratio: float, 
                                  moneyness: float, side: str) -> float:
        """
        Estimate market maker flow based on volume and positioning patterns
        
        Returns: 
        Positive = MM likely buying (providing liquidity)
        Negative = MM likely selling (removing liquidity)
        """
        
        # Base flow from volume/OI ratio
        if vol_oi_ratio > 2.0:
            base_flow = 0.5  # High volume suggests active MM participation
        elif vol_oi_ratio > 1.0:
            base_flow = 0.2
        else:
            base_flow = -0.1  # Low volume may indicate MM reluctance
        
        # Adjust for moneyness (MMs typically more active near ATM)
        if 0.95 <= moneyness <= 1.05:  # ATM
            moneyness_adj = 0.3
        elif 0.90 <= moneyness <= 1.10:  # Near ATM
            moneyness_adj = 0.1
        else:  # Far OTM
            moneyness_adj = -0.2
        
        # Adjust for absolute volume (MMs need minimum volume to be profitable)
        if volume > 1000:
            volume_adj = 0.2
        elif volume > 100:
            volume_adj = 0.0
        else:
            volume_adj = -0.3
        
        total_flow = base_flow + moneyness_adj + volume_adj
        
        # Clamp to reasonable range
        return max(-1.0, min(1.0, total_flow))
    
    def _calculate_liquidity_score(self, volume: int, open_interest: int, 
                                 spread_pct: float, implied_vol: float) -> float:
        """
        Calculate comprehensive liquidity score (0-100)
        """
        
        # Volume component (0-30 points)
        if volume >= 1000:
            volume_score = 30
        elif volume >= 500:
            volume_score = 25
        elif volume >= 100:
            volume_score = 20
        elif volume >= 50:
            volume_score = 15
        elif volume >= 10:
            volume_score = 10
        else:
            volume_score = 0
        
        # Open Interest component (0-25 points)
        if open_interest >= 5000:
            oi_score = 25
        elif open_interest >= 1000:
            oi_score = 20
        elif open_interest >= 500:
            oi_score = 15
        elif open_interest >= 100:
            oi_score = 10
        else:
            oi_score = 0
        
        # Spread component (0-30 points)
        if spread_pct <= 2:
            spread_score = 30
        elif spread_pct <= 5:
            spread_score = 25
        elif spread_pct <= 10:
            spread_score = 20
        elif spread_pct <= 20:
            spread_score = 10
        else:
            spread_score = 0
        
        # Implied Volatility component (0-15 points) - reasonable IV suggests active market
        if 0.15 <= implied_vol <= 0.50:
            iv_score = 15
        elif 0.10 <= implied_vol <= 0.80:
            iv_score = 10
        else:
            iv_score = 5
        
        total_score = volume_score + oi_score + spread_score + iv_score
        return min(100, max(0, total_score))
    
    def _identify_trading_zones(self, calls_data: Dict, puts_data: Dict, current_price: float) -> Dict[str, Any]:
        """
        Identify optimal trading zones based on liquidity analysis
        """
        
        all_contracts = calls_data.get('contracts', []) + puts_data.get('contracts', [])
        
        if not all_contracts:
            return {"zones": [], "recommendation": "No suitable trading zones identified"}
        
        # High liquidity zone
        high_liquidity = [c for c in all_contracts if c['liquidity_score'] > 70]
        
        # Tight spread zone  
        tight_spreads = [c for c in all_contracts if c['spread_pct'] < 5]
        
        # High volume zone
        high_volume = [c for c in all_contracts if c['volume'] > 100]
        
        # ATM zone (best for most strategies)
        atm_zone = [c for c in all_contracts if 0.90 <= c['moneyness'] <= 1.10]
        
        zones = []
        
        if high_liquidity:
            avg_strike = np.mean([c['strike'] for c in high_liquidity])
            zones.append({
                "name": "HIGH_LIQUIDITY_ZONE",
                "center_strike": round(avg_strike, 0),
                "contracts_count": len(high_liquidity),
                "avg_liquidity_score": round(np.mean([c['liquidity_score'] for c in high_liquidity]), 1),
                "recommendation": "Optimal for large size trades"
            })
        
        if tight_spreads:
            avg_strike = np.mean([c['strike'] for c in tight_spreads])
            zones.append({
                "name": "TIGHT_SPREADS_ZONE", 
                "center_strike": round(avg_strike, 0),
                "contracts_count": len(tight_spreads),
                "avg_spread_pct": round(np.mean([c['spread_pct'] for c in tight_spreads]), 2),
                "recommendation": "Optimal for cost-efficient entry/exit"
            })
        
        overall_recommendation = "PROCEED_WITH_CAUTION"
        if len(zones) >= 2:
            overall_recommendation = "GOOD_TRADING_CONDITIONS"
        elif len(zones) >= 1:
            overall_recommendation = "MODERATE_TRADING_CONDITIONS"
        
        return {
            "zones": zones,
            "overall_recommendation": overall_recommendation,
            "atm_liquidity_available": len(atm_zone) > 0,
            "current_price": current_price
        }
    
    def _detect_market_maker_activity(self, calls_data: Dict, puts_data: Dict) -> Dict[str, Any]:
        """
        Detect market maker activity patterns
        """
        
        all_contracts = calls_data.get('contracts', []) + puts_data.get('contracts', [])
        
        if not all_contracts:
            return {"activity_level": "NO_DATA", "signals": []}
        
        # High MM flow contracts
        high_mm_flow = [c for c in all_contracts if c['market_maker_flow'] > 0.3]
        
        # Negative MM flow (potential MM selling)
        negative_mm_flow = [c for c in all_contracts if c['market_maker_flow'] < -0.2]
        
        # Unusual volume patterns
        high_vol_contracts = sorted([c for c in all_contracts if c['volume'] > 500], 
                                  key=lambda x: x['volume'], reverse=True)[:5]
        
        signals = []
        
        if len(high_mm_flow) > 3:
            signals.append("Strong market maker buying activity detected")
        
        if len(negative_mm_flow) > 5:
            signals.append("Market makers may be reducing positions")
        
        if high_vol_contracts:
            max_volume = max([c['volume'] for c in high_vol_contracts])
            if max_volume > 2000:
                signals.append(f"Unusual volume spike detected: {max_volume} contracts")
        
        # Overall activity level
        avg_mm_flow = np.mean([c['market_maker_flow'] for c in all_contracts])
        
        if avg_mm_flow > 0.2:
            activity_level = "HIGH_BUYING"
        elif avg_mm_flow > 0.0:
            activity_level = "MODERATE_BUYING"
        elif avg_mm_flow > -0.2:
            activity_level = "NEUTRAL"
        else:
            activity_level = "SELLING_PRESSURE"
        
        return {
            "activity_level": activity_level,
            "avg_mm_flow": round(avg_mm_flow, 3),
            "signals": signals,
            "high_activity_strikes": [c['strike'] for c in high_mm_flow],
            "volume_leaders": [{"strike": c['strike'], "volume": c['volume']} for c in high_vol_contracts]
        }
    
    def _estimate_spread_from_ohlc(self, hist_data: pd.DataFrame) -> float:
        """
        Estimate bid-ask spread from OHLC data
        """
        if hist_data.empty:
            return 0.0
        
        # Simple estimation: average of (High - Low) for recent periods
        spreads = hist_data['High'] - hist_data['Low']
        return spreads.mean()
    
    def _classify_liquidity_tier(self, avg_volume: float, spread_pct: float) -> str:
        """Classify overall liquidity tier"""
        if avg_volume > 1000000 and spread_pct < 0.1:
            return "TIER_1_PREMIUM"
        elif avg_volume > 500000 and spread_pct < 0.2:
            return "TIER_2_HIGH"
        elif avg_volume > 100000 and spread_pct < 0.5:
            return "TIER_3_MODERATE"
        else:
            return "TIER_4_LOW"
    
    def _get_trading_recommendation(self, avg_volume: float, spread_pct: float, consistency: float) -> str:
        """Get trading recommendation based on liquidity metrics"""
        if avg_volume > 500000 and spread_pct < 0.2 and consistency > 0.7:
            return "EXCELLENT_FOR_TRADING"
        elif avg_volume > 100000 and spread_pct < 0.5:
            return "GOOD_FOR_TRADING"
        elif avg_volume > 50000:
            return "MODERATE_CAUTION_ADVISED"
        else:
            return "HIGH_RISK_LOW_LIQUIDITY"
    
    def _get_liquidity_tier(self, score: float) -> str:
        """Convert liquidity score to tier"""
        if score >= 80:
            return "PREMIUM"
        elif score >= 60:
            return "HIGH"
        elif score >= 40:
            return "MODERATE"
        elif score >= 20:
            return "LOW"
        else:
            return "VERY_LOW"
    
    def _assess_trading_suitability(self, volume: int, spread_pct: float) -> str:
        """Assess suitability for trading"""
        if volume >= 100 and spread_pct <= 5:
            return "SUITABLE"
        elif volume >= 50 and spread_pct <= 10:
            return "MARGINAL"
        else:
            return "NOT_RECOMMENDED"
    
    def _create_empty_liquidity_result(self, symbol: str, error: str) -> Dict[str, Any]:
        """Create empty result for error cases"""
        return {
            "symbol": symbol,
            "error": error,
            "liquidity_tier": "UNKNOWN",
            "trading_recommendation": "DATA_UNAVAILABLE"
        }
    
    def _create_empty_options_result(self, symbol: str, error: str) -> Dict[str, Any]:
        """Create empty options result for error cases"""
        return {
            "symbol": symbol,
            "error": error,
            "chain_summary": {"overall_liquidity": "UNKNOWN"},
            "trading_zones": {"zones": [], "recommendation": "DATA_UNAVAILABLE"}
        }

def test_agent2_liquidity():
    """
    Test Agent 2 with real market data
    """
    print("🚀 TESTING AGENT 2 - ADVANCED LIQUIDITY ANALYSIS")
    print("=" * 60)
    
    analyzer = AdvancedLiquidityAnalyzer()
    
    # Test symbols
    test_symbols = ['AAPL', 'TSLA', 'SPY']
    
    for symbol in test_symbols:
        print(f"\n📊 ANALYZING: {symbol}")
        print("-" * 40)
        
        # Analyze underlying
        underlying_analysis = analyzer.analyze_underlying_liquidity(symbol)
        print(f"💰 Avg Daily Volume: {underlying_analysis.get('avg_daily_volume', 0):,}")
        print(f"📈 Liquidity Tier: {underlying_analysis.get('liquidity_tier', 'Unknown')}")
        print(f"🎯 Trading Recommendation: {underlying_analysis.get('trading_recommendation', 'Unknown')}")
        
        # Analyze options
        options_analysis = analyzer.analyze_options_chain_liquidity(symbol)
        if 'error' not in options_analysis:
            chain_summary = options_analysis.get('chain_summary', {})
            print(f"⚡ Options Liquidity: {chain_summary.get('overall_liquidity', 'Unknown')}")
            print(f"📊 Avg Liquidity Score: {chain_summary.get('avg_liquidity_score', 0)}")
            
            trading_zones = options_analysis.get('trading_zones', {})
            print(f"🎯 Trading Zones: {len(trading_zones.get('zones', []))} identified")
            print(f"💡 Overall Recommendation: {trading_zones.get('overall_recommendation', 'Unknown')}")
            
            mm_activity = options_analysis.get('market_maker_signals', {})
            print(f"🏦 MM Activity Level: {mm_activity.get('activity_level', 'Unknown')}")
        else:
            print(f"❌ Options Analysis Error: {options_analysis['error']}")
        
        print()

if __name__ == "__main__":
    test_agent2_liquidity()