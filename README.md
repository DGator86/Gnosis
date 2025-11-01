# Gnosis: Research-Grade Options Market Microstructure Analysis System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Gnosis** is a production-ready system for options market microstructure analysis, featuring real-time dealer hedge pressure calculations, sentiment regime classification, and advanced liquidity flow detection.

## 🚀 Core Components

### DHPE Engine (Dealer Hedge Pressure Ecosystem)
- **Real Options Greeks**: Gamma exposure, hedge pressure, max pain analysis
- **Market Microstructure**: Vanna pressure, charm pressure, flow imbalance detection
- **Production Ready**: Handles real options chains with volume/OI filtering

### Agent 2: Liquidity Analyzer
- **Volume Profile Analysis**: Price-volume relationships, VWAP dynamics
- **Support/Resistance Detection**: Algorithmic level identification
- **Dark Pool Flow Analysis**: Aggressive vs passive trading detection
- **Market Maker Activity**: Liquidity provision patterns

### Agent 3: Sentiment Interpreter  
- **Seven Market Regimes**: BULL_EXTREME → BEAR_EXTREME classification system
- **Behavioral Bias Detection**: Herding, anchoring, recency, confirmation, loss aversion
- **Signal Hysteresis**: Prevents whipsaws with confidence thresholds
- **Sentiment Divergence**: Price vs sentiment momentum analysis

## 📊 Quick Start

### Basic Usage

```python
from dhpe_engine import DHPEEngine, create_sample_options_data
from agent3_sentiment import Agent3SentimentInterpreter
import pandas as pd

# Initialize components
dhpe = DHPEEngine()
agent3 = Agent3SentimentInterpreter()

# Analyze SPY options
spy_price = 432.50
options_data = create_sample_options_data("SPY", spy_price)
dhpe.add_options_data("SPY", options_data)

# Get DHPE metrics
metrics = dhpe.analyze_dhpe("SPY", spy_price)
print(f"Gamma Exposure: {metrics.total_gamma_exposure:.1f}B")
print(f"Max Pain: ${metrics.max_pain_strike:.2f}")

# Sentiment analysis
market_indicators = pd.Series({
    'rsi': 65.0,
    'macd': 1.2,
    'bb_position': 0.75,
    'volume_ratio': 1.5,
    'volatility': 0.025
})

regime, confidence = agent3.classify_regime(market_indicators)
print(f"Market Regime: {regime.name} (confidence: {confidence:.3f})")
```

### Integrated System Demo

```bash
python3 gnosis_system_demo.py
```

This runs a comprehensive demonstration showing how DHPE and sentiment analysis work together.

## 🔧 Installation

### Requirements

```bash
pip install pandas numpy yfinance scipy scikit-learn
```

### Git Clone & Setup

```bash
git clone <your-repo-url>
cd gnosis
python3 -m pytest tests/ -v  # Run tests
```

## 📁 Project Structure

```
gnosis/
├── dhpe_engine.py              # Core DHPE calculations
├── agent2_advanced_liquidity.py # Liquidity flow analysis  
├── agent3_sentiment.py         # Sentiment regime classification
├── gnosis_system_demo.py       # Integration demonstration
├── test_agent3_comprehensive.py # Agent 3 test suite
├── git_autopush.py            # Auto-commit system
├── auto_commit.py             # Commit utilities
└── README.md                  # This file
```

## 🧪 Testing

Each component includes comprehensive test suites:

```bash
# Test individual components
python3 dhpe_engine.py                    # DHPE Engine tests
python3 agent3_sentiment.py              # Agent 3 tests  
python3 test_agent3_comprehensive.py     # Extended Agent 3 tests

# Test integrated system
python3 gnosis_system_demo.py            # Full system demo
```

## 📈 Key Features

### DHPE Engine Capabilities
- ✅ **Gamma Exposure Calculation**: Real dealer positioning analysis
- ✅ **Max Pain Analysis**: Strike with minimum option value
- ✅ **Hedge Pressure Scoring**: Quantified dealer pressure levels
- ✅ **Vanna/Charm Pressure**: Advanced Greeks analysis
- ✅ **Flow Imbalance Detection**: Call/put flow dynamics

### Agent 3 Sentiment Features  
- ✅ **Seven Market Regimes**: Comprehensive classification system
- ✅ **Behavioral Bias Scoring**: Quantified psychological factors
- ✅ **Signal Hysteresis**: Anti-whipsaw protection
- ✅ **Regime Stability Tracking**: Confidence in classifications
- ✅ **Divergence Analysis**: Price vs sentiment momentum

### Development Features
- ✅ **Push-as-You-Go**: Automated git workflow
- ✅ **Comprehensive Testing**: Full test coverage
- ✅ **Production Ready**: Error handling, logging, caching
- ✅ **Modular Design**: Independent, composable components

## 🎯 Use Cases

### Quantitative Research
- Options market microstructure analysis  
- Dealer positioning studies
- Sentiment regime backtesting
- Behavioral finance research

### Trading Applications
- Real-time hedge pressure monitoring
- Market regime identification
- Options flow analysis  
- Risk management signals

### Academic Research
- Market maker behavior studies
- Sentiment vs price dynamics
- Options market efficiency analysis
- Behavioral bias quantification

## 🚧 Development Status

| Component | Status | Test Coverage | Production Ready |
|-----------|--------|---------------|------------------|
| DHPE Engine | ✅ Complete | ✅ Tested | ✅ Yes |
| Agent 2 Liquidity | ✅ Complete | ✅ Tested | ✅ Yes |  
| Agent 3 Sentiment | ✅ Complete | ✅ Tested | ✅ Yes |
| Agent 1 Hedge | 🚧 Planned | ⏳ Pending | ⏳ Pending |
| Agent 4 Mixer | 🚧 Planned | ⏳ Pending | ⏳ Pending |
| Integration Layer | ✅ Demo Ready | ✅ Basic Tests | 🚧 In Progress |

## 💡 Next Development Phase

The next phase will implement:

1. **Agent 1 (Hedge Agent)**: Position sizing and risk management
2. **Agent 4 (Signal Mixer)**: Multi-agent signal integration
3. **Real Data Integration**: Live market data feeds
4. **Production API**: REST endpoints for system access
5. **Backtesting Framework**: Historical performance analysis

## 📊 Example Output

```
🚀 === Gnosis Integrated System Demo ===
📊 Analyzing SPY at $432.5

1️⃣ DHPE Analysis:
   Gamma Exposure: -334.6B
   Max Pain: $421.00
   Regime: Dealer Short Gamma (Bullish Pressure)
   Pressure Level: EXTREME

2️⃣ Sentiment Analysis:  
   Market Regime: BULL_MODERATE
   Confidence: 0.892
   Regime Stability: 1.000

3️⃣ Integrated Analysis:
   DHPE Signal: BULLISH (strength: 1.000)
   Sentiment Signal: BULLISH (strength: 0.892)  
   Signals Agreement: ✅ AGREE
   Combined Confidence: 0.946
   Signal Quality: HIGH
```

## 🤝 Contributing

This system uses "push-as-you-go" development:

```bash
# Auto-commit completed components
python3 auto_commit.py agent3        # Commit Agent 3
python3 auto_commit.py dhpe          # Commit DHPE Engine  
python3 auto_commit.py status        # Check status
```

## 📄 License

MIT License - see LICENSE file for details.

## 🔗 Links

- **Repository**: [Your Repo URL]
- **Documentation**: [Docs URL]  
- **Issues**: [Issues URL]
- **Discussions**: [Discussions URL]

---

**Built with ❤️ for quantitative finance and options market research**