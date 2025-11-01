"""
Gnosis Production API Server
FastAPI-based REST API for accessing complete Gnosis system functionality
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
import uvicorn
from contextlib import asynccontextmanager
import jwt
import os
from pathlib import Path

# Import Gnosis components
from gnosis_integration_layer import GnosisOrchestrator, GnosisConfig, GnosisResult
from agent1_hedge_complete import RiskLevel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global Gnosis instance
gnosis_system: Optional[GnosisOrchestrator] = None

# Security
security = HTTPBearer()
SECRET_KEY = os.getenv("GNOSIS_SECRET_KEY", "your-secret-key-here")

# Pydantic models for API
class AnalysisRequest(BaseModel):
    symbol: str = Field(..., description="Trading symbol to analyze")
    include_options_chain: bool = Field(True, description="Include options chain analysis")
    risk_tolerance: str = Field("moderate", description="Risk tolerance level")

class BulkAnalysisRequest(BaseModel):
    symbols: List[str] = Field(..., description="List of symbols to analyze")
    concurrent_limit: int = Field(3, description="Maximum concurrent analyses")

class SystemConfigRequest(BaseModel):
    symbols: List[str] = Field(default_factory=lambda: ['SPY'])
    risk_tolerance: str = Field("moderate")
    analysis_interval: int = Field(60, description="Analysis interval in seconds")
    api_keys: Dict[str, str] = Field(default_factory=dict)

class AlertSubscription(BaseModel):
    symbols: List[str] = Field(..., description="Symbols to monitor")
    min_conviction_score: float = Field(0.7, description="Minimum conviction for alerts")
    alert_types: List[str] = Field(default_factory=lambda: ["high_risk", "strong_signal"])
    webhook_url: Optional[str] = Field(None, description="Webhook URL for alerts")

class AnalysisResponse(BaseModel):
    timestamp: str
    symbol: str
    recommendation: str
    confidence_score: float
    execution_priority: str
    processing_time: float
    mixed_signal: Dict[str, Any]
    agent_results: Dict[str, Any]
    risk_assessment: Dict[str, Any]
    alerts: List[Dict[str, Any]]

class SystemStatusResponse(BaseModel):
    status: str
    uptime_seconds: float
    active_symbols: List[str]
    agent_health: Dict[str, bool]
    performance_metrics: Dict[str, Any]
    last_analysis: Optional[str]

# Lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    global gnosis_system
    
    # Startup
    logger.info("Starting Gnosis Production API...")
    
    # Initialize with default config
    config = GnosisConfig(
        symbols=['SPY', 'QQQ', 'IWM'],
        risk_tolerance=RiskLevel.MODERATE,
        analysis_interval=60
    )
    
    gnosis_system = GnosisOrchestrator(config)
    
    # Initialize system
    if not await gnosis_system.initialize():
        logger.error("Failed to initialize Gnosis system")
        raise Exception("System initialization failed")
        
    logger.info("Gnosis system initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Gnosis system...")
    if gnosis_system:
        await gnosis_system.shutdown()
    logger.info("Gnosis system shutdown complete")

# Create FastAPI app
app = FastAPI(
    title="Gnosis Trading System API",
    description="Production API for the complete Gnosis agentic options trading framework",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Authentication
async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> str:
    """Verify JWT token"""
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=["HS256"])
        username: str = payload.get("sub")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return username
    except jwt.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/health/detailed", response_model=SystemStatusResponse)
async def detailed_health_check():
    """Detailed system health check"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    # Get agent health status
    agent_health = await gnosis_system._check_agent_health()
    
    # Calculate uptime (simplified)
    uptime_seconds = 3600.0  # Placeholder
    
    # Get performance metrics
    performance_metrics = {
        "average_analysis_time": 2.5,
        "successful_analyses_24h": 1000,
        "failed_analyses_24h": 5,
        "cache_hit_rate": 0.85
    }
    
    return SystemStatusResponse(
        status="healthy" if all(agent_health.values()) else "degraded",
        uptime_seconds=uptime_seconds,
        active_symbols=gnosis_system.config.symbols,
        agent_health=agent_health,
        performance_metrics=performance_metrics,
        last_analysis=datetime.now().isoformat()
    )

# Core analysis endpoints
@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_symbol(
    request: AnalysisRequest,
    username: str = Depends(verify_token)
):
    """Run complete Gnosis analysis for a single symbol"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        logger.info(f"User {username} requested analysis for {request.symbol}")
        
        # Run analysis
        result = await gnosis_system.run_complete_analysis(request.symbol)
        
        # Format response
        response = AnalysisResponse(
            timestamp=result.timestamp.isoformat(),
            symbol=result.symbol,
            recommendation=result.final_recommendation,
            confidence_score=result.confidence_score,
            execution_priority=result.execution_priority,
            processing_time=result.performance_metrics['processing_time_seconds'],
            mixed_signal=result.mixed_signal,
            agent_results={
                'dhpe': result.dhpe_result,
                'liquidity': result.liquidity_result,
                'sentiment': result.sentiment_result,
                'hedge': result.hedge_result
            },
            risk_assessment=result.risk_assessment,
            alerts=result.alerts
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis failed for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze/bulk")
async def bulk_analyze(
    request: BulkAnalysisRequest,
    background_tasks: BackgroundTasks,
    username: str = Depends(verify_token)
):
    """Run analysis for multiple symbols concurrently"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    if len(request.symbols) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 symbols per bulk request")
        
    try:
        logger.info(f"User {username} requested bulk analysis for {len(request.symbols)} symbols")
        
        # Run concurrent analyses
        tasks = [gnosis_system.run_complete_analysis(symbol) for symbol in request.symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Format results
        bulk_results = []
        for symbol, result in zip(request.symbols, results):
            if isinstance(result, Exception):
                bulk_results.append({
                    'symbol': symbol,
                    'status': 'error',
                    'error': str(result)
                })
            else:
                bulk_results.append({
                    'symbol': symbol,
                    'status': 'success',
                    'recommendation': result.final_recommendation,
                    'confidence_score': result.confidence_score,
                    'processing_time': result.performance_metrics['processing_time_seconds']
                })
                
        return {
            'timestamp': datetime.now().isoformat(),
            'total_symbols': len(request.symbols),
            'successful': len([r for r in bulk_results if r['status'] == 'success']),
            'failed': len([r for r in bulk_results if r['status'] == 'error']),
            'results': bulk_results
        }
        
    except Exception as e:
        logger.error(f"Bulk analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Bulk analysis failed: {str(e)}")

# Individual agent endpoints
@app.get("/agents/dhpe/{symbol}")
async def dhpe_analysis(symbol: str, username: str = Depends(verify_token)):
    """Get DHPE engine analysis only"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        market_data = await gnosis_system._get_market_data(symbol)
        result = await gnosis_system._run_dhpe_analysis(symbol, market_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"DHPE analysis failed: {str(e)}")

@app.get("/agents/liquidity/{symbol}")
async def liquidity_analysis(symbol: str, username: str = Depends(verify_token)):
    """Get liquidity agent analysis only"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        market_data = await gnosis_system._get_market_data(symbol)
        result = await gnosis_system._run_liquidity_analysis(symbol, market_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Liquidity analysis failed: {str(e)}")

@app.get("/agents/sentiment/{symbol}")
async def sentiment_analysis(symbol: str, username: str = Depends(verify_token)):
    """Get sentiment agent analysis only"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        market_data = await gnosis_system._get_market_data(symbol)
        result = await gnosis_system._run_sentiment_analysis(symbol, market_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")

@app.get("/agents/hedge/{symbol}")
async def hedge_analysis(symbol: str, username: str = Depends(verify_token)):
    """Get hedge agent analysis only"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        market_data = await gnosis_system._get_market_data(symbol)
        result = await gnosis_system._run_hedge_analysis(symbol, market_data)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Hedge analysis failed: {str(e)}")

# Market data endpoints
@app.get("/market-data/{symbol}")
async def get_market_data(symbol: str, username: str = Depends(verify_token)):
    """Get current market data for symbol"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        market_data = await gnosis_system._get_market_data(symbol)
        return market_data
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Market data retrieval failed: {str(e)}")

@app.get("/market-data/{symbol}/options")
async def get_options_chain(symbol: str, expiry: Optional[str] = None, username: str = Depends(verify_token)):
    """Get options chain for symbol"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        options_chain = await gnosis_system.market_data.get_options_chain(symbol, expiry)
        return {
            'symbol': symbol,
            'expiry': expiry,
            'options_count': len(options_chain),
            'options': [
                {
                    'symbol': opt.symbol,
                    'strike': opt.strike,
                    'expiry': opt.expiry,
                    'type': opt.option_type,
                    'bid': opt.bid,
                    'ask': opt.ask,
                    'last': opt.last,
                    'volume': opt.volume,
                    'open_interest': opt.open_interest,
                    'implied_volatility': opt.implied_volatility,
                    'delta': opt.delta,
                    'gamma': opt.gamma,
                    'theta': opt.theta,
                    'vega': opt.vega
                } for opt in options_chain
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Options chain retrieval failed: {str(e)}")

# System configuration endpoints
@app.post("/config/update")
async def update_config(
    config_request: SystemConfigRequest,
    username: str = Depends(verify_token)
):
    """Update system configuration"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        logger.info(f"User {username} updating system configuration")
        
        # Update configuration
        risk_level_map = {
            'conservative': RiskLevel.CONSERVATIVE,
            'moderate': RiskLevel.MODERATE,
            'aggressive': RiskLevel.AGGRESSIVE,
            'extreme': RiskLevel.EXTREME
        }
        
        new_config = GnosisConfig(
            symbols=config_request.symbols,
            risk_tolerance=risk_level_map.get(config_request.risk_tolerance, RiskLevel.MODERATE),
            analysis_interval=config_request.analysis_interval,
            api_keys=config_request.api_keys
        )
        
        # Recreate system with new config
        await gnosis_system.shutdown()
        gnosis_system = GnosisOrchestrator(new_config)
        
        if not await gnosis_system.initialize():
            raise Exception("Failed to reinitialize with new configuration")
            
        return {
            'status': 'success',
            'message': 'Configuration updated successfully',
            'new_config': {
                'symbols': new_config.symbols,
                'risk_tolerance': new_config.risk_tolerance.value,
                'analysis_interval': new_config.analysis_interval
            }
        }
        
    except Exception as e:
        logger.error(f"Configuration update failed: {e}")
        raise HTTPException(status_code=500, detail=f"Configuration update failed: {str(e)}")

# Continuous monitoring endpoints
@app.post("/monitor/start")
async def start_monitoring(
    symbols: Optional[List[str]] = None,
    username: str = Depends(verify_token)
):
    """Start continuous monitoring"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        logger.info(f"User {username} starting continuous monitoring")
        
        # Start continuous analysis in background
        if symbols is None:
            symbols = gnosis_system.config.symbols
            
        asyncio.create_task(gnosis_system.run_continuous_analysis(symbols))
        
        return {
            'status': 'success',
            'message': 'Continuous monitoring started',
            'symbols': symbols,
            'interval_seconds': gnosis_system.config.analysis_interval
        }
        
    except Exception as e:
        logger.error(f"Failed to start monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@app.post("/monitor/stop")
async def stop_monitoring(username: str = Depends(verify_token)):
    """Stop continuous monitoring"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    try:
        logger.info(f"User {username} stopping continuous monitoring")
        gnosis_system.stop_continuous_analysis()
        
        return {
            'status': 'success',
            'message': 'Continuous monitoring stopped'
        }
        
    except Exception as e:
        logger.error(f"Failed to stop monitoring: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

# Performance and analytics endpoints
@app.get("/analytics/performance")
async def get_performance_analytics(
    symbol: Optional[str] = None,
    hours: int = 24,
    username: str = Depends(verify_token)
):
    """Get system performance analytics"""
    global gnosis_system
    
    if not gnosis_system:
        raise HTTPException(status_code=503, detail="System not initialized")
        
    # Mock performance data (in production, would query from database)
    return {
        'timeframe_hours': hours,
        'symbol_filter': symbol,
        'metrics': {
            'total_analyses': 1000,
            'successful_analyses': 985,
            'average_processing_time': 2.3,
            'average_confidence_score': 0.72,
            'recommendations': {
                'BUY': 245,
                'SELL': 180,
                'HOLD': 560,
                'HEDGE': 15
            }
        },
        'agent_performance': {
            'dhpe': {'success_rate': 0.98, 'avg_confidence': 0.75},
            'liquidity': {'success_rate': 0.97, 'avg_confidence': 0.72},
            'sentiment': {'success_rate': 0.95, 'avg_confidence': 0.68},
            'hedge': {'success_rate': 0.99, 'avg_confidence': 0.85}
        }
    }

# Authentication endpoints
@app.post("/auth/login")
async def login(username: str, password: str):
    """Login and get JWT token"""
    # Simplified authentication (implement proper auth in production)
    if username == "admin" and password == "password":
        token_data = {"sub": username, "exp": datetime.utcnow() + timedelta(hours=24)}
        token = jwt.encode(token_data, SECRET_KEY, algorithm="HS256")
        return {"access_token": token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")

# WebSocket endpoint for real-time updates
from fastapi import WebSocket

@app.websocket("/ws/realtime/{symbol}")
async def websocket_realtime(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time updates"""
    await websocket.accept()
    
    try:
        while True:
            # In production, would stream real-time analysis updates
            await asyncio.sleep(5)
            
            # Send mock update
            update = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'type': 'market_data_update',
                'data': {
                    'price': 450.0 + np.random.normal(0, 5),
                    'volume': int(1000000 * np.random.uniform(0.5, 1.5))
                }
            }
            
            await websocket.send_text(json.dumps(update))
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Custom HTTP exception handler"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """General exception handler"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat(),
            "path": str(request.url)
        }
    )

# Startup function for production deployment
def create_app() -> FastAPI:
    """Factory function to create the FastAPI app"""
    return app

if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "gnosis_production_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )