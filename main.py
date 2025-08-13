#!/usr/bin/env python3
"""
Coinbase Trade Assistant MCP Server

A sophisticated Model Context Protocol server for cryptocurrency trading assistance
using Coinbase Advanced Trade API. Features automated technical analysis screening
and real-time market insights.

Author: AI Workspace
License: MIT
"""

import asyncio
import atexit
import json
import logging
import os
import signal
import sys
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Import MCP and FastMCP components
from mcp.server.fastmcp import Context, FastMCP
from mcp.server.fastmcp.prompts import base

# Configure logging to stderr for MCP servers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Import Coinbase SDK
try:
    from coinbase.rest import RESTClient
    from coinbase.websocket import WSClient
except ImportError as e:
    logger.error(f"Failed to import Coinbase SDK: {e}")
    logger.error("Please install with: uv add coinbase-advanced-py")
    sys.exit(1)

# Import technical analysis libraries
try:
    import numpy as np
    import pandas as pd
    import talib
except ImportError as e:
    logger.error(f"Failed to import technical analysis libraries: {e}")
    logger.error("Please install with: uv add pandas numpy ta-lib")
    sys.exit(1)

# Global variables for process management and state
running_processes: Dict[str, Any] = {}
background_tasks: set = set()
coinbase_client: Optional[RESTClient] = None
websocket_client: Optional[WSClient] = None
market_data_cache: Dict[str, Any] = {}
screening_results: Dict[str, Any] = {}

# Initialize MCP server
mcp = FastMCP("CoinbaseTradeAssistant")

def cleanup_processes():
    """Clean up all running processes and background tasks"""
    logger.info("Starting cleanup process...")

    # Cancel background tasks
    for task in background_tasks:
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled background task: {task}")

    # Clear global state
    running_processes.clear()
    background_tasks.clear()
    market_data_cache.clear()
    screening_results.clear()

    logger.info("Cleanup completed")

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {signum}, shutting down gracefully")
    cleanup_processes()
    sys.exit(0)

def get_coinbase_config() -> str:
    """Get Coinbase API configuration from JSON key file"""
    # Try environment variable first (for backward compatibility)
    key_file_path = os.getenv("COINBASE_KEY_FILE")
    
    if not key_file_path:
        # Default to the project directory - much easier setup!
        script_dir = os.path.dirname(os.path.abspath(__file__))
        key_file_path = os.path.join(script_dir, "cdp_api_key.json")
    
    if not os.path.exists(key_file_path):
        error_msg = (
            f"Coinbase API key file not found at: {key_file_path}\n"
            "Please copy your cdp_api_key.json file to the project directory, or set:\n"
            "COINBASE_KEY_FILE=/path/to/your/cdp_api_key.json"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return key_file_path

async def initialize_coinbase_client() -> RESTClient:
    """Initialize and return Coinbase REST client"""
    global coinbase_client

    if coinbase_client is None:
        try:
            key_file_path = get_coinbase_config()
            coinbase_client = RESTClient(
                key_file=key_file_path,
                rate_limit_headers=True
            )
            logger.info(f"Coinbase REST client initialized successfully using key file: {key_file_path}")
        except Exception as e:
            logger.error(f"Failed to initialize Coinbase client: {e}")
            raise

    return coinbase_client

class TechnicalAnalysis:
    """Technical analysis calculations for cryptocurrency data"""

    @staticmethod
    def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Exponential Moving Average"""
        return talib.EMA(prices, timeperiod=period)

    @staticmethod
    def calculate_sma(prices: np.ndarray, period: int) -> np.ndarray:
        """Calculate Simple Moving Average"""
        return talib.SMA(prices, timeperiod=period)

    @staticmethod
    def calculate_rsi(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """Calculate Relative Strength Index"""
        return talib.RSI(prices, timeperiod=period)

    @staticmethod
    def calculate_volume_sma(volumes: np.ndarray, period: int = 20) -> np.ndarray:
        """Calculate Simple Moving Average of volume"""
        return talib.SMA(volumes, timeperiod=period)

    @staticmethod
    def detect_ema_crossover(prices: np.ndarray) -> dict:
        """Detect 13 EMA crossing above 55 SMA"""
        ema_13 = TechnicalAnalysis.calculate_ema(prices, 13)
        sma_55 = TechnicalAnalysis.calculate_sma(prices, 55)

        if len(ema_13) < 2 or len(sma_55) < 2:
            return {"signal": False, "reason": "Insufficient data"}

        # Check if EMA crossed above SMA in the last period
        current_ema = ema_13[-1]
        current_sma = sma_55[-1]
        prev_ema = ema_13[-2]
        prev_sma = sma_55[-2]

        crossed_up = (prev_ema <= prev_sma) and (current_ema > current_sma)

        return {
            "signal": crossed_up,
            "current_ema_13": float(current_ema),
            "current_sma_55": float(current_sma),
            "crossover_strength": float(current_ema - current_sma)
        }

    @staticmethod
    def detect_rsi_recovery(prices: np.ndarray, period: int = 14) -> dict:
        """Detect RSI recovery from oversold levels (below 10)"""
        rsi = TechnicalAnalysis.calculate_rsi(prices, period)

        if len(rsi) < 5:
            return {"signal": False, "reason": "Insufficient data"}

        current_rsi = rsi[-1]

        # Check if RSI was below 10 in recent periods and is now recovering
        recent_rsi = rsi[-5:]  # Last 5 periods
        was_oversold = np.any(recent_rsi < 10)
        is_recovering = current_rsi > 10 and current_rsi < 30  # Recovery zone

        return {
            "signal": was_oversold and is_recovering,
            "current_rsi": float(current_rsi),
            "min_recent_rsi": float(np.min(recent_rsi)),
            "recovery_strength": float(current_rsi - np.min(recent_rsi))
        }

    @staticmethod
    def analyze_volume_pattern(volumes: np.ndarray, prices: np.ndarray) -> dict:
        """Analyze volume patterns for high/low relative volume"""
        if len(volumes) < 20:
            return {"signal": False, "reason": "Insufficient data"}

        # Calculate volume SMA
        volume_sma = TechnicalAnalysis.calculate_volume_sma(volumes, 20)
        current_volume = volumes[-1]
        avg_volume = volume_sma[-1]

        # Volume ratio analysis
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 0

        # Price change for volume confirmation
        price_change = (prices[-1] - prices[-2]) / prices[-2] * 100 if len(prices) > 1 else 0

        # High volume signal (3x average or more)
        high_volume_signal = volume_ratio >= 3.0

        # Low volume signal (20% of average or less)
        low_volume_signal = volume_ratio <= 0.2

        return {
            "signal": high_volume_signal or low_volume_signal,
            "volume_type": "high" if high_volume_signal else "low" if low_volume_signal else "normal",
            "volume_ratio": float(volume_ratio),
            "current_volume": float(current_volume),
            "average_volume": float(avg_volume),
            "price_change_percent": float(price_change)
        }

async def get_market_products() -> List[dict]:
    """Get all available trading products from Coinbase"""
    try:
        client = await initialize_coinbase_client()
        response = client.get_products()

        products = getattr(response, 'products', None)
        if products is None:
            logger.error("No products found in response from Coinbase API")
            return []

        # Filter for active USD products only (Product objects, not dicts)
        active_products = []
        for p in products:
            if (hasattr(p, 'status') and p.status == 'online' and 
                hasattr(p, 'trading_disabled') and not p.trading_disabled and
                hasattr(p, 'quote_name') and p.quote_name == 'US Dollar'):
                
                # Convert to dict for easier handling
                product_dict = {
                    'product_id': p.product_id,
                    'status': p.status,
                    'trading_disabled': p.trading_disabled,
                    'quote_currency': p.quote_name,
                    'volume_24h': float(p.volume_24h) if hasattr(p, 'volume_24h') and p.volume_24h else 0
                }
                active_products.append(product_dict)

        logger.info(f"Retrieved {len(active_products)} active USD trading products")
        return active_products

    except Exception as e:
        logger.error(f"Failed to get market products: {e}")
        return []

def seconds_to_granularity(seconds: int) -> str:
    """Convert seconds to Coinbase API granularity string"""
    granularity_map = {
        60: "ONE_MINUTE",
        300: "FIVE_MINUTE", 
        900: "FIFTEEN_MINUTE",
        1800: "THIRTY_MINUTE",
        3600: "ONE_HOUR",
        7200: "TWO_HOUR",
        14400: "SIX_HOUR",  # 4h maps to 6h (closest available)
        21600: "SIX_HOUR",
        86400: "ONE_DAY"
    }
    return granularity_map.get(seconds, "ONE_HOUR")  # Default to 1 hour

async def get_candle_data(product_id: str, granularity: int = 3600, limit: int = 300) -> pd.DataFrame:
    """
    Get historical candle data for a product

    Args:
        product_id: Trading pair (e.g., 'BTC-USD')
        granularity: Candle size in seconds (3600 = 1 hour)
        limit: Number of candles to retrieve (max 300)
    """
    try:
        client = await initialize_coinbase_client()

        # Calculate start and end times based on granularity and number of candles
        end_time = datetime.now()
        # Convert granularity (seconds) to timedelta and multiply by limit (number of candles)
        time_delta = timedelta(seconds=granularity * limit)
        start_time = end_time - time_delta

        response = client.get_candles(
            product_id=product_id,
            start=start_time.isoformat(),
            end=end_time.isoformat(),
            granularity=seconds_to_granularity(granularity)
        )

        candles = getattr(response, 'candles', None)
        if candles is None:
            candles = []

        if not candles:
            logger.warning(f"No candle data returned for {product_id}")
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(candles)
        df.columns = ['timestamp', 'low', 'high', 'open', 'close', 'volume']

        # Convert timestamp and ensure numeric types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        for col in ['low', 'high', 'open', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col])

        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)

        logger.info(f"Retrieved {len(df)} candles for {product_id}")
        return df

    except Exception as e:
        logger.error(f"Failed to get candle data for {product_id}: {e}")
        return pd.DataFrame()

async def screen_single_coin(product_id: str) -> dict:
    """
    Screen a single coin for trading signals

    Args:
        product_id: Trading pair to analyze

    Returns:
        Dictionary with screening results
    """
    try:
        # Get candle data
        df = await get_candle_data(product_id, granularity=3600, limit=100)

        if df.empty or len(df) < 55:
            return {
                "product_id": product_id,
                "signal": False,
                "reason": "Insufficient data",
                "data_points": len(df)
            }

        # Extract price and volume arrays
        prices = df['close'].to_numpy()
        volumes = df['volume'].to_numpy()

        # Perform technical analysis
        ema_analysis = TechnicalAnalysis.detect_ema_crossover(prices)
        rsi_analysis = TechnicalAnalysis.detect_rsi_recovery(prices)
        volume_analysis = TechnicalAnalysis.analyze_volume_pattern(volumes, prices)

        # Determine if all conditions are met
        ema_signal = ema_analysis.get("signal", False)
        rsi_signal = rsi_analysis.get("signal", False)
        volume_signal = volume_analysis.get("signal", False)

        # Calculate signal strength
        signal_count = sum([ema_signal, rsi_signal, volume_signal])
        signal_strength = signal_count / 3.0

        # Overall signal (require at least 2 out of 3 conditions)
        overall_signal = signal_count >= 2

        current_price = float(prices[-1])
        price_24h_change = ((prices[-1] - prices[-25]) / prices[-25] * 100) if len(prices) >= 25 else 0

        result = {
            "product_id": product_id,
            "signal": overall_signal,
            "signal_strength": signal_strength,
            "signal_count": signal_count,
            "current_price": current_price,
            "price_24h_change": float(price_24h_change),
            "timestamp": datetime.now().isoformat(),
            "ema_crossover": ema_analysis,
            "rsi_recovery": rsi_analysis,
            "volume_pattern": volume_analysis,
            "data_points": len(df)
        }

        if overall_signal:
            logger.info(f"🎯 SIGNAL DETECTED for {product_id}: {signal_count}/3 conditions met")

        return result

    except Exception as e:
        logger.error(f"Error screening {product_id}: {e}")
        return {
            "product_id": product_id,
            "signal": False,
            "reason": f"Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }

# MCP Tools

@mcp.tool()
async def screen_all_coins(ctx: Context, limit: int = 50, min_volume_usd: float = 100000) -> str:
    """
    Run comprehensive technical screening across all available coins

    Args:
        limit: Maximum number of coins to screen (default: 50)
        min_volume_usd: Minimum 24h volume in USD to consider (default: 100000)

    Returns:
        JSON string with screening results
    """
    try:
        logger.info(f"Starting comprehensive screening of up to {limit} coins...")

        # Get all products
        products = await get_market_products()

        if not products:
            return json.dumps({"error": "No products available"})

        # Filter products by volume (already USD pairs from get_market_products)
        filtered_products = []
        for product in products:
            if product.get('volume_24h', 0) > min_volume_usd:
                filtered_products.append(product['product_id'])

        # Limit the number of products to screen
        products_to_screen = filtered_products[:limit]

        await ctx.info(f"Screening {len(products_to_screen)} coins with volume > ${min_volume_usd:,.0f}")

        # Screen each coin
        results = []
        signals_found = []

        for i, product_id in enumerate(products_to_screen):
            await ctx.info(f"Screening {product_id} ({i+1}/{len(products_to_screen)})")

            result = await screen_single_coin(product_id)
            results.append(result)

            if result.get("signal", False):
                signals_found.append(result)

            # Update progress
            progress = (i + 1) / len(products_to_screen)
            await ctx.report_progress(progress)

        # Store results in global cache
        global screening_results
        screening_results = {
            "timestamp": datetime.now().isoformat(),
            "total_screened": len(results),
            "signals_found": len(signals_found),
            "results": results,
            "top_signals": sorted(signals_found, key=lambda x: x.get("signal_strength", 0), reverse=True)
        }

        summary = {
            "screening_summary": {
                "total_coins_screened": len(results),
                "signals_detected": len(signals_found),
                "success_rate": f"{len(signals_found)/len(results)*100:.1f}%" if results else "0%",
                "timestamp": datetime.now().isoformat()
            },
            "top_signals": signals_found[:10] if signals_found else [],
            "screening_criteria": {
                "ema_crossover": "13 EMA crosses above 55 SMA",
                "rsi_recovery": "RSI recovers from below 10",
                "volume_pattern": "High or low relative volume"
            }
        }

        logger.info(f"Screening completed: {len(signals_found)} signals found out of {len(results)} coins")

        return json.dumps(summary, indent=2)

    except Exception as e:
        error_msg = f"Screening failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def analyze_coin(product_id: str, timeframe: str = "1h") -> str:
    """
    Perform deep technical analysis on a specific cryptocurrency

    Args:
        product_id: Trading pair to analyze (e.g., 'BTC-USD')
        timeframe: Analysis timeframe ('1h', '4h', '1d')

    Returns:
        JSON string with detailed analysis
    """
    try:
        logger.info(f"Performing deep analysis on {product_id}")

        # Map timeframe to granularity in seconds (will be converted to API strings)
        # Note: 4h is not supported by Coinbase API, using 6h instead
        granularity_map = {"1h": 3600, "4h": 21600, "1d": 86400}  # 4h -> 6h (21600s)
        granularity = granularity_map.get(timeframe, 3600)

        # Get comprehensive candle data
        df = await get_candle_data(product_id, granularity=granularity, limit=200)

        if df.empty:
            return json.dumps({"error": f"No data available for {product_id}"})

        # Extract data arrays
        prices = df['close'].values
        highs = np.asarray(df['high'].values)
        lows = np.asarray(df['low'].values)
        volumes = df['volume'].values

        # Comprehensive technical analysis
        ema_13 = TechnicalAnalysis.calculate_ema(np.asarray(prices), 13)
        ema_55 = TechnicalAnalysis.calculate_ema(np.asarray(prices), 55)
        rsi = TechnicalAnalysis.calculate_rsi(np.asarray(prices), 14)
        volume_sma = TechnicalAnalysis.calculate_volume_sma(np.asarray(volumes), 20)

        # Current values
        current_price = float(prices[-1])
        current_ema_13 = float(ema_13[-1]) if len(ema_13) > 0 else None
        current_ema_55 = float(ema_55[-1]) if len(ema_55) > 0 else None
        current_rsi = float(rsi[-1]) if len(rsi) > 0 else None
        current_volume = float(volumes[-1])
        avg_volume = float(volume_sma[-1]) if len(volume_sma) > 0 else None

        # Price statistics
        price_24h = prices[-25] if len(prices) >= 25 else prices[0]
        price_change_24h = ((current_price - price_24h) / price_24h * 100)

        high_24h = float(np.max(highs[-25:])) if len(highs) >= 25 else float(np.max(highs))
        low_24h = float(np.min(lows[-25:])) if len(lows) >= 25 else float(np.min(lows))

        # Signal analysis
        ema_crossover = TechnicalAnalysis.detect_ema_crossover(np.asarray(prices))
        rsi_recovery = TechnicalAnalysis.detect_rsi_recovery(np.asarray(prices))
        volume_pattern = TechnicalAnalysis.analyze_volume_pattern(np.asarray(volumes), np.asarray(prices))

        analysis = {
            "product_id": product_id,
            "timeframe": timeframe,
            "timestamp": datetime.now().isoformat(),
            "price_data": {
                "current_price": current_price,
                "price_change_24h": float(price_change_24h),
                "high_24h": high_24h,
                "low_24h": low_24h,
                "current_volume": current_volume,
                "avg_volume_20": avg_volume,
                "volume_ratio": float(current_volume / avg_volume) if avg_volume else None
            },
            "technical_indicators": {
                "ema_13": current_ema_13,
                "ema_55": current_ema_55,
                "rsi_14": current_rsi,
                "ema_trend": "bullish" if current_ema_13 and current_ema_55 and current_ema_13 > current_ema_55 else "bearish"
            },
            "signals": {
                "ema_crossover": ema_crossover,
                "rsi_recovery": rsi_recovery,
                "volume_pattern": volume_pattern
            },
            "data_quality": {
                "candles_analyzed": len(df),
                "data_completeness": "good" if len(df) >= 100 else "limited"
            }
        }

        return json.dumps(analysis, indent=2)

    except Exception as e:
        error_msg = f"Analysis failed for {product_id}: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def get_market_data(product_ids: str = "BTC-USD,ETH-USD,SOL-USD") -> str:
    """
    Get current market data for specified cryptocurrencies

    Args:
        product_ids: Comma-separated list of trading pairs

    Returns:
        JSON string with current market data
    """
    try:
        client = await initialize_coinbase_client()
        product_list = [p.strip() for p in product_ids.split(",")]

        market_data = []

        for product_id in product_list:
            try:
                # Get current ticker
                ticker = client.get_product(product_id)

                # Get 24h stats using get_product_ticker (if available) or fallback to ticker attributes
                stats = ticker  # Use ticker as stats if no separate stats endpoint

                data = {
                    "product_id": product_id,
                    "price": float(ticker.price) if hasattr(ticker, 'price') else None,
                    "volume_24h": float(getattr(stats, 'volume_24h', 0)) if hasattr(stats, 'volume_24h') else None,
                    "price_change_24h": float(getattr(stats, 'price_change_24h', 0)) if hasattr(stats, 'price_change_24h') else None,
                    "price_change_percent_24h": float(getattr(stats, 'price_change_percent_24h', 0)) if hasattr(stats, 'price_change_percent_24h') else None,
                    "low_24h": float(getattr(stats, 'low_24h', 0)) if hasattr(stats, 'low_24h') else None,
                    "high_24h": float(getattr(stats, 'high_24h', 0)) if hasattr(stats, 'high_24h') else None,
                    "timestamp": datetime.now().isoformat()
                }

                market_data.append(data)

            except Exception as e:
                logger.warning(f"Failed to get data for {product_id}: {e}")
                market_data.append({
                    "product_id": product_id,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                })

        result = {
            "market_data": market_data,
            "timestamp": datetime.now().isoformat(),
            "data_source": "Coinbase Advanced Trade API"
        }

        return json.dumps(result, indent=2)

    except Exception as e:
        error_msg = f"Failed to get market data: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def check_signals() -> str:
    """
    Review recent trading signals from the last screening run

    Returns:
        JSON string with recent signals and their status
    """
    try:
        global screening_results

        if not screening_results:
            return json.dumps({
                "message": "No recent screening results available. Run 'screen_all_coins' first.",
                "timestamp": datetime.now().isoformat()
            })

        # Get the recent results
        results = screening_results.get("results", [])
        signals = [r for r in results if r.get("signal", False)]

        # Sort by signal strength
        signals.sort(key=lambda x: x.get("signal_strength", 0), reverse=True)

        # Prepare summary
        summary = {
            "last_screening": {
                "timestamp": screening_results.get("timestamp"),
                "total_screened": screening_results.get("total_screened", 0),
                "signals_found": len(signals)
            },
            "active_signals": signals[:15],  # Top 15 signals
            "signal_distribution": {
                "strong_signals": len([s for s in signals if s.get("signal_strength", 0) >= 0.8]),
                "moderate_signals": len([s for s in signals if 0.6 <= s.get("signal_strength", 0) < 0.8]),
                "weak_signals": len([s for s in signals if s.get("signal_strength", 0) < 0.6])
            },
            "timestamp": datetime.now().isoformat()
        }

        return json.dumps(summary, indent=2)

    except Exception as e:
        error_msg = f"Failed to check signals: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

@mcp.tool()
async def monitor_portfolio(ctx: Context, watchlist: str = "BTC-USD,ETH-USD,SOL-USD,ADA-USD,DOT-USD") -> str:
    """
    Monitor a custom watchlist of cryptocurrencies for trading signals

    Args:
        watchlist: Comma-separated list of trading pairs to monitor

    Returns:
        JSON string with monitoring results
    """
    try:
        product_list = [p.strip() for p in watchlist.split(",")]

        await ctx.info(f"Monitoring {len(product_list)} coins in watchlist")

        monitoring_results = []
        signals_detected = []

        for i, product_id in enumerate(product_list):
            await ctx.info(f"Analyzing {product_id} ({i+1}/{len(product_list)})")

            # Screen the coin
            result = await screen_single_coin(product_id)
            monitoring_results.append(result)

            if result.get("signal", False):
                signals_detected.append(result)

            # Update progress
            progress = (i + 1) / len(product_list)
            await ctx.report_progress(progress)

        # Prepare summary
        summary = {
            "watchlist_monitoring": {
                "watchlist": product_list,
                "total_monitored": len(monitoring_results),
                "signals_detected": len(signals_detected),
                "timestamp": datetime.now().isoformat()
            },
            "portfolio_signals": signals_detected,
            "detailed_results": monitoring_results,
            "recommendations": []
        }

        # Add recommendations based on signals
        if signals_detected:
            summary["recommendations"].append("🎯 Strong signals detected! Consider reviewing the flagged coins.")

            # Find strongest signal
            strongest = max(signals_detected, key=lambda x: x.get("signal_strength", 0))
            summary["recommendations"].append(f"💡 Strongest signal: {strongest['product_id']} with {strongest.get('signal_strength', 0):.1%} confidence")
        else:
            summary["recommendations"].append("📊 No strong signals in current watchlist. Market may be consolidating.")

        return json.dumps(summary, indent=2)

    except Exception as e:
        error_msg = f"Portfolio monitoring failed: {str(e)}"
        logger.error(error_msg)
        return json.dumps({"error": error_msg})

# MCP Prompts

@mcp.prompt()
def trading_analysis_prompt(coin_symbol: str) -> str:
    """
    Generate a trading analysis prompt for a specific cryptocurrency

    Args:
        coin_symbol: Cryptocurrency symbol (e.g., BTC, ETH)
    """
    return f"""
Please analyze {coin_symbol} using the following framework:

1. **Technical Analysis**:
   - Current price action and trend direction
   - Key support and resistance levels
   - Moving average analysis (13 EMA vs 55 SMA)
   - RSI momentum and potential reversal signals
   - Volume patterns and their significance

2. **Market Context**:
   - Overall cryptocurrency market sentiment
   - {coin_symbol}'s performance relative to Bitcoin and major altcoins
   - Recent news or developments affecting {coin_symbol}

3. **Risk Assessment**:
   - Key risk factors and potential downside scenarios
   - Appropriate position sizing recommendations
   - Stop-loss and take-profit levels

4. **Trading Strategy**:
   - Entry points and timing considerations
   - Short-term vs long-term outlook
   - Portfolio allocation suggestions

Focus on providing actionable insights based on current market data and technical indicators.
"""

@mcp.prompt()
def market_screening_prompt() -> str:
    """Generate a comprehensive market screening analysis prompt"""
    return """
Based on the latest cryptocurrency screening results, please provide:

1. **Market Overview**:
   - Overall market sentiment and trend direction
   - Sector rotation patterns (DeFi, Layer 1, Meme coins, etc.)
   - Volume and liquidity analysis across major pairs

2. **Signal Analysis**:
   - Quality assessment of detected technical signals
   - False signal probability and market noise considerations
   - Confluence factors that strengthen signal reliability

3. **Risk Management**:
   - Current market volatility levels
   - Correlation risks across cryptocurrency sectors
   - Recommended position sizing in current environment

4. **Trading Opportunities**:
   - Highest probability setups from screening results
   - Timing considerations for entry and exit
   - Portfolio diversification recommendations

Please synthesize this analysis into actionable trading insights with specific risk management guidelines.
"""

# Register cleanup handlers for process management
signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)
atexit.register(cleanup_processes)

def main():
    """Main entry point for the Coinbase Trade Assistant MCP server"""
    try:
        logger.info("🚀 Starting Coinbase Trade Assistant MCP Server")

        # Validate environment configuration
        try:
            key_file_path = get_coinbase_config()
            logger.info(f"✅ Coinbase API configuration validated - using key file: {key_file_path}")
        except ValueError as e:
            logger.error(f"❌ Configuration error: {e}")
            return 1

        logger.info("🔧 Server initialized successfully")
        logger.info("📊 Available tools:")
        logger.info("   - screen_all_coins: Run comprehensive technical screening")
        logger.info("   - analyze_coin: Deep analysis of specific cryptocurrency")
        logger.info("   - get_market_data: Current market data retrieval")
        logger.info("   - check_signals: Review recent trading signals")
        logger.info("   - monitor_portfolio: Monitor custom watchlist")

        # Run the MCP server
        mcp.run()

    except KeyboardInterrupt:
        logger.info("🛑 Server interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"❌ Server error: {e}")
        return 1
    finally:
        cleanup_processes()

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
