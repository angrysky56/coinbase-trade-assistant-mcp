#!/usr/bin/env python3
"""
Test script for Coinbase Trade Assistant MCP Server
Run this to validate the implementation without requiring API keys
"""

import sys
import os
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Configure test logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test that all required packages can be imported"""
    print("🧪 Testing imports...")

    try:
        import pandas as pd
        print("✅ pandas imported successfully")

        import numpy as np
        print("✅ numpy imported successfully")

        import talib
        print("✅ ta-lib imported successfully")

        from coinbase.rest import RESTClient
        print("✅ coinbase-advanced-py imported successfully")

        from mcp.server.fastmcp import FastMCP
        print("✅ mcp imported successfully")

        return True

    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_technical_analysis():
    """Test technical analysis functions with sample data"""
    print("\n📊 Testing technical analysis functions...")

    try:
        import numpy as np
        from main import TechnicalAnalysis

        # Generate sample price data
        prices = np.array([100, 102, 101, 103, 105, 104, 106, 108, 107, 109,
                          111, 110, 112, 114, 113, 115, 117, 116, 118, 120])

        volumes = np.array([1000, 1200, 800, 1500, 2000, 900, 1800, 2200,
                           1100, 2500, 3000, 1300, 2800, 3200, 1400, 3500,
                           4000, 1600, 3800, 4200])

        # Test EMA calculation
        ema_result = TechnicalAnalysis.calculate_ema(prices, 13)
        print(f"✅ EMA calculation: {len(ema_result)} values")

        # Test RSI calculation
        rsi_result = TechnicalAnalysis.calculate_rsi(prices, 14)
        print(f"✅ RSI calculation: {len(rsi_result)} values")

        # Test crossover detection
        crossover = TechnicalAnalysis.detect_ema_crossover(prices)
        print(f"✅ Crossover detection: {crossover}")

        # Test RSI recovery
        rsi_recovery = TechnicalAnalysis.detect_rsi_recovery(prices)
        print(f"✅ RSI recovery: {rsi_recovery}")

        # Test volume analysis
        volume_analysis = TechnicalAnalysis.analyze_volume_pattern(volumes, prices)
        print(f"✅ Volume analysis: {volume_analysis}")

        return True

    except Exception as e:
        print(f"❌ Technical analysis test failed: {e}")
        return False

def test_server_initialization():
    """Test MCP server initialization"""
    print("\n🖥️ Testing MCP server initialization...")

    try:
        from main import mcp

        # Check if server is properly initialized
        if hasattr(mcp, 'prompts'):
            prompts = getattr(mcp, 'prompts')
            if prompts:
                print(f"✅ MCP server initialized with {len(prompts)} prompts")
                # List available prompts
                print("📋 Available prompts:")
                for prompt_name in prompts.keys():
                    print(f"   - {prompt_name}")
                return True
            else:
                print("❌ MCP server initialized but no prompts found")
                return False
        else:
            print("❌ MCP server not properly initialized or missing public prompts attribute")
            return False

    except Exception as e:
        print(f"❌ Server initialization test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Coinbase Trade Assistant MCP Server - Test Suite")
    print("=" * 60)

    all_passed = True

    # Test imports
    if not test_imports():
        all_passed = False

    # Test technical analysis
    if not test_technical_analysis():
        all_passed = False

    # Test server initialization
    if not test_server_initialization():
        all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ All tests passed! The server is ready to use.")
        print("\n🔧 Next steps:")
        print("1. Get Coinbase CDP API keys")
        print("2. Update example_mcp_config.json with your keys")
        print("3. Add to Claude Desktop configuration")
        print("4. Restart Claude Desktop")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        print("\n💡 Try running: ./setup.sh")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
