#!/bin/bash
# Coinbase Trade Assistant MCP Setup Script

set -e

echo "🚀 Setting up Coinbase Trade Assistant MCP Server..."

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Error: Please run this script from the coinbase-trade-assistant-mcp directory"
    exit 1
fi

echo "📦 Creating virtual environment with uv..."
uv venv --python 3.12 --seed

echo "🔧 Activating virtual environment..."
source .venv/bin/activate

echo "📥 Installing dependencies..."
uv sync

echo "🧪 Testing imports..."
python -c "
import sys
try:
    import pandas
    import numpy
    import talib
    from coinbase.rest import RESTClient
    from mcp.server.fastmcp import FastMCP
    print('✅ All dependencies imported successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

echo "📝 Creating .env template..."
cat > .env.example << EOF
# Coinbase CDP API Configuration
# Get your keys from: https://docs.cdp.coinbase.com/advanced-trade/docs/auth

COINBASE_API_KEY=organizations/{org_id}/apiKeys/{key_id}
COINBASE_API_SECRET=-----BEGIN EC PRIVATE KEY-----
YOUR_PRIVATE_KEY_HERE
-----END EC PRIVATE KEY-----

# Optional: Logging configuration
LOG_LEVEL=INFO
EOF

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "📋 Next steps:"
echo "1. Get your Coinbase CDP API keys from: https://docs.cdp.coinbase.com/advanced-trade/docs/auth"
echo "2. Copy example_mcp_config.json and update with your API keys"
echo "3. Add the configuration to your Claude Desktop config"
echo "4. Restart Claude Desktop"
echo ""
echo "🔧 Configuration file location:"
echo "   ~/.config/Claude/claude_desktop_config.json (Linux)"
echo "   ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)"
echo ""
echo "📊 Available MCP tools after setup:"
echo "   - screen_all_coins: Comprehensive market screening"
echo "   - analyze_coin: Deep technical analysis"
echo "   - get_market_data: Real-time market data"
echo "   - check_signals: Review recent signals"
echo "   - monitor_portfolio: Custom watchlist monitoring"
echo ""
echo "🎯 Happy trading!"
