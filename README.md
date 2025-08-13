# Coinbase Trade Assistant MCP

A sophisticated Model Context Protocol (MCP) server that provides intelligent cryptocurrency trading assistance using Coinbase Advanced Trade API. Features automated technical analysis screening and real-time market insights.

## Features

- **Intelligent Auto-Screener**: Efficiently scans all available coins for high-probability technical setups
  - 13 EMA crosses above 55 SMA (trend confirmation)
  - RSI recovery from oversold levels (< 10)
  - Volume pattern analysis (unusual activity detection)
  - Typically returns 0-30 qualified coins depending on market conditions

- **Smart Filtering**: Focus on actionable opportunities, not information overload
- **Real-time Market Data**: Live price and volume data from Coinbase Advanced Trade API
- **Due Diligence Ready**: Curated list for manual review and selection
- **Risk-Aware**: Built-in volume and volatility filters

## Quick Start

1. **Configure API Keys**:
   - Get your free Coinbase CDP API keys from [Coinbase Developer Platform](https://docs.cdp.coinbase.com/advanced-trade/docs/auth)
   - More info [docs.cdp.coinbase.com/get-started/authentication/cdp-api-keys#creating-secret-api-keys](https://docs.cdp.coinbase.com/get-started/authentication/cdp-api-keys#creating-secret-api-keys)
   - Go to the advanced settings when getting your key and make sure to select "ECDSA"
   - Copy your `cdp_api_key.json` file to the project directory
   - That's it! No environment variables needed.

2. **Add to Claude Desktop**:
   ```json
   {
     "mcpServers": {
       "coinbase-trade-assistant": {
         "command": "uv",
         "args": [
           "--directory",
           "/path/to/coinbase-trade-assistant-mcp",
           "run",
           "main.py"
         ]
       }
     }
   }
   ```

3. **Install Dependencies- if there is an issue or you are not using the client config which should do this automatically**:
   ```bash
   cd coinbase-trade-assistant-mcp
   uv venv --python 3.12 --seed
   source .venv/bin/activate
   uv sync
   ```

## Screening Criteria

The screener uses a **confluence-based approach** requiring multiple confirming signals:

- **Trend**: 13 EMA crossing above 55 SMA (momentum shift)
- **Momentum**: RSI recovery from oversold (< 10) back above 10-30 range
- **Volume**: Unusual volume activity (3x+ average or very low < 20%)
- **Signal Strength**: Requires 2 out of 3 conditions for qualification

**Expected Results:**
- **Bull Markets**: 15-30 qualified coins
- **Bear/Sideways Markets**: 0-10 qualified coins
- **Transition Periods**: 5-15 qualified coins

All results ranked by signal strength for efficient due diligence.

## Tools Available

**Primary Workflow:**
- `screen_all_coins`: **Main tool** - Run comprehensive technical screening across all markets
- `check_signals`: Review and analyze recent screening results
- `monitor_portfolio`: Track specific watchlist for signals

**Supporting Tools:**
- `get_market_data`: Current market snapshots for specific coins
- `analyze_coin`: Basic technical data for individual coins (when needed)

## Requirements

- Python 3.12+
- Coinbase CDP API keys
- Claude Desktop or compatible MCP client

## Security

- API keys are read from local `cdp_api_key.json` file
- **Never commit your API key file to version control**
- The key file is automatically ignored by git (added to .gitignore)
- Rate limiting and error handling built-in
- If sharing your project, remove the key file first
