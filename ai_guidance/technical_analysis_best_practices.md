# Technical Analysis Best Practices for Cryptocurrency Trading

## Overview
This document outlines best practices for using technical analysis in cryptocurrency trading, specifically designed for the Coinbase Trade Assistant MCP server.

## Core Technical Indicators

### 1. Moving Average Crossovers
- **13 EMA vs 55 SMA**: Classic momentum signal
  - **Bullish Signal**: 13 EMA crosses above 55 SMA
  - **Bearish Signal**: 13 EMA crosses below 55 SMA
  - **Confirmation**: Look for volume increase during crossover
  - **False Signals**: Common in sideways markets

### 2. RSI (Relative Strength Index)
- **Oversold Recovery Strategy**:
  - RSI below 10: Extreme oversold condition
  - Recovery above 10: Potential bounce signal
  - Best when combined with support levels
  - **Warning**: Can stay oversold longer in bear markets

### 3. Volume Analysis
- **High Volume Signals**:
  - 3x+ average volume: Strong conviction move
  - Confirms price breakouts
  - Often precedes major moves
- **Low Volume Signals**:
  - <20% average volume: Lack of interest
  - Often indicates consolidation
  - Can signal accumulation phases

## Signal Combination Rules

### Strong Signal (High Probability)
- All 3 indicators align (3/3)
- Price confirms with strong volume
- Multiple timeframes agree
- **Action**: Consider larger position size

### Moderate Signal (Medium Probability)
- 2 out of 3 indicators align (2/3)
- Mixed volume confirmation
- **Action**: Standard position size with tight stops

### Weak Signal (Low Probability)
- Only 1 indicator triggers (1/3)
- No volume confirmation
- **Action**: Pass or very small position

## Risk Management Rules

### Position Sizing
- **Maximum 2-5% risk per trade**
- Scale position size based on signal strength
- Never risk more than you can afford to lose
- Diversify across multiple cryptocurrencies

### Stop Loss Guidelines
- **Trend Following**: 2-3 ATR below entry
- **Mean Reversion**: Below key support level
- **Time Stop**: Exit if no movement within expected timeframe
- **Mental Stop**: Never move stops against your position

### Take Profit Strategy
- **Partial Profits**: Take 25-50% at first target
- **Trailing Stops**: Use for trending markets
- **Support/Resistance**: Target key levels
- **Risk/Reward**: Minimum 1:2 ratio preferred

## Market Context Considerations

### Cryptocurrency Market Specifics
- **High Volatility**: Expect 10-20% daily moves
- **24/7 Trading**: No gaps like traditional markets
- **Correlation Risk**: Most altcoins follow Bitcoin
- **News Impact**: Extremely sensitive to regulatory news

### Best Trading Times
- **Asian Session**: 9 PM - 6 AM EST (Higher volatility)
- **European Session**: 2 AM - 11 AM EST (Moderate activity)
- **US Session**: 8 AM - 5 PM EST (Institutional activity)
- **Weekend**: Lower volume, higher spreads

### Market Phases
- **Bull Market**: Focus on breakout strategies
- **Bear Market**: Mean reversion and short-selling
- **Sideways Market**: Range trading and fade strategies
- **High Volatility**: Reduce position sizes

## Common Pitfalls to Avoid

### Technical Analysis Mistakes
- **Over-optimization**: Don't curve-fit indicators
- **Indicator Overload**: Stick to 3-5 key indicators
- **Ignoring Price Action**: Indicators lag price movement
- **No Context**: Consider overall market conditions

### Emotional Trading Errors
- **FOMO (Fear of Missing Out)**: Chasing prices
- **Revenge Trading**: Trying to recover losses quickly
- **Overtrading**: Taking too many marginal setups
- **Position Size Creep**: Gradually increasing risk

## Continuous Improvement

### Performance Tracking
- Keep a detailed trading journal
- Track win rate and average win/loss
- Review both winning and losing trades
- Identify recurring patterns in mistakes

### Market Evolution
- Crypto markets evolve rapidly
- Update strategies based on changing conditions
- Stay informed about new trading instruments
- Adapt position sizing to current volatility

## Integration with MCP Tools

### Using the Auto-Screener
- Run `screen_all_coins` for market-wide opportunities
- Focus on coins with multiple confirming signals
- Use `analyze_coin` for detailed technical analysis
- Monitor signals with `check_signals` regularly

### Building Watchlists
- Use `monitor_portfolio` for focused analysis
- Include diverse market cap ranges
- Balance risk across different sectors
- Regular review and updates needed

## Disclaimer
Technical analysis is not a guarantee of future performance. Cryptocurrency trading involves substantial risk and may not be suitable for all investors. Always conduct your own research and consider consulting with a financial advisor.
