# Advanced Copula-Based Pairs Trading Roadmap

## 1. Asset Selection
- List of Coins to choose from
	- BTC, ETH, BNB, SOL, ADA, DOT, LINK, ARB, AVAX, ATOM, LTC, XRP, UNI, AAVE, DOGE
- Cointegration-based Filtering
    - Run Engle-Granger Test for each pair. Select those with ADF p < 0.05

## 2. Marginal Distribution Fitting
- For each asset, fit multiple candidate distributions
	- t, normal, generalized extreme value, logistic, generalized Pareto, Laplace
- Select the best fit for each asset using 
	- KS test, Anderson-Darling test, AIC, BIC

## 3. Copula Modelling
- Transform each asset's returns to uniforms using the fitted marginal CDFs.
- For each pair, fit multiple copula families
	- Gaussian, Clayton, Gumbel, Frank
- Select the optimal copula for each pair using AIC, BIC

## 4. Signal Generation and Backtesting
- Generate trading signals using the copula-based mispricing index (conditional probability)
	- Long X, Short Y if MI_Y_given_X < 0.05, exit when MI_Y_given_X > 0.2
	- Short X, Long Y if MI_Y_given_X > 0.95, exit when MI_Y_given_X < 0.8
- For each pair, use hedge ratio computed with OLS
- Backtest the strategy, including realistic transaction costs
	- Commission = 4.5bps
	- There is no slippage because the market is highly liquid

## 5. Risk Control Integration
- Integrate risk controls
	- stop-loss, volatility targeting, position sizing, and exposure limits.
- Analyze risk-adjusted performance metrics
	- Sharpe, drawdown

## 6. Regime Detection
- Implement regime detection (Hidden Markov Models) to adapt strategy parameters or serve as features for TFT and optimization.
- Use regime information for further risk management and signal refinement.

## 7. LightGBM
- Train LightGBMClassifier on the engineered features
	- Input: (from both assets)
		- From individual asset: 
			- historical prices, volume, TA (RSI, MACD, MA, volatility)
		- Collective from both assets:
			- correlation, cointegration, historical mispricing index
	- Label: future PnL
	- Output:
		- Buy/Sell/Hold Signal

## **Best Practices**
- After each step, validate performance and risk, then iterate as needed.
- Document parameter choices, test results, and insights for future reference.
- If resources allow, experiment with advanced features (TFT, regime detection) in parallel to accelerate learning.

---

**Note:**  
The sequence reflects best practices: establish robust statistical foundations first, then incrementally add advanced modelling and optimization for incremental alpha and robustness.

I don't have advanced infrastructure. We are going to pairs instead of a basket.