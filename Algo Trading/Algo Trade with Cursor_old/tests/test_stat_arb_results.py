"""
Test cases for verifying the reliability of statistical arbitrage strategy results.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def pytest_addoption(parser):
    """Add command line options for pytest."""
    parser.addoption(
        "--run_id",
        action="store",
        default=None,
        help="Specific run ID to test"
    )

@pytest.fixture
def run_id(request):
    """Get run_id from command line option."""
    return request.config.getoption("--run_id")

class TestStatArbResults:
    @pytest.fixture
    def sample_results(self, run_id):
        """
        Load sample backtest results for testing.
        
        Args:
            run_id (str, optional): Specific run ID to test. If None, uses most recent run.
        """
        results_dir = Path("results")
        if not results_dir.exists():
            raise FileNotFoundError("Results directory not found")
            
        # Get the run directory
        if run_id:
            run_dir = results_dir / run_id
            if not run_dir.exists():
                raise FileNotFoundError(f"Run directory not found: {run_id}")
        else:
            # Get the most recent results directory
            run_dir = max(results_dir.iterdir(), key=lambda x: x.stat().st_mtime)
            
        logger.info(f"Testing results from run: {run_dir.name}")
        
        # Load results with proper error handling
        try:
            with open(run_dir / "config.json") as f:
                config = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found in {run_dir}")
            
        try:
            trades_df = pd.read_csv(run_dir / "trades" / "trades_enhanced.csv")
        except FileNotFoundError:
            raise FileNotFoundError(f"Trades file not found in {run_dir}/trades")
            
        try:
            metrics_df = pd.read_csv(run_dir / "metrics" / "metrics.csv")
        except FileNotFoundError:
            raise FileNotFoundError(f"Metrics file not found in {run_dir}/metrics")
        
        return {
            'config': config,
            'trades': trades_df,
            'metrics': metrics_df,
            'run_id': run_dir.name
        }
        
    def test_results_structure(self, sample_results):
        """Test if results have the correct structure and required fields."""
        # Test config structure
        required_config_fields = [
            'initial_capital', 'max_position_size', 'stop_loss',
            'take_profit', 'commission', 'strategy_version'
        ]
        for field in required_config_fields:
            assert field in sample_results['config'], f"Missing config field: {field}"

        # Test trades structure
        required_trade_fields = [
            'timestamp', 'symbol', 'action', 'size', 'entry_price', 'exit_price',
            'pnl', 'cumulative_pnl', 'cumulative_return', 'drawdown'
        ]
        for field in required_trade_fields:
            assert field in sample_results['trades'].columns, f"Missing trade field: {field}"
            
    def test_trade_consistency(self, sample_results):
        """Test if trades are consistent and follow strategy rules."""
        trades = sample_results['trades']
        config = sample_results['config']
        
        # Check position sizes (normalized)
        max_position = trades['size'].abs().max()
        max_allowed = config['max_position_size']
        assert max_position <= max_allowed, f"Position size {max_position} exceeds limit {max_allowed}"
        
        # Check trade types
        valid_actions = ['buy', 'sell', 'close']
        assert trades['action'].isin(valid_actions).all(), "Invalid action types found"
        
        # Check PnL consistency with commission and spread
        for _, trade in trades.iterrows():
            if trade['action'] == 'close':
                expected_pnl = (trade['exit_price'] - trade['entry_price']) * trade['size']
                expected_pnl -= trade['commission'] + trade['spread']
                assert abs(trade['pnl'] - expected_pnl) < 0.01, f"PnL mismatch for trade at {trade['timestamp']}"
                
    def test_risk_metrics(self, sample_results):
        """Test if risk metrics are within reasonable bounds."""
        metrics = sample_results['metrics'].iloc[0]
        
        # Check Sharpe ratio
        assert metrics['sharpe_ratio'] > -5, "Unrealistic Sharpe ratio"
        assert metrics['sharpe_ratio'] < 10, "Unrealistic Sharpe ratio"
        
        # Check max drawdown (0% to 100%)
        assert 0 <= metrics['max_drawdown'] <= 1, "Invalid max drawdown"
        
        # Check win rate
        assert 0 <= metrics['win_rate'] <= 1, "Invalid win rate"
        
        # Check profit factor
        assert metrics['profit_factor'] >= 0, "Invalid profit factor"
        
    def test_data_quality(self, sample_results):
        """Test if the data quality is sufficient for reliable results."""
        trades = sample_results['trades']
        
        # Check for missing values in critical columns
        critical_columns = [
            'timestamp', 'symbol', 'action', 'size', 'pnl',
            'entry_price', 'exit_price', 'spread', 'zscore'
        ]
        for col in critical_columns:
            assert not trades[col].isnull().any(), f"Missing values detected in {col}"
            
        # Check timestamp format
        try:
            pd.to_datetime(trades['timestamp'])
        except:
            assert False, "Invalid timestamp format"
            
        # Check numeric columns
        numeric_columns = ['size', 'entry_price', 'exit_price', 'pnl', 'spread', 'zscore']
        for col in numeric_columns:
            assert pd.to_numeric(trades[col], errors='coerce').notnull().all(), f"Invalid numeric values in {col}"
            
    def test_performance_consistency(self, sample_results):
        """Test if performance metrics are consistent with trade data."""
        trades = sample_results['trades']
        metrics = sample_results['metrics'].iloc[0]
        
        # Check total return consistency
        calculated_return = trades['pnl'].sum() / sample_results['config']['initial_capital']
        assert abs(metrics['total_return'] - calculated_return) < 0.01, "Total return mismatch"
        
        # Check number of trades
        assert metrics['num_trades'] == len(trades), "Trade count mismatch"
        
    def test_strategy_specific_rules(self, sample_results):
        """Test if strategy-specific rules are followed."""
        trades = sample_results['trades']
        
        # Group trades by timestamp to check pairs
        grouped_trades = trades.groupby('timestamp')
        for timestamp, group in grouped_trades:
            # Skip if no trades at this timestamp
            if len(group) == 0:
                continue
                
            # Check if trades are in pairs
            assert len(group) % 2 == 0, f"Unpaired trades at {timestamp}"
            
            # Check if opposite positions are taken
            for i in range(0, len(group), 2):
                trade1 = group.iloc[i]
                trade2 = group.iloc[i+1]
                assert trade1['action'] != trade2['action'], f"Same position type at {timestamp}"
                assert abs(trade1['size']) == abs(trade2['size']), f"Unequal position sizes at {timestamp}"
                
            # Check spread and zscore for entry trades
            entry_trades = group[group['action'] != 'close']
            if not entry_trades.empty:
                assert 'spread' in entry_trades.columns, "Missing spread data"
                assert 'zscore' in entry_trades.columns, "Missing zscore data"
                assert (entry_trades['spread'].abs() > 0).all(), "Zero spread detected"
                assert (entry_trades['zscore'].abs() > 0).all(), "Zero zscore detected"
            
    def test_error_handling(self, sample_results):
        """Test if error handling and logging are implemented correctly."""
        run_dir = Path("results") / sample_results['run_id']
        
        # Check for error log file
        error_log = run_dir / "errors.log"
        assert error_log.exists(), "Error log file not found"
        
        # Check for summary file
        summary_file = run_dir / "summary.json"
        assert summary_file.exists(), "Summary file not found"
        
    def test_results_persistence(self, sample_results):
        """Test if all required files are saved and can be reloaded."""
        run_dir = Path("results") / sample_results['run_id']
        
        required_files = [
            "config.json",
            "summary.json",
            "errors.log",
            "trades/trades_enhanced.csv",
            "metrics/metrics.csv",
            "plots/equity_curve.png",
            "plots/drawdown_curve.png",
            "plots/trade_distribution.png",
            "plots/monthly_returns.png"
        ]
        
        for file in required_files:
            assert (run_dir / file).exists(), f"Required file not found: {file}"
            
    def test_visualization_quality(self, sample_results):
        """Test if visualizations are generated and readable."""
        run_dir = Path("results") / sample_results['run_id']
        plots_dir = run_dir / "plots"
        
        # Check plot files
        plot_files = [
            "equity_curve.png",
            "drawdown_curve.png",
            "trade_distribution.png",
            "monthly_returns.png"
        ]
        
        for plot in plot_files:
            plot_path = plots_dir / plot
            assert plot_path.exists(), f"Plot file not found: {plot}"
            assert plot_path.stat().st_size > 0, f"Empty plot file: {plot}"
            
    def test_parameter_sensitivity(self, sample_results):
        """Test if results are sensitive to small parameter changes."""
        config = sample_results['config']
        metrics = sample_results['metrics'].iloc[0]
        
        # Check if strategy parameters are within reasonable ranges
        assert 0 < config['max_position_size'] <= 1, "Invalid max_position_size"
        assert 0 < config['stop_loss'] < 1, "Invalid stop_loss"
        assert 0 < config['take_profit'] < 1, "Invalid take_profit"
        assert config['stop_loss'] < config['take_profit'], "Stop loss should be less than take profit"
        assert 0 <= config['commission'] < 0.01, "Invalid commission rate" 