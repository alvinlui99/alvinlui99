import pandas as pd
import numpy as np
from hmmlearn import hmm
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from datetime import datetime

# Read the backtest results
df = pd.read_csv('backtest_results/backtest_results_BTCUSDT_ETHUSDT_20250529_030824.csv')

# Extract zscore column, drop NaN values, and reshape for HMM
zscore_data = df['zscore'].dropna().values.reshape(-1, 1)

# Load the existing model
model_filename = 'hmm_model_20250529_110326.pkl'
with open(model_filename, 'rb') as f:
    model = pickle.load(f)
print(f"\nLoaded model from: {model_filename}")

# Get the number of states from the loaded model
n_states = model.n_components

# Get the hidden states
hidden_states = model.predict(zscore_data)

# Calculate state means and standard deviations
state_means = []
state_stds = []
for i in range(n_states):
    state_data = zscore_data[hidden_states == i]
    state_means.append(np.mean(state_data))
    state_stds.append(np.std(state_data))

# Create a DataFrame with results (only for non-NaN values)
valid_indices = df['zscore'].dropna().index
results_df = pd.DataFrame({
    'timestamp': df.loc[valid_indices, 'timestamp'],
    'zscore': df.loc[valid_indices, 'zscore'],
    'regime': hidden_states
})

# Plotting
plt.figure(figsize=(15, 10))

# Plot 1: Z-score with regime colors
plt.subplot(2, 1, 1)
for i in range(n_states):
    mask = hidden_states == i
    plt.scatter(results_df['timestamp'][mask], results_df['zscore'][mask], 
                label=f'Regime {i+1}', alpha=0.6)
plt.title('Z-score with Regime Detection')
plt.xlabel('Timestamp')
plt.ylabel('Z-score')
plt.legend()

# Plot 2: Distribution of Z-scores by regime
plt.subplot(2, 1, 2)
sns.kdeplot(data=results_df, x='zscore', hue='regime', common_norm=False)
plt.title('Distribution of Z-scores by Regime')
plt.xlabel('Z-score')
plt.ylabel('Density')

plt.tight_layout()
plt.savefig('regime_detection_results.png')
plt.close()

# Print regime statistics
print("\nRegime Statistics:")
for i in range(n_states):
    print(f"\nRegime {i+1}:")
    print(f"Mean Z-score: {state_means[i]:.4f}")
    print(f"Std Dev: {state_stds[i]:.4f}")
    print(f"Duration: {np.sum(hidden_states == i)} periods")
    print(f"Percentage: {(np.sum(hidden_states == i) / len(hidden_states) * 100):.2f}%")

# Save results to CSV
results_df.to_csv('regime_detection_results.csv', index=False) 