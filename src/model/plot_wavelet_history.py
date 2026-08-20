import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"

# Load training and test wavelet coefficients
train_data = torch.load(DATA_ROOT / 'wavelets/train wavelet/level4_swt_train_windows_norm.pt')
test_data = torch.load(DATA_ROOT / 'wavelets/test wavelet/level4_swt_test_windows_norm.pt')

# Load the original data
train_df = pd.read_csv(DATA_ROOT / 'Training Data/bitcoin_2010-07-29_2025-04-25_train.csv')
test_df = pd.read_csv(DATA_ROOT / 'Testing Data/bitcoin_2010-07-29_2025-04-25_test.csv')

# Convert dates to datetime
train_df['Date'] = pd.to_datetime(train_df['Date'])
test_df['Date'] = pd.to_datetime(test_df['Date'])

# Sort dataframes by date in ascending order (oldest to newest)
train_df = train_df.sort_values('Date').reset_index(drop=True)
test_df = test_df.sort_values('Date').reset_index(drop=True)

# The Close column contains the daily log returns
train_df['log_returns'] = train_df['Close']
test_df['log_returns'] = test_df['Close']

# Print first few log returns to verify
print("\nFirst few log returns (train):")
print(train_df[['Date', 'log_returns']].head())
print("\nLast few log returns (train):")
print(train_df[['Date', 'log_returns']].tail())

# Convert log returns back to price levels
# Date with Bitcoin's first trading price on Mt. Gox: $0.04951 on July 17, 2010
initial_price = 0.04951

# Calculate cumulative returns and convert to price
train_df['price'] = initial_price * np.exp(np.cumsum(train_df['log_returns']))
test_df['price'] = train_df['price'].iloc[-1] * np.exp(np.cumsum(test_df['log_returns']))

# Print some statistics
print("\nTraining period:")
print(f"First date: {train_df['Date'].iloc[0]}")
print(f"Last date: {train_df['Date'].iloc[-1]}")
print(f"Initial price: ${train_df['price'].iloc[0]:.2f}")
print(f"Final price: ${train_df['price'].iloc[-1]:.2f}")
print(f"Max price: ${train_df['price'].max():.2f}")
print(f"Min price: ${train_df['price'].min():.2f}")
print(f"Returns range: [{train_df['log_returns'].min():.4f}, {train_df['log_returns'].max():.4f}]")
print(f"Returns std: {train_df['log_returns'].std():.4f}")

print("\nTest period:")
print(f"First date: {test_df['Date'].iloc[0]}")
print(f"Last date: {test_df['Date'].iloc[-1]}")
print(f"Initial price: ${test_df['price'].iloc[0]:.2f}")
print(f"Final price: ${test_df['price'].iloc[-1]:.2f}")
print(f"Max price: ${test_df['price'].max():.2f}")
print(f"Min price: ${test_df['price'].min():.2f}")
print(f"Returns range: [{test_df['log_returns'].min():.4f}, {test_df['log_returns'].max():.4f}]")
print(f"Returns std: {test_df['log_returns'].std():.4f}")

# Load normalization factors
means = torch.load(DATA_ROOT / 'wavelets/train wavelet/wavelet_means.pt')
stds = torch.load(DATA_ROOT / 'wavelets/train wavelet/wavelet_stds.pt')

# Convert to numpy for easier plotting
train_np = train_data.numpy()
test_np = test_data.numpy()
means_np = means.numpy()
stds_np = stds.numpy()

print(f"\nTrain shape: {train_np.shape}")
print(f"Test shape: {test_np.shape}")

# Reshape data to match dimensions if needed
if train_np.ndim == 3:  # If shape is [N, T, 5]
    train_np = train_np.reshape(train_np.shape[0], train_np.shape[1], -1, 1)
if test_np.ndim == 3:  # If shape is [N, T, 5]
    test_np = test_np.reshape(test_np.shape[0], test_np.shape[1], -1, 1)

# Unnormalize the coefficients
def unnormalize(data, means, stds):
    # Ensure means and stds are the right shape [5, 1]
    if means.ndim == 1:
        means = means.reshape(-1, 1)
    if stds.ndim == 1:
        stds = stds.reshape(-1, 1)
    return data * stds[None, None, :, :] + means[None, None, :, :]

train_unnorm = unnormalize(train_np, means_np, stds_np)
test_unnorm = unnormalize(test_np, means_np, stds_np)

print(f"Unnormalized train shape: {train_unnorm.shape}")
print(f"Unnormalized test shape: {test_unnorm.shape}")

# Plot settings
band_names = ['cD1', 'cD2', 'cD3', 'cD4', 'cA4']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

# Create figure with 7 subplots (price, returns, and 5 wavelet bands)
plt.figure(figsize=(20, 20))

# Plot 1: Price (log scale)
plt.subplot(7, 1, 1)
plt.plot(train_df['Date'], train_df['price'], label='Train', color='blue', alpha=0.6)
plt.plot(test_df['Date'], test_df['price'], label='Test', color='red', alpha=0.6)
plt.title('Bitcoin Price (USD, Log Scale)')
plt.yscale('log')  # Use log scale for better visualization
plt.grid(True, alpha=0.3)
plt.legend()
plt.gcf().autofmt_xdate()

# Add price statistics
train_stats = f'Train range: ${train_df["price"].min():.2f} - ${train_df["price"].max():.2f}'
test_stats = f'Test range: ${test_df["price"].min():.2f} - ${test_df["price"].max():.2f}'
plt.text(0.02, 0.95, train_stats, transform=plt.gca().transAxes)
plt.text(0.60, 0.95, test_stats, transform=plt.gca().transAxes)

# Plot 2: Log Returns
plt.subplot(7, 1, 2)
plt.plot(train_df['Date'], train_df['log_returns'], label='Train', color='blue', alpha=0.6)
plt.plot(test_df['Date'], test_df['log_returns'], label='Test', color='red', alpha=0.6)
plt.title('Daily Log Returns')
plt.grid(True, alpha=0.3)
plt.legend()
plt.gcf().autofmt_xdate()

# Add statistics for returns
train_stats = f'Train σ: {np.std(train_df["log_returns"]):.4f}'
test_stats = f'Test σ: {np.std(test_df["log_returns"]):.4f}'
plt.text(0.02, 0.95, train_stats, transform=plt.gca().transAxes)
plt.text(0.85, 0.95, test_stats, transform=plt.gca().transAxes)

# Plot each wavelet band
for band_idx in range(5):
    plt.subplot(7, 1, band_idx + 3)
    
    # Get coefficients for this band - only first timestep of each window
    train_coeffs = train_unnorm[:, 0, band_idx, 0]  # Shape: [N_train]
    test_coeffs = test_unnorm[:, 0, band_idx, 0]    # Shape: [N_test]
    
    # Create date arrays for each window's first timestep
    train_dates = train_df['Date'].iloc[:len(train_coeffs)]
    test_dates = test_df['Date'].iloc[:len(test_coeffs)]
    
    # Plot coefficients as points and lines
    plt.plot(train_dates, train_coeffs, color=colors[band_idx], alpha=0.6, 
             label='Train', linewidth=1)
    plt.scatter(train_dates, train_coeffs, color=colors[band_idx], alpha=0.3, s=1)
    
    plt.plot(test_dates, test_coeffs, color=colors[band_idx], alpha=0.6,
             label='Test', linewidth=1)
    plt.scatter(test_dates, test_coeffs, color=colors[band_idx], alpha=0.3, s=1)
    
    # Add vertical line separating train/test
    plt.axvline(x=test_dates.iloc[0], color='black', linestyle='--', alpha=0.3)
    
    # Add band statistics
    train_stats = f'Train σ: {np.std(train_coeffs):.4f}'
    test_stats = f'Test σ: {np.std(test_coeffs):.4f}'
    plt.text(0.02, 0.95, train_stats, transform=plt.gca().transAxes)
    plt.text(0.85, 0.95, test_stats, transform=plt.gca().transAxes)
    
    plt.title(f'Wavelet Band: {band_names[band_idx]}')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.gcf().autofmt_xdate()

plt.tight_layout()
plt.savefig(OUTPUT_ROOT / 'wavelet_history_with_price.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ Saved full wavelet history plot to outputs/wavelet_history_with_price.png") 
