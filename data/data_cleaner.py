import pandas as pd
import numpy as np
import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load the raw data
df = pd.read_csv(PROJECT_ROOT / 'data' / 'Raw' / 'bitcoin_2010-07-29_2025-04-25.csv')

# Convert dates to datetime and remove time component
df['Date'] = pd.to_datetime(df['Start']).dt.date

# Keep only Date and Close columns for raw prices
raw_df = df[['Date', 'Close']].copy()

# Sort by date in ascending order (oldest to newest)
raw_df = raw_df.sort_values('Date').reset_index(drop=True)

# Calculate log returns
log_df = raw_df.copy()
log_df['Close'] = np.log(raw_df['Close'] / raw_df['Close'].shift(1))
log_df = log_df.dropna().reset_index(drop=True)  # Drop first row with NaN return

print("Full dataset:")
print(f"Date range: {raw_df['Date'].min()} to {raw_df['Date'].max()}")
print(f"Total rows: {len(raw_df)}")

print("\nRaw price statistics:")
print(f"Min price: ${raw_df['Close'].min():.2f}")
print(f"Max price: ${raw_df['Close'].max():.2f}")
print(f"Mean price: ${raw_df['Close'].mean():.2f}")
print(f"Std price: ${raw_df['Close'].std():.2f}")

print("\nLog returns statistics:")
print(f"Mean: {log_df['Close'].mean():.6f}")
print(f"Std: {log_df['Close'].std():.6f}")
print(f"Min: {log_df['Close'].min():.6f}")
print(f"Max: {log_df['Close'].max():.6f}")

# Split into training and test sets (80-20 split)
split_idx = int(len(raw_df) * 0.8)
train_raw = raw_df.iloc[:split_idx].copy()
test_raw = raw_df.iloc[split_idx:].copy()

# Same split for log returns
split_idx_log = int(len(log_df) * 0.8)
train_log = log_df.iloc[:split_idx_log].copy()
test_log = log_df.iloc[split_idx_log:].copy()

print("\nTraining set (80%):")
print(f"Date range: {train_raw['Date'].min()} to {train_raw['Date'].max()}")
print(f"Rows: {len(train_raw)}")
print(f"Raw price range: ${train_raw['Close'].min():.2f} - ${train_raw['Close'].max():.2f}")
print(f"Log returns range: [{train_log['Close'].min():.4f}, {train_log['Close'].max():.4f}]")

print("\nTest set (20%):")
print(f"Date range: {test_raw['Date'].min()} to {test_raw['Date'].max()}")
print(f"Rows: {len(test_raw)}")
print(f"Raw price range: ${test_raw['Close'].min():.2f} - ${test_raw['Close'].max():.2f}")
print(f"Log returns range: [{test_log['Close'].min():.4f}, {test_log['Close'].max():.4f}]")

# Create output directories if they don't exist
training_dir = PROJECT_ROOT / 'data' / 'Training Data'
testing_dir = PROJECT_ROOT / 'data' / 'Testing Data'
os.makedirs(training_dir, exist_ok=True)
os.makedirs(testing_dir, exist_ok=True)

# Convert dates to strings for CSV storage
train_raw['Date'] = train_raw['Date'].astype(str)
test_raw['Date'] = test_raw['Date'].astype(str)
train_log['Date'] = train_log['Date'].astype(str)
test_log['Date'] = test_log['Date'].astype(str)

# Save raw price datasets
train_raw.to_csv(training_dir / 'bitcoin_raw_prices_train.csv', index=False)
test_raw.to_csv(testing_dir / 'bitcoin_raw_prices_test.csv', index=False)

# Save log returns datasets (rename the existing files)
train_log.to_csv(training_dir / 'bitcoin_2010-07-29_2025-04-25_train.csv', index=False)
test_log.to_csv(testing_dir / 'bitcoin_2010-07-29_2025-04-25_test.csv', index=False)

print("\n✅ Saved files:")
print("Raw prices:")
print("  - data/Training Data/bitcoin_raw_prices_train.csv")
print("  - data/Testing Data/bitcoin_raw_prices_test.csv")
print("Log returns:")
print("  - data/Training Data/bitcoin_2010-07-29_2025-04-25_train.csv")
print("  - data/Testing Data/bitcoin_2010-07-29_2025-04-25_test.csv") 
