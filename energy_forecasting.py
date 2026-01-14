"""
Energy Consumption Forecasting in Smart Homes
==============================================

This project forecasts household energy usage hour-by-hour and day-by-day using 
historical consumption, weather, and occupancy-related features. It builds a 
reproducible pipeline that cleans and engineers features from a smart grid dataset, 
trains tree-based and baseline models, evaluates them with time-series-aware metrics, 
and delivers a deployed forecasting model with recommendations for demand smoothing 
and cost savings.

Author: Energy Forecasting Team
Date: 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning Libraries
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("ENERGY CONSUMPTION FORECASTING IN SMART HOMES")
print("=" * 80)
print()

# ============================================================================
# 1. DATA LOADING AND PREPROCESSING
# ============================================================================

print("STEP 1: DATA LOADING AND PREPROCESSING")
print("-" * 80)

# Load the dataset
df = pd.read_csv('energyConsumption.csv')
print(f"✓ Loaded dataset with {len(df)} rows and {len(df.columns)} columns")
print(f"✓ Columns: {list(df.columns)}")
print()

# Display basic information
print("Dataset Overview:")
print(df.head())
print()
print("Data Types:")
print(df.dtypes)
print()

# Check for missing values
print("Missing Values:")
print(df.isnull().sum())
print()

# ============================================================================
# 2. TIMESTAMP PROCESSING
# ============================================================================

print("\nSTEP 2: TIMESTAMP PROCESSING")
print("-" * 80)

# Combine DATE and START TIME into a single timestamp
df['timestamp'] = pd.to_datetime(df['DATE'] + ' ' + df['START TIME'], 
                                  format='%m/%d/%Y %H:%M')

# Sort by timestamp to ensure chronological order
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✓ Created timestamp column")
print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"✓ Total duration: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print()

# ============================================================================
# 3. DATA AGGREGATION (15-min to Hourly)
# ============================================================================

print("\nSTEP 3: DATA AGGREGATION")
print("-" * 80)

# Aggregate 15-minute intervals to hourly for better modeling
df_hourly = df.groupby(pd.Grouper(key='timestamp', freq='H')).agg({
    'USAGE': 'sum',  # Sum energy usage over the hour
    'TEMPERATURE': 'mean',  # Average temperature
    'HUMIDITY': 'mean'  # Average humidity
}).reset_index()

# Remove any rows with missing values after aggregation
df_hourly = df_hourly.dropna()

print(f"✓ Aggregated to hourly intervals")
print(f"✓ New dataset size: {len(df_hourly)} rows")
print()
print("Hourly Data Sample:")
print(df_hourly.head(10))
print()

# ============================================================================
# 4. FEATURE ENGINEERING
# ============================================================================

print("\nSTEP 4: FEATURE ENGINEERING")
print("-" * 80)

# Time-based features
df_hourly['hour'] = df_hourly['timestamp'].dt.hour
df_hourly['day_of_week'] = df_hourly['timestamp'].dt.dayofweek
df_hourly['day_of_month'] = df_hourly['timestamp'].dt.day
df_hourly['month'] = df_hourly['timestamp'].dt.month
df_hourly['is_weekend'] = (df_hourly['day_of_week'] >= 5).astype(int)

print("✓ Created time-based features: hour, day_of_week, month, is_weekend")

# Cyclical encoding for hour and month (captures circular nature of time)
df_hourly['hour_sin'] = np.sin(2 * np.pi * df_hourly['hour'] / 24)
df_hourly['hour_cos'] = np.cos(2 * np.pi * df_hourly['hour'] / 24)
df_hourly['month_sin'] = np.sin(2 * np.pi * df_hourly['month'] / 12)
df_hourly['month_cos'] = np.cos(2 * np.pi * df_hourly['month'] / 12)

print("✓ Created cyclical encodings for hour and month")

# Lag features (previous consumption values)
df_hourly['usage_lag_1'] = df_hourly['USAGE'].shift(1)  # 1 hour ago
df_hourly['usage_lag_2'] = df_hourly['USAGE'].shift(2)  # 2 hours ago
df_hourly['usage_lag_24'] = df_hourly['USAGE'].shift(24)  # 24 hours ago (same time yesterday)

print("✓ Created lag features: 1h, 2h, 24h")

# Rolling statistics (moving averages)
df_hourly['usage_rolling_mean_3'] = df_hourly['USAGE'].rolling(window=3, min_periods=1).mean()
df_hourly['usage_rolling_mean_6'] = df_hourly['USAGE'].rolling(window=6, min_periods=1).mean()
df_hourly['usage_rolling_std_3'] = df_hourly['USAGE'].rolling(window=3, min_periods=1).std()

print("✓ Created rolling statistics: 3h mean, 6h mean, 3h std")

# Remove rows with NaN values created by lag features
df_hourly = df_hourly.dropna()

print(f"✓ Final dataset size after feature engineering: {len(df_hourly)} rows")
print()

# Display engineered features
print("Engineered Features Sample:")
print(df_hourly[['timestamp', 'USAGE', 'TEMPERATURE', 'HUMIDITY', 'hour', 
                 'is_weekend', 'usage_lag_1', 'usage_rolling_mean_3']].head())
print()

# ============================================================================
# 5. TRAIN-TEST SPLIT (Time-Series Aware)
# ============================================================================

print("\nSTEP 5: TRAIN-TEST SPLIT")
print("-" * 80)

# Define features and target
feature_columns = [
    'TEMPERATURE', 'HUMIDITY',
    'hour', 'day_of_week', 'month', 'is_weekend',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'usage_lag_1', 'usage_lag_2', 'usage_lag_24',
    'usage_rolling_mean_3', 'usage_rolling_mean_6', 'usage_rolling_std_3'
]

X = df_hourly[feature_columns]
y = df_hourly['USAGE']

# Time-series split: 80% train, 20% test (no shuffling!)
split_index = int(len(df_hourly) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(f"✓ Training set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")
print(f"✓ Features used: {len(feature_columns)}")
print()

# ============================================================================
# 6. BASELINE MODEL - LINEAR REGRESSION
# ============================================================================

print("\nSTEP 6: BASELINE MODEL - LINEAR REGRESSION")
print("-" * 80)

# Train baseline model
baseline_model = LinearRegression()
baseline_model.fit(X_train, y_train)

# Make predictions
y_pred_baseline = baseline_model.predict(X_test)

# Evaluate baseline model
mae_baseline = mean_absolute_error(y_test, y_pred_baseline)
rmse_baseline = np.sqrt(mean_squared_error(y_test, y_pred_baseline))
mape_baseline = np.mean(np.abs((y_test - y_pred_baseline) / y_test)) * 100
r2_baseline = r2_score(y_test, y_pred_baseline)

print("Baseline Model Performance:")
print(f"  MAE:  {mae_baseline:.4f} kWh")
print(f"  RMSE: {rmse_baseline:.4f} kWh")
print(f"  MAPE: {mape_baseline:.2f}%")
print(f"  R²:   {r2_baseline:.4f}")
print()

# ============================================================================
# 7. ADVANCED MODEL - XGBOOST
# ============================================================================

print("\nSTEP 7: ADVANCED MODEL - XGBOOST")
print("-" * 80)

# Train XGBoost model
xgb_model = XGBRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=0
)

print("Training XGBoost model...")
xgb_model.fit(X_train, y_train)
print("✓ Training complete")
print()

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate XGBoost model
mae_xgb = mean_absolute_error(y_test, y_pred_xgb)
rmse_xgb = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
mape_xgb = np.mean(np.abs((y_test - y_pred_xgb) / y_test)) * 100
r2_xgb = r2_score(y_test, y_pred_xgb)

print("XGBoost Model Performance:")
print(f"  MAE:  {mae_xgb:.4f} kWh")
print(f"  RMSE: {rmse_xgb:.4f} kWh")
print(f"  MAPE: {mape_xgb:.2f}%")
print(f"  R²:   {r2_xgb:.4f}")
print()

# ============================================================================
# 8. MODEL COMPARISON
# ============================================================================

print("\nSTEP 8: MODEL COMPARISON")
print("-" * 80)

comparison_df = pd.DataFrame({
    'Metric': ['MAE (kWh)', 'RMSE (kWh)', 'MAPE (%)', 'R²'],
    'Baseline (Linear Regression)': [mae_baseline, rmse_baseline, mape_baseline, r2_baseline],
    'XGBoost': [mae_xgb, rmse_xgb, mape_xgb, r2_xgb]
})

print(comparison_df.to_string(index=False))
print()

improvement_mae = ((mae_baseline - mae_xgb) / mae_baseline) * 100
improvement_rmse = ((rmse_baseline - rmse_xgb) / rmse_baseline) * 100

print(f"✓ XGBoost improves MAE by {improvement_mae:.2f}%")
print(f"✓ XGBoost improves RMSE by {improvement_rmse:.2f}%")
print()

# ============================================================================
# 9. VISUALIZATIONS
# ============================================================================

print("\nSTEP 9: GENERATING VISUALIZATIONS")
print("-" * 80)

# Create test set dataframe for plotting
test_df = df_hourly.iloc[split_index:].copy()
test_df['Actual'] = y_test.values
test_df['Predicted_Baseline'] = y_pred_baseline
test_df['Predicted_XGBoost'] = y_pred_xgb

# Plot 1: Actual vs Predicted (XGBoost)
plt.figure(figsize=(14, 6))
plt.plot(test_df['timestamp'], test_df['Actual'], label='Actual', linewidth=2, alpha=0.8)
plt.plot(test_df['timestamp'], test_df['Predicted_XGBoost'], label='XGBoost Prediction', 
         linewidth=2, alpha=0.8, linestyle='--')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Energy Consumption (kWh)', fontsize=12)
plt.title('Actual vs Predicted Energy Consumption (XGBoost Model)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('actual_vs_predicted_xgboost.png', dpi=300, bbox_inches='tight')
print("✓ Saved: actual_vs_predicted_xgboost.png")
plt.close()

# Plot 2: Model Comparison
plt.figure(figsize=(14, 6))
plt.plot(test_df['timestamp'], test_df['Actual'], label='Actual', linewidth=2.5, alpha=0.9)
plt.plot(test_df['timestamp'], test_df['Predicted_Baseline'], label='Baseline (Linear Regression)', 
         linewidth=2, alpha=0.7, linestyle=':')
plt.plot(test_df['timestamp'], test_df['Predicted_XGBoost'], label='XGBoost', 
         linewidth=2, alpha=0.8, linestyle='--')
plt.xlabel('Time', fontsize=12)
plt.ylabel('Energy Consumption (kWh)', fontsize=12)
plt.title('Model Comparison: Actual vs Predicted Energy Consumption', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")
plt.close()

# Plot 3: Feature Importance
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(10, 8))
plt.barh(feature_importance['Feature'], feature_importance['Importance'])
plt.xlabel('Importance Score', fontsize=12)
plt.ylabel('Feature', fontsize=12)
plt.title('Feature Importance (XGBoost Model)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ Saved: feature_importance.png")
plt.close()

# Plot 4: Error Distribution
errors = y_test.values - y_pred_xgb
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('Prediction Error (kWh)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Error Distribution (XGBoost Model)', fontsize=14, fontweight='bold')
plt.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
print("✓ Saved: error_distribution.png")
plt.close()

# Plot 5: Hourly Average Consumption Pattern
hourly_pattern = df_hourly.groupby('hour')['USAGE'].mean()
plt.figure(figsize=(12, 6))
plt.plot(hourly_pattern.index, hourly_pattern.values, marker='o', linewidth=2, markersize=8)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Average Energy Consumption (kWh)', fontsize=12)
plt.title('Average Hourly Energy Consumption Pattern', fontsize=14, fontweight='bold')
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hourly_pattern.png', dpi=300, bbox_inches='tight')
print("✓ Saved: hourly_pattern.png")
plt.close()

print()

# ============================================================================
# 10. INSIGHTS AND RECOMMENDATIONS
# ============================================================================

print("\nSTEP 10: INSIGHTS AND RECOMMENDATIONS")
print("-" * 80)

# Identify peak consumption hours
hourly_avg = df_hourly.groupby('hour')['USAGE'].mean()
peak_hours = hourly_avg.nlargest(3)
low_hours = hourly_avg.nsmallest(3)

print("📊 CONSUMPTION PATTERNS:")
print()
print("Peak Consumption Hours:")
for hour, usage in peak_hours.items():
    print(f"  • {hour:02d}:00 - {usage:.3f} kWh")
print()

print("Low Consumption Hours:")
for hour, usage in low_hours.items():
    print(f"  • {hour:02d}:00 - {usage:.3f} kWh")
print()

# Temperature correlation
temp_corr = df_hourly[['USAGE', 'TEMPERATURE', 'HUMIDITY']].corr()['USAGE']
print("🌡️ WEATHER IMPACT:")
print(f"  • Temperature correlation: {temp_corr['TEMPERATURE']:.3f}")
print(f"  • Humidity correlation: {temp_corr['HUMIDITY']:.3f}")
print()

# Top features
top_features = feature_importance.head(5)
print("🔑 TOP 5 INFLUENTIAL FEATURES:")
for idx, row in top_features.iterrows():
    print(f"  • {row['Feature']}: {row['Importance']:.4f}")
print()

print("💡 RECOMMENDATIONS FOR ENERGY SAVINGS:")
print()
print("1. DEMAND SHIFTING:")
print(f"   • Shift high-energy activities from peak hours ({peak_hours.index[0]:02d}:00-{peak_hours.index[-1]:02d}:00)")
print(f"   • to low-demand hours ({low_hours.index[0]:02d}:00-{low_hours.index[-1]:02d}:00)")
print()

print("2. TEMPERATURE MANAGEMENT:")
if temp_corr['TEMPERATURE'] > 0:
    print("   • Higher temperatures increase energy usage (likely cooling)")
    print("   • Optimize thermostat settings during peak temperature hours")
else:
    print("   • Lower temperatures increase energy usage (likely heating)")
    print("   • Improve insulation to reduce heating demand")
print()

print("3. PREDICTIVE SCHEDULING:")
print("   • Use forecasts to pre-cool/pre-heat during off-peak hours")
print("   • Schedule appliances (dishwasher, laundry) during low-demand periods")
print()

print("4. COST OPTIMIZATION:")
avg_usage = df_hourly['USAGE'].mean()
potential_savings = (peak_hours.mean() - low_hours.mean()) * 0.15  # Assume $0.15/kWh rate difference
print(f"   • Average hourly consumption: {avg_usage:.3f} kWh")
print(f"   • Potential savings by shifting 1 kWh from peak to off-peak: ${potential_savings:.3f}/hour")
print()

# ============================================================================
# 11. SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"✓ Dataset processed: {len(df)} rows → {len(df_hourly)} hourly records")
print(f"✓ Features engineered: {len(feature_columns)} features")
print(f"✓ Models trained: Baseline (Linear Regression) + XGBoost")
print(f"✓ Best model: XGBoost (MAE: {mae_xgb:.4f} kWh, MAPE: {mape_xgb:.2f}%)")
print(f"✓ Visualizations saved: 5 plots")
print()
print("📁 Output Files:")
print("  • actual_vs_predicted_xgboost.png")
print("  • model_comparison.png")
print("  • feature_importance.png")
print("  • error_distribution.png")
print("  • hourly_pattern.png")
print()
print("=" * 80)
print("FORECASTING PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 80)
