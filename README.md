# Energy Consumption Forecasting - Quick Start Guide

## 📋 Overview

This guide helps you run the energy consumption forecasting system and understand the outputs.

---

## 🚀 Quick Start

### Step 1: Ensure Dependencies are Installed

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Step 2: Run the Forecasting Pipeline

```bash
python energy_forecasting.py
```

### Step 3: View Results

The script will:
- Process your data
- Train models
- Generate visualizations
- Print performance metrics and recommendations

---

## 📁 File Structure

```
subin pr/
├── energyConsumption.csv          # Input data (with temperature & humidity)
├── energy_forecasting.py          # Main forecasting script
├── updatecsv.py                   # Utility to add weather data
├── plan.md                        # Project plan
├── RESULTS.md                     # Detailed results summary
├── README.md                      # This file
│
├── actual_vs_predicted_xgboost.png    # Prediction accuracy plot
├── model_comparison.png               # Baseline vs XGBoost comparison
├── feature_importance.png             # Feature importance chart
├── error_distribution.png             # Error analysis
└── hourly_pattern.png                 # Daily consumption pattern
```

---

## 🔄 Using with Your Own Data

### Option 1: Replace Existing Data

1. Replace `energyConsumption.csv` with your data
2. Ensure it has these columns:
   - `TYPE` (e.g., "Electric usage")
   - `DATE` (format: MM/DD/YYYY)
   - `START TIME` (format: H:MM)
   - `END TIME` (format: H:MM)
   - `USAGE` (energy consumption in kWh)
   - `UNITS` (e.g., "kWh")
   - `COST` (optional)
   - `NOTES` (optional)
   - `TEMPERATURE` (in Celsius)
   - `HUMIDITY` (in percentage)

3. Run the script:
   ```bash
   python energy_forecasting.py
   ```

### Option 2: Add Weather Data to Existing CSV

If your CSV doesn't have temperature and humidity:

1. Ensure your CSV has: `TYPE`, `DATE`, `START TIME`, `END TIME`, `USAGE`, `UNITS`, `COST`, `NOTES`
2. Run the weather data generator:
   ```bash
   python updatecsv.py
   ```
3. This will add `TEMPERATURE` and `HUMIDITY` columns with realistic fake data
4. Then run the forecasting:
   ```bash
   python energy_forecasting.py
   ```

---

## 📊 Understanding the Outputs

### Console Output

The script prints:
1. **Data Loading**: Confirms data is loaded correctly
2. **Feature Engineering**: Shows features created
3. **Model Performance**: 
   - MAE (Mean Absolute Error) - lower is better
   - RMSE (Root Mean Squared Error) - lower is better
   - MAPE (Mean Absolute Percentage Error) - lower is better
   - R² (R-squared) - higher is better (max 1.0)
4. **Insights**: Peak hours, weather impact, recommendations

### Visualization Files

1. **actual_vs_predicted_xgboost.png**
   - Blue line = actual consumption
   - Orange dashed line = predicted consumption
   - Closer lines = better predictions

2. **model_comparison.png**
   - Compares all models
   - Shows which model performs best

3. **feature_importance.png**
   - Shows which features matter most
   - Longer bars = more important features

4. **error_distribution.png**
   - Shows prediction errors
   - Centered around zero = good
   - Narrow distribution = consistent predictions

5. **hourly_pattern.png**
   - Shows average consumption by hour
   - Helps identify peak and off-peak hours

---

## 🎯 Key Metrics Explained

### MAE (Mean Absolute Error)
- Average prediction error in kWh
- Example: MAE = 0.05 means predictions are off by 0.05 kWh on average
- **Good**: < 0.10 kWh for hourly predictions

### RMSE (Root Mean Squared Error)
- Similar to MAE but penalizes large errors more
- Always ≥ MAE
- **Good**: < 0.15 kWh for hourly predictions

### MAPE (Mean Absolute Percentage Error)
- Percentage error
- Example: MAPE = 15% means predictions are off by 15% on average
- **Good**: < 20% for energy forecasting

### R² (R-squared)
- How well the model explains variance
- Range: 0 to 1 (1 = perfect)
- **Good**: > 0.70 for energy forecasting

---

## 💡 Interpreting Recommendations

### Peak Hours
- Hours with highest average consumption
- **Action**: Avoid running high-power appliances during these times

### Low Hours
- Hours with lowest average consumption
- **Action**: Schedule energy-intensive tasks during these times

### Weather Impact
- Positive temperature correlation = cooling dominates
- Negative temperature correlation = heating dominates
- **Action**: Optimize thermostat based on forecasts

### Top Features
- Shows what drives your consumption
- **Action**: Focus efficiency efforts on top factors

---

## 🔧 Customization

### Modify Model Parameters

Edit `energy_forecasting.py` around line 180:

```python
xgb_model = XGBRegressor(
    n_estimators=100,      # Increase for more complex patterns
    learning_rate=0.1,     # Decrease for more careful learning
    max_depth=6,           # Increase for more complex patterns
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

### Change Train-Test Split

Edit around line 150:

```python
split_index = int(len(df_hourly) * 0.8)  # Change 0.8 to 0.7 for 70-30 split
```

### Add More Features

Add your custom features around line 120-140 in the feature engineering section.

---

## ⚠️ Troubleshooting

### Error: "ModuleNotFoundError: No module named 'xgboost'"
**Solution**: Install dependencies
```bash
pip install xgboost scikit-learn matplotlib seaborn
```

### Error: "FileNotFoundError: energyConsumption.csv"
**Solution**: Ensure CSV file is in the same directory as the script

### Warning: "Not enough data for lag features"
**Solution**: Need at least 25+ hours of data for proper lag features

### Poor Model Performance (R² < 0.5)
**Possible causes**:
- Dataset too small (need more data)
- Missing important features
- Data quality issues
**Solution**: Get more data or check data quality

---

## 📈 Scaling to Larger Datasets

### Current Sample: 3 days (288 rows)
- Works for testing and development
- Limited pattern recognition

### Recommended: 6-12 months
- Better seasonal pattern detection
- More robust model training
- Improved accuracy

### Steps to Scale:
1. Obtain larger dataset with same format
2. Replace `energyConsumption.csv`
3. Run `updatecsv.py` if weather data needed
4. Run `energy_forecasting.py` (no code changes needed!)

The code automatically handles any dataset size with the same format.

---

## 📞 Support

### Common Questions

**Q: How accurate should the model be?**
A: For hourly forecasting, MAPE < 20% and R² > 0.70 is good. With more data, aim for MAPE < 15%.

**Q: Can I use this for real-time forecasting?**
A: Yes, but you'll need to modify the code to:
- Load new data continuously
- Retrain periodically
- Make predictions on-demand

**Q: What if I don't have weather data?**
A: Use `updatecsv.py` to generate realistic fake weather data for testing. For production, integrate with a weather API.

**Q: How do I improve accuracy?**
A: 
- Get more historical data (6-12 months minimum)
- Add more features (occupancy, appliance usage, etc.)
- Tune hyperparameters
- Try ensemble methods

---

## ✅ Checklist for Success

- [ ] Dependencies installed
- [ ] CSV file has required columns
- [ ] Weather data added (temperature & humidity)
- [ ] Script runs without errors
- [ ] Visualizations generated
- [ ] Model performance is acceptable (R² > 0.70)
- [ ] Recommendations reviewed
- [ ] Ready to scale with full dataset

---

## 🎓 Next Steps

1. **Collect More Data**: Get 6-12 months of historical consumption data
2. **Integrate Weather API**: Use real weather data instead of fake data
3. **Add Features**: Include occupancy, day type (holiday), appliance data
4. **Deploy**: Create API or web interface for real-time forecasting
5. **Automate**: Set up scheduled retraining and prediction updates
6. **Monitor**: Track actual vs predicted and retrain when accuracy drops

---

**Happy Forecasting! 🚀**

For detailed results and insights, see `RESULTS.md`
#   E n e r g y - C o n s u m p t i o n - F o r e c a s t i n g - I n - S m a r t - H o m e s  
 