"""
Script to add Temperature and Humidity columns to energyConsumption.csv
Generates realistic weather data based on time of day and date patterns
"""

import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
INPUT_FILE = 'energyConsumption.csv'
OUTPUT_FILE = 'energyConsumption.csv'  # Overwrite the original file

def generate_realistic_weather(df):
    """
    Generate realistic temperature and humidity based on:
    - Time of day (cooler at night, warmer during day)
    - Date progression (slight day-to-day variation)
    - Random noise for realism
    """
    
    temperatures = []
    humidities = []
    
    # Base temperature for October (in Celsius)
    base_temp = 18.0  # ~64°F
    
    # Base humidity for October
    base_humidity = 65.0
    
    for idx, row in df.iterrows():
        # Parse time
        time_str = row['START TIME']
        hour = int(time_str.split(':')[0])
        
        # Parse date to get day progression
        date_str = row['DATE']
        date_obj = datetime.strptime(date_str, '%m/%d/%Y')
        day_of_year = date_obj.timetuple().tm_yday
        
        # Temperature calculation
        # Daily cycle: cooler at night (3-6 AM), warmer in afternoon (2-4 PM)
        time_factor = np.sin((hour - 6) * np.pi / 12)  # Peak at 2-3 PM
        daily_variation = 6.0 * time_factor  # ±6°C variation
        
        # Day-to-day variation (slight trend)
        day_variation = np.sin(day_of_year * 0.1) * 2.0
        
        # Random noise
        noise = np.random.normal(0, 0.5)
        
        temperature = base_temp + daily_variation + day_variation + noise
        temperature = round(temperature, 1)
        
        # Humidity calculation
        # Inverse relationship with temperature (higher temp = lower humidity)
        humidity_variation = -time_factor * 15.0  # Inverse of temp pattern
        humidity_noise = np.random.normal(0, 2.0)
        
        humidity = base_humidity + humidity_variation + humidity_noise
        humidity = max(30, min(95, round(humidity, 1)))  # Keep between 30-95%
        
        temperatures.append(temperature)
        humidities.append(humidity)
    
    return temperatures, humidities


def main():
    print("=" * 60)
    print("Energy Consumption CSV Updater")
    print("=" * 60)
    
    # Read the CSV
    print(f"\n📂 Reading {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    
    print(f"   ✓ Loaded {len(df)} rows")
    print(f"   ✓ Columns: {list(df.columns)}")
    
    # Analyze the data
    print("\n📊 Analyzing data structure...")
    
    # Get unique dates
    unique_dates = df['DATE'].unique()
    print(f"   ✓ Date range: {unique_dates[0]} to {unique_dates[-1]}")
    print(f"   ✓ Number of days: {len(unique_dates)}")
    
    # Count rows per day
    rows_per_day = len(df) // len(unique_dates)
    print(f"   ✓ Rows per day: ~{rows_per_day} (15-minute intervals)")
    
    # Generate weather data
    print("\n🌡️  Generating realistic weather data...")
    temperatures, humidities = generate_realistic_weather(df)
    
    print(f"   ✓ Temperature range: {min(temperatures):.1f}°C to {max(temperatures):.1f}°C")
    print(f"   ✓ Humidity range: {min(humidities):.1f}% to {max(humidities):.1f}%")
    
    # Add new columns
    df['TEMPERATURE'] = temperatures
    df['HUMIDITY'] = humidities
    
    # Reorder columns for better readability
    # Put weather data after time columns
    column_order = ['TYPE', 'DATE', 'START TIME', 'END TIME', 
                   'TEMPERATURE', 'HUMIDITY', 
                   'USAGE', 'UNITS', 'COST', 'NOTES']
    df = df[column_order]
    
    # Save the updated CSV
    print(f"\n💾 Saving updated CSV to {OUTPUT_FILE}...")
    df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"   ✓ File saved successfully!")
    
    # Display sample
    print("\n📋 Sample of updated data (first 5 rows):")
    print(df.head().to_string())
    
    print("\n" + "=" * 60)
    print("✅ CSV update completed successfully!")
    print("=" * 60)
    print("\n💡 Note: This script works with any size CSV with the same format.")
    print("   You can replace the data later and run this script again.")


if __name__ == "__main__":
    main()
