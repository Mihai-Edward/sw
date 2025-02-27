import pandas as pd
import numpy as np
from lottery_ml_predictor import LotteryMLPredictor
from datetime import datetime
import os

def load_data(file_path):
    """Load historical lottery data from CSV"""
    df = pd.read_csv(file_path)
    
    # Convert date column if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
    
    return df

def main():
    # Initialize predictor
    predictor = LotteryMLPredictor(numbers_range=(1, 80), numbers_to_draw=20)
    
    # Load historical data
    data_file = 'historical_draws.csv'  # Update this to your data file path
    print(f"Loading data from {data_file}...")
    historical_data = load_data(data_file)
    
    # Prepare data for training
    print("Preparing training data...")
    X, y = predictor.prepare_data(historical_data)
    
    # Split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train models
    print("Training models...")
    predictor.train_models(X_train, y_train)
    
    # Save models
    models_dir = 'ml_models'
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    predictor.save_models(f'{models_dir}/lottery_predictor_{timestamp}')
    
    # Generate prediction for next draw
    print("\nGenerating prediction for next draw...")
    recent_draws = historical_data.tail(5)
    predicted_numbers, probabilities = predictor.predict(recent_draws)
    
    print("\nPredicted numbers for next draw:", predicted_numbers)
    
    # Print probability information for each number
    print("\nTop 10 most likely numbers and their probabilities:")
    number_probs = [(i+1, prob) for i, prob in enumerate(probabilities)]
    number_probs.sort(key=lambda x: x[1], reverse=True)
    for number, prob in number_probs[:10]:
        print(f"Number {number}: {prob:.4f}")

if __name__ == "__main__":
    main()