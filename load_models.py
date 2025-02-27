import pandas as pd
import numpy as np
from lottery_ml_model import LotteryMLPredictor
import os
import ast

def load_data(file_path):
    """Load historical lottery data from CSV"""
    df = pd.read_csv(file_path)
    
    # Convert date column if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    
    # Convert 'numbers' column from string representation of lists to actual lists
    if 'numbers' in df.columns:
        df['numbers'] = df['numbers'].apply(ast.literal_eval)
    
    return df

def prepare_recent_draws(df):
    """Prepare recent draws for prediction"""
    # Convert lists in 'numbers' column to individual columns
    numbers_df = pd.DataFrame(df['numbers'].tolist(), index=df.index)
    return numbers_df

def main():
    # Initialize predictor
    predictor = LotteryMLPredictor(numbers_range=(1, 80), numbers_to_draw=20)
    
    # Load historical data
    data_file = 'historical_draws.csv'
    print(f"Loading data from {data_file}...")
    historical_data = load_data(data_file)
    
    # Read the saved timestamp
    with open('model_timestamp.txt', 'r') as f:
        timestamp = f.read().strip()
    
    # Load models
    print("Loading existing ML models...")
    models_dir = 'ml_models'
    model_path = f'{models_dir}/lottery_predictor_{timestamp}'
    predictor.load_models(model_path)
    
    # Generate prediction for next draw
    print("\nGenerating prediction for next draw...")
    recent_draws = historical_data.tail(5).drop(columns=['date'])
    recent_draws_prepared = prepare_recent_draws(recent_draws)
    predicted_numbers, probabilities = predictor.predict(recent_draws_prepared)
    
    # Sort predicted numbers
    predicted_numbers_sorted = sorted(predicted_numbers)
    
    print("\nPredicted numbers for next draw:", predicted_numbers_sorted)
    
    # Print probability information for each number
    print("\nTop 10 most likely numbers and their probabilities:")
    number_probs = [(i+1, prob) for i, prob in enumerate(probabilities)]
    number_probs.sort(key=lambda x: x[1], reverse=True)
    for number, prob in number_probs[:10]:
        print(f"Number {number}: {prob:.4f}")

if __name__ == "__main__":
    main()