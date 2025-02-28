import sys
import os
import pandas as pd
import numpy as np
from lottery_ml_model import LotteryMLPredictor
from datetime import datetime
import ast

def load_data(file_path):
    """Load historical lottery data from CSV"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    
    df = pd.read_csv(file_path)
    
    # Handle date column with mixed formats
    try:
        # First try to convert dates that are in DD-MM-YYYY format
        df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
        
        # For any dates that failed (resulted in NaT), try the other format
        mask = df['date'].isna()
        df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'date'], format='%Y-%m-%d', errors='coerce')
        
        # For any remaining NaT values, try parsing with mixed format
        mask = df['date'].isna()
        df.loc[mask, 'date'] = pd.to_datetime(df.loc[mask, 'date'], format='mixed')
        
    except Exception as e:
        print(f"Warning: Date conversion issue: {e}")
    
    # Clean up the data structure
    if 'numbers' in df.columns:
        # If we have a 'numbers' column, use it to fill in the individual number columns
        try:
            # Convert string representation of list to actual list
            df['numbers'] = df['numbers'].apply(ast.literal_eval)
            
            # Create or update the individual number columns
            for i in range(20):
                col_name = f'number{i+1}'
                if col_name not in df.columns:
                    df[col_name] = np.nan
                df[col_name] = df['numbers'].apply(lambda x: x[i] if len(x) > i else np.nan)
        except Exception as e:
            print(f"Warning: Could not process 'numbers' column: {e}")
    else:
        # Handle the case where numbers are not in a list format
        try:
            number_cols = [f'number{i+1}' for i in range(20)]
            for col in number_cols:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Assuming the numbers are in the second column and separated by commas
            df[number_cols] = df.iloc[:, 1].str.split(',', expand=True).astype(float)
        except Exception as e:
            print(f"Warning: Could not process numbers: {e}")
    
    # Drop the 'numbers' column as we now have individual columns
    if 'numbers' in df.columns:
        df = df.drop('numbers', axis=1)
    
    # Fill any remaining NaN values with the mode (most common value) of each column
    number_cols = [f'number{i+1}' for i in range(1, 21)]
    for col in number_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    return df

def prepare_data(df):
    """Prepare data for training"""
    # Get all number columns
    number_cols = [f'number{i}' for i in range(1, 21)]
    required_cols = ['date'] + number_cols
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in data: {missing_cols}")
    
    # Create feature matrix X from the number columns
    X = df[number_cols].astype(float)
    
    # Create target matrix y (80 columns, one for each possible number)
    y = np.zeros((len(df), 80))
    for i in range(80):
        number = i + 1
        y[:, i] = (X == number).any(axis=1).astype(int)
    
    return X, y

def main():
    try:
        # Initialize predictor
        predictor = LotteryMLPredictor(numbers_range=(1, 80), numbers_to_draw=20)
        
        # Check if there is a previously saved model
        models_dir = 'ml_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        model_timestamp_file = 'model_timestamp.txt'
        model_loaded = False
        
        if os.path.exists(model_timestamp_file):
            with open(model_timestamp_file, 'r') as f:
                timestamp = f.read().strip()
                model_path = f'{models_dir}/lottery_predictor_{timestamp}'
                try:
                    predictor.load_models(model_path)
                    print(f"Model loaded from {model_path}")
                    model_loaded = True
                except Exception as e:
                    print(f"Error loading model: {e}")
        
        if not model_loaded:
            # Load historical data
            data_file = 'C:/Users/MihaiNita/OneDrive - Prime Batteries/Desktop/proiectnow/Versiune1.4/historical_draws.csv'
            print(f"Loading data from {data_file}...")
            historical_data = load_data(data_file)
            
            # Prepare data for training
            print("Preparing training data...")
            X, y = prepare_data(historical_data)
            
            # Check the shapes of X and y
            print(f"Shape of X: {X.shape}")
            print(f"Shape of y: {y.shape}")
            
            # Split into training and testing sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train models
            print("Training models...")
            predictor.train_models(X_train, y_train)
            
            # Save models
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'{models_dir}/lottery_predictor_{timestamp}'
            predictor.save_models(model_path)
            
            # Save the timestamp to a file for later use
            with open(model_timestamp_file, 'w') as f:
                f.write(timestamp)
            
            print("\nModel training and saving complete.\n")
        
        # Predict the next draw
        print("Generating ML prediction for next draw...")
        # Here you need to provide recent draws to the predict function
        # For demonstration, let's assume recent_draws is the last 5 draws from historical data
        recent_draws = historical_data.tail(5)
        predicted_numbers, probabilities = predictor.predict(recent_draws)
        
        print(f"Predicted numbers for the next draw: {predicted_numbers}")
        print(f"Prediction probabilities: {probabilities}")
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        print("\nDebug information:")
        if 'historical_data' in locals():
            print("\nFirst few rows of processed data:")
            print(historical_data.head())
            print("\nColumns in data:", historical_data.columns.tolist())

if __name__ == "__main__":
    main()