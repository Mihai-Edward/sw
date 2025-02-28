import sys
import os
import pandas as pd
import numpy as np
from lottery_ml_model import LotteryMLPredictor
from datetime import datetime, timedelta

def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file {file_path} not found")
    
    df = pd.read_csv(file_path)
    try:
        df['date'] = pd.to_datetime(df['date'], format='%H:%M %d-%m-%Y', errors='coerce')
        df.loc[df['date'].isna(), 'date'] = pd.to_datetime(df.loc[df['date'].isna(), 'date'], errors='coerce')
    except Exception as e:
        print(f"Warning: Date conversion issue: {e}")
    
    number_cols = [f'number{i+1}' for i in range(20)]
    try:
        df[number_cols] = df[number_cols].astype(float)
    except Exception as e:
        print(f"Warning: Could not process number columns: {e}")
    
    for col in number_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 0)
    
    return df

def get_next_draw_time(current_time):
    minutes = (current_time.minute // 5 + 1) * 5
    next_draw_time = current_time.replace(minute=0, second=0, microsecond=0) + timedelta(minutes=minutes)
    return next_draw_time

def save_predictions_to_csv(predicted_numbers, probabilities, next_draw_time, file_path):
    data = {
        'Timestamp': [next_draw_time.strftime('%Y-%m-%d %H:%M:%S')],
        'Predicted Numbers': [','.join(map(str, predicted_numbers))],
        'Probabilities': [','.join(map(str, [probabilities[num - 1] for num in predicted_numbers]))]
    }
    df = pd.DataFrame(data)
    
    if not os.path.exists(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=False)
    else:
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)

def main():
    try:
        predictor = LotteryMLPredictor(numbers_range=(1, 80), numbers_to_draw=20)
        models_dir = 'src/ml_models'
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
        
        model_timestamp_file = 'src/model_timestamp.txt'
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
            data_file = 'src/historical_draws.csv'
            print(f"Loading data from {data_file}...")
            historical_data = load_data(data_file)
            
            print("Preparing training data...")
            X, y = predictor.prepare_data(historical_data)
            
            print(f"Shape of X: {X.shape}")
            print(f"Shape of y: {y.shape}")
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            print("Training models...")
            predictor.train_models(X_train, y_train)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_path = f'{models_dir}/lottery_predictor_{timestamp}'
            predictor.save_models(model_path)
            
            with open(model_timestamp_file, 'w') as f:
                f.write(timestamp)
            
            print("\nModel training and saving complete.\n")
        else:
            # Load historical data if the model is already trained
            data_file = 'src/historical_draws.csv'
            print(f"Loading data from {data_file}...")
            historical_data = load_data(data_file)
        
        print("Generating ML prediction for next draw...")
        recent_draws = historical_data.tail(5).copy()
        predicted_numbers, probabilities = predictor.predict(recent_draws)
        
        formatted_numbers = ','.join(map(str, predicted_numbers))
        next_draw_time = get_next_draw_time(datetime.now())
        print(f"Predicted numbers for the next draw at {next_draw_time.strftime('%H:%M %d-%m-%Y')}: {formatted_numbers}")
        print(f"Prediction probabilities: {[probabilities[num - 1] for num in predicted_numbers]}")

        predictions_file = 'data/processed/predictions.csv'
        save_predictions_to_csv(predicted_numbers, probabilities, next_draw_time, predictions_file)
        
    except Exception as e:
        print(f"\nError: {str(e)}")
        if 'historical_data' in locals():
            print("\nFirst few rows of processed data:")
            print(historical_data.head())
            print("\nColumns in data:", historical_data.columns.tolist())

if __name__ == "__main__":
    main()