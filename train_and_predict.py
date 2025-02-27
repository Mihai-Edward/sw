import pandas as pd
import numpy as np
from lottery_ml_model import LotteryMLPredictor
from datetime import datetime
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

def prepare_data(df):
    """Prepare data for training"""
    # Debug: Print columns to verify the presence of 'numbers'
    print("Columns in DataFrame:", df.columns)
    
    if 'numbers' not in df.columns:
        raise KeyError("'numbers' column is missing in the DataFrame")
    
    X = df.drop(columns=['date'])
    
    # Convert lists in 'numbers' column to individual columns
    numbers_df = pd.DataFrame(df['numbers'].tolist(), index=df.index)
    X = pd.concat([X.drop(columns=['numbers']), numbers_df], axis=1)
    
    y_dict = {num: (numbers_df == num).astype(int).sum(axis=1) for num in range(1, 81)}
    
    return X, y_dict

def main():
    # Initialize predictor
    predictor = LotteryMLPredictor(numbers_range=(1, 80), numbers_to_draw=20)
    
    # Load historical data
    data_file = 'historical_draws.csv'
    print(f"Loading data from {data_file}...")
    historical_data = load_data(data_file)
    
    # Prepare data for training
    print("Preparing training data...")
    X, y_dict = prepare_data(historical_data)
    
    # Stack target arrays into a 2D array
    y = np.column_stack([y_dict[num] for num in range(1, 81)])
    
    # Check the shapes of X and y
    print("Shape of X:", X.shape)
    print("Shape of y:", y.shape)
    
    # Split into training and testing sets
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Check the shapes of X_train and y_train
    print("Shape of X_train:", X_train.shape)
    print("Shape of y_train:", y_train.shape)
    
    # Train models
    print("Training models...")
    predictor.train_models(X_train, y_train)
    
    # Save models
    models_dir = 'ml_models'
    os.makedirs(models_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f'{models_dir}/lottery_predictor_{timestamp}'
    predictor.save_models(model_path)
    
    # Save the timestamp to a file for later use
    with open('model_timestamp.txt', 'w') as f:
        f.write(timestamp)
    
    print("\nModel training and saving complete.\n")

if __name__ == "__main__":
    main()