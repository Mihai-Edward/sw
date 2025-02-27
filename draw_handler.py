import pandas as pd
from datetime import datetime
import os
import numpy as np
from openpyxl import Workbook, load_workbook
from lottery_ml_model import LotteryMLPredictor  # Updated import

def save_draw_to_csv(draw_date, draw_numbers, csv_file='historical_draws.csv'):
    """
    Save draw numbers to CSV file and display them in console
    
    Args:
        draw_date: Date/time of the draw
        draw_numbers: List of numbers from the draw
        csv_file: Path to the CSV file
    """
    # Display in console
    print(f"\nSaving draw to ML database: {draw_date}")
    print("Numbers:", ', '.join(map(str, sorted(draw_numbers))))
    
    # Ensure draw_numbers has exactly 20 numbers
    if len(draw_numbers) > 20:
        draw_numbers = sorted(draw_numbers)[:20]
    elif len(draw_numbers) < 20:
        print("Warning: Draw has less than 20 numbers")
        return
    
    # Prepare data for CSV
    draw_dict = {f'number{i+1}': num for i, num in enumerate(sorted(draw_numbers))}
    draw_dict['date'] = draw_date
    
    # Check if file exists
    if not os.path.exists(csv_file):
        # Create new DataFrame with headers
        df = pd.DataFrame([draw_dict])
    else:
        # Read existing file
        try:
            df = pd.read_csv(csv_file)
            # Check if this draw already exists
            if not df.empty and 'date' in df.columns:
                if draw_date in df['date'].values:
                    print(f"Draw for {draw_date} already exists in database")
                    return
            df = pd.concat([df, pd.DataFrame([draw_dict])], ignore_index=True)
        except Exception as e:
            print(f"Error reading existing CSV: {str(e)}")
            df = pd.DataFrame([draw_dict])
    
    # Save to CSV
    try:
        df.to_csv(csv_file, index=False)
        print(f"Successfully saved draw to {csv_file}")
    except Exception as e:
        print(f"Error saving to CSV: {str(e)}")

def save_predictions_to_excel(predictions, probabilities, timestamp, excel_file='data/processed/predictions.xlsx'):
    """
    Save ML predictions and their probabilities to an Excel file in the 'data/processed' directory
    
    Args:
        predictions: List of predicted numbers
        probabilities: List of probabilities corresponding to the predicted numbers
        timestamp: Time when the prediction was made
        excel_file: Path to the Excel file
    """
    # Prepare data for Excel
    data = {
        'Timestamp': [timestamp] * len(predictions),
        'Number': predictions,
        'Probability': probabilities
    }
    df = pd.DataFrame(data)

    if os.path.exists(excel_file):
        # Load existing workbook and sheet
        book = load_workbook(excel_file)
        writer = pd.ExcelWriter(excel_file, engine='openpyxl')
        writer.book = book
        writer.sheets = {ws.title: ws for ws in book.worksheets}
        
        # Get the last row in the existing Excel sheet
        startrow = writer.sheets['Predictions'].max_row
        
        # Append the new data
        df.to_excel(writer, sheet_name='Predictions', index=False, header=False, startrow=startrow)
        
        # Save the workbook
        writer.close()
    else:
        # Create a new workbook and sheet
        book = Workbook()
        sheet = book.active
        sheet.title = 'Predictions'
        
        # Write the data
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Predictions', index=False)

    print(f"\nPredictions saved to {excel_file}")

def train_ml_models(csv_file='historical_draws.csv', models_dir='ml_models'):
    """
    Train ML models based on historical data
    
    Args:
        csv_file: Path to the CSV file containing historical data
        models_dir: Directory to save the trained models
    """
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            print(f"\nError: No historical data found at {csv_file}")
            return None
        
        # Load historical data
        historical_data = pd.read_csv(csv_file)
        
        # Verify data format
        required_columns = ['date'] + [f'number{i}' for i in range(1, 21)]
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            print(f"\nError: Missing columns in data: {missing_columns}")
            return None
        
        # Check if we have enough historical data (need at least 6 draws)
        if len(historical_data) < 6:
            print(f"\nNot enough historical data. Need at least 6 draws, but only have {len(historical_data)}.")
            return None
        
        print(f"\nFound {len(historical_data)} historical draws in database.")
        
        # Initialize predictor
        predictor = LotteryMLPredictor(numbers_range=(1, 80), numbers_to_draw=20)
        
        print("Training new ML models...")
        # Prepare and train models
        X, y = predictor.prepare_data(historical_data)
        predictor.train_models(X, y)
        
        # Save models
        os.makedirs(models_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        predictor.save_models(f'{models_dir}/lottery_predictor_{timestamp}')
        print("Models trained and saved successfully!")
        
    except Exception as e:
        print(f"\nError in training ML models: {str(e)}")

def get_ml_prediction(csv_file='historical_draws.csv'):
    """
    Get prediction from ML model based on historical data
    """
    try:
        # Check if CSV file exists
        if not os.path.exists(csv_file):
            print(f"\nError: No historical data found at {csv_file}")
            print("Please use option 3 first to collect some draws.")
            return None
            
        # Load historical data
        historical_data = pd.read_csv(csv_file)
        
        # Verify data format
        required_columns = ['date'] + [f'number{i}' for i in range(1, 21)]
        missing_columns = [col for col in required_columns if col not in historical_data.columns]
        if missing_columns:
            print(f"\nError: Missing columns in data: {missing_columns}")
            print("Please ensure your data has 'date' and 'number1' through 'number20' columns.")
            return None
        
        # Check if we have enough historical data (need at least 6 draws)
        if len(historical_data) < 6:
            print(f"\nNot enough historical data. Need at least 6 draws, but only have {len(historical_data)}.")
            print("Please use option 3 to collect more draws.")
            return None
        
        print(f"\nFound {len(historical_data)} historical draws in database.")
        
        # Initialize predictor
        predictor = LotteryMLPredictor(numbers_range=(1, 80), numbers_to_draw=20)
        
        # Load existing models if available
        models_dir = 'ml_models'
        latest_model = None
        if os.path.exists(models_dir):
            model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.pkl')]
            if model_files:
                latest_model = os.path.join(models_dir, max(model_files))
        
        if latest_model:
            print("Loading existing ML models...")
            predictor.load_models(latest_model.replace('_model.pkl', ''))
        else:
            print("Training new ML models...")
            # Prepare and train models
            X, y = predictor.prepare_data(historical_data)
            predictor.train_models(X, y)
            
            # Save models
            os.makedirs(models_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            predictor.save_models(f'{models_dir}/lottery_predictor_{timestamp}')
        
        # Generate prediction
        recent_draws = historical_data.tail(5)
        predicted_numbers, probabilities = predictor.predict(recent_draws)
        
        # Display prediction results
        print("\n=== ML Prediction Results ===")
        print("Predicted numbers for next draw:", sorted(predicted_numbers))
        
        print("\nTop 10 most likely numbers and their probabilities:")
        number_probs = [(number, prob) for number, prob in zip(predicted_numbers, probabilities)]
        number_probs.sort(key=lambda x: x[1], reverse=True)
        for number, prob in number_probs[:10]:
            print(f"Number {number}: {prob:.4f}")
        
        # Save predictions to Excel
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        save_predictions_to_excel([num for num, _ in number_probs[:10]], [prob for _, prob in number_probs[:10]], timestamp)
        
        return predicted_numbers
        
    except Exception as e:
        print(f"\nError in ML prediction: {str(e)}")
        print("\nDebug information:")
        print(f"CSV file exists: {os.path.exists(csv_file)}")
        if os.path.exists(csv_file):
            try:
                df = pd.read_csv(csv_file)
                print(f"Number of rows in data: {len(df)}")
                print("Columns in data:", df.columns.tolist())
                print("\nFirst few rows of data:")
                print(df.head())
            except Exception as read_error:
                print(f"Error reading CSV file: {str(read_error)}")
        return None