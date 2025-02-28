from data_collector_selenium import KinoDataCollector
from data_analysis import DataAnalysis
from draw_handler import save_draw_to_csv, get_ml_prediction
import sys
import os
import pandas as pd

def ensure_directories():
    """Create necessary directories if they don't exist"""
    directories = ['src/ml_models', 'src/data/processed']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def check_and_train_model():
    """Check if model exists and is trained, if not train it"""
    model_timestamp_file = 'src/model_timestamp.txt'
    models_dir = 'src/ml_models'
    
    needs_training = False
    
    # Check if model timestamp exists
    if not os.path.exists(model_timestamp_file):
        needs_training = True
    else:
        # Read timestamp and check if model files exist
        with open(model_timestamp_file, 'r') as f:
            timestamp = f.read().strip()
            model_path = f'{models_dir}/lottery_predictor_{timestamp}'
            if not os.path.exists(f"{model_path}_prob_model_0.pkl"):
                needs_training = True
    
    if needs_training:
        print("No trained model found. Training new model...")
        from train_and_predict import main as train_main
        train_main()

def main():
    # Ensure all necessary directories exist
    ensure_directories()
    
    collector = KinoDataCollector()
    draws = None

    while True:
        print("\n==========================")
        print("1. Show number frequency")
        print("2. Get number suggestion")
        print("3. Fetch latest draws from lotostats.ro")
        print("4. Find common pairs")
        print("5. Find consecutive numbers")
        print("6. Number range analysis")
        print("7. Find hot and cold numbers")
        print("8. Save analysis to Excel")
        print("9. Get ML prediction")
        print("10. Evaluate prediction accuracy")
        print("11. Exit")
        print("==========================\n")

        try:
            choice = input("Choose an option (1-11): ")

            if choice == '1':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                top_numbers = analysis.get_top_numbers(20)
                print("\nTop 20 most frequent numbers:", ', '.join(map(str, top_numbers)))
                sorted_top_numbers = sorted(top_numbers)
                print("Sorted from 1 to 80:", ', '.join(map(str, sorted_top_numbers)))
            
            elif choice == '2':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                suggested_numbers = analysis.suggest_numbers()
                print("\nSuggested numbers:", ', '.join(map(str, suggested_numbers)))
            
            elif choice == '3':
                draws = collector.fetch_latest_draws()
                if draws:
                    print("\nDraws collected successfully:")
                    for i, draw in enumerate(draws, 1):
                        draw_date, numbers = draw
                        print(f"Draw {i}: Date: {draw_date}, Numbers: {', '.join(map(str, numbers))}")
                        # Save each draw to CSV for ML analysis
                        save_draw_to_csv(draw_date, numbers)
                else:
                    print("\nFailed to fetch draws")
            
            elif choice == '4':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                common_pairs = analysis.find_common_pairs()
                print("\nCommon pairs:", common_pairs)
            
            elif choice == '5':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                consecutive_numbers = analysis.find_consecutive_numbers()
                print("\nConsecutive numbers:", consecutive_numbers)
            
            elif choice == '6':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                range_analysis = analysis.number_range_analysis()
                print("\nNumber range analysis:", range_analysis)
            
            elif choice == '7':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                hot, cold = analysis.hot_and_cold_numbers()
                print("\nHot numbers:", hot)
                print("Cold numbers:", cold)
            
            elif choice == '8':
                if not draws:
                    draws = collector.fetch_latest_draws()
                analysis = DataAnalysis(draws)
                analysis.save_to_excel("lottery_analysis.xlsx")
                print("\nAnalysis results saved to lottery_analysis.xlsx")
            
            elif choice == '9':
                # Check if we have a trained model, if not train it
                check_and_train_model()
                
                print("\nGenerating ML prediction for next draw...")
                predicted_numbers = get_ml_prediction()
                if predicted_numbers:
                    print("\nPredicted numbers for next draw:", sorted(predicted_numbers))
            
            elif choice == '10':
                try:
                    from prediction_evaluator import PredictionEvaluator
                    evaluator = PredictionEvaluator()
                    
                    # Load the most recent prediction
                    predictions_file = 'src/data/processed/predictions.xlsx'
                    if not os.path.exists(predictions_file):
                        print("\nNo predictions found. Please use option 9 first to generate predictions.")
                        continue
                        
                    try:
                        predictions_df = pd.read_excel(predictions_file)
                        if len(predictions_df) == 0:
                            print("\nNo predictions found in the file. Please use option 9 first to generate predictions.")
                            continue
                            
                        latest_prediction = predictions_df.iloc[-10:]['Number'].tolist()
                        
                        # Get actual numbers from user
                        print("\nEnter the actual draw numbers (20 numbers, space-separated):")
                        try:
                            actual_numbers = list(map(int, input().strip().split()))
                            if len(actual_numbers) != 20:
                                print("\nError: Please enter exactly 20 numbers.")
                                continue
                                
                            if not all(1 <= x <= 80 for x in actual_numbers):
                                print("\nError: All numbers must be between 1 and 80.")
                                continue
                                
                            # Compare and save results
                            result = evaluator.save_comparison(latest_prediction, actual_numbers)
                            evaluator.display_results(result)
                            
                        except ValueError:
                            print("\nError: Please enter valid numbers separated by spaces.")
                            continue
                            
                    except Exception as e:
                        print(f"\nError reading predictions file: {e}")
                        
                except Exception as e:
                    print(f"\nError in evaluation: {str(e)}")
                    print("Please try again.")

            elif choice == '11':
                print("\nExiting program...")
                sys.exit(0)
            
            else:
                print("\nInvalid option. Please choose 1-11")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")
            print("Please try again.")

if __name__ == "__main__":
    main()