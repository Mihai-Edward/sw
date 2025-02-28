import pandas as pd
from datetime import datetime
import numpy as np
import os

class PredictionEvaluator:
    def __init__(self):
        self.predictions_file = 'data/processed/predictions.xlsx'
        self.historical_file = 'historical_draws.csv'
        self.results_dir = 'data/processed'
        self.results_file = os.path.join(self.results_dir, 'prediction_results.xlsx')
        
        # Ensure directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    def compare_prediction_with_actual(self, predicted_numbers, actual_numbers):
        """
        Compare predicted numbers with actual draw numbers
        
        Args:
            predicted_numbers (list): List of predicted numbers
            actual_numbers (list): List of actual draw numbers
            
        Returns:
            dict: Dictionary containing comparison results
        """
        predicted_set = set(predicted_numbers)
        actual_set = set(actual_numbers)
        
        correct_numbers = predicted_set.intersection(actual_set)
        accuracy = len(correct_numbers) / 20  # Since we draw 20 numbers
        
        return {
            'correct_numbers': sorted(list(correct_numbers)),
            'num_correct': len(correct_numbers),
            'accuracy': accuracy,
            'predicted_numbers': sorted(list(predicted_numbers)),
            'actual_numbers': sorted(list(actual_numbers))
        }

    def save_comparison(self, predicted_numbers, actual_numbers, draw_date=None):
        """
        Save prediction comparison to Excel
        
        Args:
            predicted_numbers (list): List of predicted numbers
            actual_numbers (list): List of actual draw numbers
            draw_date (str, optional): Date of the draw. Defaults to current datetime.
        
        Returns:
            dict: Comparison results
        """
        if draw_date is None:
            draw_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
        result = self.compare_prediction_with_actual(predicted_numbers, actual_numbers)
        
        data = {
            'Date': [draw_date],
            'Predicted_Numbers': [str(result['predicted_numbers'])],
            'Actual_Numbers': [str(result['actual_numbers'])],
            'Correct_Numbers': [str(result['correct_numbers'])],
            'Number_Correct': [result['num_correct']],
            'Accuracy': [f"{result['accuracy']*100:.2f}%"]
        }
        
        df = pd.DataFrame(data)
        
        try:
            # Try to load existing file
            if os.path.exists(self.results_file):
                existing_df = pd.read_excel(self.results_file)
                df = pd.concat([existing_df, df], ignore_index=True)
        except Exception as e:
            print(f"Warning: Could not load existing results file: {e}")
        
        # Save to Excel
        try:
            df.to_excel(self.results_file, index=False)
            print(f"\nResults saved to {self.results_file}")
        except Exception as e:
            print(f"Warning: Could not save results: {e}")
        
        return result

    def get_performance_stats(self):
        """
        Calculate overall prediction performance statistics
        
        Returns:
            dict: Dictionary containing performance statistics
        """
        try:
            if not os.path.exists(self.results_file):
                return None
                
            df = pd.read_excel(self.results_file)
            if len(df) == 0:
                return None
                
            stats = {
                'total_predictions': len(df),
                'average_correct': df['Number_Correct'].mean(),
                'best_prediction': df['Number_Correct'].max(),
                'worst_prediction': df['Number_Correct'].min(),
                'average_accuracy': df['Number_Correct'].mean() / 20 * 100
            }
            
            return stats
            
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            return None

    def display_results(self, result):
        """
        Display comparison results in a formatted way
        
        Args:
            result (dict): Comparison results from compare_prediction_with_actual
        """
        print("\n=== Prediction Evaluation Results ===")
        print(f"Numbers correctly predicted: {result['num_correct']} out of 20")
        print(f"Accuracy: {result['accuracy']*100:.1f}%")
        print("\nCorrect numbers:")
        print(', '.join(map(str, result['correct_numbers'])))
        print("\nPredicted numbers:")
        print(', '.join(map(str, result['predicted_numbers'])))
        print("\nActual numbers:")
        print(', '.join(map(str, result['actual_numbers'])))
        
        # Display overall statistics
        stats = self.get_performance_stats()
        if stats:
            print("\n=== Overall Performance ===")
            print(f"Total predictions evaluated: {stats['total_predictions']}")
            print(f"Average correct numbers: {stats['average_correct']:.1f}")
            print(f"Best prediction: {stats['best_prediction']} correct numbers")
            print(f"Worst prediction: {stats['worst_prediction']} correct numbers")
            print(f"Average accuracy: {stats['average_accuracy']:.1f}%")

if __name__ == "__main__":
    # Example usage
    evaluator = PredictionEvaluator()
    
    # Example data
    predicted = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    actual = [1, 2, 3, 4, 5, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
    
    # Test comparison
    result = evaluator.save_comparison(predicted, actual)
    evaluator.display_results(result)