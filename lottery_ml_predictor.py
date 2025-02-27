import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib
from datetime import datetime
import os

class LotteryMLPredictor:
    def __init__(self, numbers_range=(1, 80), numbers_to_draw=20):
        """
        Initialize the ML predictor for lottery numbers.
        
        Args:
            numbers_range (tuple): The range of possible numbers (min, max)
            numbers_to_draw (int): Number of numbers drawn in each game
        """
        self.numbers_range = numbers_range
        self.numbers_to_draw = numbers_to_draw
        
        # Initialize the models
        self.scaler = StandardScaler()
        self.probabilistic_model = GaussianNB()
        self.pattern_model = MLPClassifier(
            hidden_layer_sizes=(256, 128, 64),  # Three layers for deep learning
            activation='relu',
            solver='adam',
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            verbose=True
        )
        
        # Initialize tracking variables
        self.is_trained = False
        self.training_date = None
        
    def prepare_data(self, historical_data):
        """
        Prepare historical lottery data for training.
        
        Args:
            historical_data (pd.DataFrame): DataFrame with lottery draw history
        
        Returns:
            tuple: (features array, labels array)
        """
        try:
            features = []
            labels = []
            
            # Ensure data is properly sorted by date
            if 'date' in historical_data.columns:
                historical_data = historical_data.sort_values('date')
            
            # Get number columns
            number_cols = [col for col in historical_data.columns if col.startswith('number')]
            
            # Convert all number columns to numeric
            for col in number_cols:
                historical_data[col] = pd.to_numeric(historical_data[col], errors='coerce')
            
            # Remove any rows with invalid data
            historical_data = historical_data.dropna()
            
            # We need at least 6 draws for meaningful patterns (5 for features, 1 for label)
            if len(historical_data) < 6:
                raise ValueError(f"Need at least 6 draws, but only have {len(historical_data)}")
            
            # Create sliding windows of 5 draws each
            for i in range(len(historical_data) - 5):
                window = historical_data.iloc[i:i+5]
                next_draw = historical_data.iloc[i+5]
                
                try:
                    # Create feature vector from window
                    feature_vector = self._create_feature_vector(window[number_cols])
                    features.append(feature_vector)
                    
                    # Create label vector (one-hot encoding for next draw numbers)
                    label = np.zeros(self.numbers_range[1] - self.numbers_range[0] + 1)
                    next_numbers = next_draw[number_cols].values.astype(int)
                    for num in next_numbers:
                        if self.numbers_range[0] <= num <= self.numbers_range[1]:
                            label[num - self.numbers_range[0]] = 1
                    labels.append(label)
                    
                except Exception as e:
                    print(f"Warning: Skipping window due to error: {str(e)}")
                    continue
            
            return np.array(features), np.array(labels)
            
        except Exception as e:
            print(f"Error in prepare_data: {str(e)}")
            raise
        
    def _create_feature_vector(self, window_numbers):
        """
        Create features from a window of previous draws.
        
        Args:
            window_numbers (pd.DataFrame): DataFrame containing previous draws
            
        Returns:
            np.array: Feature vector
        """
        features = []
        
        try:
            # Convert all numbers to integers
            all_numbers = window_numbers.values.astype(int).flatten()
            
            # 1. Frequency features
            number_counts = np.zeros(self.numbers_range[1] - self.numbers_range[0] + 1)
            for num in all_numbers:
                if self.numbers_range[0] <= num <= self.numbers_range[1]:
                    number_counts[num - self.numbers_range[0]] += 1
            features.extend(number_counts / len(window_numbers))
            
            # 2. Statistical features from last draw
            last_draw = window_numbers.iloc[-1].values.astype(int)
            features.extend([
                np.mean(last_draw),  # Average of last drawn numbers
                np.std(last_draw),   # Standard deviation
                np.median(last_draw), # Median
                np.max(last_draw) - np.min(last_draw)  # Range
            ])
            
            # 3. Range-based features
            ranges = [(1,20), (21,40), (41,60), (61,80)]
            for start, end in ranges:
                count = sum(1 for num in last_draw if start <= num <= end)
                features.append(count / len(last_draw))
            
            # 4. Pattern features
            features.extend([
                len(set(last_draw) & set(range(1, 41))),  # Numbers in first half
                len(set(last_draw) & set(range(41, 81))), # Numbers in second half
                sum(1 for i in range(len(last_draw)-1) if last_draw[i+1] - last_draw[i] == 1),  # Consecutive numbers
                len(set(last_draw) & set(range(1, 81, 2))),  # Odd numbers
                len(set(last_draw) & set(range(0, 81, 2)))   # Even numbers
            ])
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error in _create_feature_vector: {str(e)}")
            raise
        
    def train_models(self, X_train, y_train):
        """
        Train both probabilistic and pattern recognition models.
        
        Args:
            X_train (np.array): Training features
            y_train (np.array): Training labels
        """
        try:
            print("Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            print("Training probabilistic model...")
            self.probabilistic_model.fit(X_train_scaled, y_train)
            
            print("Training pattern recognition model...")
            self.pattern_model.fit(X_train_scaled, y_train)
            
            self.is_trained = True
            self.training_date = datetime.now()
            print("Training completed successfully!")
            
        except Exception as e:
            print(f"Error in train_models: {str(e)}")
            raise
        
    def predict(self, recent_draws):
        """
        Generate predictions for next draw.
        
        Args:
            recent_draws (pd.DataFrame): DataFrame containing recent draws
            
        Returns:
            tuple: (predicted numbers list, probabilities array)
        """
        try:
            if not self.is_trained:
                raise ValueError("Models have not been trained yet!")
                
            # Get number columns
            number_cols = [col for col in recent_draws.columns if col.startswith('number')]
            
            # Create feature vector
            feature_vector = self._create_feature_vector(recent_draws[number_cols])
            feature_vector = self.scaler.transform([feature_vector])
            
            # Get predictions from both models
            prob_pred = self.probabilistic_model.predict_proba(feature_vector)[0]
            pattern_pred = self.pattern_model.predict_proba(feature_vector)[0]
            
            # Combine predictions (weighted average)
            combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
            
            # Select top N numbers based on combined predictions
            predicted_indices = np.argsort(combined_pred)[-self.numbers_to_draw:]
            predicted_numbers = sorted(predicted_indices + self.numbers_range[0])
            
            return predicted_numbers, combined_pred
            
        except Exception as e:
            print(f"Error in predict: {str(e)}")
            raise
        
    def save_models(self, path_prefix):
        """
        Save trained models and scaler to files.
        
        Args:
            path_prefix (str): Prefix for saved model files
        """
        try:
            if not self.is_trained:
                raise ValueError("Cannot save untrained models!")
                
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path_prefix), exist_ok=True)
            
            # Save models and scaler
            joblib.dump(self.probabilistic_model, f'{path_prefix}_prob_model.pkl')
            joblib.dump(self.pattern_model, f'{path_prefix}_pattern_model.pkl')
            joblib.dump(self.scaler, f'{path_prefix}_scaler.pkl')
            
            # Save metadata
            metadata = {
                'training_date': self.training_date,
                'numbers_range': self.numbers_range,
                'numbers_to_draw': self.numbers_to_draw
            }
            joblib.dump(metadata, f'{path_prefix}_metadata.pkl')
            
            print(f"Models saved to {path_prefix}_*.pkl")
            
        except Exception as e:
            print(f"Error saving models: {str(e)}")
            raise
        
    def load_models(self, path_prefix):
        """
        Load trained models and scaler from files.
        
        Args:
            path_prefix (str): Prefix for saved model files
        """
        try:
            # Load models and scaler
            self.probabilistic_model = joblib.load(f'{path_prefix}_prob_model.pkl')
            self.pattern_model = joblib.load(f'{path_prefix}_pattern_model.pkl')
            self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')
            
            # Load metadata
            metadata = joblib.load(f'{path_prefix}_metadata.pkl')
            self.training_date = metadata['training_date']
            self.numbers_range = metadata['numbers_range']
            self.numbers_to_draw = metadata['numbers_to_draw']
            
            self.is_trained = True
            print("Models loaded successfully!")
            print(f"Models were trained on: {self.training_date}")
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            raise

    def get_model_info(self):
        """
        Get information about the current model state.
        
        Returns:
            dict: Model information
        """
        return {
            'is_trained': self.is_trained,
            'training_date': self.training_date,
            'numbers_range': self.numbers_range,
            'numbers_to_draw': self.numbers_to_draw
        }