import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import joblib

class LotteryPredictor:
    def __init__(self, numbers_range=(1, 80), numbers_to_draw=20):
        self.numbers_range = numbers_range
        self.numbers_to_draw = numbers_to_draw
        self.scaler = StandardScaler()
        self.probabilistic_model = GaussianNB()
        # Using MLPClassifier instead of TensorFlow
        self.pattern_model = MLPClassifier(
            hidden_layer_sizes=(256, 128),
            activation='relu',
            solver='adam',
            max_iter=500
        )
        
    def prepare_data(self, historical_data):
        """
        Prepare historical lottery data for training
        
        Args:
            historical_data (pd.DataFrame): DataFrame with columns ['draw_date', 'numbers']
                where 'numbers' is a list of drawn numbers
        """
        # Sort data by date
        historical_data = historical_data.sort_values('draw_date')
        
        # Create features for each number
        features = []
        labels = []
        
        for i in range(len(historical_data) - 5):  # Use 5 previous draws as features
            window = historical_data.iloc[i:i+5]
            next_draw = historical_data.iloc[i+5]
            
            # Feature engineering
            feature_vector = self._create_feature_vector(window)
            features.append(feature_vector)
            
            # Create label (1 for drawn numbers, 0 for others)
            label = np.zeros(self.numbers_range[1] - self.numbers_range[0] + 1)
            label[next_draw['numbers'] - self.numbers_range[0]] = 1
            labels.append(label)
            
        return np.array(features), np.array(labels)
    
    def _create_feature_vector(self, window):
        """Create feature vector from window of previous draws"""
        features = []
        
        # Frequency features
        number_counts = np.zeros(self.numbers_range[1] - self.numbers_range[0] + 1)
        for _, row in window.iterrows():
            for num in row['numbers']:
                number_counts[num - self.numbers_range[0]] += 1
        features.extend(number_counts / len(window))
        
        # Pattern features
        last_draw = window.iloc[-1]['numbers']
        features.extend([
            np.mean(last_draw),  # Average of last drawn numbers
            np.std(last_draw),   # Standard deviation of last drawn numbers
            len(set(last_draw) & set(range(1, 41))),  # Numbers in first half
            len(set(last_draw) & set(range(41, 81)))  # Numbers in second half
        ])
        
        return np.array(features)
    
    def train_models(self, X_train, y_train):
        """Train both probabilistic and pattern recognition models"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train probabilistic model
        self.probabilistic_model.fit(X_train_scaled, y_train)
        
        # Train pattern recognition model
        self.pattern_model.fit(X_train_scaled, y_train)
    
    def predict(self, recent_draws):
        """
        Generate predictions for next draw
        
        Args:
            recent_draws: Last 5 draws to use for prediction
        
        Returns:
            list: Predicted numbers for next draw
        """
        # Create feature vector from recent draws
        feature_vector = self._create_feature_vector(recent_draws)
        feature_vector = self.scaler.transform([feature_vector])
        
        # Get predictions from both models
        prob_pred = self.probabilistic_model.predict_proba(feature_vector)[0]
        pattern_pred = self.pattern_model.predict_proba(feature_vector)[0]
        
        # Combine predictions (weighted average)
        combined_pred = 0.4 * prob_pred + 0.6 * pattern_pred
        
        # Select top N numbers based on combined predictions
        predicted_numbers = np.argsort(combined_pred)[-self.numbers_to_draw:]
        
        # Convert back to actual number range
        return sorted(predicted_numbers + self.numbers_range[0])
    
    def save_models(self, path_prefix):
        """Save trained models to files"""
        joblib.dump(self.probabilistic_model, f'{path_prefix}_prob_model.pkl')
        joblib.dump(self.pattern_model, f'{path_prefix}_pattern_model.pkl')
        joblib.dump(self.scaler, f'{path_prefix}_scaler.pkl')
    
    def load_models(self, path_prefix):
        """Load trained models from files"""
        self.probabilistic_model = joblib.load(f'{path_prefix}_prob_model.pkl')
        self.pattern_model = joblib.load(f'{path_prefix}_pattern_model.pkl')
        self.scaler = joblib.load(f'{path_prefix}_scaler.pkl')