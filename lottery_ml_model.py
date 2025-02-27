import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

class LotteryMLPredictor:
    def __init__(self, numbers_range=(1, 80), numbers_to_draw=20):
        self.numbers_range = numbers_range
        self.numbers_to_draw = numbers_to_draw
        self.models = {num: RandomForestClassifier() for num in range(numbers_range[0], numbers_range[1] + 1)}

    def prepare_data(self, historical_data):
        X = historical_data.drop(columns=['date'])
        y = {num: (X == num).astype(int).sum(axis=1) for num in range(self.numbers_range[0], self.numbers_range[1] + 1)}
        return X, y

    def train_models(self, X, y):
        for num, model in self.models.items():
            y_num = y[:, num - 1]  # Select the corresponding column for the target number
            model.fit(X, y_num)

    def predict(self, recent_draws):
        probabilities = {num: model.predict_proba(recent_draws)[:, 1] for num, model in self.models.items()}
        avg_probabilities = {num: np.mean(probs) for num, probs in probabilities.items()}
        sorted_probs = sorted(avg_probabilities.items(), key=lambda item: item[1], reverse=True)
        predicted_numbers = [num for num, prob in sorted_probs[:self.numbers_to_draw]]
        probabilities = [prob for num, prob in sorted_probs[:self.numbers_to_draw]]
        return predicted_numbers, probabilities

    def save_models(self, model_path):
        for num, model in self.models.items():
            model_file = f"{model_path}_{num}_model.pkl"
            print(f"Saving model to {model_file}")
            with open(model_file, 'wb') as f:
                pickle.dump(model, f)

    def load_models(self, model_path):
        for num in self.models.keys():
            model_file = f"{model_path}_{num}_model.pkl"
            print(f"Loading model from {model_file}")
            if os.path.exists(model_file):
                with open(model_file, 'rb') as f:
                    self.models[num] = pickle.load(f)
            else:
                raise FileNotFoundError(f"Model file {model_file} not found.")