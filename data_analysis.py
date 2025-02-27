from collections import Counter
from itertools import combinations
import pandas as pd
import os

class DataAnalysis:
    def __init__(self, draws):
        self.draws = draws

    def count_frequency(self):
        all_numbers = [number for draw in self.draws for number in draw[1]]
        frequency = Counter(all_numbers)
        return frequency

    def get_top_numbers(self, top_n=20):
        frequency = self.count_frequency()
        most_common_numbers = [number for number, count in frequency.most_common(top_n)]
        return most_common_numbers

    def suggest_numbers(self, top_n=20):
        frequency = self.count_frequency()
        most_common_numbers = [number for number, count in frequency.most_common(top_n)]
        return most_common_numbers

    def find_common_pairs(self, top_n=10):
        pairs = Counter()
        for draw in self.draws:
            numbers = draw[1]
            pairs.update(combinations(numbers, 2))
        return pairs.most_common(top_n)

    def find_consecutive_numbers(self, top_n=10):
        consecutive_pairs = Counter()
        for draw in self.draws:
            numbers = sorted(draw[1])
            for i in range(len(numbers) - 1):
                if numbers[i] + 1 == numbers[i + 1]:
                    consecutive_pairs.update([(numbers[i], numbers[i + 1])])
        return consecutive_pairs.most_common(top_n)

    def number_range_analysis(self):
        ranges = {
            '1-20': 0,
            '21-40': 0,
            '41-60': 0,
            '61-80': 0
        }
        for draw in self.draws:
            for number in draw[1]:
                if 1 <= number <= 20:
                    ranges['1-20'] += 1
                elif 21 <= number <= 40:
                    ranges['21-40'] += 1
                elif 41 <= number <= 60:
                    ranges['41-60'] += 1
                elif 61 <= number <= 80:
                    ranges['61-80'] += 1
        return ranges

    def hot_and_cold_numbers(self, top_n=10):
        frequency = self.count_frequency()
        hot_numbers = frequency.most_common(top_n)
        cold_numbers = frequency.most_common()[:-top_n-1:-1]
        return hot_numbers, cold_numbers

    def save_to_excel(self, filename="lottery_analysis.xlsx"):
        # Frequency Analysis
        frequency = self.count_frequency()
        frequency_df = pd.DataFrame(frequency.items(), columns=["Number", "Frequency"])

        # Common Pairs Analysis
        common_pairs = self.find_common_pairs()
        common_pairs_df = pd.DataFrame(common_pairs, columns=["Pair", "Frequency"])
        common_pairs_df["Number 1"] = common_pairs_df["Pair"].apply(lambda x: x[0])
        common_pairs_df["Number 2"] = common_pairs_df["Pair"].apply(lambda x: x[1])
        common_pairs_df = common_pairs_df.drop(columns=["Pair"])

        # Consecutive Numbers Analysis
        consecutive_numbers = self.find_consecutive_numbers()
        consecutive_numbers_df = pd.DataFrame(consecutive_numbers, columns=["Pair", "Frequency"])
        consecutive_numbers_df["Number 1"] = consecutive_numbers_df["Pair"].apply(lambda x: x[0])
        consecutive_numbers_df["Number 2"] = consecutive_numbers_df["Pair"].apply(lambda x: x[1])
        consecutive_numbers_df = consecutive_numbers_df.drop(columns=["Pair"])

        # Number Range Analysis
        range_analysis = self.number_range_analysis()
        range_analysis_df = pd.DataFrame(range_analysis.items(), columns=["Range", "Count"])

        # Hot and Cold Numbers Analysis
        hot_numbers, cold_numbers = self.hot_and_cold_numbers()
        hot_numbers_df = pd.DataFrame(hot_numbers, columns=["Number", "Frequency"])
        cold_numbers_df = pd.DataFrame(cold_numbers, columns=["Number", "Frequency"])

        # Ensure the processed data directory exists
        processed_data_dir = "data/processed"
        os.makedirs(processed_data_dir, exist_ok=True)

        # Save to Excel file in the processed data directory
        file_path = os.path.join(processed_data_dir, filename)
        with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
            frequency_df.to_excel(writer, sheet_name='Frequency', index=False)
            common_pairs_df.to_excel(writer, sheet_name='Common Pairs', index=False)
            consecutive_numbers_df.to_excel(writer, sheet_name='Consecutive Numbers', index=False)
            range_analysis_df.to_excel(writer, sheet_name='Number Range', index=False)
            hot_numbers_df.to_excel(writer, sheet_name='Hot Numbers', index=False)
            cold_numbers_df.to_excel(writer, sheet_name='Cold Numbers', index=False)

        print(f"\nAnalysis results saved to {file_path}")

if __name__ == "__main__":
    # Example usage
    example_draws = [
        ("20:15 26-02-2025", [1, 2, 2, 9, 12, 14, 17, 25, 26, 30, 38, 44, 54, 57, 58, 61, 65, 71, 72, 76, 79]),
        ("20:10 26-02-2025", [4, 5, 7, 7, 9, 18, 24, 27, 29, 34, 40, 45, 48, 52, 55, 57, 70, 71, 72, 74, 77]),
        # Add more draws as needed
    ]
    analysis = DataAnalysis(example_draws)
    print("Frequency of numbers:", analysis.count_frequency())
    print("Top 20 most frequent numbers:", ', '.join(map(str, analysis.get_top_numbers(20))))
    analysis.save_to_excel("lottery_analysis.xlsx")