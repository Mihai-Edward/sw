from data_collector_selenium import KinoDataCollector
from data_analysis import DataAnalysis
import sys

def main():
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
        print("9. Exit")
        print("==========================\n")

        try:
            choice = input("Choose an option (1-9): ")

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
                sys.exit(0)
            else:
                print("\nInvalid option. Please choose 1-9")

        except Exception as e:
            print(f"\nAn error occurred: {str(e)}")

if __name__ == "__main__":
    main()