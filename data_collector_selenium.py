from selenium import webdriver
from selenium.webdriver.edge.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from datetime import datetime
import pytz
import time

class KinoDataCollector:
    def __init__(self, user_login="777MNM", debug=True):
        self.debug = debug
        self.user_login = user_login
        self.greek_tz = pytz.timezone('Europe/Athens')
        self.utc_tz = pytz.timezone('UTC')
        self.update_timestamps()
        self.base_url = "https://lotostats.ro/toate-rezultatele-grecia-kino-20-80"

    def update_timestamps(self):
        current_utc = datetime.now(self.utc_tz)
        self.current_utc_str = current_utc.strftime('%Y-%m-%d %H:%M:%S')

    def extract_numbers(self, number_cells):
        numbers = []
        for cell in number_cells:
            try:
                num = int(cell.text.strip())
                if 1 <= num <= 80:
                    numbers.append(num)
            except ValueError:
                continue
        return numbers

    def fetch_latest_draws(self, num_draws=24, delay=1):
        driver = None
        try:
            print(f"\nFetching {num_draws} latest draws...")

            # Setup Edge options
            edge_options = Options()
            edge_options.add_argument('--headless')
            edge_options.add_argument('--disable-gpu')
            edge_options.add_argument('--no-sandbox')
            edge_options.add_argument('--ignore-certificate-errors')

            # Initialize Edge driver
            driver = webdriver.Edge(options=edge_options)
            print("Browser initialized")

            # Load the page
            driver.get(self.base_url)
            print("Page loaded")

            # Wait for the table to be present
            wait = WebDriverWait(driver, 40)
            wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "#all_results > tbody")))
            print("Results table found")

            # Give extra time for JavaScript to load
            time.sleep(10)

            draws = []

            # Find all draw rows in the results table
            rows = driver.find_elements(By.CSS_SELECTOR, "#all_results > tbody > tr")
            print(f"Found {len(rows)} rows in the results table")

            for row in rows:
                try:
                    # Get the date cell
                    date_cell = row.find_element(By.CSS_SELECTOR, "td:first-child")
                    draw_date = date_cell.text.strip()

                    # Get all number cells in the row
                    number_cells = row.find_elements(By.CSS_SELECTOR, "td > div.nrr")
                    numbers = self.extract_numbers(number_cells)

                    if len(numbers) >= 20:
                        draws.append((draw_date, sorted(numbers)))
                        if len(draws) >= num_draws:
                            break

                    # Introduce a delay between processing each draw
                    time.sleep(delay)

                except Exception as e:
                    print(f"Error processing row: {str(e)}")
                    continue

            if draws:
                print(f"\nSuccessfully collected {len(draws)} draws")
                return draws
            else:
                print("\nNo draws found. Debug information:")
                table = driver.find_element(By.CSS_SELECTOR, "#all_results > tbody")
                print("\nTable HTML:")
                print(table.get_attribute('innerHTML')[:1000])
                return []

        except Exception as e:
            print(f"Error: {str(e)}")
            return []

        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass