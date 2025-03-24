import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import urllib.parse
import urllib.request
import logging
import time  # For rate limiting
from urllib.robotparser import RobotFileParser  # For robots.txt compliance
from functools import lru_cache  # For caching robots.txt
from time import sleep  # For retry mechanism
import argparse  # For command-line arguments
import json  # For configuration file support
import PyPDF2  # For extracting text from PDF files
import pdfplumber  # For extracting tables from PDF files
from PIL import Image  # For image processing
import pytesseract  # For OCR (Optical Character Recognition)
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os  # For environment variable management
import threading  # For running background tasks
import schedule  # For periodic tasks
from concurrent.futures import ThreadPoolExecutor, as_completed  # For parallel execution

# Parse command-line arguments for configuration
parser = argparse.ArgumentParser(description="Financial Analyzer Configuration")
parser.add_argument("--config_file", type=str, help="Path to a JSON configuration file")
parser.add_argument("--user_agent", type=str, default="FinancialAnalyzerBot/1.0 (+https://github.com/knigh/FinancialAnalysisApp)", help="User-Agent for HTTP requests")
parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
parser.add_argument("--retry_delay", type=int, default=5, help="Initial delay (in seconds) between retries")
args = parser.parse_args()

# Load configuration from file if provided, or use a default configuration
DEFAULT_CONFIG = {
    "user_agent": "FinancialAnalyzerBot/1.0 (+https://github.com/knigh/FinancialAnalysisApp)",
    "max_retries": 3,
    "retry_delay": 5,
    "robots_timeout": 5,
    "scraping_timeout": 10,
    "news_api_key": None,  # Add default value for news API key
    "alpha_vantage_api_key": None,  # Add default value for Alpha Vantage API
    "enable_selenium": False,  # Add default value for selenium
    "news_source": "NewsAPI",  # Default news source.
    "finnhub_api_key": None,  # Add default value for Finnhub API Key
}

if args.config_file:
    try:
        with open(args.config_file, 'r') as config_file:
            config = json.load(config_file)
            if not isinstance(config, dict):
                raise ValueError("Configuration file must contain a JSON object.")

            USER_AGENT = config.get("user_agent", DEFAULT_CONFIG["user_agent"])
            MAX_RETRIES = config.get("max_retries", DEFAULT_CONFIG["max_retries"])
            RETRY_DELAY = config.get("retry_delay", DEFAULT_CONFIG["retry_delay"])
            ROBOTS_TIMEOUT = config.get("robots_timeout", DEFAULT_CONFIG["robots_timeout"])
            SCRAPING_TIMEOUT = config.get("scraping_timeout", DEFAULT_CONFIG["scraping_timeout"])
            NEWS_API_KEY = config.get("news_api_key", DEFAULT_CONFIG["news_api_key"])  # Load news API key
            ALPHA_VANTAGE_API_KEY = config.get("alpha_vantage_api_key", DEFAULT_CONFIG["alpha_vantage_api_key"])  # Load Alpha Vantage API Key
            ENABLE_SELENIUM = config.get("enable_selenium", DEFAULT_CONFIG["enable_selenium"])
            NEWS_SOURCE = config.get("news_source", DEFAULT_CONFIG["news_source"])
            FINNHUB_API_KEY = config.get("finnhub_api_key", DEFAULT_CONFIG["finnhub_api_key"])
            if not isinstance(MAX_RETRIES, int) or MAX_RETRIES <= 0:
                raise ValueError("max_retries must be a positive integer.")
            if not isinstance(RETRY_DELAY, (int, float)) or RETRY_DELAY < 0:
                raise ValueError("retry_delay must be a non-negative number.")
            if not isinstance(ROBOTS_TIMEOUT, (int, float)) or ROBOTS_TIMEOUT <= 0:
                raise ValueError("robots_timeout must be a positive number.")
            if not isinstance(SCRAPING_TIMEOUT, (int, float)) or SCRAPING_TIMEOUT <= 0:
                raise ValueError("scraping_timeout must be a positive number.")
            if not isinstance(ENABLE_SELENIUM, bool):
                raise ValueError("enable_selenium must be a boolean.")
            if NEWS_SOURCE not in ["NewsAPI", "Finnhub"]:
                raise ValueError("news_source must be either 'NewsAPI' or 'Finnhub'.")
    except Exception as e:
        logging.warning(f"Failed to load or validate configuration file {args.config_file}: {e}")
        USER_AGENT = DEFAULT_CONFIG["user_agent"]
        MAX_RETRIES = DEFAULT_CONFIG["max_retries"]
        RETRY_DELAY = DEFAULT_CONFIG["retry_delay"]
        ROBOTS_TIMEOUT = DEFAULT_CONFIG["robots_timeout"]
        SCRAPING_TIMEOUT = DEFAULT_CONFIG["scraping_timeout"]
        NEWS_API_KEY = DEFAULT_CONFIG["news_api_key"]  # Use default
        ALPHA_VANTAGE_API_KEY = DEFAULT_CONFIG["alpha_vantage_api_key"]
        ENABLE_SELENIUM = DEFAULT_CONFIG["enable_selenium"]
        NEWS_SOURCE = DEFAULT_CONFIG["news_source"]
        FINNHUB_API_KEY = DEFAULT_CONFIG["finnhub_api_key"]

else:
    USER_AGENT = DEFAULT_CONFIG["user_agent"]
    MAX_RETRIES = DEFAULT_CONFIG["max_retries"]
    RETRY_DELAY = DEFAULT_CONFIG["retry_delay"]
    ROBOTS_TIMEOUT = DEFAULT_CONFIG["robots_timeout"]
    SCRAPING_TIMEOUT = DEFAULT_CONFIG["scraping_timeout"]
    NEWS_API_KEY = DEFAULT_CONFIG["news_api_key"]  # Use default
    ALPHA_VANTAGE_API_KEY = DEFAULT_CONFIG["alpha_vantage_api_key"]
    ENABLE_SELENIUM = DEFAULT_CONFIG["enable_selenium"]
    NEWS_SOURCE = DEFAULT_CONFIG["news_source"]
    FINNHUB_API_KEY = DEFAULT_CONFIG["finnhub_api_key"]

# Validate command-line arguments
if MAX_RETRIES <= 0:
    raise ValueError("MAX_RETRIES must be a positive integer.")
if RETRY_DELAY < 0:
    raise ValueError("RETRY_DELAY must be a non-negative number.")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FinancialAnalyzer:
    """
    A class for analyzing financial data of a company.
    """

    def __init__(self, company_name, website_url=None, financial_data=None):
        """
        Initializes the FinancialAnalyzer.

        Args:
            company_name (str): The name of the company.
            website_url (str, optional): The URL of the company's website. Defaults to None.
            financial_data (pd.DataFrame, optional): Pre-loaded financial data. Defaults to None.
        """
        self.company_name = company_name
        self.website_url = website_url
        self.financial_data = financial_data

        if website_url and financial_data is None:
            self.financial_data = self.scrape_financial_data()

    @lru_cache(maxsize=10)
    def fetch_robots_txt(self, robots_url, timeout=None):
        """
        Fetches and parses the robots.txt file, with caching for performance.

        Args:
            robots_url (str): The URL of the robots.txt file.
            timeout (int): Timeout for fetching the robots.txt file. Defaults to ROBOTS_TIMEOUT.

        Returns:
            RobotFileParser: A parsed RobotFileParser object.
        """
        timeout = timeout or ROBOTS_TIMEOUT
        rp = RobotFileParser()
        rp.set_url(robots_url)
        try:
            rp.read()
            return rp
        except Exception as e:
            st.warning(f"Could not fetch or parse robots.txt at {robots_url} (timeout={timeout}): {e}")
            logging.warning(f"Could not fetch or parse robots.txt at {robots_url} (timeout={timeout}): {e}")
            return None

    def scrape_financial_data(self):
        """
        Scrapes financial data from the company's website.

        Returns:
            pd.DataFrame: A DataFrame containing the scraped financial data, or None if an error occurs.
        """
        if not self.website_url:
            st.error("Website URL not provided.")
            logging.error("Website URL not provided.")
            return None

        # Check robots.txt compliance
        parsed_url = urllib.parse.urlparse(self.website_url)
        robots_url = f"{parsed_url.scheme}://{parsed_url.netloc}/robots.txt"
        try:
            rp = self.fetch_robots_txt(robots_url, timeout=ROBOTS_TIMEOUT)
            if rp and not rp.can_fetch("*", self.website_url):
                st.error(f"Scraping is not allowed by robots.txt at {robots_url} for URL: {self.website_url}")
                logging.error(f"Scraping is not allowed by robots.txt at {robots_url} for URL: {self.website_url}")
                return None
        except Exception as e:
            st.warning(f"Error while parsing robots.txt at {robots_url} for URL: {self.website_url}: {e}")
            logging.warning(f"Error while parsing robots.txt at {robots_url} for URL: {self.website_url}: {e}")

        retries = 0
        delay = RETRY_DELAY
        while retries < MAX_RETRIES:
            try:
                # Dynamic rate limiting: Adjust delay based on response time
                start_time = time.time()
                headers = {"User-Agent": USER_AGENT}
                response = requests.get(self.website_url, headers=headers, timeout=SCRAPING_TIMEOUT)
                response.raise_for_status()
                elapsed_time = time.time() - start_time
                time.sleep(max(1, elapsed_time))  # Ensure at least 1-second delay

                soup = BeautifulSoup(response.content, "html.parser")

                tables = soup.find_all("table")
                for table in tables:
                    if "balance sheet" in table.text.lower() or "income statement" in table.text.lower() or "cash flow" in table.text.lower():
                        data = []
                        rows = table.find_all("tr")
                        for row in rows:
                            cols = row.find_all(["td", "th"])  # Handle both td and th
                            cols = [ele.text.strip() for ele in cols]
                            data.append(cols)

                        df = pd.DataFrame(data)

                        if not df.empty:
                            # Ensure the first row is suitable for headers
                            if df.iloc[0].isnull().any():
                                st.error(f"Table headers are missing or invalid in URL: {self.website_url}")
                                logging.error(f"Table headers are missing or invalid in URL: {self.website_url}")
                                return None

                            df.columns = df.iloc[0]
                            df = df[1:]
                            df = df.dropna(axis=1, how='all')
                            df = df.dropna(axis=0, how='all')

                            for col in df.columns:
                                try:
                                    df[col] = pd.to_numeric(df[col].str.replace(r'[$,()]', '', regex=True), errors='coerce')
                                except (ValueError, AttributeError):
                                    pass

                            # Validate for negative values
                            if (df.select_dtypes(include=['number']) < 0).any().any():
                                st.warning(f"Negative values detected in financial data from URL: {self.website_url}")
                                logging.warning(f"Negative values detected in financial data from URL: {self.website_url}")

                            return df

                st.error(f"Financial data table not found in URL: {self.website_url}")
                logging.error(f"Financial data table not found in URL: {self.website_url}")
                return None

            except requests.exceptions.Timeout:
                st.error(f"Request to {self.website_url} timed out.")
                logging.error(f"Request to {self.website_url} timed out.")
                return None
            except requests.exceptions.HTTPError as e:
                if e.response.status_code == 429:
                    retries += 1
                    st.warning(f"Too many requests to {self.website_url}. Retrying in {delay} seconds... (Attempt {retries}/{MAX_RETRIES})")
                    logging.warning(f"HTTP 429 Too Many Requests for URL {self.website_url}: Retrying in {delay} seconds... (Attempt {retries}/{MAX_RETRIES})")
                    sleep(delay)
                    delay *= 2  # Exponential backoff
                else:
                    st.error(f"HTTP error occurred while accessing {self.website_url}: {e}")
                    logging.error(f"HTTP error occurred while accessing {self.website_url}: {e}")
                    return None
            except requests.exceptions.RequestException as e:
                st.error(f"Error during scraping from URL {self.website_url}: {e}")
                logging.error(f"Error during scraping from URL {self.website_url}: {e}")
                return None
            except Exception as e:
                st.error(f"An unexpected error occurred while scraping from URL {self.website_url}: {e}")
                logging.error(f"An unexpected error occurred while scraping from URL {self.website_url}: {e}")
                return None

        st.error(f"Failed to scrape data from {self.website_url} after {MAX_RETRIES} retries.")
        logging.error(f"Failed to scrape data from {self.website_url} after {MAX_RETRIES} retries.")
        return None

    def calculate_ratios(self):
        """
        Calculates financial ratios based on the scraped or uploaded data.

        Returns:
            dict: A dictionary containing the calculated ratios, or None if an error occurs.
        """
        if self.financial_data is None or not isinstance(self.financial_data, pd.DataFrame):
            st.error("Financial data is not available or is in the wrong format.")
            logging.error("Financial data is not available or is in the wrong format.")
            return None

        try:
            data = self.financial_data

            def find_column(patterns):
                """
                Finds a column in the DataFrame that matches any of the given patterns.

                Args:
                    patterns (list): A list of regex patterns to search for.

                Returns:
                    str: The name of the matching column, or None if no match is found.
                """
                for pattern in patterns:
                    match = next((col for col in data.columns if re.search(pattern, col, re.IGNORECASE)), None)
                    if match:
                        return match
                return None

            # Define patterns for financial terms
            revenue_patterns = [r'revenue', r'sales', r'total revenue', r'net sales']
            net_income_patterns = [r'net income', r'profit', r'net profit', r'earnings']
            total_assets_patterns = [r'total assets', r'assets']
            total_liabilities_patterns = [r'total liabilities', r'liabilities']
            current_assets_patterns = [r'current assets', r'short-term assets']
            current_liabilities_patterns = [r'current liabilities', r'short-term liabilities']
            total_equity_patterns = [r'total equity', r'shareholders equity', r'owners equity']
            cash_patterns = [r'cash', r'cash equivalents', r'cash and cash equivalents']
            inventory_patterns = [r'inventory', r'stock']
            cogs_patterns = [r'cost of goods sold', r'cogs', r'cost of sales']

            # Identify required columns
            revenue_col = find_column(revenue_patterns)
            net_income_col = find_column(net_income_patterns)
            total_assets_col = find_column(total_assets_patterns)
            total_liabilities_col = find_column(total_liabilities_patterns)
            current_assets_col = find_column(current_assets_patterns)
            current_liabilities_col = find_column(current_liabilities_patterns)
            total_equity_col = find_column(total_equity_patterns)
            cash_col = find_column(cash_patterns)
            inventory_col = find_column(inventory_patterns)
            cogs_col = find_column(cogs_patterns)

            # Check for missing columns
            missing_columns = []
            if not revenue_col:
                missing_columns.append("Revenue")
            if not net_income_col:
                missing_columns.append("Net Income")
            if not total_assets_col:
                missing_columns.append("Total Assets")
            if not total_liabilities_col:
                missing_columns.append("Total Liabilities")
            if not current_assets_col:
                missing_columns.append("Current Assets")
            if not current_liabilities_col:
                missing_columns.append("Current Liabilities")
            if not total_equity_col:
                missing_columns.append("Total Equity")
            if not cash_col:
                missing_columns.append("Cash")
            if not inventory_col:
                missing_columns.append("Inventory")
            if not cogs_col:
                missing_columns.append("Cost of Goods Sold (COGS)")

            if missing_columns:
                st.error(f"The following required columns are missing: {', '.join(missing_columns)}")
                logging.error(f"The following required columns are missing: {', '.join(missing_columns)}")
                return None

            # Calculate financial ratios
            ratios = {}

            def format_number(value, currency="USD"):
                """
                Formats a number with commas, two decimal places, and a currency type.

                Args:
                    value (float): The number to format.
                    currency (str): The currency type (e.g., "USD", "EUR").

                Returns:
                    str: The formatted number as a string with currency.
                """
                return f"{currency} {value:,.2f}" if value is not None else "N/A"

            if revenue_col and net_income_col:
                revenue = pd.to_numeric(data[revenue_col], errors='coerce').iloc[-1]
                net_income = pd.to_numeric(data[net_income_col], errors='coerce').iloc[-1]
                ratios["Profit Margin"] = format_number(net_income / revenue * 100) if revenue != 0 else "N/A"

            if total_assets_col and total_liabilities_col:
                total_assets = pd.to_numeric(data[total_assets_col], errors='coerce').iloc[-1]
                total_liabilities = pd.to_numeric(data[total_liabilities_col], errors='coerce').iloc[-1]
                ratios["Debt-to-Asset Ratio"] = format_number(total_liabilities / total_assets * 100) if total_assets != 0 else "N/A"

            if current_assets_col and current_liabilities_col:
                current_assets = pd.to_numeric(data[current_assets_col], errors='coerce').iloc[-1]
                current_liabilities = pd.to_numeric(data[current_liabilities_col], errors='coerce').iloc[-1]
                ratios["Current Ratio"] = format_number(current_assets / current_liabilities) if current_liabilities != 0 else "N/A"

            if total_equity_col and total_liabilities_col:
                total_equity = pd.to_numeric(data[total_equity_col], errors='coerce').iloc[-1]
                ratios["Debt-to-Equity Ratio"] = format_number(total_liabilities / total_equity) if total_equity != 0 else "N/A"

            if cash_col and current_liabilities_col:
                cash = pd.to_numeric(data[cash_col], errors='coerce').iloc[-1]
                ratios["Cash Ratio"] = format_number(cash / current_liabilities) if current_liabilities != 0 else "N/A"

            if inventory_col and cogs_col:
                inventory = pd.to_numeric(data[inventory_col], errors='coerce').iloc[-1]
                cogs = pd.to_numeric(data[cogs_col], errors='coerce').iloc[-1]
                ratios["Inventory Turnover"] = format_number(cogs / inventory) if inventory != 0 else "N/A"

            if revenue_col and total_assets_col:
                ratios["Asset Turnover Ratio"] = format_number(revenue / total_assets) if total_assets != 0 else "N/A"

            if net_income_col and total_assets_col:
                ratios["Return on Assets (ROA)"] = format_number(net_income / total_assets * 100) if total_assets != 0 else "N/A"

            if net_income_col and total_equity_col:
                ratios["Return on Equity (ROE)"] = format_number(net_income / total_equity * 100) if total_equity != 0 else "N/A"

            # Validate for negative values in key columns
            key_columns = [revenue_col, net_income_col, total_assets_col, total_liabilities_col]
            for col in key_columns:
                if col and (data[col].astype(float) < 0).any():
                    st.warning(f"Negative values detected in column '{col}' for financial ratios.")
                    logging.warning(f"Negative values detected in column '{col}' for financial ratios.")

            return ratios

        except KeyError as e:
            st.error(f"KeyError: {e}. Required financial data columns not found.")
            logging.error(f"KeyError: {e}. Required financial data columns not found.")
            return None
        except TypeError as e:
            st.error(f"TypeError: {e}. Check the type of your data. Likely a data format issue.")
            logging.error(f"TypeError: {e}. Check the type of your data. Likely a data format issue.")
            return None
        except IndexError as e:
            st.error(f"IndexError: {e}. Check the structure of your data. Data may be missing.")
            logging.error(f"IndexError: {e}. Check the structure of your data. Data may be missing.")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.error(f"An unexpected error occurred: {e}")
            return None

    def analyze_financial_health(self):
        """
        Analyzes the financial health of the company based on calculated ratios.

        Returns:
            dict: A dictionary containing the analysis results, or None if an error occurs.
        """
        ratios = self.calculate_ratios()
        if ratios is None:
            return None

        analysis = {}

def extract_unstructured_pdf_data(pdf_file):
    """
    Extracts unstructured data from a PDF file and attempts to process it into a structured format.

    Args:
        pdf_file: The uploaded PDF file.

    Returns:
        pd.DataFrame: A DataFrame containing the extracted data, or None if extraction fails.
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            raw_text = ""
            for page in pdf.pages:
                raw_text += page.extract_text() or ""

            if not raw_text.strip():
                st.warning("The PDF contains no readable text. Please ensure the file is not scanned or corrupted.")
                return None

            # Process raw text to extract financial data using regular expressions
            data = []
            lines = raw_text.split("\n")
            for line in lines:
                # Match patterns like "Revenue: $123,456" or "Net Income 123,456"
                match = re.match(r"(.+?):\s*([\d,.\(\)\$]+)", line) or re.match(r"(.+?)\s+([\d,.\(\)\$]+)", line)
                if match:
                    key, value = match.groups()
                    data.append([key.strip(), value.strip()])
                else:
                    # Fallback: Extract numeric values from unstructured lines
                    numbers = re.findall(r"[\d,.\(\)\$]+", line)
                    if numbers:
                        data.append([line.strip(), " | ".join(numbers)])

            if data:
                return pd.DataFrame(data, columns=["Description", "Values"])
            else:
                st.warning("No recognizable financial data could be extracted from the PDF.")
                return None
    except Exception as e:
        logging.error(f"Error extracting unstructured data from PDF: {e}")
        st.error(f"An error occurred while processing the PDF: {e}")
        return None

def convert_pdf_to_csv(pdf_file, output_csv_path="extracted_data.csv"):
    """
    Converts a PDF file into a CSV file by extracting tables or unstructured text.

    Args:
        pdf_file: The uploaded PDF file.
        output_csv_path (str): The path to save the generated CSV file.

    Returns:
        str: The path to the generated CSV file, or None if no data was found.
    """
    try:
        with pdfplumber.open(pdf_file) as pdf:
            tables = []
            raw_text = ""

            # Extract tables
            for page in pdf.pages:
                extracted_tables = page.extract_tables()
                for table in extracted_tables:
                    if table:  # Ensure the table is not empty
                        tables.append(pd.DataFrame(table))
                raw_text += page.extract_text() or ""

            # If tables are found, save them to CSV
            if tables:
                combined_data = pd.concat(tables, ignore_index=True)
                combined_data.to_csv(output_csv_path, index=False)
                return output_csv_path

            # If no tables are found, process unstructured text
            if raw_text.strip():
                lines = raw_text.split("\n")
                structured_data = []

                for line in lines:
                    # Match patterns like "Revenue: $123,456" or "Net Income 123,456"
                    match = re.match(r"(.+?):\s*([\d,.\(\)\$]+)", line) or re.match(r"(.+?)\s+([\d,.\(\)\$]+)", line)
                    if match:
                        key, value = match.groups()
                        structured_data.append([key.strip(), value.strip()])
                    else:
                        # Fallback: Extract numeric values from unstructured lines
                        numbers = re.findall(r"[\d,.\(\)\$]+", line)
                        if numbers:
                            structured_data.append([line.strip(), " | ".join(numbers)])

                if structured_data:
                    df = pd.DataFrame(structured_data, columns=["Description", "Values"])
                    df.to_csv(output_csv_path, index=False)
                    return output_csv_path

            return None
    except Exception as e:
        logging.error(f"Error converting PDF to CSV: {e}")
        st.error(f"An error occurred while processing the PDF: {e}")
        return None

def preprocess_image(image):
    """
    Preprocesses an image to improve OCR accuracy.

    Args:
        image: The image to preprocess.

    Returns:
        Image: The preprocessed image.
    """
    from PIL import ImageEnhance, ImageFilter

    image = image.convert("L")  # Convert to grayscale
    image = image.filter(ImageFilter.MedianFilter())  # Noise reduction
    enhancer = ImageEnhance.Contrast(image)
    image = enhancer.enhance(2)  # Increase contrast
    return image

def extract_text_from_image(image_file):
    """
    Extracts text from an image file using OCR.

    Args:
        image_file: The uploaded image file.

    Returns:
        str: The extracted text from the image.
    """
    try:
        image = Image.open(image_file)
        image = preprocess_image(image)  # Preprocess the image
        text = pytesseract.image_to_string(image)
        return text
    except Exception as e:
        logging.error(f"Error extracting text from image: {e}")
        return None

def process_extracted_text_to_dataframe(text):
    """
    Processes extracted text into a structured DataFrame.

    Args:
        text (str): The extracted text from an image or unstructured source.

    Returns:
        pd.DataFrame: A DataFrame containing structured financial data, or None if processing fails.
    """
    try:
        lines = text.split("\n")
        structured_data = []

        for line in lines:
            # Match patterns like "Revenue: $123,456" or "Net Income 123,456"
            match = re.match(r"(.+?):\s*([\d,.\(\)\$]+)", line) or re.match(r"(.+?)\s+([\d,.\(\)\$]+)", line)
            if match:
                key, value = match.groups()
                structured_data.append([key.strip(), value.strip()])
            else:
                # Fallback: Extract numeric values from unstructured lines
                numbers = re.findall(r"[\d,.\(\)\$]+", line)
                if numbers:
                    structured_data.append([line.strip(), " | ".join(numbers)])

        if structured_data:
            return pd.DataFrame(structured_data, columns=["Description", "Values"])
        return None
    except Exception as e:
        logging.error(f"Error processing extracted text into DataFrame: {e}")
        return None

def scrape_google_financial_data(company_name):
    """
    Scrapes financial data and news from Google for the specified company using Selenium.

    Args:
        company_name (str): The name of the company.

    Returns:
        dict: A dictionary containing financial data and relevant news.
    """
    try:
        # Set up Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # Navigate to Google search
        base_url = "https://www.google.com/search"
        query = f"{company_name} financial data"
        driver.get(f"{base_url}?q={query}")

        # Wait for the page to load
        time.sleep(3)  # Adjust this delay if necessary for slower connections

        # Extract financial data
        financial_data = {}
        try:
            stock_price = driver.find_element(By.CLASS_NAME, "BNeawe.iBp4i.AP7Wnd").text
            financial_data["Stock Price"] = stock_price
        except Exception as e:
            logging.warning(f"Stock price not found: {e}")

        try:
            metrics = driver.find_elements(By.CLASS_NAME, "BNeawe.s3v9rd.AP7Wnd")
            for metric in metrics:
                text = metric.text.lower()
                if "market cap" in text:
                    financial_data["Market Cap"] = text.split(":")[-1].strip()
                elif "revenue" in text:
                    financial_data["Revenue"] = text.split(":")[-1].strip()
                elif "net income" in text:
                    financial_data["Net Income"] = text.split(":")[-1].strip()
        except Exception as e:
            logging.warning(f"Financial metrics not found: {e}")

        # Extract relevant news
        news = []
        try:
            news_items = driver.find_elements(By.CLASS_NAME, "BNeawe.vvjwJb.AP7Wnd")
            for item in news_items:
                title = item.text
                link = item.find_element(By.XPATH, "..").get_attribute("href")
                news.append({"title": title, "link": link})
        except Exception as e:
            logging.warning(f"News not found: {e}")

        driver.quit()

        # Log if no data is found
        if not financial_data and not news:
            logging.warning(f"No financial data or news found for {company_name}.")

        # Return the extracted data
        return {"financial_data": financial_data, "news": news}
    except Exception as e:
        logging.error(f"Error scraping Google financial data for {company_name}: {e}")
        return None

def scrape_yahoo_finance(stock_symbol):
    """
    Scrapes financial data and news from Yahoo Finance for the specified stock symbol.

    Args:
        stock_symbol (str): The stock symbol (e.g., "NVDA" for NVIDIA).

    Returns:
        dict: A dictionary containing financial data and relevant news.
    """
    try:
        base_url = f"https://finance.yahoo.com/quote/{stock_symbol}"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Log the raw HTML content for debugging
        logging.info(f"HTML Content for {stock_symbol}:\n{soup.prettify()}")

        # Extract financial data
        financial_data = {}
        try:
            stock_price = soup.find("fin-streamer", {"data-field": "regularMarketPrice"}).text
            financial_data["Stock Price"] = stock_price
        except Exception as e:
            logging.warning(f"Yahoo Finance: Stock price not found for {stock_symbol}: {e}")

        # Extract relevant news
        news = []
        try:
            # Use the updated selector based on browser testing
            news_items = soup.select('a[data-test="quote-news-headlines"]')  # Updated selector
            logging.info(f"News items found: {len(news_items)}")
            for item in news_items:
                title = item.text.strip()
                link = item["href"]
                if not link.startswith("http"):
                    link = f"https://finance.yahoo.com{link}"  # Ensure full URL
                news.append({"title": title, "link": link})

            # If no news items are found, log and switch to Selenium
            if not news_items:
                logging.warning("No news items found using BeautifulSoup. Switching to Selenium for JavaScript rendering.")
                news = scrape_yahoo_finance_with_selenium(stock_symbol)
        except Exception as e:
            logging.warning(f"Yahoo Finance: News not found for {stock_symbol}: {e}")
            logging.warning(f"Exception during news extraction: {e}")

        logging.info(f"Extracted news: {news}")
        return {"financial_data": financial_data, "news": news}
    except requests.exceptions.HTTPError as e:
        logging.error(f"Yahoo Finance: HTTP error for {stock_symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error scraping Yahoo Finance for {stock_symbol}: {e}")
        return None

def scrape_yahoo_finance_with_selenium(stock_symbol):
    """
    Scrapes financial news from Yahoo Finance using Selenium for JavaScript-rendered content.

    Args:
        stock_symbol (str): The stock symbol (e.g., "NVDA" for NVIDIA).

    Returns:
        list: A list of news articles with titles and links.
    """
    try:
        base_url = f"https://finance.yahoo.com/quote/{stock_symbol}"
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)

        # Navigate to the Yahoo Finance page
        driver.get(base_url)
        time.sleep(5)  # Wait for JavaScript to load the content

        # Extract news items
        news = []
        try:
            news_elements = driver.find_elements(By.CSS_SELECTOR, "a[data-test='quote-news-headlines']")  # Adjusted selector
            logging.info(f"News items found with Selenium: {len(news_elements)}")
            for element in news_elements:
                title = element.text.strip()
                link = element.get_attribute("href")
                news.append({"title": title, "link": link})
        except Exception as e:
            logging.warning(f"Selenium: News not found for {stock_symbol}: {e}")

        driver.quit()
        return news
    except Exception as e:
        logging.error(f"Error scraping Yahoo Finance with Selenium for {stock_symbol}: {e}")
        return []

# API Key Management: Use environment variables for Alpha Vantage API key
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "YOUR_DEFAULT_API_KEY")

def scrape_alpha_vantage(stock_symbol, api_key=ALPHA_VANTAGE_API_KEY):
    """
    Fetches financial data from Alpha Vantage for the specified stock symbol.

    Args:
        stock_symbol (str): The stock symbol (e.g., "NVDA" for NVIDIA).
        api_key (str): The API key for Alpha Vantage.

    Returns:
        dict: A dictionary containing financial data.
    """
    try:
        base_url = "https://www.alphavantage.co/query"
        params = {
            "function": "OVERVIEW",
            "symbol": stock_symbol,
            "apikey": api_key
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()

        if "Symbol" not in data:
            logging.warning(f"Alpha Vantage: No data found for {stock_symbol}.")
            return None

        financial_data = {
            "Market Cap": data.get("MarketCapitalization"),
            "Revenue": data.get("RevenueTTM"),
            "Net Income": data.get("NetIncomeTTM"),
            "PE Ratio": data.get("PERatio"),
        }
        return {"financial_data": financial_data, "news": []}
    except requests.exceptions.HTTPError as e:
        logging.error(f"Alpha Vantage: HTTP error for {stock_symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error fetching data from Alpha Vantage for {stock_symbol}: {e}")
        return None

def scrape_forbes(company_name):
    """
    Scrapes news from Forbes for the specified company.

    Args:
        company_name (str): The name of the company.

    Returns:
        dict: A dictionary containing relevant news.
    """
    try:
        base_url = "https://www.forbes.com/search/"
        params = {"q": company_name}
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract relevant news
        news = []
        try:
            news_items = soup.find_all("div", class_="stream-item__text")
            for item in news_items:
                title = item.find("a").text
                link = item.find("a")["href"]
                news.append({"title": title, "link": link})
        except Exception as e:
            logging.warning(f"Forbes: News not found: {e}")

        return {"financial_data": {}, "news": news}
    except Exception as e:
        logging.error(f"Error scraping Forbes for {company_name}: {e}")
        return None

def scrape_nairobi_stock_exchange(stock_symbol):
    """
    Scrapes financial data from the Nairobi Stock Exchange (NSE) for the specified stock symbol.

    Args:
        stock_symbol (str): The stock symbol (e.g., "KCB" for KCB Group).

    Returns:
        dict: A dictionary containing financial data.
    """
    try:
        base_url = f"https://www.nse.co.ke/market-statistics/equities.html"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract financial data for the stock symbol
        financial_data = {}
        try:
            table = soup.find("table", {"id": "equities-table"})
            rows = table.find_all("tr")
            for row in rows:
                cols = row.find_all("td")
                if len(cols) > 0 and stock_symbol.upper() in cols[0].text:
                    financial_data["Stock Symbol"] = cols[0].text.strip()
                    financial_data["Last Price"] = cols[1].text.strip()
                    financial_data["Change"] = cols[2].text.strip()
                    financial_data["Volume"] = cols[3].text.strip()
                    break
        except Exception as e:
            logging.warning(f"Nairobi Stock Exchange: Data not found for {stock_symbol}: {e}")

        return {"financial_data": financial_data, "news": []}
    except Exception as e:
        logging.error(f"Error scraping Nairobi Stock Exchange for {stock_symbol}: {e}")
        return None

def scrape_world_stock_exchanges(stock_symbol, exchange_code):
    """
    Scrapes financial data from major world stock exchanges for the specified stock symbol.

    Args:
        stock_symbol (str): The stock symbol (e.g., "AAPL" for Apple).
        exchange_code (str): The exchange code (e.g., "NASDAQ", "LSE", "HKEX").

    Returns:
        dict: A dictionary containing financial data.
    """
    try:
        # Example: Use Yahoo Finance for global stock exchanges
        base_url = f"https://finance.yahoo.com/quote/{stock_symbol}.{exchange_code}"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract financial data
        financial_data = {}
        try:
            stock_price = soup.find("fin-streamer", {"data-field": "regularMarketPrice"}).text
            financial_data["Stock Price"] = stock_price
        except Exception as e:
            logging.warning(f"World Stock Exchanges: Stock price not found for {stock_symbol} on {exchange_code}: {e}")

        return {"financial_data": financial_data, "news": []}
    except Exception as e:
        logging.error(f"Error scraping world stock exchanges for {stock_symbol} on {exchange_code}: {e}")
        return None

def validate_data(data):
    """
    Validates the scraped data to ensure it is in the correct format.

    Args:
        data: The data to validate.

    Returns:
        bool: True if the data is valid, False otherwise.
    """
    if not data or not isinstance(data, dict):
        logging.error("Invalid data format: Data is empty or not a dictionary.")
        return False
    return True

def rate_limit():
    """
    Implements a delay to respect rate limits.
    """
    time.sleep(2)  # Delay of 2 seconds between requests

def scrape_cnbc_africa(company_name):
    """
    Scrapes financial news from CNBC Africa for the specified company.

    Args:
        company_name (str): The name of the company.

    Returns:
        dict: A dictionary containing relevant news.
    """
    try:
        base_url = "https://www.cnbcafrica.com/search/"
        params = {"q": company_name}
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract relevant news
        news = []
        try:
            news_items = soup.find_all("div", class_="search-result")
            for item in news_items:
                title = item.find("a").text.strip()
                link = item.find("a")["href"]
                news.append({"title": title, "link": link})
        except Exception as e:
            logging.warning(f"CNBC Africa: News not found for {company_name}: {e}")

        return {"financial_data": {}, "news": news}
    except Exception as e:
        logging.error(f"Error scraping CNBC Africa for {company_name}: {e}")
        return None

def scrape_cnbc(company_name):
    """
    Scrapes financial news from CNBC for the specified company.

    Args:
        company_name (str): The name of the company.

    Returns:
        dict: A dictionary containing relevant news.
    """
    try:
        base_url = "https://www.cnbc.com/search/"
        params = {"query": company_name}
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract relevant news
        news = []
        try:
            news_items = soup.find_all("div", class_="SearchResultCard")
            for item in news_items:
                title = item.find("a").text.strip()
                link = item.find("a")["href"]
                news.append({"title": title, "link": link})
        except Exception as e:
            logging.warning(f"CNBC: News not found for {company_name}: {e}")

        return {"financial_data": {}, "news": news}
    except Exception as e:
        logging.error(f"Error scraping CNBC for {company_name}: {e}")
        return None

def scrape_business_daily_africa(company_name):
    """
    Scrapes financial news from Business Daily Africa for the specified company.

    Args:
        company_name (str): The name of the company.

    Returns:
        dict: A dictionary containing relevant news.
    """
    try:
        base_url = "https://www.businessdailyafrica.com/search"
        params = {"q": company_name}
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract relevant news
        news = []
        try:
            news_items = soup.find_all("div", class_="search-result")
            for item in news_items:
                title = item.find("a").text.strip()
                link = item.find("a")["href"]
                news.append({"title": title, "link": link})
        except Exception as e:
            logging.warning(f"Business Daily Africa: News not found for {company_name}: {e}")

        return {"financial_data": {}, "news": news}
    except Exception as e:
        logging.error(f"Error scraping Business Daily Africa for {company_name}: {e}")
        return None

def scrape_ft(company_name):
    """
    Scrapes financial news from Financial Times (FT) for the specified company.

    Args:
        company_name (str): The name of the company.

    Returns:
        dict: A dictionary containing relevant news.
    """
    try:
        base_url = "https://www.ft.com/search"
        params = {"q": company_name}
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers, params=params)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract relevant news
        news = []
        try:
            news_items = soup.find_all("li", class_="o-teaser-collection__item")
            for item in news_items:
                title = item.find("a", class_="js-teaser-heading-link").text.strip()
                link = item.find("a", class_="js-teaser-heading-link")["href"]
                if not link.startswith("http"):
                    link = f"https://www.ft.com{link}"  # Ensure full URL
                news.append({"title": title, "link": link})
        except Exception as e:
            logging.warning(f"FT: News not found for {company_name}: {e}")

        return {"financial_data": {}, "news": news}
    except Exception as e:
        logging.error(f"Error scraping FT for {company_name}: {e}")
        return None

def aggregate_data(stock_symbol, exchange_code=None):
    """
    Aggregates financial data and news from multiple sources, including specific stock exchanges.

    Args:
        stock_symbol (str): The stock symbol (e.g., "NVDA" for NVIDIA).
        exchange_code (str, optional): The exchange code (e.g., "NSE", "NASDAQ"). Defaults to None.

    Returns:
        dict: A dictionary containing aggregated financial data and news.
    """
    sources = [
        scrape_yahoo_finance,
        scrape_alpha_vantage,
        scrape_forbes,
        scrape_cnbc_africa,
        scrape_cnbc,
        scrape_business_daily_africa,
        scrape_ft,  # Added Financial Times (FT)
    ]

    # Add specific exchange scraping based on the exchange code
    if exchange_code == "NSE":
        sources.append(scrape_nairobi_stock_exchange)
    elif exchange_code:
        sources.append(lambda symbol: scrape_world_stock_exchanges(symbol, exchange_code))

    aggregated_data = {"financial_data": {}, "news": []}

    for source in sources:
        try:
            result = source(stock_symbol)
            if result:
                aggregated_data["financial_data"].update(result.get("financial_data", {}))
                news_items = result.get("news", [])
                if news_items:
                    logging.info(f"News found from {source.__name__}: {len(news_items)} items.")
                aggregated_data["news"].extend(news_items)
        except Exception as e:
            logging.error(f"Error in source {source.__name__}: {e}")

    return aggregated_data

# Mapping of company names to stock symbols
COMPANY_TO_SYMBOL = {
    "nvidia": "NVDA",
    "apple": "AAPL",
    "kcb group": "KCB",
    "safaricom": "SCOM",
    # Add more companies as needed
}

# Mapping of stock symbols to exchange codes
STOCK_TO_EXCHANGE = {
    "NVDA": "NASDAQ",
    "AAPL": "NASDAQ",
    "KCB": "Nairobi Stock Exchange",
    "SCOM": "Nairobi Stock Exchange",
    # Add more mappings as needed
}

def resolve_company_and_exchange(company_name):
    """
    Resolves the stock symbol and exchange code based on the company name.

    Args:
        company_name (str): The name of the company.

    Returns:
        tuple: A tuple containing the stock symbol and exchange code, or (None, None) if not found.
    """
    company_name = company_name.strip().lower()  # Normalize case and whitespace
    logging.debug(f"Resolving company name: {company_name}")

    # Find the stock symbol
    stock_symbol = COMPANY_TO_SYMBOL.get(company_name)
    if not stock_symbol:
        logging.error(f"Company name '{company_name}' not found in the mapping.")
        return None, None

    # Resolve the exchange code
    exchange_code = STOCK_TO_EXCHANGE.get(stock_symbol)
    if not exchange_code:
        logging.error(f"Exchange name for stock symbol '{stock_symbol}' not found in the mapping.")
        return stock_symbol, None

    logging.debug(f"Resolved stock symbol: {stock_symbol}, exchange code: {exchange_code}")
    return stock_symbol, exchange_code

# Ensure debug logging is enabled
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Global variable to store the latest data
latest_data = {"financial_data": {}, "news": []}

def fetch_and_update_data(stock_symbol, exchange_code=None):
    """
    Fetches and updates the latest financial data and news for a given stock symbol.
    Also calculates and displays financial ratios.

    Args:
        stock_symbol (str): The stock symbol (e.g., "AAPL").
        exchange_code (str, optional): The exchange code (e.g., "NASDAQ"). Defaults to None.
    """
    global latest_data
    logging.info(f"Fetching new data for {stock_symbol}...")
    new_data = aggregate_data(stock_symbol, exchange_code)
    if new_data:
        if detect_changes(latest_data, new_data):
            logging.info("New data detected. Updating...")
            latest_data = new_data

            # Display updated financial data
            st.subheader("Latest Financial Data")
            if latest_data["financial_data"]:
                logging.debug(f"Fetched financial data structure: {latest_data['financial_data']}")
                display_table(latest_data["financial_data"], "Latest Financial Data")
            else:
                st.info("No financial data available yet.")

            # Calculate and display financial ratios
            if "financial_data" in latest_data and isinstance(latest_data["financial_data"], dict):
                financial_data_df = pd.DataFrame.from_dict(latest_data["financial_data"], orient="index", columns=["Value"])
                ratios_df = calculate_ratios_with_standards(financial_data_df)
                if not ratios_df.empty:
                    st.subheader("Financial Ratios with Industry Standards")
                    display_table(ratios_df, "Financial Ratios with Industry Standards")
                else:
                    st.warning("Could not calculate financial ratios.")
        else:
            logging.info("No changes detected in the data.")
    else:
        logging.warning(f"Failed to fetch new data for {stock_symbol}.")
        st.error(f"Failed to fetch new data for {stock_symbol}.")

def detect_changes(old_data, new_data):
    """
    Detects changes between old and new data using hashing.

    Args:
        old_data (dict): The previously stored data.
        new_data (dict): The newly fetched data.

    Returns:
        bool: True if changes are detected, False otherwise.
    """
    import hashlib
    old_hash = hashlib.md5(json.dumps(old_data, sort_keys=True).encode()).hexdigest()
    new_hash = hashlib.md5(json.dumps(new_data, sort_keys=True).encode()).hexdigest()
    return old_hash != new_hash

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "YOUR_FINNHUB_API_KEY")

def fetch_finnhub_news(stock_symbol):
    """
    Fetches financial news for a specific stock symbol using Finnhub API.

    Args:
        stock_symbol (str): The stock symbol (e.g., "AAPL" for Apple).

    Returns:
        list: A list of news articles with titles and links.
    """
    try:
        base_url = f"https://finnhub.io/api/v1/news"
        params = {
            "category": "general",
            "token": FINNHUB_API_KEY
        }
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        news_data = response.json()

        # Extract relevant news
        news = []
        for item in news_data:
            if "headline" in item and "url" in item:
                news.append({"title": item["headline"], "link": item["url"]})

        logging.info(f"Fetched {len(news)} news articles from Finnhub.")
        return news
    except requests.exceptions.HTTPError as e:
        logging.error(f"Finnhub API: HTTP error for {stock_symbol}: {e}")
        return []
    except Exception as e:
        logging.error(f"Error fetching news from Finnhub for {stock_symbol}: {e}")
        return []

def fetch_robots_txt(robots_url, timeout=5):
    """
    Fetches and parses the robots.txt file.

    Args:
        robots_url (str): The URL of the robots.txt file.
        timeout (int): Timeout for fetching the robots.txt file.

    Returns:
        RobotFileParser: A parsed RobotFileParser object, or None if an error occurs.
    """
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        rp.read()
        return rp
    except Exception as e:
        logging.warning(f"Could not fetch or parse robots.txt at {robots_url}: {e}")
        return None

def scrape_financial_data_with_retries(url, retries=3, delay=5):
    """
    Scrapes financial data from a URL with retries and rate limiting.

    Args:
        url (str): The URL to scrape.
        retries (int): Number of retry attempts.
        delay (int): Delay between retries in seconds.

    Returns:
        BeautifulSoup: Parsed HTML content, or None if scraping fails.
    """
    for attempt in range(retries):
        try:
            headers = {"User-Agent": USER_AGENT}
            response = requests.get(url, headers=headers, timeout=SCRAPING_TIMEOUT)
            response.raise_for_status()
            return BeautifulSoup(response.text, "html.parser")
        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:  # Too Many Requests
                logging.warning(f"HTTP 429 Too Many Requests. Retrying in {delay} seconds... (Attempt {attempt + 1}/{retries})")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logging.error(f"HTTP error occurred: {e}")
                break
        except Exception as e:
            logging.error(f"Error during scraping: {e}")
            break
    return None

def fetch_news_with_api(company_name):
    """
    Fetches the latest news about a company using the News API.

    Args:
        company_name (str): The name of the company.

    Returns:
        list: A list of news articles, or an empty list if no news is found.
    """
    if not NEWS_API_KEY:
        logging.warning("News API key is missing. Skipping news fetching.")
        st.warning("News API key is not configured. Unable to fetch news.")
        return []

    url = f"https://newsapi.org/v2/everything?q={urllib.parse.quote_plus(company_name)}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        articles = response.json().get("articles", [])
        if not articles:
            st.info(f"No recent news found for {company_name}.")
        return [{"title": article["title"], "url": article["url"]} for article in articles]
    except Exception as e:
        logging.error(f"Error fetching news for {company_name}: {e}")
        st.error(f"An error occurred while fetching news for {company_name}.")
        return []

def analyze_financial_health(ratios):
    """
    Analyzes the financial health of a company based on calculated ratios.

    Args:
        ratios (dict): A dictionary of financial ratios.

    Returns:
        dict: A dictionary containing the financial health analysis.
    """
    analysis = {}
    try:
        if "Profit Margin" in ratios:
            profit_margin = float(ratios["Profit Margin"].replace(",", "").replace("%", ""))
            analysis["Profitability"] = "High" if profit_margin > 20 else "Moderate" if profit_margin > 10 else "Low"

        if "Debt-to-Equity Ratio" in ratios:
            debt_to_equity = float(ratios["Debt-to-Equity Ratio"].replace(",", "").replace("%", ""))
            analysis["Leverage"] = "Low" if debt_to_equity < 50 else "Moderate" if debt_to_equity < 100 else "High"

        if "Current Ratio" in ratios:
            current_ratio = float(ratios["Current Ratio"].replace(",", ""))
            analysis["Liquidity"] = "High" if current_ratio > 2 else "Moderate" if current_ratio > 1.5 else "Low"
    except Exception as e:
        logging.error(f"Error analyzing financial health: {e}")
    return analysis

def display_table(data, title, currency="USD"):
    """
    Displays data in a table format with clear borders using Streamlit.

    Args:
        data (dict or pd.DataFrame): The data to display.
        title (str): The title of the table.
        currency (str): The currency type for formatting amounts.
    """
    st.subheader(title)
    if isinstance(data, pd.DataFrame):
        # Format numeric values with currency
        for col in data.select_dtypes(include=["number"]).columns:
            data[col] = data[col].apply(lambda x: format_number(x, currency))
        st.dataframe(data.style.set_table_styles(
            [{'selector': 'th', 'props': [('border', '1px solid black')]},
             {'selector': 'td', 'props': [('border', '1px solid black')]}]
        ))
    elif isinstance(data, dict):
        df = pd.DataFrame(list(data.items()), columns=["Metric", "Value"])
        st.dataframe(df.style.set_table_styles(
            [{'selector': 'th', 'props': [('border', '1px solid black')]},
             {'selector': 'td', 'props': [('border', '1px solid black')]}]
        ))
    else:
        st.write("No data available to display.")

def calculate_ratios_with_standards(financial_data, currency="USD"):
    """
    Calculates financial ratios and compares them with industry standards.

    Args:
        financial_data (pd.DataFrame): The financial data to calculate ratios from.
        currency (str): The currency type for formatting amounts.

    Returns:
        pd.DataFrame: A DataFrame containing the ratios, their values, industry standards, and explanations.
    """
    ratios = []
    try:
        # Define required columns
        required_columns = {
            "Revenue": 0,
            "Net Income": 0,
            "Total Assets": 0,
            "Total Liabilities": 0,
            "Current Assets": 0,
            "Current Liabilities": 0,
            "Total Equity": 0
        }

        # Add missing columns with default values
        for col, default_value in required_columns.items():
            if col not in financial_data.columns:
                logging.warning(f"Missing column '{col}'. Adding default value: {default_value}.")
                financial_data[col] = default_value

        # Convert columns to numeric for calculations
        for col in required_columns.keys():
            financial_data[col] = pd.to_numeric(financial_data[col], errors='coerce').fillna(0)

        # Profit Margin
        revenue = financial_data["Revenue"].iloc[-1]
        net_income = financial_data["Net Income"].iloc[-1]
        profit_margin = (net_income / revenue * 100) if revenue != 0 else None
        ratios.append({
            "Metric": "Profit Margin",
            "Value": f"{profit_margin:,.2f}%" if profit_margin is not None else "N/A",
            "Industry Standard": "10-20%",
            "Explanation": "Indicates the percentage of revenue that turns into profit."
        })

        # Debt-to-Equity Ratio
        total_liabilities = financial_data["Total Liabilities"].iloc[-1]
        total_equity = financial_data["Total Equity"].iloc[-1]
        debt_to_equity = (total_liabilities / total_equity) if total_equity != 0 else None
        ratios.append({
            "Metric": "Debt-to-Equity Ratio",
            "Value": f"{debt_to_equity:,.2f}" if debt_to_equity is not None else "N/A",
            "Industry Standard": "1.0-2.0",
            "Explanation": "Measures the company's financial leverage."
        })

        # Current Ratio
        current_assets = financial_data["Current Assets"].iloc[-1]
        current_liabilities = financial_data["Current Liabilities"].iloc[-1]
        current_ratio = (current_assets / current_liabilities) if current_liabilities != 0 else None
        ratios.append({
            "Metric": "Current Ratio",
            "Value": f"{current_ratio:,.2f}" if current_ratio is not None else "N/A",
            "Industry Standard": "1.5-2.0",
            "Explanation": "Indicates the company's ability to pay short-term obligations."
        })

        # Add currency to financial data
        for ratio in ratios:
            if "Value" in ratio and ratio["Value"] != "N/A" and "%" not in ratio["Value"]:
                ratio["Value"] = format_number(float(ratio["Value"].replace(",", "")), currency)

        return pd.DataFrame(ratios)
    except Exception as e:
        logging.error(f"Error calculating ratios: {e}")
        st.error("An unexpected error occurred while calculating financial ratios.")
        return pd.DataFrame()

def scrape_nyse_indices():
    """
    Scrapes major world indices data from the NYSE website.

    Returns:
        pd.DataFrame: A DataFrame containing indices data, or None if an error occurs.
    """
    try:
        base_url = "https://www.nyse.com/index"
        headers = {"User-Agent": USER_AGENT}
        response = requests.get(base_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract indices data
        indices_data = []
        try:
            table = soup.find("table", {"class": "indices-table"})  # Adjust class based on actual HTML
            rows = table.find_all("tr")
            for row in rows[1:]:  # Skip header row
                cols = row.find_all("td")
                cols = [col.text.strip() for col in cols]
                if len(cols) >= 3:  # Ensure sufficient columns
                    indices_data.append({
                        "Index": cols[0],
                        "Last Price": cols[1],
                        "Change": cols[2],
                        "Change (%)": cols[3] if len(cols) > 3 else "N/A"
                    })
        except Exception as e:
            logging.warning(f"Error extracting indices data: {e}")

        if indices_data:
            return pd.DataFrame(indices_data)
        else:
            logging.warning("No indices data found.")
            return None
    except Exception as e:
        logging.error(f"Error scraping NYSE indices: {e}")
        return None

def display_indices_chart(indices_df):
    """
    Displays a chart of major world indices using Streamlit.

    Args:
        indices_df (pd.DataFrame): The DataFrame containing indices data.
    """
    try:
        if indices_df is not None and not indices_df.empty:
            st.subheader("Major World Indices")
            st.dataframe(indices_df)

            # Plot indices data
            indices_df["Change (%)"] = pd.to_numeric(indices_df["Change (%)"].str.replace("%", ""), errors="coerce")
            st.line_chart(indices_df.set_index("Index")["Change (%)"])
        else:
            st.warning("No indices data available to display.")
    except Exception as e:
        logging.error(f"Error displaying indices chart: {e}")
        st.error("An error occurred while displaying indices data.")

def handle_uploaded_file(uploaded_file):
    """
    Handles the uploaded file and ensures proper error handling.

    Args:
        uploaded_file: The uploaded file object.

    Returns:
        str or None: The file path if successful, or None if an error occurs.
    """
    try:
        file_path = os.path.join("temp", uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        return file_path
    except Exception as e:
        logging.error(f"Error handling uploaded file: {e}")
        st.error("An error occurred while processing the uploaded file.")
        return None

def cleanup_file(file_path):
    """
    Cleans up the temporary file if it exists.

    Args:
        file_path (str): The path to the file to be removed.
    """
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logging.info(f"Temporary file {file_path} removed successfully.")
    except Exception as e:
        logging.error(f"Error removing temporary file {file_path}: {e}")

def main():
    """
    Main function to run the Streamlit application.
    """
    st.title("Real-Time Financial Analyzer App")
    st.write("Analyze and track financial data of companies in real time.")

    # Input field for company name
    company_name = st.text_input("Enter the company name (e.g., 'NVIDIA', 'Apple'):").strip()

    # Option to provide a URL for financial statements
    financial_url = st.text_input("Enter the URL of the company's financial statements (optional):").strip()

    # Option to upload a financial statement file
    uploaded_file = st.file_uploader(
        "Upload financial data (PDF, Excel, Image)", type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg"]
    )

    file_path = None  # Initialize file_path to None
    if uploaded_file:
        file_path = handle_uploaded_file(uploaded_file)

    # Automatically resolve exchange name
    stock_symbol, exchange_code = None, None
    if company_name:
        stock_symbol, exchange_code = resolve_company_and_exchange(company_name)
        if stock_symbol and exchange_code:
            st.success(f"Resolved: {company_name} ({stock_symbol}) on {exchange_code}")
        elif not stock_symbol:
            st.error(f"Could not resolve the stock symbol for '{company_name}'. Please check the company name.")
        elif not exchange_code:
            st.error(f"Could not resolve the exchange code for stock symbol '{stock_symbol}'. Please check the mappings.")

    # Process financial data from URL or uploaded file
    financial_data = None
    if financial_url:
        st.info(f"Fetching financial data from URL: {financial_url}")
        analyzer = FinancialAnalyzer(company_name, website_url=financial_url)
        financial_data = analyzer.scrape_financial_data()
        if financial_data is not None:
            st.success("Financial data successfully fetched from the URL.")
        else:
            st.error("Failed to fetch financial data from the URL.")

    if file_path:
        try:
            file_extension = uploaded_file.name.split(".")[-1].lower()
            if file_extension == "pdf":
                structured_data_df = extract_structured_pdf_data(file_path) if 'extract_structured_pdf_data' in globals() else None
                if structured_data_df is not None and not structured_data_df.empty:
                    financial_data = structured_data_df
                    st.success("Structured data (table) extracted from PDF.")
                else:
                    unstructured_data_df = extract_unstructured_pdf_data(file_path)
                    if unstructured_data_df is not None and not unstructured_data_df.empty:
                        financial_data = unstructured_data_df
                        st.success("Unstructured data extracted from PDF.")
                    else:
                        st.error("Failed to extract data from PDF.")
            elif file_extension in ["xlsx", "xls"]:
                try:
                    financial_data = pd.read_excel(file_path)
                    st.success("Data extracted from Excel file.")
                except Exception as e:
                    st.error(f"Error reading Excel file: {e}")
                    logging.error(f"Error reading Excel file: {e}")
            elif file_extension in ["png", "jpg", "jpeg"]:
                image_df = process_extracted_text_to_dataframe(extract_text_from_image(file_path))
                if image_df is not None:
                    financial_data = image_df
                    st.success("Data extracted from image.")
                else:
                    st.error("Failed to extract data from image.")
            else:
                st.error("Unsupported file type. Please upload a PDF, Excel, or image file.")
        finally:
            cleanup_file(file_path)

    # Analyze financial data
    if st.button("Analyze Financial Data"):
        if financial_data is not None and not financial_data.empty:
            ratios_df = calculate_ratios_with_standards(financial_data)
            if not ratios_df.empty:
                st.subheader("Financial Ratios with Industry Standards")
                display_table(ratios_df, "Financial Ratios with Industry Standards")
            else:
                st.error("Could not calculate financial ratios.")
        else:
            st.warning("No financial data available to analyze. Please provide a URL or upload a file.")

    # Button to fetch and display news
    if st.button("Fetch News"):
        if not company_name:
            st.error("Please provide the company name.")
            return

        if not stock_symbol:
            st.error("Could not resolve the stock symbol. Please check your inputs.")
            return

        st.info(f"Fetching news for {company_name} ({stock_symbol})...")
        news = fetch_news_with_api(company_name)
        if news:
            news_df = pd.DataFrame(news)
            display_table(news_df, "Latest News")
        else:
            st.warning("No news found.")

    # Display the last known data if available
    st.subheader("Last Known Financial Data")
    if latest_data["financial_data"]:
        display_table(latest_data["financial_data"], "Last Known Financial Data")
    else:
        st.info("No financial data available yet.")

    st.subheader("Last Known News")
    if latest_data.get("news") and isinstance(latest_data["news"], list):
        news_df = pd.DataFrame(latest_data["news"])
        if not news_df.empty:
            display_table(news_df, "Last Known News")
        else:
            st.info("No news available yet.")
    else:
        st.info("No news available yet.")

    # Button to start tracking
    if st.button("Start Tracking"):
        if not company_name:
            st.error("Please provide the company name.")
            return

        if not stock_symbol or not exchange_code:
            st.error("Could not resolve the stock symbol or exchange code. Please check your inputs.")
            return

        st.info(f"Fetching initial data for {company_name} ({stock_symbol}) on {exchange_code}...")
        fetch_and_update_data(stock_symbol, exchange_code)

    # Button to refresh data
    if st.button("Refresh Data"):
        if not company_name:
            st.error("Please provide the company name.")
            return

        if not stock_symbol or not exchange_code:
            st.error("Could not resolve the stock symbol or exchange code. Please check your inputs.")
            return

        st.info(f"Refreshing data for {company_name} ({stock_symbol}) on {exchange_code}...")
        fetch_and_update_data(stock_symbol, exchange_code)

    # Button to fetch and display major world indices
    if st.button("Fetch Major World Indices"):
        st.info("Fetching major world indices data...")
        indices_df = scrape_nyse_indices()
        if indices_df is not None:
            display_indices_chart(indices_df)
        else:
            st.error("Failed to fetch indices data.")

    # Display the latest data
    st.subheader("Latest Financial Data")
    if latest_data["financial_data"]:
        display_table(latest_data["financial_data"], "Latest Financial Data")
    else:
        st.info("No financial data available yet.")

    st.subheader("Latest News")
    if latest_data["news"]:
        news_df = pd.DataFrame(latest_data["news"])
        display_table(news_df, "Latest News")
    else:
        st.info("No news available yet.")

if __name__ == "__main__":
    main()

def resolve_company_ticker(self, company_name):
    """
    Resolves the company name to its stock ticker using Alpha Vantage.

    Args:
        company_name (str): The name of the company.

    Returns:
        str: The stock ticker symbol, or None if not found.
    """
    if not ALPHA_VANTAGE_API_KEY:
        logging.warning("Alpha Vantage API key is missing. Company ticker resolution will be skipped.")
        return None

    url = f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={ALPHA_VANTAGE_API_KEY}"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data and "bestMatches" in data and len(data["bestMatches"]) > 0:
            return data["bestMatches"][0]["1. symbol"]
        else:
            logging.warning(f"No ticker found for company: {company_name}")
            return None
    except requests.exceptions.RequestException as e:
        logging.error(f"Error resolving company ticker: {e}")
        return None
    except json.JSONDecodeError:
        logging.error("Error decoding JSON response from Alpha Vantage.")
        return None

def fetch_major_world_indices(self):
    """
    Fetches major world indices data. Uses Alpha Vantage or RapidAPI.
    """
    self.indices_data.clear()  # Clear stale data
    if WORLD_INDICES_API == "AlphaVantage":
        self._fetch_world_indices_alpha_vantage()
    elif WORLD_INDICES_API == "RapidAPI":
        self._fetch_world_indices_rapidapi()
    else:
        logging.error(f"Invalid world indices API: {WORLD_INDICES_API}")

def display_financial_data(self):
    """
    Displays the financial data, ratios, stock price, and news in the Streamlit app.
    """
    import tempfile  # Use tempfile for safer temporary file handling

    st.title("Real-Time Financial Analyzer")

    company_name = st.text_input("Enter the company name (e.g., 'NVIDIA', 'Apple'):", self.company_name)
    website_url = st.text_input("Enter the URL of the company's financial statements (optional):", self.website_url)

    uploaded_file = st.file_uploader("Upload financial data (PDF, Excel, Image)", type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg"])

    if company_name != self.company_name:
        self.company_name = company_name
        self.ticker = self.resolve_company_ticker(company_name)
        if self.ticker:
            st.success(f"Resolved: {company_name} ({self.ticker})")
        else:
            st.warning(f"Could not resolve ticker for {company_name}. Some features may be limited.")

    if website_url != self.website_url:
        self.website_url = website_url

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as temp_file:
            temp_file.write(uploaded_file.getbuffer())
            temp_file_path = temp_file.name
        try:
            self.financial_data = self.extract_financial_data(temp_file_path)
            if self.financial_data is not None:
                st.success("Financial data uploaded successfully!")
            else:
                st.error("Failed to extract financial data from the uploaded file.")
        finally:
            os.remove(temp_file_path)  # Clean up the temporary file