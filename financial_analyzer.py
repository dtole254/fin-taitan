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
    "scraping_timeout": 10
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
            if not isinstance(MAX_RETRIES, int) or MAX_RETRIES <= 0:
                raise ValueError("max_retries must be a positive integer.")
            if not isinstance(RETRY_DELAY, (int, float)) or RETRY_DELAY < 0:
                raise ValueError("retry_delay must be a non-negative number.")
            if not isinstance(ROBOTS_TIMEOUT, (int, float)) or ROBOTS_TIMEOUT <= 0:
                raise ValueError("robots_timeout must be a positive number.")
            if not isinstance(SCRAPING_TIMEOUT, (int, float)) or SCRAPING_TIMEOUT <= 0:
                raise ValueError("scraping_timeout must be a positive number.")
    except Exception as e:
        logging.warning(f"Failed to load or validate configuration file {args.config_file}: {e}")
        USER_AGENT = DEFAULT_CONFIG["user_agent"]
        MAX_RETRIES = DEFAULT_CONFIG["max_retries"]
        RETRY_DELAY = DEFAULT_CONFIG["retry_delay"]
        ROBOTS_TIMEOUT = DEFAULT_CONFIG["robots_timeout"]
        SCRAPING_TIMEOUT = DEFAULT_CONFIG["scraping_timeout"]
else:
    USER_AGENT = DEFAULT_CONFIG["user_agent"]
    MAX_RETRIES = DEFAULT_CONFIG["max_retries"]
    RETRY_DELAY = DEFAULT_CONFIG["retry_delay"]
    ROBOTS_TIMEOUT = DEFAULT_CONFIG["robots_timeout"]
    SCRAPING_TIMEOUT = DEFAULT_CONFIG["scraping_timeout"]

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

            if revenue_col and net_income_col:
                revenue = pd.to_numeric(data[revenue_col], errors='coerce').iloc[-1]
                net_income = pd.to_numeric(data[net_income_col], errors='coerce').iloc[-1]
                ratios["Profit Margin"] = net_income / revenue if revenue != 0 else None

            if total_assets_col and total_liabilities_col:
                total_assets = pd.to_numeric(data[total_assets_col], errors='coerce').iloc[-1]
                total_liabilities = pd.to_numeric(data[total_liabilities_col], errors='coerce').iloc[-1]
                ratios["Debt-to-Asset Ratio"] = total_liabilities / total_assets if total_assets != 0 else None

            if current_assets_col and current_liabilities_col:
                current_assets = pd.to_numeric(data[current_assets_col], errors='coerce').iloc[-1]
                current_liabilities = pd.to_numeric(data[current_liabilities_col], errors='coerce').iloc[-1]
                ratios["Current Ratio"] = current_assets / current_liabilities if current_liabilities != 0 else None

            if total_equity_col and total_liabilities_col:
                total_equity = pd.to_numeric(data[total_equity_col], errors='coerce').iloc[-1]
                ratios["Debt-to-Equity Ratio"] = total_liabilities / total_equity if total_equity != 0 else None

            if cash_col and current_liabilities_col:
                cash = pd.to_numeric(data[cash_col], errors='coerce').iloc[-1]
                ratios["Cash Ratio"] = cash / current_liabilities if current_liabilities != 0 else None

            if inventory_col and cogs_col:
                inventory = pd.to_numeric(data[inventory_col], errors='coerce').iloc[-1]
                cogs = pd.to_numeric(data[cogs_col], errors='coerce').iloc[-1]
                ratios["Inventory Turnover"] = cogs / inventory if inventory != 0 else None

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
        st.error("An unexpected error occurred while fetching data. Please try again later.")
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
            news_items = soup.find_all("li", class_="js-stream-content")
            for item in news_items:
                title = item.find("a").text
                link = f"https://finance.yahoo.com{item.find('a')['href']}"
                news.append({"title": title, "link": link})
        except Exception as e:
            logging.warning(f"Yahoo Finance: News not found for {stock_symbol}: {e}")

        return {"financial_data": financial_data, "news": news}
    except requests.exceptions.HTTPError as e:
        logging.error(f"Yahoo Finance: HTTP error for {stock_symbol}: {e}")
        return None
    except Exception as e:
        logging.error(f"Error scraping Yahoo Finance for {stock_symbol}: {e}")
        return None

def scrape_alpha_vantage(stock_symbol, api_key="YOUR_ALPHA_VANTAGE_API_KEY"):
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
    "NVIDIA": "NVDA",
    "Apple": "AAPL",
    "KCB Group": "KCB",
    "Safaricom": "SCOM",
    # Add more companies as needed
}

# Mapping of exchange names to exchange codes
EXCHANGE_TO_CODE = {
    "NASDAQ": "NASDAQ",
    "New York Stock Exchange": "NYSE",
    "London Stock Exchange": "LSE",
    "Hong Kong Stock Exchange": "HKEX",
    "Nairobi Stock Exchange": "NSE",
    # Add more exchanges as needed
}

def resolve_company_and_exchange(company_name, exchange_name):
    """
    Resolves the stock symbol and exchange code based on the company name and exchange name.

    Args:
        company_name (str): The name of the company.
        exchange_name (str): The name of the exchange.

    Returns:
        tuple: A tuple containing the stock symbol and exchange code, or (None, None) if not found.
    """
    stock_symbol = COMPANY_TO_SYMBOL.get(company_name)
    exchange_code = EXCHANGE_TO_CODE.get(exchange_name)
    if not stock_symbol:
        logging.error(f"Company name '{company_name}' not found in the mapping.")
    if not exchange_code:
        logging.error(f"Exchange name '{exchange_name}' not found in the mapping.")
    return stock_symbol, exchange_code

def main():
    st.title("Financial Analyzer App")
    st.write("Analyze financial data of companies.")

    # Input fields for company name and exchange name
    company_name = st.text_input("Enter the company name (e.g., 'NVIDIA', 'Apple'):")
    exchange_name = st.text_input("Enter the exchange name (e.g., 'NASDAQ', 'Nairobi Stock Exchange'):")


    # File uploader for CSV, Excel, PDF, and image files
    uploaded_file = st.file_uploader("Upload a financial data file (CSV, Excel, PDF, or Image):", type=["csv", "xlsx", "pdf", "png", "jpg", "jpeg"])

    if st.button("Analyze"):
        if not company_name and not uploaded_file:
            st.error("Please provide either the company name and exchange name or upload a file.")
            return

        try:
            financial_data = None

            # Handle uploaded files
            if uploaded_file:
                if uploaded_file.name.endswith(".csv"):
                    financial_data = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(".xlsx"):
                    financial_data = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith(".pdf"):
                    # Convert PDF to CSV
                    st.info("Converting PDF to CSV for efficient data extraction...")
                    csv_path = convert_pdf_to_csv(uploaded_file)
                    if csv_path:
                        st.success(f"PDF successfully converted to CSV: {csv_path}")
                        with open(csv_path, "rb") as f:
                            st.download_button(
                                label="Download Extracted CSV",
                                data=f,
                                file_name="extracted_data.csv",
                                mime="text/csv"
                            )
                        financial_data = pd.read_csv(csv_path)
                        st.write("Extracted Data from PDF (as CSV):")
                        st.dataframe(financial_data)
                    else:
                        st.error("No data could be extracted from the PDF. Please ensure the data is structured.")
                elif uploaded_file.name.endswith((".png", ".jpg", ".jpeg")):
                    # Extract text from image
                    st.info("Extracting text from image...")
                    extracted_text = extract_text_from_image(uploaded_file)
                    if extracted_text:
                        st.write("Extracted Text from Image:")
                        st.text(extracted_text)
                        financial_data = process_extracted_text_to_dataframe(extracted_text)
                        if financial_data is not None and not financial_data.empty:
                            st.write("Structured Data from Image:")
                            st.dataframe(financial_data)
                        else:
                            st.error("Failed to structure data from the extracted text. Please ensure the image contains readable financial data.")
                    else:
                        st.error("Failed to extract text from the image. Please ensure the image is clear and contains readable text.")

            # Resolve stock symbol and exchange code
            stock_symbol, exchange_code = resolve_company_and_exchange(company_name, exchange_name)
            if not stock_symbol or not exchange_code:
                st.error("Could not resolve the stock symbol or exchange code. Please check your inputs.")
                return

            # Handle scraping if no file is uploaded
            if not financial_data and stock_symbol:
                st.info(f"Scraping financial data and news for {company_name} ({stock_symbol}) from {exchange_name}...")
                aggregated_data = aggregate_data(stock_symbol, exchange_code)
                if aggregated_data:
                    st.write("Aggregated Financial Data:")
                    st.json(aggregated_data.get("financial_data", {}))

                    st.write("Relevant News:")
                    news_items = aggregated_data.get("news", [])
                    if news_items:
                        for news_item in news_items:
                            st.markdown(f"- [{news_item['title']}]({news_item['link']})")
                    else:
                        st.info("No relevant news found for this company.")
                else:
                    st.error(f"Failed to scrape financial data or news for {company_name}. Please try again later.")

            # Display and analyze financial data
            if financial_data is not None and not financial_data.empty:
                st.write("Financial Data:")
                st.dataframe(financial_data)

                st.write("Calculated Financial Ratios:")
                analyzer = FinancialAnalyzer(company_name=stock_symbol, financial_data=financial_data)
                ratios = analyzer.calculate_ratios()
                if ratios:
                    st.json(ratios)
                else:
                    st.warning("Could not calculate financial ratios. Check the data format.")
            else:
                st.error("Failed to process financial data. Check the file or URL.")

        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.error(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    main()