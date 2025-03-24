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
parser.add_argument("--user_agent", type=str, default="FinancialAnalyzerBot/1.0 (+https://github.com/yourusername/FinancialAnalysisApp)",
                    help="User-Agent for HTTP requests")  # Replace yourusername
parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for HTTP requests")
parser.add_argument("--retry_delay", type=int, default=5, help="Initial delay (in seconds) between retries")
args = parser.parse_args()

# Load configuration from file if provided, or use a default configuration
DEFAULT_CONFIG = {
    "user_agent": "FinancialAnalyzerBot/1.0 (+https://github.com/yourusername/FinancialAnalysisApp)",  # Replace yourusername
    "max_retries": 3,
    "retry_delay": 5,
    "robots_timeout": 5,
    "scraping_timeout": 10,
    "news_api_key": None,  # Add default value for news API key
    "alpha_vantage_api_key": None,  # Add default value for Alpha Vantage API
    "enable_selenium": False,  # Add default value for selenium
    "news_source": "NewsAPI",  # Default news source.
    "finnhub_api_key": None,  # Add default value for Finnhub API Key
    "world_indices_api": "AlphaVantage",  # Default for world indices
    "rapidapi_key": None, # Default for RapidAPI
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
            WORLD_INDICES_API = config.get("world_indices_api", DEFAULT_CONFIG["world_indices_api"])
            RAPIDAPI_KEY = config.get("rapidapi_key", DEFAULT_CONFIG["rapidapi_key"])
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
            if WORLD_INDICES_API not in ["AlphaVantage", "RapidAPI"]:
                raise ValueError("world_indices_api must be either 'AlphaVantage' or 'RapidAPI'")

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
        WORLD_INDICES_API = DEFAULT_CONFIG["world_indices_api"]
        RAPIDAPI_KEY = DEFAULT_CONFIG["rapidapi_key"]

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
    WORLD_INDICES_API = DEFAULT_CONFIG["world_indices_api"]
    RAPIDAPI_KEY = DEFAULT_CONFIG["rapidapi_key"]

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
        self.stock_data = None  # To store stock price
        self.news = []  # To store news articles
        self.ticker = None
        self.indices_data = {}

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
                # Return the symbol of the top match
                return data["bestMatches"][0]["1. symbol"]
            else:
                logging.warning(f"Could not resolve ticker for company: {company_name} using Alpha Vantage.")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error resolving company ticker with Alpha Vantage: {e}")
            return None
        except json.JSONDecodeError:
            logging.error("Error decoding JSON response from Alpha Vantage.")
            return None

    def fetch_stock_price(self, ticker):
        """
        Fetches the latest stock price for the given ticker symbol using Alpha Vantage.

        Args:
            ticker (str): The stock ticker symbol.

        Returns:
            float: The stock price, or None if an error occurs.
        """
        if not ALPHA_VANTAGE_API_KEY:
            logging.warning("Alpha Vantage API key is missing. Stock price fetching will be skipped.")
            return None

        url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={ticker}&apikey={ALPHA_VANTAGE_API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and "Global Quote" in data and data["Global Quote"]["02. symbol"]:
                # Check if the price is available.
                if data["Global Quote"]["05. price"] and data["Global Quote"]["05. price"] != "None":
                    price = float(data["Global Quote"]["05. price"])
                    return price
                else:
                    logging.warning(f"Could not retrieve stock price for ticker: {ticker} from Alpha Vantage.")
                    return None
            else:
                logging.warning(f"Could not retrieve stock price for ticker: {ticker} from Alpha Vantage.")
                return None
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching stock price from Alpha Vantage: {e}")
            return None
        except json.JSONDecodeError:
            logging.error("Error decoding JSON response from Alpha Vantage.")
            return None

    def fetch_company_news(self, company_name):
        """
        Fetches the latest news for the company.  Uses either NewsAPI or Finnhub.

        Args:
            company_name (str): The name of the company.

        Returns:
            list: A list of news articles, or an empty list if no news is found or an error occurs.
        """
        if NEWS_SOURCE == "NewsAPI":
            return self._fetch_news_from_newsapi(company_name)
        elif NEWS_SOURCE == "Finnhub":
            return self._fetch_news_from_finnhub(company_name)
        else:
            logging.error(f"Invalid news source: {NEWS_SOURCE}")
            return []

    def _fetch_news_from_newsapi(self, company_name):
        """Fetches news from NewsAPI"""
        if not NEWS_API_KEY:
            logging.warning("News API key is missing. News fetching from NewsAPI will be skipped.")
            return []

        encoded_company_name = urllib.parse.quote_plus(company_name)
        url = f"https://newsapi.org/v2/everything?q={encoded_company_name}&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            articles = data.get('articles', [])  # Ensure it returns a list
            if not articles:
                logging.warning(f"No news found for {company_name} in News API.")
            return articles
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching news for {company_name} from News API: {e}")
            return []
        except json.JSONDecodeError:
            logging.error("Error decoding JSON response from News API.")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching news: {e}")
            return []

    def _fetch_news_from_finnhub(self, company_name):
        """Fetches news from Finnhub"""
        if not FINNHUB_API_KEY:
            logging.warning("Finnhub API key is missing. News fetching from Finnhub will be skipped.")
            return []

        if not self.ticker:
            logging.warning(f"Ticker is required to fetch news from Finnhub for {company_name}")
            return []

        url = f"https://finnhub.io/api/v3/company-news?symbol={self.ticker}&token={FINNHUB_API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            news_list = response.json()
            if not news_list:
                logging.warning(f"No news found for {company_name} (ticker: {self.ticker}) from Finnhub.")
            return news_list
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching news for {company_name} from Finnhub: {e}")
            return []
        except json.JSONDecodeError:
            logging.error("Error decoding JSON response from Finnhub.")
            return []
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching news from Finnhub: {e}")
            return []

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
                st.error(
                    f"Scraping is not allowed by robots.txt at {robots_url} for URL: {self.website_url}")
                logging.error(
                    f"Scraping is not allowed by robots.txt at {robots_url} for URL: {self.website_url}")
                return None
        except Exception as e:
            st.warning(f"Error while parsing robots.txt at {robots_url} for URL: {self.website_url}: {e}")
            logging.warning(
                f"Error while parsing robots.txt at {robots_url} for URL: {self.website_url}: {e}")

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
                        rows = table.find_all("tr")  # tr for rows
                        for row in rows:
                            cols = row.find_all(
                                ["td", "th"])  # both td and th
                            cols = [ele.text.strip() for ele in cols]
                            data.append(cols)

                        df = pd.DataFrame(data)

                        if not df.empty:
                            # Ensure the first row is suitable for headers
                            if df.iloc[0].isnull().any():
                                st.error(
                                    f"Table headers are missing or invalid in URL: {self.website_url}")
                                logging.error(
                                    f"Table headers are missing or invalid in URL: {self.website_url}")
                                return None

                            df.columns = df.iloc[0]
                            df = df[1:]
                            df = df.dropna(axis=1, how='all')
                            df = df.dropna(axis=0, how='all')

                            for col in df.columns:
                                try:
                                    df[col] = pd.to_numeric(df[col].str.replace(r'[$,()]', '', regex=True),
                                                           errors='coerce')
                                except (ValueError, AttributeError):
                                    pass

                            # Validate for negative values
                            if (df.select_dtypes(include=['number']) < 0).any().any():
                                st.warning(
                                    f"Negative values detected in financial data from URL: {self.website_url}")
                                logging.warning(
                                    f"Negative values detected in financial data from URL: {self.website_url}")

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
                    st.warning(
                        f"Too many requests to {self.website_url}. Retrying in {delay} seconds... (Attempt {retries}/{MAX_RETRIES})")
                    logging.warning(
                        f"HTTP 429 Too Many Requests for URL {self.website_url}: Retrying in {delay} seconds... (Attempt {retries}/{MAX_RETRIES})")
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
                st.error(
                    f"An unexpected error occurred while scraping from URL {self.website_url}: {e}")
                logging.error(
                    f"An unexpected error occurred while scraping from URL {self.website_url}: {e}")
                return None

        st.error(f"Failed to scrape data from {self.website_url} after {MAX_RETRIES} retries.")
        logging.error(f"Failed toscrape data from {self.website_url} after {MAX_RETRIES} retries.")
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
                    match = next(
                        (col for col in data.columns if re.search(pattern, col, re.IGNORECASE)), None)
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
            if not cash_col:missing_columns.append("Cash")
            if not inventory_col:
                missing_columns.append("Inventory")
            if not cogs_col:
                missing_columns.append("COGS")

            if missing_columns:
                st.error(f"Missing required financial data columns: {', '.join(missing_columns)}")
                logging.error(f"Missing required financial data columns: {', '.join(missing_columns)}")
                return None

            # --- Calculate Ratios ---
            ratios = {}
            try:
                # Profitability Ratios
                if revenue_col and net_income_col:
                    ratios['net_profit_margin'] = data[net_income_col].sum() / data[revenue_col].sum()
                if revenue_col and cogs_col:
                  ratios['gross_profit_margin'] = (data[revenue_col].sum() - data[cogs_col].sum()) / data[revenue_col].sum()
                if total_equity_col and net_income_col:
                    ratios['return_on_equity'] = data[net_income_col].sum() / data[total_equity_col].sum()
                if total_assets_col and net_income_col:
                    ratios['return_on_assets'] = data[net_income_col].sum() / data[total_assets_col].sum()

                # Liquidity Ratios
                if current_assets_col and current_liabilities_col:
                    ratios['current_ratio'] = data[current_assets_col].sum() / data[current_liabilities_col].sum()
                    ratios['quick_ratio'] = (data[current_assets_col].sum() - data[inventory_col].sum()) / data[current_liabilities_col].sum()
                if cash_col and current_liabilities_col:
                    ratios['cash_ratio'] = data[cash_col].sum() / data[current_liabilities_col].sum()

                # Solvency Ratios
                if total_liabilities_col and total_equity_col:
                    ratios['debt_to_equity_ratio'] = data[total_liabilities_col].sum() / data[total_equity_col].sum()
                if total_assets_col and total_equity_col:
                    ratios['equity_multiplier'] = data[total_assets_col].sum() / data[total_equity_col].sum()

                # Efficiency Ratios
                if revenue_col and total_assets_col:
                    ratios['asset_turnover_ratio'] = data[revenue_col].sum() / data[total_assets_col].sum()
                if cogs_col and inventory_col:
                    ratios['inventory_turnover_ratio'] = data[cogs_col].sum() / data[inventory_col].sum()

                return ratios
            except ZeroDivisionError:
                st.error("Error: Division by zero encountered while calculating ratios.")
                logging.error("Error: Division by zero encountered while calculating ratios.")
                return None
            except Exception as e:
                st.error(f"An error occurred while calculating ratios: {e}")
                logging.error(f"An error occurred while calculating ratios: {e}")
                return None

    def analyze_financial_health(self, ratios):
        """
        Analyzes the financial health of the company based on the calculated ratios.

        Args:
            ratios (dict): A dictionary containing the calculated financial ratios.

        Returns:
            dict: A dictionary containing the analysis of the financial health, or None if an error occurs.
        """
        if ratios is None:
            st.error("No ratios provided for analysis.")
            logging.error("No ratios provided for analysis.")
            return None

        analysis = {}
        try:
            # Profitability Analysis
            analysis['net_profit_margin_analysis'] = "Good" if ratios.get('net_profit_margin', 0) > 0.1 else "Poor"
            analysis['gross_profit_margin_analysis'] = "Good" if ratios.get('gross_profit_margin', 0) > 0.3 else "Poor"
            analysis['return_on_equity_analysis'] = "Good" if ratios.get('return_on_equity', 0) > 0.15 else "Poor"
            analysis['return_on_assets_analysis'] = "Good" if ratios.get('return_on_assets', 0) > 0.05 else "Poor"

            # Liquidity Analysis
            analysis['current_ratio_analysis'] = "Good" if ratios.get('current_ratio', 0) > 1.5 else "Poor"
            analysis['quick_ratio_analysis'] = "Good" if ratios.get('quick_ratio', 0) > 1 else "Poor"
            analysis['cash_ratio_analysis'] = "Good" if ratios.get('cash_ratio', 0) > 0.5 else "Poor"

            # Solvency Analysis
            analysis['debt_to_equity_ratio_analysis'] = "Low" if ratios.get('debt_to_equity_ratio', 0) < 1 else "High"
            analysis['equity_multiplier_analysis'] = "Low" if ratios.get('equity_multiplier', 0) < 2 else "High"

            # Efficiency Analysis
            analysis['asset_turnover_ratio_analysis'] = "Efficient" if ratios.get('asset_turnover_ratio', 0) > 1 else "Inefficient"
            analysis['inventory_turnover_ratio_analysis'] = "Efficient" if ratios.get('inventory_turnover_ratio', 0) > 6 else "Inefficient"

            return analysis
        except Exception as e:
            st.error(f"An error occurred during financial health analysis: {e}")
            logging.error(f"An error occurred during financial health analysis: {e}")
            return None

    def extract_text_from_pdf(self, file_path):
        """
        Extracts text from a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            str: The extracted text, or None if an error occurs.
        """
        try:
            with open(file_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""  # changed from extractText to extract_text
                return text
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            logging.error(f"Error extracting text from PDF: {e}")
            return None

    def extract_tables_from_pdf(self, file_path):
        """
        Extracts tables from a PDF file.

        Args:
            file_path (str): The path to the PDF file.

        Returns:
            list: A list of pandas DataFrames, or None if no tables are found or an error occurs.
        """
        try:
            tables = []
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    page_tables = page.extract_tables()
                    if page_tables:
                        for table in page_tables:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            tables.append(df)
            return tables
        except Exception as e:
            st.error(f"Error extracting tables from PDF: {e}")
            logging.error(f"Error extracting tables from PDF: {e}")
            return None

    def extract_text_from_image(self, image_path):
        """
        Extracts text from an image using OCR.

        Args:
            image_path (str): The path to the image file.

        Returns:
            str: The extracted text, or None if an error occurs.
        """
        try:
            img = Image.open(image_path)
            text = pytesseract.image_to_string(img)
            return text
        except Exception as e:
            st.error(f"Error extracting text from image: {e}")
            logging.error(f"Error extracting text from image: {e}")
            return None

    def load_financial_data(self, file_path):
        """
        Loads financial data from a file (CSV, Excel, PDF, image).

        Args:
            file_path (str): The path to the file.

        Returns:
            pd.DataFrame: The loaded financial data, or None if an error occurs.
        """
        if not file_path:
            st.error("File path is empty.")
            logging.error("File path is empty.")
            return None

        try:
            if file_path.lower().endswith(('.csv', '.txt')):
                return pd.read_csv(file_path)
            elif file_path.lower().endswith(('.xls', '.xlsx')):
                return pd.read_excel(file_path)
            elif file_path.lower().endswith('.pdf'):
                # First, try to extract tables.
                tables = self.extract_tables_from_pdf(file_path)
                if tables:
                    #  For simplicity, use the first table if multiple are found.
                    return tables[0]
                else:
                    # If no tables, then extract text and attempt to parse.
                    text = self.extract_text_from_pdf(file_path)
                    if text:
                        #  basic parsing, assuming comma or space separated
                        lines = text.split('\n')
                        data = [line.split(',') for line in lines]  #  comma
                        if not data or not any(data):
                           data = [line.split() for line in lines] # space
                        # Remove any empty lists
                        data = [row for row in data if row]
                        return pd.DataFrame(data)
                    else:
                        return None  # Return None if no table and no text
            elif file_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                text = self.extract_text_from_image(file_path)
                if text:
                    lines = text.split('\n')
                    data = [line.split(',') for line in lines] # comma
                    if not data or not any(data):
                        data = [line.split() for line in lines]  # space
                    data = [row for row in data if row]
                    return pd.DataFrame(data)
                else:
                    return None
            else:
                st.error("Unsupported file format. Please upload a CSV, Excel, PDF, or image file.")
                logging.error("Unsupported file format.")
                return None
        except FileNotFoundError:
            st.error(f"File not found at {file_path}")
            logging.error(f"File not found at {file_path}")
            return None
        except Exception as e:
            st.error(f"Error loading data from {file_path}: {e}")
            logging.error(f"Error loading data from {file_path}: {e}")
            return None

    def fetch_world_indices(self):
        """
        Fetches data for major world indices.  Uses AlphaVantage or RapidAPI

        Returns:
            dict: A dictionary containing the world indices data, or None if an error occurs.
        """
        if WORLD_INDICES_API == "AlphaVantage":
            return self._fetch_world_indices_alpha_vantage()
        elif WORLD_INDICES_API == "RapidAPI":
            return self._fetch_world_indices_rapidapi()
        else:
            logging.error(f"Invalid world indices API: {WORLD_INDICES_API}")
            return {}

    def _fetch_world_indices_alpha_vantage(self):
        """Fetches world indices using Alpha Vantage."""
        if not ALPHA_VANTAGE_API_KEY:
            logging.warning("Alpha Vantage API key is missing. Fetching world indices will be skipped.")
            return {}

        indices = {
            "S&P 500": "SPX",
            "Dow Jones": "DJIA",
            "Nasdaq": "IXIC",
            "FTSE 100": "FTSE",
            "Nikkei 225": "N225",
            "Hang Seng": "HSI"
        }
        data = {}
        for index_name, symbol in indices.items():
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                index_data = response.json()
                if "Time Series (Daily)" in index_data:
                    # Get the latest day's data
                    latest_date = sorted(index_data["Time Series (Daily)"].keys())[0]
                    data[index_name] = {
                        "price": float(index_data["Time Series (Daily)"][latest_date]["4. close"]),
                        "date": latest_date
                    }
                else:
                    logging.warning(f"Could not retrieve data for {index_name} from Alpha Vantage.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching world indices data from Alpha Vantage for {index_name}: {e}")
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON response for world indices from Alpha Vantage for {index_name}.")
            except KeyError:
                logging.error(f"Key error while processing data for {index_name} from Alpha Vantage.")
        return data

    def _fetch_world_indices_rapidapi(self):
        """Fetches world indices using RapidAPI."""
        if not RAPIDAPI_KEY:
            logging.warning("RapidAPI key is missing. Fetching world indices will be skipped.")
            return {}

        url = "https://world-stock-index.p.rapidapi.com/v1/world-stock-index/list-most-traded"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "world-stock-index.p.rapidapi.com"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            indices_data = {}
            for item in data.get('data', []):
                index_name = item.get('index_name')
                price = item.get('price')
                last_updated = item.get('last_updated')
                if index_name and price and last_updated:
                  indices_data[index_name] = {
                    "price": price,
                    "date": last_updated
                }
            return indices_data
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching world indices data from RapidAPI: {e}")
            return {}
        except json.JSONDecodeError:
            logging.error("Error decoding JSON response for world indices from RapidAPI.")
            return {}
        except KeyError:
                logging.error(f"Key error while processing data from RapidAPI.")
                return {}

    def display_financial_data(self):
        """
        Displays the financial data, ratios, analysis, stock price, and news in the Streamlit app.
        """
        st.title("Financial Analysis of " + self.company_name)

        # Display company news
        st.header("Latest News")
        if self.news:
            for article in self.news:
                if NEWS_SOURCE == "NewsAPI":
                    st.subheader(article['title'])
                    st.write(article['description'])
                    st.write(f"Source: {article['source']['name']}")
                    st.write(f"Published At: {article['publishedAt']}")
                    st.write(f"[Link]({article['url']})")
                elif NEWS_SOURCE == "Finnhub":
                    st.subheader(article['headline'])
                    st.write(article['summary'])
                    st.write(f"Source: {article['source']}")
                    st.write(f"Published At: {article['datetime']}") # needs conversion
                    st.write(f"[Link]({article['url']})")
                st.markdown("---")
        else:
            st.write("No news available.")

        # Display stock price
        st.header("Stock Price")
        if self.stock_data is not None:
            st.write(f"Current Price: {self.stock_data}")
            st.write(f"Ticker Symbol: {self.ticker}")
        else:
            st.write("Stock price not available.")

        # Display financial data
        st.header("Financial Data")
        if self.financial_data is not None:
            st.dataframe(self.financial_data)
        else:
            st.write("No financial data available.")

        # Calculate and display ratios and analysis
        st.header("Financial Ratios")
        ratios = self.calculate_ratios()
        if ratios:
            st.json(ratios)
            analysis = self.analyze_financial_health(ratios)
            st.header("Financial Health Analysis")
            if analysis:
                st.json(analysis)
            else:
                st.write("Could not perform financial health analysis.")
        else:
            st.write("Could not calculate financial ratios.")

        # Display world indices
        st.header("World Indices")
        if self.indices_data:
            st.dataframe(pd.DataFrame(self.indices_data).T)
        else:
            st.write("Could not fetch world indices.")

def main():
    """
    Main function to run the financial analysis.
    """
    st.title("Financial Analyzer")

    # Input for company name
    company_name = st.text_input("Enter the name of the company:")
    if not company_name:
        st.stop()

    # Input for website URL
    website_url = st.text_input("Enter the company's website URL (optional):")

    # File upload for financial data
    uploaded_file = st.file_uploader("Upload financial data (CSV, Excel, PDF, image)",
                                    type=["csv", "xls", "xlsx", "pdf", "jpg", "jpeg", "png"])
    file_path = None
    if uploaded_file:
        # Save the uploaded file to a temporary location
        file_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

    # Initialize FinancialAnalyzer
    analyzer = FinancialAnalyzer(company_name, website_url)

    # Resolve company ticker
    analyzer.ticker = analyzer.resolve_company_ticker(company_name)

    # Fetch stock price
    if analyzer.ticker:
        analyzer.stock_data = analyzer.fetch_stock_price(analyzer.ticker)

    # Fetch company news
    analyzer.news = analyzer.fetch_company_news(company_name)

    # Fetch world indices
    analyzer.indices_data = analyzer.fetch_world_indices()

    # Load financial data
    if uploaded_file:
        analyzer.financial_data = analyzer.load_financial_data(file_path)
    elif website_url:
        analyzer.financial_data = analyzer.scrape_financial_data()

    # Perform analysis and display results
    analyzer.display_financial_data()

    # Clean up the temporary file
    if file_path and os.path.exists(file_path):
        os.remove(file_path)


if __name__ == "__main__":
    main()
