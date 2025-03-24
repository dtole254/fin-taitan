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
        self.tracking_thread = None

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
            data = self.financial_data.copy() # Create a copy to avoid modifying original

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
            if not total_equity_