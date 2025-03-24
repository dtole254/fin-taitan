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
        self.stop_tracking = False
        self.last_known_financial_data = None # Store the last known good financial data
        self.last_known_news = []
        self.driver = None # Selenium webdriver

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
            if not total_equity_col:
                missing_columns.append("Total Equity")
            if not cash_col:
                missing_columns.append("Cash")
            if not inventory_col:
                missing_columns.append("Inventory")
            if not cogs_col:
                missing_columns.append("COGS")

            if missing_columns:
                st.error(f"Missing required financial data: {', '.join(missing_columns)}")
                logging.error(f"Missing required financial data: {', '.join(missing_columns)}")
                return None

            # Perform calculations
            ratios = {}
            try:
                # Basic Ratios
                ratios['gross_profit_margin'] = ((data[revenue_col] - data[cogs_col]) / data[revenue_col]).mean() if revenue_col and cogs_col else None
                ratios['net_profit_margin'] = (data[net_income_col] / data[revenue_col]).mean() if net_income_col and revenue_col else None
                ratios['return_on_assets'] = (data[net_income_col] / data[total_assets_col]).mean() if net_income_col and total_assets_col else None
                ratios['return_on_equity'] = (data[net_income_col] / data[total_equity_col]).mean() if net_income_col and total_equity_col else None
                ratios['debt_to_equity_ratio'] = (data[total_liabilities_col] / data[total_equity_col]).mean() if total_liabilities_col and total_equity_col else None
                ratios['current_ratio'] = (data[current_assets_col] / data[current_liabilities_col]).mean() if current_assets_col and current_liabilities_col else None
                ratios['quick_ratio'] = ((data[current_assets_col] - data[inventory_col]) / data[current_liabilities_col]).mean() if current_assets_col and inventory_col and current_liabilities_col else None
                ratios['cash_ratio'] = (data[cash_col] / data[current_liabilities_col]).mean() if cash_col and current_liabilities_col else None
                ratios['inventory_turnover'] = (data[cogs_col] / data[inventory_col]).mean() if cogs_col and inventory_col else None
                ratios['asset_turnover'] = (revenue_col / data[total_assets_col]).mean() if revenue_col and total_assets_col else None


                # Additional ratios - requires more specific columns
                #EBITDA  Earnings Before Interest, Taxes, Depreciation and Amortization.
                #ocf: operating cash flow
                ebitda_col = find_column([r'ebitda', r'earnings before interest, taxes, depreciation and amortization'])
                depreciation_col = find_column([r'depreciation', r'depreciation and amortization'])
                interest_expense_col = find_column([r'interest expense', r'interest paid'])
                operating_cash_flow_col = find_column([r'operating cash flow', r'cash flow from operations', r'net cash from operating activities'])

                if ebitda_col is None:
                    if revenue_col and cogs_col and depreciation_col and interest_expense_col:
                        data['ebitda_calculated'] = data[revenue_col] - data[cogs_col] + data[depreciation_col] + data[interest_expense_col]
                        ebitda_col = 'ebitda_calculated'
                    else:
                        ebitda_col = None

                ratios['ebitda_margin'] = (data[ebitda_col] / data[revenue_col]).mean() if ebitda_col and revenue_col else None

                ratios['free_cash_flow'] = (data[operating_cash_flow_col] - data[depreciation_col]).mean() if operating_cash_flow_col and depreciation_col else None
                ratios['free_cash_flow_per_share'] = (ratios['free_cash_flow'] / data['Shares Outstanding']).mean() if 'free_cash_flow' in ratios and 'Shares Outstanding' in data else None #Need to find shares outstanding.

                ratios['price_to_earnings'] = (self.stock_data / (data[net_income_col] / data['Shares Outstanding']).mean()) if self.stock_data and net_income_col and 'Shares Outstanding' in data else None
                ratios['price_to_book'] = (self.stock_data / (data[total_equity_col] / data['Shares Outstanding']).mean()) if self.stock_data and total_equity_col and 'Shares Outstanding' in data else None
                ratios['earnings_per_share'] = (data[net_income_col] / data['Shares Outstanding']).mean() if net_income_col and 'Shares Outstanding' in data else None
                ratios['book_value_per_share'] = (data[total_equity_col] / data['Shares Outstanding']).mean() if total_equity_col and 'Shares Outstanding' in data else None

                #Leverage Ratios
                ratios['debt_to_assets_ratio'] = (data[total_liabilities_col] / data[total_assets_col]).mean() if total_liabilities_col and total_assets_col else None
                ratios['financial_leverage'] = (data[total_assets_col] / data[total_equity_col]).mean() if total_assets_col and total_equity_col else None
                ratios['interest_coverage_ratio'] = (data[ebitda_col] / data[interest_expense_col]).mean() if ebitda_col and interest_expense_col else None #Check

                return ratios
            except Exception as e:
                st.error(f"Error calculating financial ratios: {e}")
                logging.error(f"Error calculating financial ratios: {e}")
                return None

        return None

    def display_financial_data(self):
        """
        Displays the financial data and calculated ratios.
        """
        if self.financial_data is not None and isinstance(self.financial_data, pd.DataFrame):
            st.subheader("Financial Data")
            st.dataframe(self.financial_data)  # Use st.dataframe for better display
        else:
            st.info("No financial data available to display.")

        ratios = self.calculate_ratios()
        if ratios:
            st.subheader("Financial Ratios")
            ratios_df = pd.DataFrame(list(ratios.items()), columns=['Ratio', 'Value'])  # Convert to DataFrame
            st.dataframe(ratios_df)
        else:
            st.info("No financial ratios to display.")

    def display_stock_data(self):
        """
        Displays the stock price.
        """
        if self.stock_data is not None:
            st.subheader("Stock Data")
            st.metric(label="Stock Price", value=self.stock_data)
        else:
            st.info("No stock data available to display.")

    def display_company_news(self):
        """
        Displays the latest news for the company.
        """
        if self.news:
            st.subheader("Company News")
            for article in self.news:
                if NEWS_SOURCE == "NewsAPI":
                    st.markdown(f"**{article['title']}**")
                    st.markdown(f"_{article['source']['name']}_ - {article['publishedAt']}")
                    st.markdown(f"{article['description']}")
                    st.markdown(f"[Read more]({article['url']})")
                    st.write("---")  # Separator
                elif NEWS_SOURCE == "Finnhub":
                    st.markdown(f"**{article['headline']}**")
                    st.markdown(f"_Source: {article['source']}_ - {article['datetime']}")
                    st.markdown(f"{article['summary']}")
                    st.markdown(f"[Read more]({article['url']})")
                    st.write("---")  # Separator
        else:
            st.info("No news available to display.")

    def extract_financial_data_from_pdf(self, pdf_file):
        """
        Extracts financial data from a PDF file.

        Args:
            pdf_file (UploadedFile): The PDF file uploaded by the user.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted financial data, or None if an error occurs.
        """
        try:
            # Save the uploaded file to a temporary location
            temp_file_path = f"/tmp/{pdf_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(pdf_file.read())

            # Extract tables using pdfplumber
            tables = []
            with pdfplumber.open(temp_file_path) as pdf:
                for page in pdf.pages:
                    tables.extend(page.extract_tables())

            # Process the tables
            for table in tables:
                df = pd.DataFrame(table[1:], columns=table[0])
                df = df.dropna(axis=1, how='all')
                df = df.dropna(axis=0, how='all')

                for col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col].str.replace(r'[$,()]', '', regex=True),
                                                       errors='coerce')
                    except (ValueError, AttributeError):
                        pass
                # Basic validation: Check for non-empty and numeric data
                if not df.empty and df.select_dtypes(include=['number']).any().any():
                    return df

            # If no suitable table is found
            st.error("No suitable financial data table found in the PDF.")
            logging.error("No suitable financial data table found in the PDF.")
            return None

        except PyPDF2.errors.PdfReadError as e:
            st.error(f"Error reading PDF: {e}.  It may be encrypted or corrupted.")
            logging.error(f"Error reading PDF: {e}.  It may be encrypted or corrupted.")
            return None
        except Exception as e:
            st.error(f"An error occurred while extracting data from PDF: {e}")
            logging.error(f"An error occurred while extracting data from PDF: {e}")
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def extract_text_from_image(self, image_file):
        """
        Extracts text from an image file using OCR.

        Args:
            image_file (UploadedFile): The image file uploaded by the user.

        Returns:
            str: The extracted text, or None if an error occurs.
        """
        try:
            # Save the image to a temporary file
            temp_file_path = f"/tmp/{image_file.name}"
            with open(temp_file_path, "wb") as f:
                f.write(image_file.read())

            # Open the image using PIL
            image = Image.open(temp_file_path)

            # Use pytesseract to extract text
            text = pytesseract.image_to_string(image)
            return text

        except Exception as e:
            st.error(f"Error extracting text from image: {e}")
            logging.error(f"Error extracting text from image: {e}")
            return None
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)

    def extract_financial_data_from_text(self, text):
        """
        Extracts financial data from text using regular expressions.

        Args:
            text (str): The text to extract data from.

        Returns:
            pd.DataFrame: A DataFrame containing the extracted financial data, or None if no data is found.
        """
        # Define regular expressions for financial data extraction
        patterns = {
            "Revenue": r"Revenue\s*([$0-9.,]+)",
            "Net Income": r"Net Income\s*([$0-9.,]+)",
            "Total Assets": r"Total Assets\s*([$0-9.,]+)",
            "Total Liabilities": r"Total Liabilities\s*([$0-9.,]+)",
            "Current Assets": r"Current Assets\s*([$0-9.,]+)",
            "Current Liabilities": r"Current Liabilities\s*([$0-9.,]+)",
            "Total Equity": r"Total Equity\s*([$0-9.,]+)",
            "Cash": r"Cash\s*([$0-9.,]+)",
            "Inventory": r"Inventory\s*([$0-9.,]+)",
            "COGS": r"Cost of Goods Sold\s*([$0-9.,]+)",
        }

        data = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, text)
            if match:
                data[key] = float(match.group(1).replace(",", "").replace("$", ""))
            else:
                data[key] = None  # Important: Explicitly set None for missing data

        # Create a DataFrame
        df = pd.DataFrame([data])

        # If all values are None, return None
        if all(value is None for value in data.values()):
            return None

        return df
    
    def initialize_selenium(self):
        """Initializes the Selenium webdriver."""
        if not ENABLE_SELENIUM:
            return

        try:
            chrome_options = Options()
            chrome_options.add_argument("--headless")  # Run in headless mode
            chrome_options.add_argument("--disable-gpu")  # Disable GPU acceleration
            chrome_options.add_argument("--no-sandbox")  # Bypass OS security restrictions
            chrome_options.add_argument("--disable-dev-shm-usage")  # Overcome limited resource
            # Set user agent
            chrome_options.add_argument(f"user-agent={USER_AGENT}")

            # Use ChromeDriverManager to automatically download and manage the driver
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            self.driver.set_page_load_timeout(SCRAPING_TIMEOUT)  # Set page load timeout

        except Exception as e:
            st.error(f"Failed to initialize Selenium: {e}")
            logging.error(f"Failed to initialize Selenium: {e}")
            self.driver = None  # Ensure driver is None on failure

    def scrape_financial_data_selenium(self):
        """
        Scrapes financial data from the company's website using Selenium.  This is used
        for websites that heavily rely on JavaScript.

        Returns:
            pd.DataFrame: A DataFrame containing the scraped financial data, or None if an error occurs.
        """
        if not ENABLE_SELENIUM:
            st.error("Selenium is not enabled in the configuration.")
            logging.error("Selenium is not enabled in the configuration.")
            return None

        if not self.driver:
            st.error("Selenium driver is not initialized.")
            logging.error("Selenium driver is not initialized.")
            return None

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

        try:
            self.driver.get(self.website_url)
            # Add a delay to allow JavaScript to render (adjust as needed)
            time.sleep(5)

            # Get the page source after JavaScript execution
            page_source = self.driver.page_source
            soup = BeautifulSoup(page_source, "html.parser")

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

        except Exception as e:
            st.error(f"Error during scraping with Selenium from URL {self.website_url}: {e}")
            logging.error(f"Error during scraping with Selenium from URL {self.website_url}: {e}")
            return None
    
    def fetch_world_indices(self):
        """
        Fetches major world indices data.  Uses either AlphaVantage or RapidAPI.
        """
        if WORLD_INDICES_API == "AlphaVantage":
            self._fetch_world_indices_alpha_vantage()
        elif WORLD_INDICES_API == "RapidAPI":
            self._fetch_world_indices_rapidapi()
        else:
            logging.error(f"Invalid world indices API: {WORLD_INDICES_API}")

    def _fetch_world_indices_alpha_vantage(self):
        """Fetches world indices using Alpha Vantage."""
        if not ALPHA_VANTAGE_API_KEY:
            logging.warning("Alpha Vantage API key is missing. World indices fetching will be skipped.")
            return

        indices = {
            "S&P 500": "S&P500",
            "Dow Jones": "DJIA",
            "Nasdaq": "NASDAQ",
            "FTSE 100": "FTSE",
            "Nikkei 225": "N225",
            "Hang Seng": "HSI",
            "Euro Stoxx 50": "STOXX50E"
        }
        for index_name, symbol in indices.items():
            url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
            try:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                data = response.json()
                if data and "Time Series (Daily)" in data:
                    # Get the latest day's data
                    latest_date = sorted(data["Time Series (Daily)"].keys())[0]
                    index_data = data["Time Series (Daily)"][latest_date]
                    self.indices_data[index_name] = {
                        "price": float(index_data["4. close"]),
                        "change": float(index_data["4. close"]) - float(index_data["1. open"]),
                        "date": latest_date
                    }
                else:
                    logging.warning(f"Could not retrieve data for {index_name} from Alpha Vantage.")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error fetching world indices data from Alpha Vantage for {index_name}: {e}")
            except json.JSONDecodeError:
                logging.error(f"Error decoding JSON response from Alpha Vantage for {index_name}.")
            except Exception as e:
                logging.error(f"An unexpected error occurred while fetching world indices data from Alpha Vantage for {index_name}: {e}")

    def _fetch_world_indices_rapidapi(self):
        """Fetches world indices using RapidAPI."""
        if not RAPIDAPI_KEY:
            logging.warning("RapidAPI key is missing. World indices fetching will be skipped.")
            return
        
        url = "https://world-stock-index.p.rapidapi.com/v1/worldindices"
        headers = {
            "X-RapidAPI-Key": RAPIDAPI_KEY,
            "X-RapidAPI-Host": "world-stock-index.p.rapidapi.com"
        }
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            data = response.json()
            if data and "data" in data:
                for item in data["data"]:
                    index_name = item["name"]
                    # Map RapidAPI names to standard names
                    name_mapping = {
                        "S&P 500": "S&P 500",
                        "Dow Jones Industrial Average": "Dow Jones",
                        "NASDAQ Composite": "Nasdaq",
                        "FTSE 100 Index": "FTSE 100",
                        "Nikkei 225 Index": "Nikkei 225",
                        "Hang Seng Index": "Hang Seng",
                        "Euro Stoxx 50 Index": "Euro Stoxx 50"
                    }
                    if index_name in name_mapping:
                        index_name = name_mapping[index_name]
                        self.indices_data[index_name] = {
                            "price": item["price"],
                            "change": item["change"],
                            "date": item["date"]
                        }
            else:
                logging.warning(f"Could not retrieve world indices data from RapidAPI.")
        except requests.exceptions.RequestException as e:
            logging.error(f"Error fetching world indices data from RapidAPI: {e}")
        except json.JSONDecodeError:
            logging.error("Error decoding JSON response from RapidAPI.")
        except Exception as e:
            logging.error(f"An unexpected error occurred while fetching world indices data from RapidAPI: {e}")

    def display_world_indices(self):
        """Displays the fetched world indices data."""
        if self.indices_data:
            st.subheader("World Indices")
            for index_name, data in self.indices_data.items():
                st.markdown(f"**{index_name}**: {data['price']:.2f} ({data['change']:.2f}%) - {data['date']}")
        else:
            st.info("No world indices data available to display.")

    def track_financial_data(self):
        """
        Periodically tracks financial data and sends an email if significant changes are detected.
        """
        while not self.stop_tracking:
            try:
                # Fetch the latest financial data
                latest_data = self.scrape_financial_data()
                latest_news = self.fetch_company_news(self.company_name)

                if latest_data is not None and isinstance(latest_data, pd.DataFrame):
                    # Calculate ratios for the latest data
                    latest_ratios = self.calculate_ratios()

                    if self.last_known_financial_data is not None and isinstance(self.last_known_financial_data, pd.DataFrame):
                        # Calculate ratios for the last known data
                        last_known_ratios = self.calculate_ratios()
                        # Compare the latest ratios with the last known ratios
                        significant_changes = self.compare_ratios(latest_ratios, last_known_ratios)

                        if significant_changes:
                            # Send an email notification
                            self.send_email_notification(significant_changes, latest_news)

                    # Update the last known financial data and news
                    self.last_known_financial_data = latest_data.copy()
                    self.last_known_news = latest_news
                else:
                    logging.warning("Failed to retrieve financial data.  Retaining previous data.")