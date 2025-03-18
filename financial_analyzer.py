import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
import urllib.parse
import urllib.request
import logging
from io import BytesIO

try:
    import PyPDF2  # Add this import for PDF handling
    logging.info("Script has been executed successfully.")
except ModuleNotFoundError:
    logging.error("PyPDF2 module is not installed. Please install it using 'pip install PyPDF2'.")
    raise

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

    def scrape_financial_data(self):
        """
        Scrapes financial data from the company's website, including tables and PDFs.

        Returns:
            pd.DataFrame: A DataFrame containing the scraped financial data, or None if an error occurs.
        """
        if not self.website_url:
            st.error("Website URL not provided.")
            logging.error("Website URL not provided.")
            return None

        try:
            response = requests.get(self.website_url, timeout=10)  # Add timeout
            response.raise_for_status()
            soup = BeautifulSoup(response.content, "html.parser")

            # Check for tables on the webpage
            tables = soup.find_all("table")
            if tables:
                for table in tables:
                    if any(keyword in table.text.lower() for keyword in ["balance sheet", "income statement", "cash flow"]):
                        data = []
                        rows = table.find_all("tr")
                        for row in rows:
                            cols = row.find_all(["td", "th"])  # Handle both td and th
                            cols = [ele.text.strip() for ele in cols]
                            data.append(cols)

                        df = pd.DataFrame(data)

                        if not df.empty:
                            df.columns = df.iloc[0]
                            df = df[1:]
                            df = df.dropna(axis=1, how='all')
                            df = df.dropna(axis=0, how='all')

                            for col in df.columns:
                                try:
                                    df[col] = df[col].str.replace(r'[$,()]', '', regex=True).astype(float)
                                except (ValueError, AttributeError):
                                    pass
                            return df

            # If no tables are found, check for PDF links
            pdf_links = [a['href'] for a in soup.find_all('a', href=True) if a['href'].lower().endswith('.pdf')]
            if pdf_links:
                for pdf_link in pdf_links:
                    pdf_url = urllib.parse.urljoin(self.website_url, pdf_link)
                    pdf_response = requests.get(pdf_url, timeout=10)
                    pdf_response.raise_for_status()

                    # Extract text from the PDF
                    pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_response.content))
                    pdf_text = ""
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:  # Ensure text is not None
                            pdf_text += page_text

                    # Attempt to parse financial data from the PDF text
                    df = self.parse_pdf_text_to_dataframe(pdf_text)
                    if df is not None:
                        return df

            st.warning("No relevant financial data found in tables or PDFs. Please verify the webpage content.")
            logging.warning("No relevant financial data found in tables or PDFs.")
            return None

        except requests.exceptions.RequestException as e:
            st.error(f"Error during scraping: {e}")
            logging.error(f"Error during scraping: {e}")
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.error(f"An unexpected error occurred: {e}")
            return None

    def parse_pdf_text_to_dataframe(self, pdf_text):
        """
        Parses financial data from extracted PDF text into a DataFrame.

        Args:
            pdf_text (str): The extracted text from the PDF.

        Returns:
            pd.DataFrame: A DataFrame containing the parsed financial data, or None if parsing fails.
        """
        try:
            # Split text into lines and attempt to identify tabular data
            lines = pdf_text.splitlines()
            data = []
            for line in lines:
                # Split line into columns based on whitespace or delimiters
                cols = re.split(r'\s{2,}|\t', line.strip())
                if len(cols) > 1:  # Ensure it's a valid row with multiple columns
                    data.append(cols)

            if data:
                df = pd.DataFrame(data)
                df.columns = df.iloc[0]  # Use the first row as column headers
                df = df[1:]  # Remove the header row from the data
                df = df.dropna(axis=1, how='all')
                df = df.dropna(axis=0, how='all')

                for col in df.columns:
                    try:
                        df[col] = df[col].str.replace(r'[$,()]', '', regex=True).astype(float)
                    except (ValueError, AttributeError):
                        pass
                return df

            return None
        except Exception as e:
            logging.error(f"Failed to parse PDF text into DataFrame: {e}")
            return None

    def calculate_ratios(self):
        """
        Calculates financial ratios based on the scraped data.

        Returns:
            dict: A dictionary containing the calculated ratios, or None if an error occurs.
        """
        if self.financial_data is None or not isinstance(self.financial_data, pd.DataFrame):
            st.error("Financial data is not available or is in the wrong format.")
            logging.error("Financial data is not available or is in the wrong format.")
            return None

        try:
            data = self.financial_data

            revenue_col = next((col for col in data.columns if re.search(r'revenue|sales', col, re.IGNORECASE)), None)
            net_income_col = next((col for col in data.columns if re.search(r'net income|profit', col, re.IGNORECASE)), None)
            total_assets_col = next((col for col in data.columns if re.search(r'total assets', col, re.IGNORECASE)), None)
            total_liabilities_col = next((col for col in data.columns if re.search(r'total liabilities', col, re.IGNORECASE)), None)
            current_assets_col = next((col for col in data.columns if re.search(r'current assets', col, re.IGNORECASE)), None)
            current_liabilities_col = next((col for col in data.columns if re.search(r'current liabilities', col, re.IGNORECASE)), None)
            total_equity_col = next((col for col in data.columns if re.search(r'total equity', col, re.IGNORECASE)), None)
            cash_col = next((col for col in data.columns if re.search(r'cash', col, re.IGNORECASE)), None)
            inventory_col = next((col for col in data.columns if re.search(r'inventory', col, re.IGNORECASE)), None)
            cogs_col = next((col for col in data.columns if re.search(r'cost of goods sold|cogs', col, re.IGNORECASE)), None)

            ratios = {}

            if revenue_col and net_income_col:
                ratios["Profit Margin"] = data[net_income_col].iloc[-1] / data[revenue_col].iloc[-1] if data[revenue_col].iloc[-1] != 0 else None

            if total_assets_col and total_liabilities_col:
                ratios["Debt-to-Asset Ratio"] = data[total_liabilities_col].iloc[-1] / data[total_assets_col].iloc[-1] if data[total_assets_col].iloc[-1] != 0 else None

            if current_assets_col and current_liabilities_col:
                ratios["Current Ratio"] = data[current_assets_col].iloc[-1] / data[current_liabilities_col].iloc[-1] if data[current_liabilities_col].iloc[-1] != 0 else None

            if total_equity_col and total_liabilities_col:
                ratios["Debt-to-Equity Ratio"] = data[total_liabilities_col].iloc[-1] / data[total_equity_col].iloc[-1] if data[total_equity_col].iloc[-1] != 0 else None

            if cash_col and current_liabilities_col:
                ratios["Cash Ratio"] = data[cash_col].iloc[-1] / data[current_liabilities_col].iloc[-1] if data[current_liabilities_col].iloc[-1] != 0 else None

            if inventory_col and cogs_col:
                ratios["Inventory Turnover"] = data[cogs_col].iloc[-1] / data[inventory_col].iloc[-1] if data[inventory_col].iloc[-1] != 0 else None

            return ratios

        except KeyError as e:
            st.error(f"KeyError: {e}. Required financial data columns not found.")
            logging.error(f"KeyError: {e}. Required financial data columns not found.")
            return None
        except TypeError as e:
            st.error(f"TypeError: {e}. Check the type of your data. Likely a data format issue: {e}")
            logging.error(f"TypeError: {e}. Check the type of your data. Likely a data format issue: {e}")
            return None
        except IndexError as e:
            st.error(f"IndexError: {e}. Check the structure of your data. Data may be missing: {e}")
            logging.error(f"IndexError: {e}. Check the structure of your data. Data may be missing: {e}")
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

        # Example analysis based on ratios
        if "Profit Margin" in ratios:
            if ratios["Profit Margin"] > 0.2:
                analysis["Profitability"] = "High"
            elif ratios["Profit Margin"] > 0.1:
                analysis["Profitability"] = "Moderate"
            else:
                analysis["Profitability"] = "Low"

        if "Debt-to-Asset Ratio" in ratios:
            if ratios["Debt-to-Asset Ratio"] < 0.5:
                analysis["Debt Level"] = "Low"
            elif ratios["Debt-to-Asset Ratio"] < 0.7:
                analysis["Debt Level"] = "Moderate"
            else:
                analysis["Debt Level"] = "High"

        if "Current Ratio" in ratios:
            if ratios["Current Ratio"] > 2:
                analysis["Liquidity"] = "High"
            elif ratios["Current Ratio"] > 1:
                analysis["Liquidity"] = "Moderate"
            else:
                analysis["Liquidity"] = "Low"

        return analysis

def main():
    st.title("Financial Analyzer")
    st.sidebar.header("Input Parameters")

    # Input fields for company name and website URL
    company_name = st.sidebar.text_input("Company Name", value="Example Corp")
    website_url = st.sidebar.text_input("Website URL", value="https://example.com/financials")

    # Button to trigger analysis
    if st.sidebar.button("Analyze"):
        if not company_name or not website_url:
            st.error("Please provide both the company name and website URL.")
        else:
            analyzer = FinancialAnalyzer(company_name, website_url)
            financial_data = analyzer.financial_data

            if financial_data is not None:
                st.subheader("Scraped Financial Data")
                st.dataframe(financial_data)

                # Display calculated ratios
                ratios = analyzer.calculate_ratios()
                if ratios:
                    st.subheader("Calculated Financial Ratios")
                    for ratio, value in ratios.items():
                        st.write(f"{ratio}: {value:.2f}" if value is not None else f"{ratio}: N/A")

                # Display financial health analysis
                analysis = analyzer.analyze_financial_health()
                if analysis:
                    st.subheader("Financial Health Analysis")
                    for key, value in analysis.items():
                        st.write(f"{key}: {value}")
            else:
                st.error("Failed to retrieve financial data. Please check the website URL.")

if __name__ == "__main__":
    main()
````
