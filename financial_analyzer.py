import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

class FinancialAnalyzer:
    def __init__(self, company_name, website_url=None, financial_data=None):
        self.company_name = company_name
        self.website_url = website_url
        self.financial_data = financial_data  # Can be a dict or a DataFrame

        if website_url and financial_data is None:
            self.financial_data = self.scrape_financial_data()

    def scrape_financial_data(self):
        """Scrapes financial data from a company's website."""
        if not self.website_url:
            print("Website URL not provided.")
            return None

        try:
            response = requests.get(self.website_url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            soup = BeautifulSoup(response.content, "html.parser")

            # Example: Searching for a table with financial data (adapt to website structure)
            # This is a highly simplified example. Real-world scraping requires much more robust logic.
            tables = soup.find_all("table")
            for table in tables:
                if "balance sheet" in table.text.lower() or "income statement" in table.text.lower() or "cash flow" in table.text.lower():
                    data = []
                    rows = table.find_all("tr")
                    for row in rows:
                        cols = row.find_all("td") or row.find_all("th") #handle th tags in headers
                        cols = [ele.text.strip() for ele in cols]
                        data.append(cols)

                    df = pd.DataFrame(data)

                    #clean dataframe
                    if not df.empty:
                        df.columns = df.iloc[0]
                        df = df[1:]
                        df = df.dropna(axis=1, how='all')
                        df = df.dropna(axis=0, how='all')

                        #try to convert to numeric
                        for col in df.columns:
                            try:
                                df[col] = df[col].str.replace('[$,()]', '', regex=True).astype(float)
                            except (ValueError, AttributeError):
                                pass #ignore columns that cannot be converted.
                        return df

            print("Financial data table not found.")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Error during scraping: {e}")
            return None

    def calculate_ratios(self):
        """Calculates various financial ratios."""
        if self.financial_data is None or not isinstance(self.financial_data, pd.DataFrame):
            print("Financial data is not available or is in the wrong format.")
            return None

        try:
            # Example calculations (adapt to available data)
            # Example: Assuming data is in a DataFrame with columns like 'Revenue', 'Net Income', 'Total Assets', 'Total Liabilities'
            data = self.financial_data

            # Find columns that look like the ones we need, allowing for variations in naming.
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
                 ratios["Current Ratio"] = data[current_assets_col].iloc[-1] / data[current_liabilities_col].iloc[-1] if data[current_liabilities_col].iloc[-1] !=0 else None

            if total_equity_col and total_liabilities_col:
                ratios["Debt-to-Equity Ratio"] = data[total_liabilities_col].iloc[-1] / data[total_equity_col].iloc[-1] if data[total_equity_col].iloc[-1] != 0 else None

            if cash_col and current_liabilities_col:
                ratios["Cash Ratio"] = data[cash_col].iloc[-1] / data[current_liabilities_col].iloc[-1] if data[current_liabilities_col].iloc[-1] != 0 else None

            if inventory_col and cogs_col:
                ratios["Inventory Turnover"] = data[cogs_col].iloc[-1] / data[inventory_col].iloc[-1] if data[inventory_col].iloc[-1] != 0 else None

            return ratios

        except KeyError as e:
            print(f"KeyError: {e}. Required financial data columns not found.")
            return None
        except TypeError as e:
            print(f"TypeError: {e}. Check the type of your data.")
            return None
        except IndexError as e:
            print(f"IndexError: {e}. Check the structure of your data.")
            return None

    def analyze_financial_health(self):
        """Analyzes the financial health based on calculated ratios."""
        ratios = self.calculate_ratios()
        if ratios is None:
            return None

        analysis = {}
        # Example interpretation (adapt to specific ratios and thresholds)
        if "Profit Margin" in ratios and ratios["Profit Margin"] is not None:
            analysis["Profit Margin"] = "Good" if ratios["Profit Margin"] > 0.1 else "Needs Improvement"
        if "Debt-to-Asset Ratio" in ratios and ratios["Debt-to-Asset Ratio"] is not None:
            analysis["Debt-to-Asset Ratio"] = "Low Risk" if ratios["Debt-to-Asset Ratio"] < 0.5 else "High Risk"
        if "Current Ratio" in ratios and ratios["Current Ratio"] is not None:
            analysis["Current Ratio"] = "Good Liquidity" if ratios["Current Ratio"] > 1.5 else "Poor Liquidity"
        if "Debt-to-Equity Ratio" in ratios and ratios["Debt-to-Equity Ratio"] is not None:
            analysis["Debt-to-Equity Ratio"] = "Low Leverage" if ratios["Debt-to-Equity Ratio"] < 1 else "High Leverage"
        if "Cash Ratio" in ratios and ratios["Cash Ratio"] is not None:
            analysis["Cash Ratio"] = "Strong immediate liquidity" if ratios["Cash Ratio"] > 0.5 else "Weak immediate liquidity"
        if "Inventory Turnover" in ratios and ratios["Inventory Turnover"] is not None:
            analysis["Inventory Turnover"] = "Efficient Inventory Management" if ratios["Inventory Turnover"] > 5 else "Inefficient Inventory Management"

        return analysis

# Example Usage
company_url = "https://investors.apple.com/financials/default.aspx" #Example website
company_name = "Apple Inc."
analyzer = FinancialAnalyzer(company_name, company_url)
ratios = analyzer.calculate_ratios()
analysis = analyzer.analyze_financial_health()

if ratios:
    print
