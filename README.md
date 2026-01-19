# Financial Chatbot

A production-ready Streamlit application that serves as a Financial Chatbot, allowing users to query financial data from CSV files using natural language queries powered by Together AI and LangChain.

## Features

- 💬 **Natural Language Interface**: Ask questions about financial data in plain English
- 📊 **Dual Data Source Support**: Query both holdings and trades data simultaneously
- 🔒 **Data-Only Responses**: Strictly answers only from provided CSV data
- 🎯 **Smart Column Mapping**: Automatically handles column name mappings (Fund → PortfolioName, Yearly Profit and Loss → PL_YTD)
- 📁 **Flexible File Input**: Upload CSV files or use default files from the root directory
- 🔄 **Chat History**: Maintains conversation context throughout the session

## Tech Stack

- **UI**: Streamlit with modern chat interface
- **LLM**: Together AI (via LangChain integration)
- **Data Processing**: LangChain's pandas dataframe agent for accurate calculations
- **Python**: 3.8+

## Installation

1. **Clone or download this repository**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Place your CSV files** (optional):
   - `holdings.csv` - Portfolio holdings data
   - `trades.csv` - Trade transaction data
   
   These files can also be uploaded through the UI.

## Usage

1. **Run the application**:
   ```bash
   streamlit run app.py
   ```

2. **Configure the application**:
   - Enter your Together AI API key in the sidebar (default key is pre-filled)
   - Upload your CSV files or use the default files if they exist in the root directory
   - Wait for the agent to initialize (you'll see a success message)

3. **Start chatting**:
   - Type your question in the chat input
   - The bot will analyze the data and provide answers based only on the CSV files

## Example Questions

- "What is the total number of holdings for Garfield fund?"
- "Which funds performed better based on yearly Profit and Loss?"
- "How many trades are there for MNC Investment Fund?"
- "Show me the total PL_YTD for all portfolios"
- "Rank the funds by PL_YTD"

## Data Schema

### Holdings CSV
- **PortfolioName**: The fund/portfolio identifier (referred to as "Fund" in queries)
- **PL_YTD**: Yearly Profit and Loss (referred to as "Yearly Profit and Loss" in queries)
- Other columns: AsOfDate, OpenDate, CloseDate, SecurityId, Qty, Price, etc.

### Trades CSV
- **PortfolioName**: The fund/portfolio identifier (referred to as "Fund" in queries)
- Other columns: TradeDate, SettleDate, Quantity, Price, TradeTypeName, SecurityId, etc.

## Important Notes

- The chatbot **only answers questions** based on the provided CSV data
- For questions outside the data scope, it responds with: "Sorry can not find the answer"
- The bot does not use external knowledge or real-time market data
- Column name mappings are handled automatically (Fund → PortfolioName, Yearly Profit and Loss → PL_YTD)

## Troubleshooting

- **Agent not initializing**: Check your API key and ensure both CSV files are loaded
- **No response**: Verify that your question can be answered from the CSV data
- **File upload errors**: Ensure CSV files are properly formatted and not corrupted

## License

This project is provided as-is for educational and professional use.
