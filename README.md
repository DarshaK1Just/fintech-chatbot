# Financial Chatbot

A production-ready Streamlit application that serves as an intelligent Financial Chatbot, allowing users to query financial data from CSV files using natural language queries powered by Together AI and LangChain.

## ✨ Features

- 💬 **Natural Language Interface**: Ask questions about financial data in plain English
- 📊 **Dual Data Source Support**: Query both holdings and trades data simultaneously
- 🔒 **Data-Only Responses**: Strictly answers only from provided CSV data
- 🎯 **Smart Column Mapping**: Automatically handles column name mappings (Fund → PortfolioName, Yearly Profit and Loss → PL_YTD)
- 📁 **Flexible File Input**: Upload CSV files or use default files from the data directory
- 🔄 **Chat History**: Maintains conversation context throughout the session
- 🏗️ **Modular Architecture**: Professional code structure following best practices
- 🔐 **Secure Configuration**: Environment variable management with .env files
- 🐛 **Debug Mode**: Toggle detailed logging for troubleshooting

## 🛠️ Tech Stack

- **UI Framework**: Streamlit with modern chat interface
- **LLM Provider**: Together AI (Mixtral-8x7B-Instruct-v0.1)
- **Agent Framework**: LangChain with pandas dataframe agent
- **Data Processing**: Pandas for CSV handling
- **Configuration**: python-dotenv for environment management
- **Python**: 3.8+

## 📁 Project Structure

```
financial-chatbot/
├── app.py                          # Main application entry point
├── .env                            # Environment variables (create from .env.example)
├── .env.example                    # Environment variables template
├── .gitignore                      # Git ignore rules
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── data/                           # Data directory
│   ├── holdings.csv               # Holdings data
│   └── trades.csv                 # Trades data
└── src/                            # Source code
    ├── __init__.py
    ├── core/                       # Core functionality
    │   ├── __init__.py
    │   ├── config.py              # Configuration management
    │   └── session.py             # Session state management
    ├── agents/                     # LangChain agents
    │   ├── __init__.py
    │   └── financial_agent.py     # Financial data agent
    ├── utils/                      # Utility functions
    │   ├── __init__.py
    │   ├── data_loader.py         # CSV data loading
    │   ├── response_cleaner.py    # Response processing
    │   └── validators.py          # Input validation
    └── ui/                         # UI components
        ├── __init__.py
        └── pages/
            ├── __init__.py
            └── main_page.py       # Main chat interface
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Together AI API key ([Get one here](https://api.together.xyz/settings/api-keys))

### Step 1: Clone or Download Repository

```bash
git clone <repository-url>
cd financial-chatbot
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env and add your Together AI API key
# TOGETHER_API_KEY=your_actual_api_key_here
```

**Note**: Never commit your `.env` file to version control. It's already included in `.gitignore`.

### Step 5: Add Your Data Files

Place your CSV files in the `data/` directory:
- `data/holdings.csv` - Portfolio holdings data
- `data/trades.csv` - Trade transaction data

Alternatively, you can upload files through the UI after starting the application.

### Step 6: Create Required Directories

```bash
# Create the data directory if it doesn't exist
mkdir -p data
```

## 💻 Usage

### Starting the Application

```bash
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### Configuration

1. **API Key Setup**:
   - The API key is loaded from the `.env` file automatically
   - You can also enter it manually in the sidebar if needed

2. **Upload Data Files** (if not using default files):
   - Click "Upload holdings.csv" in the sidebar
   - Click "Upload trades.csv" in the sidebar
   - Wait for the success message confirming files are loaded

3. **Initialize Agent**:
   - The agent initializes automatically when both CSV files and API key are available
   - Click "Reinitialize Agent" button if you need to reload

4. **Debug Mode** (Optional):
   - Enable "Show Debug Info" checkbox in sidebar
   - View detailed agent responses and error traces

### Asking Questions

Simply type your question in the chat input at the bottom of the page. The chatbot will:
1. Validate your question
2. Query the pandas dataframes
3. Return a clear, concise answer

## 📝 Example Questions

### Holdings Queries

- "What is the total number of holdings for Garfield fund?"
- "Which funds performed better based on yearly Profit and Loss?"
- "Show me the total PL_YTD for all portfolios"
- "What is the average price of holdings?"
- "Rank all funds by their PL_YTD performance"
- "How many different security types are in the holdings?"

### Trades Queries

- "How many trades are there for MNC Investment Fund?"
- "What is the total principal amount for all trades?"
- "Show me all trade types"
- "What is the average trade price for Garfield fund?"
- "How many buy trades vs sell trades are there?"

### Cross-Dataset Queries

- "Compare the number of holdings vs number of trades for Garfield fund"
- "Show me portfolios ranked by both PL_YTD and trade count"
- "Which custodian manages the most holdings and trades?"

## 📊 Data Schema

### Holdings CSV (data/holdings.csv)

Required columns:
- **PortfolioName**: Fund/portfolio identifier (also referenced as "Fund" in queries)
- **PL_YTD**: Yearly Profit and Loss (also referenced as "Yearly Profit and Loss")
- **Qty**: Quantity of holdings
- **Price**: Current price
- **SecurityId**: Security identifier

Optional columns: AsOfDate, OpenDate, CloseDate, SecurityTypeName, SecName, MV_Local, MV_Base, PL_DTD, PL_QTD, PL_MTD, etc.

### Trades CSV (data/trades.csv)

Required columns:
- **PortfolioName**: Fund/portfolio identifier (also referenced as "Fund")
- **TradeDate**: Date of trade execution
- **Quantity**: Number of units traded
- **Price**: Trade execution price
- **TradeTypeName**: Type of trade (Buy, Sell, etc.)

Optional columns: SettleDate, SecurityId, Principal, Interest, TotalCash, CustodianName, Counterparty, etc.

## 🔧 Configuration

### Environment Variables

Edit `.env` file to configure:

```bash
# Together AI Configuration
TOGETHER_API_KEY=your_api_key_here

# Model Configuration
DEFAULT_MODEL=mistralai/Mixtral-8x7B-Instruct-v0.1
TEMPERATURE=0
MAX_ITERATIONS=20

# Debug Settings
DEBUG_MODE=False
```

### Available Models

- `mistralai/Mixtral-8x7B-Instruct-v0.1` (Recommended - Best performance)
- `meta-llama/Llama-3.1-70B-Instruct-Turbo`
- `meta-llama/Llama-3.3-70B-Instruct-Turbo`


## 🧪 Development

### Running in Development Mode

```bash
# Enable debug mode in .env
DEBUG_MODE=True

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

### Code Structure

- **`app.py`**: Main entry point, minimal code
- **`src/core/`**: Configuration and session management
- **`src/agents/`**: LangChain agent implementation
- **`src/utils/`**: Reusable utility functions
- **`src/ui/`**: Streamlit UI components


## 📦 Dependencies

Main dependencies (see `requirements.txt` for complete list):

- `streamlit==1.32.0` - Web UI framework
- `pandas==2.2.0` - Data manipulation
- `langchain==0.1.0` - LLM framework
- `langchain-experimental==0.0.50` - Pandas dataframe agent
- `langchain-together==0.0.1` - Together AI integration
- `python-dotenv==1.0.0` - Environment variable management

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request
