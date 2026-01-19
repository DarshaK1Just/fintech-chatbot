"""Main chat interface page"""

import streamlit as st
import os
from src.core.config import Config
from src.core.session import SessionState
from src.utils.data_loader import DataLoader
from src.agents.financial_agent import FinancialAgent


def render_sidebar():
    """Render sidebar with configuration options"""
    with st.sidebar:
        st.header("Configuration")
        
        # API Key Input
        api_key_input = st.text_input(
            "Together AI API Key",
            value=SessionState.get("api_key") or Config.TOGETHER_API_KEY,
            type="password",
            help="Enter your Together AI API key"
        )
        
        if api_key_input:
            SessionState.set("api_key", api_key_input)
            os.environ["TOGETHER_API_KEY"] = api_key_input
        
        st.divider()
        
        # File Uploaders
        st.subheader("Data Files")
        
        uploaded_holdings = st.file_uploader(
            "Upload holdings.csv",
            type=["csv"],
            help="Upload the holdings CSV file"
        )
        
        uploaded_trades = st.file_uploader(
            "Upload trades.csv",
            type=["csv"],
            help="Upload the trades CSV file"
        )
        
        # Load data
        holdings_df = None
        trades_df = None
        
        if uploaded_holdings:
            holdings_df = DataLoader.load_from_file(uploaded_holdings)
            if holdings_df is not None:
                st.success(f"Loaded holdings.csv ({len(holdings_df)} rows)")
        else:
            holdings_df, _ = DataLoader.load_default_files()
            if holdings_df is not None:
                st.info("Using default holdings.csv")
            else:
                st.warning("No holdings file available")
        
        if uploaded_trades:
            trades_df = DataLoader.load_from_file(uploaded_trades)
            if trades_df is not None:
                st.success(f"Loaded trades.csv ({len(trades_df)} rows)")
        else:
            _, trades_df = DataLoader.load_default_files()
            if trades_df is not None:
                st.info("Using default trades.csv")
            else:
                st.warning("No trades file available")
        
        # Update session state
        SessionState.set_dataframes(holdings_df, trades_df)
        
        # Debug toggle
        show_debug = st.checkbox("Show Debug Info", value=SessionState.get("show_debug", False))
        SessionState.set("show_debug", show_debug)
        
        # Model selection
        st.divider()
        st.subheader("Model Selection")
        st.info(f"Using **{Config.DEFAULT_MODEL}**")
        
        # Initialize agent
        if holdings_df is not None and trades_df is not None and SessionState.get("api_key"):
            if SessionState.get("agent") is None or st.button("Reinitialize Agent"):
                with st.spinner("Initializing agent..."):
                    agent = FinancialAgent(
                        api_key=SessionState.get("api_key"),
                        model=Config.DEFAULT_MODEL
                    )
                    
                    if agent.create(holdings_df, trades_df, show_debug):
                        SessionState.set("agent", agent)
                        st.success("Agent initialized successfully!")
                    else:
                        st.error("Failed to create agent")
        
        # Display data info
        if holdings_df is not None:
            with st.expander("Holdings Data Info"):
                info = DataLoader.get_dataframe_info(holdings_df)
                st.write(f"**Rows:** {info['rows']}")
                st.write(f"**Columns:** {', '.join(info['columns'][:5])}...")
        
        if trades_df is not None:
            with st.expander("Trades Data Info"):
                info = DataLoader.get_dataframe_info(trades_df)
                st.write(f"**Rows:** {info['rows']}")
                st.write(f"**Columns:** {', '.join(info['columns'][:5])}...")


def render_chat_interface():
    """Render main chat interface"""
    st.subheader("Chat with Financial Data")
    
    # Display chat history
    for message in SessionState.get_messages():
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your financial data..."):
        agent = SessionState.get("agent")
        
        if agent is None:
            st.error("Agent not initialized. Please check configuration in sidebar.")
            st.stop()
        
        # Add user message
        SessionState.add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent.query(prompt)
                st.markdown(response)
        
        # Add assistant response
        SessionState.add_message("assistant", response)


def render_instructions():
    """Render usage instructions"""
    with st.expander("How to Use"):
        st.markdown("""
        **This chatbot answers questions about your financial data from CSV files.**
        
        **Example Questions:**
        
        **Holdings:**
        - "What is the total number of holdings for Garfield fund?"
        - "Which funds performed better based on yearly Profit and Loss?"
        - "Show me the total PL_YTD for all portfolios"
        - "What is the average price of holdings?"
        
        **Trades:**
        - "How many trades are there for MNC Investment Fund?"
        - "What is the total principal amount for all trades?"
        - "Show me all trade types"
        
        **Important Notes:**
        - The bot only answers questions based on the uploaded CSV data
        - It cannot answer general knowledge questions or provide real-time market data
        - Column mappings: "Fund" = PortfolioName, "Yearly Profit and Loss" = PL_YTD
        - For best results, ask specific questions about the data
        """)


def render():
    """Render the main page"""
    # Initialize session state
    SessionState.initialize()
    
    # Header
    st.markdown('<div class="main-header">Financial Chatbot</div>', unsafe_allow_html=True)
    
    # Render components
    render_sidebar()
    render_chat_interface()
    render_instructions()