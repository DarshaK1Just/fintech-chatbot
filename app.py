# """
# Financial Chatbot Application
# A Streamlit-based chatbot that queries financial data from CSV files using LangChain and Together AI.
# """

# import streamlit as st
# import pandas as pd
# import os
# from langchain_experimental.agents import create_pandas_dataframe_agent
# from langchain_together import ChatTogether
# from typing import Optional
# import traceback


# # Page configuration
# st.set_page_config(
#     page_title="Financial Chatbot",
#     page_icon="💼",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better UI
# st.markdown("""
#     <style>
#     .main-header {
#         font-size: 2.5rem;
#         font-weight: bold;
#         color: #1f77b4;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .stChatMessage {
#         padding: 1rem;
#     }
#     </style>
#     """, unsafe_allow_html=True)


# def initialize_session_state():
#     """Initialize session state variables."""
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "holdings_df" not in st.session_state:
#         st.session_state.holdings_df = None
#     if "trades_df" not in st.session_state:
#         st.session_state.trades_df = None
#     if "agent" not in st.session_state:
#         st.session_state.agent = None
#     if "api_key" not in st.session_state:
#         st.session_state.api_key = None


# def load_default_files():
#     """Load default CSV files from the root directory."""
#     try:
#         holdings_path = "holdings.csv"
#         trades_path = "trades.csv"
        
#         if os.path.exists(holdings_path) and os.path.exists(trades_path):
#             st.session_state.holdings_df = pd.read_csv(holdings_path)
#             st.session_state.trades_df = pd.read_csv(trades_path)
#             return True
#         return False
#     except Exception as e:
#         st.error(f"Error loading default files: {str(e)}")
#         return False


# def validate_question(question: str) -> bool:
#     """
#     Basic validation to check if question is about general knowledge.
#     This is a heuristic check - the agent will also validate against data.
#     """
#     # List of keywords that indicate questions outside the data scope
#     external_keywords = [
#         "capital of", "what is bitcoin", "price of bitcoin", "current price",
#         "stock price", "market price", "real-time", "live price",
#         "what is", "who is", "when did", "history of", "explain"
#     ]
    
#     question_lower = question.lower()
    
#     # Check for financial context keywords
#     financial_keywords = [
#         "fund", "portfolio", "holding", "trade", "profit", "loss",
#         "pl_ytd", "ytd", "transaction", "security", "quantity"
#     ]
    
#     # If question has financial keywords, it's likely about the data
#     has_financial_context = any(keyword in question_lower for keyword in financial_keywords)
    
#     # If it's a general knowledge question without financial context
#     if any(keyword in question_lower for keyword in external_keywords) and not has_financial_context:
#         # Check if it's asking about something that could be in our data
#         if "fund" in question_lower or "portfolio" in question_lower or "holding" in question_lower:
#             return True
#         return False
    
#     return True


# def create_agent(llm, holdings_df: pd.DataFrame, trades_df: pd.DataFrame):
#     """
#     Create a LangChain pandas dataframe agent with custom instructions.
#     """
#     system_prompt = """You are a financial data analyst. Answer questions using ONLY the two provided DataFrames.

# KEY INFORMATION:
# - df (or dfs[0] or holdings_df): Holdings data with columns: PortfolioName, PL_YTD, Qty, Price, SecurityId, SecName, etc.
# - dfs[1] (or trades_df): Trades data with columns: PortfolioName, TradeDate, Quantity, Price, TradeTypeName, etc.

# IMPORTANT MAPPINGS:
# - "Fund" = PortfolioName column in BOTH DataFrames
# - "Yearly Profit and Loss" or "PL YTD" = PL_YTD column (only in holdings DataFrame)
# - Use df or dfs[0] for holdings, dfs[1] for trades

# CRITICAL RULES:
# 1. ALWAYS use the correct tool name: python_repl_ast (not python\_repl\_ast)
# 2. For holdings data, use: df or dfs[0]
# 3. For trades data, use: dfs[1]
# 4. Use .sum() only ONCE - do NOT chain .sum().sum() unless aggregating grouped data

# WORKFLOW - Be efficient and direct:
# 1. For counting holdings: df[df['PortfolioName'] == 'Fund Name'].shape[0]
# 2. For counting trades: dfs[1][dfs[1]['PortfolioName'] == 'Fund Name'].shape[0]
# 3. For total PL_YTD: df['PL_YTD'].sum()
# 4. For PL_YTD by fund: df.groupby('PortfolioName')['PL_YTD'].sum().sort_values(ascending=False)

# RESPONSE FORMAT:
# - Provide ONLY the final numerical answer or result
# - Be direct and concise
# - Do NOT include code snippets in the final answer
# - Do NOT repeat the question
# - Do NOT add prefixes

# Examples:
# Q: "Total PL_YTD for all portfolios?"
# Good: "The total PL_YTD for all portfolios is 15234.56"
# Bad: "You can use: holdings_df.groupby('PortfolioName')['PL_YTD'].sum().sum()"

# Q: "Holdings for Garfield fund?"
# Good: "The total number of holdings for Garfield fund is 221"
# Bad: "To calculate this, use: df[df['PortfolioName'] == 'Garfield'].shape[0]"
# """

#     try:
#         # First, try openai-tools which is most reliable
#         try:
#             agent = create_pandas_dataframe_agent(
#                 llm=llm,
#                 df=[holdings_df, trades_df],
#                 verbose=True,
#                 agent_type="openai-tools",
#                 prefix=system_prompt,
#                 allow_dangerous_code=True,
#                 max_iterations=20,
#                 return_intermediate_steps=False,
#                 include_df_in_prompt=True,
#                 number_of_head_rows=3,
#                 agent_executor_kwargs={"handle_parsing_errors": True}
#             )
#             if st.session_state.get("show_debug", False):
#                 st.success("Agent created with type: openai-tools")
#             return agent
#         except Exception as e1:
#             if st.session_state.get("show_debug", False):
#                 st.warning(f"openai-tools failed: {str(e1)}, trying zero-shot-react-description...")
            
#             # Fallback to zero-shot-react-description
#             try:
#                 agent = create_pandas_dataframe_agent(
#                     llm=llm,
#                     df=[holdings_df, trades_df],
#                     verbose=True,
#                     agent_type="zero-shot-react-description",
#                     prefix=system_prompt,
#                     allow_dangerous_code=True,
#                     max_iterations=20,
#                     return_intermediate_steps=False,
#                     include_df_in_prompt=True,
#                     number_of_head_rows=3,
#                     agent_executor_kwargs={"handle_parsing_errors": True}
#                 )
#                 if st.session_state.get("show_debug", False):
#                     st.success("Agent created with type: zero-shot-react-description")
#                 return agent
#             except Exception as e2:
#                 raise Exception(f"Failed to create agent. Error with openai-tools: {str(e1)}. Error with zero-shot: {str(e2)}")
#     except Exception as e:
#         st.error(f"Error creating agent: {str(e)}")
#         if st.session_state.get("show_debug", False):
#             st.code(traceback.format_exc())
#         return None


# def clean_response(response_text: str) -> str:
#     """
#     Clean the response to remove duplicates, code snippets, and formatting issues.
#     """
#     if not response_text:
#         return ""
    
#     # Remove common prefixes that might cause duplication
#     prefixes_to_remove = [
#         "extracted answer:",
#         "the answer is:",
#         "answer:",
#         "result:",
#         "output:",
#         "final answer:"
#     ]
    
#     # Remove code snippets and technical instructions
#     code_patterns = [
#         "you can use:",
#         "use the following command:",
#         "you would need to",
#         "however, i cannot perform",
#         "once the dataframes are available:",
#         ".groupby(",
#         ".sum(",
#         "holdings_df",
#         "trades_df",
#         "dfs[0]",
#         "dfs[1]"
#     ]
    
#     lines = response_text.strip().split('\n')
#     cleaned_lines = []
#     seen_lines = set()
    
#     for line in lines:
#         line = line.strip()
#         if not line:
#             continue
        
#         # Convert to lowercase for comparison
#         line_lower = line.lower()
        
#         # Skip lines with code patterns
#         if any(pattern in line_lower for pattern in code_patterns):
#             continue
        
#         # Skip lines that are just code (contain backticks or common code syntax)
#         if '`' in line or 'df[' in line or '.shape[0]' in line:
#             continue
        
#         # Remove prefixes
#         for prefix in prefixes_to_remove:
#             if line_lower.startswith(prefix):
#                 line = line[len(prefix):].strip()
#                 line_lower = line.lower()
        
#         # Only add unique lines (case-insensitive comparison)
#         if line_lower not in seen_lines and line and len(line) > 10:  # Ignore very short lines
#             seen_lines.add(line_lower)
#             cleaned_lines.append(line)
    
#     # Join and return the first unique line (the actual answer)
#     if cleaned_lines:
#         # Return only the first meaningful line
#         return cleaned_lines[0]
    
#     # If no cleaned lines, check if original contains useful info
#     response_lower = response_text.lower()
#     if any(pattern in response_lower for pattern in code_patterns):
#         return "Sorry can not find the answer"
    
#     return response_text.strip()


# def get_agent_response(agent, question: str) -> str:
#     """
#     Get response from the agent and handle errors appropriately.
#     """
#     show_debug = st.session_state.get("show_debug", False)
    
#     try:
#         # Validate question format
#         if not validate_question(question):
#             return "Sorry can not find the answer"
        
#         # Run the agent
#         response = agent.invoke({"input": question})
        
#         # Extract the answer from the response
#         answer = None
#         if isinstance(response, dict):
#             answer = response.get("output") or response.get("answer") or response.get("result")
#             if answer is None and show_debug:
#                 st.warning(f"No standard key found in response. Available keys: {list(response.keys())}")
#                 for value in response.values():
#                     if isinstance(value, str) and value.strip():
#                         answer = value
#                         break
#             if answer is None:
#                 answer = str(response)
#         elif hasattr(response, 'output'):
#             answer = response.output
#         elif hasattr(response, 'result'):
#             answer = response.result
#         else:
#             answer = str(response)
        
#         # Clean up the answer
#         if answer is None:
#             answer = ""
#         answer = str(answer).strip()
        
#         # Clean the response to remove duplicates
#         answer = clean_response(answer)
        
#         # Debug: Show raw response in debug mode only
#         if show_debug:
#             with st.expander("Debug: Raw Response"):
#                 st.write(f"**Response Type:** {type(response).__name__}")
#                 try:
#                     st.json(response)
#                 except:
#                     st.write(f"**Response (string):** {str(response)[:500]}")
#                 st.write(f"**Cleaned answer:** {answer[:500]}")
        
#         # Check if answer is empty
#         if not answer or len(answer.strip()) == 0:
#             if show_debug:
#                 st.error("Answer is empty!")
#             return "Sorry can not find the answer"
        
#         # Check for iteration/time limit messages
#         answer_lower = answer.lower()
#         if "stopped due to iteration limit" in answer_lower or "iteration limit" in answer_lower:
#             st.warning("Agent hit iteration limit. Try rephrasing or breaking down your question.")
#             if show_debug:
#                 st.info(f"Full response: {answer}")
#             if len(answer) > 50:
#                 return answer
#             return "Sorry can not find the answer"
        
#         if "stopped due to time limit" in answer_lower or "time limit" in answer_lower:
#             st.warning("Agent hit time limit. Please try a simpler query.")
#             return "Sorry can not find the answer"
        
#         # Check for explicit error phrases or code instructions
#         explicit_error_phrases = [
#             "cannot find", "can't find", "unable to find", 
#             "not found in the data", "no data available",
#             "does not exist in", "not present in the data",
#             "cannot answer", "unable to answer", 
#             "sorry, can not find", "sorry can not find",
#             "cannot perform", "you can use", "you would need to",
#             "once the dataframes", "use the following"
#         ]
        
#         if any(phrase in answer_lower for phrase in explicit_error_phrases):
#             return "Sorry can not find the answer"
        
#         # Check if response contains code instead of answer
#         if '.groupby(' in answer or 'df[' in answer or '.shape[0]' in answer or '`' in answer:
#             if show_debug:
#                 st.warning(f"Agent returned code instead of answer: {answer[:200]}")
#             return "Sorry can not find the answer"
        
#         # Check for errors in response
#         if "traceback" in answer_lower or "exception:" in answer_lower or "error:" in answer_lower:
#             if show_debug:
#                 st.warning(f"Agent returned an error: {answer[:500]}")
#             return "Sorry can not find the answer"
        
#         return answer
        
#     except Exception as e:
#         error_msg = str(e)
#         error_msg_lower = error_msg.lower()
        
#         # Check for model availability errors
#         if "model_not_available" in error_msg_lower or "non-serverless model" in error_msg_lower:
#             with st.expander("Model Error (Click to view)", expanded=True):
#                 st.error("**Error:** Model not available as serverless")
#                 st.warning("The selected model is not available as a serverless endpoint.")
#                 st.info("Please go to the sidebar and select a different serverless model (e.g., Llama-3.1-70B-Instruct-Turbo)")
#                 if show_debug:
#                     st.code(traceback.format_exc())
#             return "Sorry can not find the answer"
        
#         # Show errors for debugging
#         with st.expander("Error Details (Click to view)", expanded=True):
#             st.error(f"**Exception:** {type(e).__name__}")
#             st.error(f"**Error Message:** {error_msg}")
#             if show_debug:
#                 st.code(traceback.format_exc())
#             else:
#                 st.info("Enable 'Show Debug Info' in sidebar for full traceback")
        
#         return "Sorry can not find the answer"


# def main():
#     """Main application function."""
#     initialize_session_state()
    
#     # Header
#     st.markdown('<div class="main-header">Financial Chatbot</div>', unsafe_allow_html=True)
    
#     # Sidebar
#     with st.sidebar:
#         st.header("Configuration")
        
#         # API Key Input
#         api_key_input = st.text_input(
#             "Together AI API Key",
#             value=st.session_state.api_key or "15945d687619194107b4d5cb281e4659e467500daa4a15a378fe7b9be9fa77b1",
#             type="password",
#             help="Enter your Together AI API key"
#         )
        
#         if api_key_input:
#             st.session_state.api_key = api_key_input
#             os.environ["TOGETHER_API_KEY"] = api_key_input
        
#         st.divider()
        
#         # File Uploaders
#         st.subheader("Data Files")
        
#         uploaded_holdings = st.file_uploader(
#             "Upload holdings.csv",
#             type=["csv"],
#             help="Upload the holdings CSV file. If not uploaded, the default file will be used."
#         )
        
#         uploaded_trades = st.file_uploader(
#             "Upload trades.csv",
#             type=["csv"],
#             help="Upload the trades CSV file. If not uploaded, the default file will be used."
#         )
        
#         # Load data
#         holdings_df = None
#         trades_df = None
        
#         if uploaded_holdings is not None:
#             try:
#                 holdings_df = pd.read_csv(uploaded_holdings)
#                 st.success(f"Loaded holdings.csv ({len(holdings_df)} rows)")
#             except Exception as e:
#                 st.error(f"Error loading holdings.csv: {str(e)}")
#         else:
#             if load_default_files():
#                 holdings_df = st.session_state.holdings_df
#                 st.info("Using default holdings.csv")
#             else:
#                 st.warning("No holdings file available")
        
#         if uploaded_trades is not None:
#             try:
#                 trades_df = pd.read_csv(uploaded_trades)
#                 st.success(f"Loaded trades.csv ({len(trades_df)} rows)")
#             except Exception as e:
#                 st.error(f"Error loading trades.csv: {str(e)}")
#         else:
#             if st.session_state.trades_df is not None:
#                 trades_df = st.session_state.trades_df
#                 st.info("Using default trades.csv")
#             elif os.path.exists("trades.csv"):
#                 trades_df = pd.read_csv("trades.csv")
#                 st.session_state.trades_df = trades_df
#                 st.info("Using default trades.csv")
#             else:
#                 st.warning("No trades file available")
        
#         # Update session state
#         if holdings_df is not None:
#             st.session_state.holdings_df = holdings_df
#         if trades_df is not None:
#             st.session_state.trades_df = trades_df
        
#         # Debug toggle
#         st.session_state.show_debug = st.checkbox("Show Debug Info", value=st.session_state.get("show_debug", False))
        
#         # Model selection
#         st.divider()
#         st.subheader("Model Selection")
#         selected_model = "mistralai/Mixtral-8x7B-Instruct-v0.1"
#         st.info("Using **Mixtral-8x7B-Instruct-v0.1** - Best performing model for this use case")
        
#         # Initialize agent
#         if holdings_df is not None and trades_df is not None and st.session_state.api_key:
#             if st.session_state.agent is None or st.session_state.get("selected_model") != selected_model or st.button("Reinitialize Agent"):
#                 with st.spinner("Initializing agent..."):
#                     try:
#                         llm = ChatTogether(
#                             model=selected_model,
#                             temperature=0,
#                             api_key=st.session_state.api_key,
#                         )
#                         agent = create_agent(llm, holdings_df, trades_df)
#                         if agent:
#                             st.session_state.agent = agent
#                             st.session_state.selected_model = selected_model
#                             st.success(f"Agent initialized successfully with {selected_model}!")
#                         else:
#                             st.error("Failed to create agent")
#                     except Exception as e:
#                         error_str = str(e)
#                         if "model_not_available" in error_str or "non-serverless model" in error_str:
#                             st.error(f"Model not available as serverless: {selected_model}")
#                             st.warning("This model requires a dedicated endpoint. Please select a serverless model.")
#                             st.info("Try: **meta-llama/Llama-3.1-70B-Instruct-Turbo**")
#                         else:
#                             st.error(f"Error initializing agent: {error_str}")
#                         if st.session_state.get("show_debug", False):
#                             st.code(traceback.format_exc())
        
#         # Display data info
#         if holdings_df is not None:
#             with st.expander("Holdings Data Info"):
#                 st.write(f"**Rows:** {len(holdings_df)}")
#                 st.write(f"**Columns:** {', '.join(holdings_df.columns[:5].tolist())}...")
        
#         if trades_df is not None:
#             with st.expander("Trades Data Info"):
#                 st.write(f"**Rows:** {len(trades_df)}")
#                 st.write(f"**Columns:** {', '.join(trades_df.columns[:5].tolist())}...")
    
#     # Main chat interface
#     st.subheader("Chat with Financial Data")
    
#     # Display chat history
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question about your financial data..."):
#         # Check if agent is ready
#         if st.session_state.agent is None:
#             st.error("Agent not initialized. Please check your configuration in the sidebar.")
#             st.stop()
        
#         # Add user message to chat history
#         st.session_state.messages.append({"role": "user", "content": prompt})
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         # Get agent response
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 response = get_agent_response(st.session_state.agent, prompt)
#                 st.markdown(response)
        
#         # Add assistant response to chat history
#         st.session_state.messages.append({"role": "assistant", "content": response})
    
#     # Instructions
#     with st.expander("How to Use"):
#         st.markdown("""
#         **This chatbot answers questions about your financial data from the CSV files.**
        
#         **Example Questions:**
#         - "What is the total number of holdings for Garfield fund?"
#         - "Which funds performed better based on yearly Profit and Loss?"
#         - "How many trades are there for MNC Investment Fund?"
#         - "Show me the total PL_YTD for all portfolios"
        
#         **Important Notes:**
#         - The bot only answers questions based on the uploaded CSV data
#         - It cannot answer general knowledge questions or provide real-time market data
#         - Column mappings: "Fund" = PortfolioName, "Yearly Profit and Loss" = PL_YTD
#         """)


# if __name__ == "__main__":
#     main()


"""
Financial Chatbot Application - Main Entry Point
A Streamlit-based chatbot that queries financial data from CSV files using LangChain and Together AI.
"""

import streamlit as st
from src.ui.pages import main_page
from src.core.config import setup_page_config

def main():
    """Main application entry point."""
    setup_page_config()
    main_page.render()

if __name__ == "__main__":
    main()