"""Financial data agent using LangChain"""

import streamlit as st
import pandas as pd
import traceback
from typing import Optional
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_together import ChatTogether
from src.core.config import Config
from src.utils.response_cleaner import ResponseCleaner
from src.utils.validators import QuestionValidator


class FinancialAgent:
    """Agent for querying financial data"""
    
    SYSTEM_PROMPT = """You are a financial data analyst. Answer questions using ONLY the two provided DataFrames.

KEY INFORMATION:
- df (or dfs[0] or holdings_df): Holdings data with columns: PortfolioName, PL_YTD, Qty, Price, SecurityId, SecName, etc.
- dfs[1] (or trades_df): Trades data with columns: PortfolioName, TradeDate, Quantity, Price, TradeTypeName, etc.

IMPORTANT MAPPINGS:
- "Fund" = PortfolioName column in BOTH DataFrames
- "Yearly Profit and Loss" or "PL YTD" = PL_YTD column (only in holdings DataFrame)
- Use df or dfs[0] for holdings, dfs[1] for trades

CRITICAL RULES:
1. ALWAYS use the correct tool name: python_repl_ast (not python\_repl\_ast)
2. For holdings data, use: df or dfs[0]
3. For trades data, use: dfs[1]
4. Use .sum() only ONCE - do NOT chain .sum().sum() unless aggregating grouped data

WORKFLOW:
1. For counting holdings: df[df['PortfolioName'] == 'Fund Name'].shape[0]
2. For counting trades: dfs[1][dfs[1]['PortfolioName'] == 'Fund Name'].shape[0]
3. For total PL_YTD: df['PL_YTD'].sum()
4. For PL_YTD by fund: df.groupby('PortfolioName')['PL_YTD'].sum().sort_values(ascending=False)

RESPONSE FORMAT:
- Provide ONLY the final numerical answer or result
- Be direct and concise
- Do NOT include code snippets in the final answer
- Do NOT repeat the question
- Do NOT add prefixes

Examples:
Q: "Total PL_YTD for all portfolios?"
Good: "The total PL_YTD for all portfolios is 15234.56"
Bad: "You can use: holdings_df.groupby('PortfolioName')['PL_YTD'].sum().sum()"
"""
    
    def __init__(self, api_key: str, model: str = None):
        """
        Initialize the financial agent.
        
        Args:
            api_key: Together AI API key
            model: Model name to use (defaults to Config.DEFAULT_MODEL)
        """
        self.api_key = api_key
        self.model = model or Config.DEFAULT_MODEL
        self.agent = None
        self.show_debug = False
    
    def create(self, holdings_df: pd.DataFrame, trades_df: pd.DataFrame, 
               show_debug: bool = False) -> bool:
        """
        Create the LangChain agent.
        
        Args:
            holdings_df: Holdings DataFrame
            trades_df: Trades DataFrame
            show_debug: Whether to show debug information
            
        Returns:
            True if successful, False otherwise
        """
        self.show_debug = show_debug
        
        try:
            llm = ChatTogether(
                model=self.model,
                temperature=Config.TEMPERATURE,
                api_key=self.api_key,
            )
            
            # Try openai-tools first
            try:
                self.agent = create_pandas_dataframe_agent(
                    llm=llm,
                    df=[holdings_df, trades_df],
                    verbose=True,
                    agent_type="openai-tools",
                    prefix=self.SYSTEM_PROMPT,
                    allow_dangerous_code=True,
                    max_iterations=Config.MAX_ITERATIONS,
                    return_intermediate_steps=False,
                    include_df_in_prompt=True,
                    number_of_head_rows=3,
                    agent_executor_kwargs={"handle_parsing_errors": True}
                )
                if self.show_debug:
                    st.success("Agent created with type: openai-tools")
                return True
                
            except Exception as e1:
                if self.show_debug:
                    st.warning(f"openai-tools failed, trying zero-shot-react-description...")
                
                # Fallback to zero-shot
                self.agent = create_pandas_dataframe_agent(
                    llm=llm,
                    df=[holdings_df, trades_df],
                    verbose=True,
                    agent_type="zero-shot-react-description",
                    prefix=self.SYSTEM_PROMPT,
                    allow_dangerous_code=True,
                    max_iterations=Config.MAX_ITERATIONS,
                    return_intermediate_steps=False,
                    include_df_in_prompt=True,
                    number_of_head_rows=3,
                    agent_executor_kwargs={"handle_parsing_errors": True}
                )
                if self.show_debug:
                    st.success("Agent created with type: zero-shot-react-description")
                return True
                
        except Exception as e:
            st.error(f"Error creating agent: {str(e)}")
            if self.show_debug:
                st.code(traceback.format_exc())
            return False
    
    def query(self, question: str) -> str:
        """
        Query the agent with a question.
        
        Args:
            question: User's question
            
        Returns:
            Agent's response
        """
        if not self.agent:
            return "Agent not initialized"
        
        # Validate question
        if not QuestionValidator.validate(question):
            return "Sorry can not find the answer"
        
        try:
            # Run agent
            response = self.agent.invoke({"input": question})
            
            # Extract answer
            answer = self._extract_answer(response)
            
            # Clean response
            answer = ResponseCleaner.clean(answer)
            
            # Debug output
            if self.show_debug:
                with st.expander("Debug: Raw Response"):
                    st.write(f"**Response Type:** {type(response).__name__}")
                    try:
                        st.json(response)
                    except:
                        st.write(f"**Response:** {str(response)[:500]}")
                    st.write(f"**Cleaned answer:** {answer[:500]}")
            
            # Validate answer
            return self._validate_answer(answer)
            
        except Exception as e:
            return self._handle_error(e)
    
    def _extract_answer(self, response) -> str:
        """Extract answer from response object"""
        answer = None
        
        if isinstance(response, dict):
            answer = response.get("output") or response.get("answer") or response.get("result")
            if answer is None:
                for value in response.values():
                    if isinstance(value, str) and value.strip():
                        answer = value
                        break
            if answer is None:
                answer = str(response)
        elif hasattr(response, 'output'):
            answer = response.output
        elif hasattr(response, 'result'):
            answer = response.result
        else:
            answer = str(response)
        
        return str(answer).strip() if answer else ""
    
    def _validate_answer(self, answer: str) -> str:
        """Validate and clean the answer"""
        if not answer or len(answer.strip()) == 0:
            if self.show_debug:
                st.error("Answer is empty!")
            return "Sorry can not find the answer"
        
        answer_lower = answer.lower()
        
        # Check for iteration/time limits
        if "iteration limit" in answer_lower:
            st.warning("Agent hit iteration limit. Try rephrasing your question.")
            return "Sorry can not find the answer" if len(answer) < 50 else answer
        
        if "time limit" in answer_lower:
            st.warning("Agent hit time limit. Please try a simpler query.")
            return "Sorry can not find the answer"
        
        # Check for error phrases
        error_phrases = [
            "cannot find", "can't find", "unable to find",
            "not found in the data", "no data available",
            "cannot answer", "unable to answer",
            "cannot perform", "you can use", "you would need to"
        ]
        
        if any(phrase in answer_lower for phrase in error_phrases):
            return "Sorry can not find the answer"
        
        # Check for code in response
        if ResponseCleaner.contains_code(answer):
            if self.show_debug:
                st.warning(f"Agent returned code: {answer[:200]}")
            return "Sorry can not find the answer"
        
        # Check for errors
        if "traceback" in answer_lower or "exception:" in answer_lower:
            if self.show_debug:
                st.warning(f"Agent error: {answer[:500]}")
            return "Sorry can not find the answer"
        
        return answer
    
    def _handle_error(self, error: Exception) -> str:
        """Handle query errors"""
        error_msg = str(error)
        error_msg_lower = error_msg.lower()
        
        # Model availability errors
        if "model_not_available" in error_msg_lower or "non-serverless" in error_msg_lower:
            with st.expander("Model Error", expanded=True):
                st.error("Model not available as serverless")
                st.warning("Please select a different model in the sidebar")
                if self.show_debug:
                    st.code(traceback.format_exc())
            return "Sorry can not find the answer"
        
        # General errors
        with st.expander("Error Details", expanded=True):
            st.error(f"**Exception:** {type(error).__name__}")
            st.error(f"**Message:** {error_msg}")
            if self.show_debug:
                st.code(traceback.format_exc())
            else:
                st.info("Enable 'Show Debug Info' in sidebar for details")
        
        return "Sorry can not find the answer"