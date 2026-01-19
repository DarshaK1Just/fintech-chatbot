"""Session state management"""

import streamlit as st
from typing import Optional, Any
import pandas as pd


class SessionState:
    """Manage Streamlit session state"""
    
    @staticmethod
    def initialize():
        """Initialize all session state variables"""
        defaults = {
            "messages": [],
            "holdings_df": None,
            "trades_df": None,
            "agent": None,
            "api_key": None,
            "show_debug": False,
            "selected_model": None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    @staticmethod
    def get(key: str, default: Any = None) -> Any:
        """Get value from session state"""
        return st.session_state.get(key, default)
    
    @staticmethod
    def set(key: str, value: Any):
        """Set value in session state"""
        st.session_state[key] = value
    
    @staticmethod
    def add_message(role: str, content: str):
        """Add a message to chat history"""
        st.session_state.messages.append({
            "role": role,
            "content": content
        })
    
    @staticmethod
    def get_messages() -> list:
        """Get all chat messages"""
        return st.session_state.get("messages", [])
    
    @staticmethod
    def clear_messages():
        """Clear chat history"""
        st.session_state.messages = []
    
    @staticmethod
    def set_dataframes(holdings_df: Optional[pd.DataFrame], trades_df: Optional[pd.DataFrame]):
        """Set both dataframes"""
        st.session_state.holdings_df = holdings_df
        st.session_state.trades_df = trades_df