"""Configuration management using environment variables and Streamlit config"""

import os
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration"""
    
    # API Configuration
    TOGETHER_API_KEY: str = os.getenv("TOGETHER_API_KEY", "")
    
    # Model Configuration
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "mistralai/Mixtral-8x7B-Instruct-v0.1")
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0"))
    MAX_ITERATIONS: int = int(os.getenv("MAX_ITERATIONS", "20"))
    
    # Data Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    HOLDINGS_PATH: Path = DATA_DIR / "holdings.csv"
    TRADES_PATH: Path = DATA_DIR / "trades.csv"
    
    # UI Configuration
    PAGE_TITLE: str = "Financial Chatbot"
    PAGE_ICON: str = "💼"
    LAYOUT: str = "wide"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate that required configuration exists"""
        if not cls.TOGETHER_API_KEY:
            return False
        return True
    
    @classmethod
    def get_data_paths(cls) -> dict:
        """Get all data file paths"""
        return {
            "holdings": cls.HOLDINGS_PATH,
            "trades": cls.TRADES_PATH
        }


def setup_page_config():
    """Configure Streamlit page settings"""
    st.set_page_config(
        page_title=Config.PAGE_TITLE,
        page_icon=Config.PAGE_ICON,
        layout=Config.LAYOUT,
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        .stChatMessage {
            padding: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)