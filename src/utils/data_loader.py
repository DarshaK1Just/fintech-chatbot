"""Data loading utilities"""

import pandas as pd
import streamlit as st
from pathlib import Path
from typing import Optional, Tuple
from src.core.config import Config


class DataLoader:
    """Handle loading of CSV data files"""
    
    @staticmethod
    def load_from_file(file) -> Optional[pd.DataFrame]:
        """Load DataFrame from uploaded file"""
        try:
            return pd.read_csv(file)
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            return None
    
    @staticmethod
    def load_from_path(path: Path) -> Optional[pd.DataFrame]:
        """Load DataFrame from file path"""
        try:
            if path.exists():
                return pd.read_csv(path)
            return None
        except Exception as e:
            st.error(f"Error loading {path.name}: {str(e)}")
            return None
    
    @staticmethod
    def load_default_files() -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load default CSV files from data directory"""
        holdings_df = DataLoader.load_from_path(Config.HOLDINGS_PATH)
        trades_df = DataLoader.load_from_path(Config.TRADES_PATH)
        return holdings_df, trades_df
    
    @staticmethod
    def get_dataframe_info(df: pd.DataFrame) -> dict:
        """Get information about a DataFrame"""
        return {
            "rows": len(df),
            "columns": list(df.columns),
            "column_count": len(df.columns)
        }