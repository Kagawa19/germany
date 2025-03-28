import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import logging
import json

from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Import local modules
from extraction.extractor import WebExtractor
from database.operations import store_extract_data, fetch_data, get_all_content, create_schema
from database.connection import get_sqlalchemy_engine
from ui.dashboard import dashboard_page, initiative_dashboard, initiative_comparison
from ui.data_explorer import view_data_page, export_content_by_language
from ui.extraction_ui import run_web_extraction, initialization_page
from utils.logging import configure_logging

# Configure logging
logger = configure_logging("main")

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Initiative Analysis Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Sidebar for application navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode", 
    ["Dashboard", "Initiative Analysis", "Initiative Comparison", "View Data", "Web Extraction", "Database Initialization", "Export Data"]
)
logger.info(f"User selected app mode: {app_mode}")

# Main application logic
if app_mode == "Dashboard":
    st.title("Initiative Analysis Dashboard")
    dashboard_page()

elif app_mode == "Initiative Analysis":
    initiative_dashboard()

elif app_mode == "Initiative Comparison":
    initiative_comparison()

elif app_mode == "View Data":
    st.title("Content Data Explorer")
    view_data_page()

elif app_mode == "Export Data":
    st.title("Export Content Data")
    export_content_by_language()
    
elif app_mode == "Web Extraction":
    st.title("Web Content Extraction")
    run_web_extraction()

elif app_mode == "Database Initialization":
    initialization_page()

# Custom styling
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
</style>
""", unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Initiative Analysis Dashboard")

# Run the main app if this file is executed directly
if __name__ == "__main__":
    logger.info("Application main method executed")
    print("Application started via main method")