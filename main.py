import streamlit as st
import pandas as pd
import requests
import os
import json
from sqlalchemy import create_engine, text
import time

# Get database connection from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/appdb")

# Set page config
st.set_page_config(
    page_title="Data Processing Application",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Add a title and description
st.title("Data Processing Application")
st.write("Use this application to process and manage content data.")

# Create a connection to the database
@st.cache_resource
def get_db_connection():
    return create_engine(DATABASE_URL)

# Function to fetch data from the database
def fetch_data():
    engine = get_db_connection()
    query = text("SELECT id, link, summary, theme, organization FROM content_data")
    df = pd.read_sql(query, engine)
    return df

# Function to process data
def process_data():
    with st.spinner("Processing data..."):
        # Simulate processing with a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)
        
        # Show success message after processing
        st.success("Data processed successfully!")
        
        # Display the latest data
        st.subheader("Latest Content Data")
        st.dataframe(fetch_data())

# Sidebar for application navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "View Data", "Process Data"])

# Home page
if app_mode == "Home":
    st.write("Welcome to the Data Processing Application!")
    st.write("This application allows you to view and process content data.")
    st.write("Use the sidebar to navigate to different sections.")
    
    # Add a big process data button on the home page
    if st.button("Process Data", key="home_process", use_container_width=True):
        process_data()

# View Data page
elif app_mode == "View Data":
    st.subheader("Content Data")
    data = fetch_data()
    st.dataframe(data)
    
    # Add filtering options
    if not data.empty:
        st.subheader("Filter Data")
        col1, col2 = st.columns(2)
        with col1:
            selected_theme = st.selectbox("Filter by Theme", ["All"] + data["theme"].unique().tolist())
        with col2:
            selected_org = st.selectbox("Filter by Organization", ["All"] + data["organization"].unique().tolist())
            
        # Apply filters
        filtered_data = data.copy()
        if selected_theme != "All":
            filtered_data = filtered_data[filtered_data["theme"] == selected_theme]
        if selected_org != "All":
            filtered_data = filtered_data[filtered_data["organization"] == selected_org]
            
        st.subheader("Filtered Results")
        st.dataframe(filtered_data)

# Process Data page
elif app_mode == "Process Data":
    st.subheader("Process Content Data")
    st.write("Click the button below to process the content data.")
    
    # Add a button to trigger the processing
    if st.button("Process Data", key="process_page_button", use_container_width=True):
        process_data()
    
    # Add a form to add new content
    st.subheader("Add New Content")
    with st.form("add_content_form"):
        link = st.text_input("Link")
        summary = st.text_area("Summary")
        theme = st.text_input("Theme")
        organization = st.text_input("Organization")
        
        # Submit button
        submitted = st.form_submit_button("Add Content")
        if submitted:
            try:
                # Make API call to FastAPI backend
                api_url = "http://app:8000/content/"
                payload = {
                    "link": link,
                    "summary": summary,
                    "theme": theme,
                    "organization": organization
                }
                
                response = requests.post(api_url, json=payload)
                if response.status_code == 200:
                    st.success("Content added successfully!")
                    st.json(response.json())
                else:
                    st.error(f"Error adding content: {response.text}")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2025 Data Processing Application")