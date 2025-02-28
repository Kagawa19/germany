import streamlit as st
import pandas as pd
import requests
import os
import json
import logging
from sqlalchemy import create_engine, text
import time
from web_extractor import WebExtractor  # Import the WebExtractor class

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)

# Create a logger for this module
logger = logging.getLogger(__name__)

# Get database connection from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/appdb")
logger.info(f"Using database URL: {DATABASE_URL}")

# Set page config
st.set_page_config(
    page_title="Data Processing Application",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

logger.info("Application started")
print("Data Processing Application started")

# Add a title and description
st.title("Data Processing Application")
st.write("Use this application to process and manage content data.")

# Create a connection to the database
@st.cache_resource
def get_db_connection():
    logger.info("Creating database connection")
    try:
        engine = create_engine(DATABASE_URL)
        logger.info("Database connection created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database connection: {str(e)}")
        raise

# Function to fetch data from the database
def fetch_data():
    logger.info("Fetching data from database")
    try:
        engine = get_db_connection()
        query = text("SELECT id, link, summary, theme, organization FROM content_data")
        logger.debug(f"Executing query: {query}")
        df = pd.read_sql(query, engine)
        logger.info(f"Fetched {len(df)} rows from database")
        print(f"Fetched {len(df)} rows from content_data table")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        print(f"ERROR: Failed to fetch data: {str(e)}")
        raise

# Function to process data
def process_data():
    logger.info("Starting data processing")
    print("Beginning data processing routine")
    with st.spinner("Processing data..."):
        # Simulate processing with a progress bar
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.05)
            progress_bar.progress(i + 1)
            if i % 25 == 0:
                logger.debug(f"Processing progress: {i+1}%")
                print(f"Processing data: {i+1}% complete")
        
        # Show success message after processing
        logger.info("Data processing completed successfully")
        print("Data processing completed successfully")
        st.success("Data processed successfully!")
        
        # Display the latest data
        st.subheader("Latest Content Data")
        st.dataframe(fetch_data())

# Function to run web extraction and save results to database
def run_web_extraction():
    logger.info("Starting web extraction process")
    print("Starting web content extraction")
    with st.spinner("Extracting web content... This may take a few minutes."):
        # Create progress bar
        progress_bar = st.progress(0)
        
        # Initialize WebExtractor
        logger.info("Initializing WebExtractor")
        extractor = WebExtractor()
        
        # Set up a placeholder for status updates
        status_placeholder = st.empty()
        status_placeholder.info("Initializing web extraction...")
        
        # Run the extractor
        logger.info("Running web extractor")
        print("Searching for relevant web content...")
        status_placeholder.info("Searching the web for relevant content...")
        progress_bar.progress(25)
        
        try:
            results = extractor.run()
            logger.debug(f"Web extraction results: {json.dumps(results, default=str)[:1000]}...")
            progress_bar.progress(75)
            
            # Check if we got results
            if results["status"] == "success" and results["results"]:
                result_count = len(results["results"])
                logger.info(f"Web extraction successful. Found {result_count} results")
                print(f"Found {result_count} relevant web pages")
                status_placeholder.info(f"Found {result_count} relevant web pages. Saving to database...")
                
                # Save results to database
                engine = get_db_connection()
                saved_count = 0
                
                for i, result in enumerate(results["results"]):
                    try:
                        # Extract a summary (first 500 chars of content)
                        summary = result["content"][:500] + "..." if len(result["content"]) > 500 else result["content"]
                        
                        # Determine theme and organization from the result
                        # This is a simplified approach - you might want to implement a more sophisticated categorization
                        theme = "Environmental Sustainability"
                        organization = result.get("source", "")
                        
                        logger.debug(f"Saving result {i+1}/{result_count}: {result['link']}")
                        
                        # Insert into database using API call
                        api_url = "http://app:8000/content/"
                        payload = {
                            "link": result["link"],
                            "summary": summary,
                            "theme": theme,
                            "organization": organization,
                            "content": result["content"]  # Add full content if your API/DB supports it
                        }
                        
                        logger.debug(f"Sending API request to {api_url}")
                        print(f"Saving content from: {result['link']}")
                        response = requests.post(api_url, json=payload)
                        if response.status_code == 200:
                            saved_count += 1
                            logger.info(f"Successfully saved content: {result['link']}")
                        else:
                            logger.warning(f"Failed to save content: {response.status_code} - {response.text}")
                            print(f"WARNING: Failed to save content from {result['link']}: {response.status_code}")
                            
                    except Exception as e:
                        logger.error(f"Error saving result {i+1}: {str(e)}")
                        print(f"ERROR: Failed to save result {i+1}: {str(e)}")
                        st.error(f"Error saving result: {str(e)}")
                        continue
                
                progress_bar.progress(100)
                logger.info(f"Web extraction completed. Saved {saved_count}/{result_count} items to database")
                print(f"Web extraction completed! Saved {saved_count}/{result_count} items")
                status_placeholder.success(f"Web extraction completed! Saved {saved_count} items to database.")
                
                # Display the latest data
                st.subheader("Latest Content Data")
                st.dataframe(fetch_data())
            else:
                error_msg = results.get('error', 'Unknown error')
                logger.error(f"Web extraction failed or found no results: {error_msg}")
                print(f"ERROR: Web extraction failed: {error_msg}")
                progress_bar.progress(100)
                status_placeholder.error(f"Web extraction failed or found no results: {error_msg}")
        except Exception as e:
            logger.exception(f"Exception during web extraction: {str(e)}")
            print(f"CRITICAL ERROR: Web extraction process failed: {str(e)}")
            progress_bar.progress(100)
            status_placeholder.error(f"Web extraction failed with exception: {str(e)}")

# Sidebar for application navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "View Data", "Process Data", "Web Extraction"])
logger.info(f"User selected app mode: {app_mode}")
print(f"User navigated to: {app_mode}")

# Home page
if app_mode == "Home":
    logger.info("Rendering Home page")
    st.write("Welcome to the Data Processing Application!")
    st.write("This application allows you to view and process content data.")
    st.write("Use the sidebar to navigate to different sections.")
    
    # Add buttons on the home page
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Data", key="home_process", use_container_width=True):
            logger.info("User clicked Process Data button from Home page")
            print("Process Data button clicked from Home page")
            process_data()
    with col2:
        if st.button("Run Web Extraction", key="home_extract", use_container_width=True):
            logger.info("User clicked Run Web Extraction button from Home page")
            print("Web Extraction button clicked from Home page")
            run_web_extraction()

# View Data page
elif app_mode == "View Data":
    logger.info("Rendering View Data page")
    st.subheader("Content Data")
    try:
        data = fetch_data()
        st.dataframe(data)
        
        # Add filtering options
        if not data.empty:
            logger.info(f"Loaded data with {len(data)} rows for viewing")
            st.subheader("Filter Data")
            col1, col2 = st.columns(2)
            with col1:
                theme_options = ["All"] + data["theme"].unique().tolist()
                selected_theme = st.selectbox("Filter by Theme", theme_options)
                logger.debug(f"User selected theme filter: {selected_theme}")
            with col2:
                org_options = ["All"] + data["organization"].unique().tolist()
                selected_org = st.selectbox("Filter by Organization", org_options)
                logger.debug(f"User selected organization filter: {selected_org}")
                
            # Apply filters
            filtered_data = data.copy()
            if selected_theme != "All":
                filtered_data = filtered_data[filtered_data["theme"] == selected_theme]
                logger.debug(f"Applied theme filter, remaining rows: {len(filtered_data)}")
                print(f"Filtered data by theme '{selected_theme}': {len(filtered_data)} rows remaining")
            if selected_org != "All":
                filtered_data = filtered_data[filtered_data["organization"] == selected_org]
                logger.debug(f"Applied organization filter, remaining rows: {len(filtered_data)}")
                print(f"Filtered data by organization '{selected_org}': {len(filtered_data)} rows remaining")
                
            st.subheader("Filtered Results")
            st.dataframe(filtered_data)
            logger.info(f"Displaying filtered data with {len(filtered_data)} rows")
        else:
            logger.warning("No data available to display")
            print("WARNING: No data available to display in View Data page")
    except Exception as e:
        logger.exception(f"Error in View Data page: {str(e)}")
        print(f"ERROR in View Data page: {str(e)}")
        st.error(f"Error loading data: {str(e)}")

# Process Data page
elif app_mode == "Process Data":
    logger.info("Rendering Process Data page")
    st.subheader("Process Content Data")
    st.write("Click the button below to process the content data.")
    
    # Add a button to trigger the processing
    if st.button("Process Data", key="process_page_button", use_container_width=True):
        logger.info("User clicked Process Data button from Process Data page")
        print("Process Data button clicked from Process Data page")
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
            logger.info("User submitted Add Content form")
            print(f"Adding new content with link: {link}")
            try:
                # Make API call to FastAPI backend
                api_url = "http://app:8000/content/"
                payload = {
                    "link": link,
                    "summary": summary,
                    "theme": theme,
                    "organization": organization
                }
                
                logger.debug(f"Sending API request to {api_url} with payload: {json.dumps(payload)}")
                response = requests.post(api_url, json=payload)
                if response.status_code == 200:
                    logger.info("Content added successfully")
                    print(f"Successfully added content: {link}")
                    st.success("Content added successfully!")
                    st.json(response.json())
                else:
                    logger.error(f"Error adding content: {response.status_code} - {response.text}")
                    print(f"ERROR: Failed to add content: {response.status_code} - {response.text}")
                    st.error(f"Error adding content: {response.text}")
            except Exception as e:
                logger.exception(f"Exception when adding content: {str(e)}")
                print(f"CRITICAL ERROR: Could not add content: {str(e)}")
                st.error(f"Error: {str(e)}")

# Web Extraction page (new)
elif app_mode == "Web Extraction":
    logger.info("Rendering Web Extraction page")
    st.subheader("Web Content Extraction")
    st.write("Click the button below to extract web content about Germany's international cooperation efforts in environmental sustainability.")
    
    # Add options for extraction
    st.subheader("Extraction Options")
    col1, col2 = st.columns(2)
    with col1:
        num_results = st.slider("Maximum number of results to save", min_value=5, max_value=50, value=20)
        logger.debug(f"User set maximum results to: {num_results}")
        print(f"User set maximum results to: {num_results}")
    with col2:
        use_custom_prompt = st.checkbox("Use custom prompt file")
        logger.debug(f"Use custom prompt: {use_custom_prompt}")
        print(f"Use custom prompt option set to: {use_custom_prompt}")
    
    if use_custom_prompt:
        prompt_file = st.text_input("Prompt file path (relative to project root)", value="prompts/extract.txt")
        logger.debug(f"Custom prompt file path: {prompt_file}")
        print(f"Custom prompt file path set to: {prompt_file}")
    
    # Add a button to trigger the extraction
    if st.button("Start Web Extraction", key="extract_button", use_container_width=True):
        logger.info("User clicked Start Web Extraction button")
        print("Starting web extraction process...")
        run_web_extraction()
    
    # Show extraction history (you would need to implement this)
    st.subheader("Previous Extractions")
    st.info("This feature will show the history of web extractions performed.")
    logger.info("Displayed web extraction history placeholder")

# Footer
st.markdown("---")
st.markdown("© 2025 Data Processing Application")
logger.info("Page rendering completed")