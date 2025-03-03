import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import os
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# Import local modules
from web_extractor import WebExtractor
from content_db import store_extract_data, fetch_data, get_all_content

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
logger = logging.getLogger("main")

# Load environment variables
load_dotenv()

# Get database connection from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/appdb")
logger.info(f"Using database URL: {DATABASE_URL}")

# Initialize API keys
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Set page config
st.set_page_config(
    page_title="Germany Environmental Cooperation Analysis",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Create a connection to the database
@st.cache_resource
def get_sqlalchemy_engine():
    logger.info("Creating database connection")
    try:
        engine = create_engine(DATABASE_URL)
        logger.info("Database connection created successfully")
        return engine
    except Exception as e:
        logger.error(f"Failed to create database connection: {str(e)}")
        raise

# Function to run web extraction
def run_web_extraction(max_queries=None, max_results_per_query=None):
    """
    Run web extraction process with configurable parameters.
    
    Args:
        max_queries: Maximum number of queries to run
        max_results_per_query: Maximum results per query
    """
    # Log the start of the extraction process with detailed parameters
    logging.info(
        f"Starting web extraction process | "
        f"Max Queries: {max_queries or 'Unlimited'} | "
        f"Max Results Per Query: {max_results_per_query or 'Unlimited'}"
    )
    
    # Print user-friendly startup message
    print(f"🌐 Initiating web content extraction...")
    print(f"   Configuration:")
    print(f"   - Max Queries: {max_queries or 'Unlimited'}")
    print(f"   - Max Results Per Query: {max_results_per_query or 'Unlimited'}")

    # Use Streamlit spinner for visual feedback
    with st.spinner("Extracting web content... This may take a few minutes."):
        # Create progress bar
        progress_bar = st.progress(0)

        # Log extractor initialization
        logging.info("Initializing WebExtractor")
        print("🔍 Initializing web content extractor...")

        # Initialize WebExtractor
        extractor = WebExtractor(
            search_api_key=SERPER_API_KEY,
            max_workers=10
        )

        # Status placeholders
        status_placeholder = st.empty()
        status_placeholder.info("Searching the web for relevant content...")
        progress_bar.progress(25)

        try:
            # Run web extraction
            results = extractor.run(
                max_queries=max_queries, 
                max_results_per_query=max_results_per_query
            )
            progress_bar.progress(40)

            # First check the status
            if results["status"] == "success":
                # This is a successful extraction regardless of whether new results were found
                
                if results["results"]:
                    # Case 1: Success with results
                    result_count = len(results["results"])
                    
                    # Detailed logging of extraction results
                    logging.info(f"Web extraction successful. Found {result_count} results")
                    print(f"✅ Web extraction complete. {result_count} items retrieved.")
                    
                    status_placeholder.info(f"Processing {result_count} items...")
                    progress_bar.progress(50)

                    try:
                        # Database operations
                        engine = get_sqlalchemy_engine()
                        existing_urls_query = text("SELECT link FROM content_data")
                        existing_urls = set(pd.read_sql(existing_urls_query, engine)['link'])

                        # Filter out duplicate URLs
                        original_count = len(results["results"])
                        results["results"] = [r for r in results["results"] if r.get("link") not in existing_urls]
                        filtered_count = original_count - len(results["results"])

                        if filtered_count > 0:
                            logging.info(f"Filtered out {filtered_count} duplicate results")
                            print(f"🚫 Removed {filtered_count} duplicate entries.")

                        # Store extracted data
                        stored_ids = store_extract_data(results["results"])
                        stored_count = len(stored_ids)

                        # Final progress and logging
                        progress_bar.progress(100)
                        logging.info(f"Web extraction completed. Saved {stored_count}/{result_count} items to database")
                        print(f"💾 Saved {stored_count} new items to database.")
                        
                        status_placeholder.success(f"Saved {stored_count} new items to database.")

                        # Display saved items if any
                        if stored_ids:
                            try:
                                engine = get_sqlalchemy_engine()
                                ids_str = ','.join(str(id) for id in stored_ids)
                                query = text(f"SELECT id, link, title, date, summary, themes, organization, sentiment FROM content_data WHERE id IN ({ids_str})")
                                saved_df = pd.read_sql(query, engine)

                                if not saved_df.empty:
                                    st.subheader("Newly Extracted Content")
                                    st.dataframe(saved_df)
                            except Exception as e:
                                logging.error(f"Error displaying saved items: {str(e)}")
                                print(f"❌ Error displaying saved items: {str(e)}")

                    except Exception as e:
                        logging.error(f"Error handling extraction results: {str(e)}")
                        status_placeholder.error(f"Error handling extraction results: {str(e)}")
                        print(f"❌ Error processing extraction results: {str(e)}")

                    # Display latest content data
                    st.subheader("Latest Content Data")
                    st.dataframe(fetch_data())
                    
                else:
                    # Case 2: Success but no new results
                    # This is the critical part that was missing
                    message = results.get('message', 'No new content found')
                    skipped = results.get('skipped_urls', 0)
                    
                    logging.info(f"Web extraction successful: {message}")
                    print(f"✅ Web extraction complete. {message}")
                    
                    if skipped > 0:
                        print(f"⏭️ Skipped {skipped} already processed URLs")
                    
                    progress_bar.progress(100)
                    status_placeholder.success(f"Web extraction complete: {message}")
                    
                    # Still display latest content data
                    st.subheader("Latest Content Data")
                    st.dataframe(fetch_data())
                    
            elif results["status"] == "warning":
                # Case 3: Warning status - not an error but needs attention
                message = results.get('message', 'Warning during extraction')
                
                logging.warning(f"Web extraction warning: {message}")
                print(f"⚠️ Web extraction completed with warnings: {message}")
                
                progress_bar.progress(100)
                status_placeholder.warning(f"Web extraction warning: {message}")
                
                # Still display latest content data
                st.subheader("Latest Content Data")
                st.dataframe(fetch_data())
                
            else:
                # Case 4: True error
                error_msg = results.get('error', results.get('message', 'Unknown error'))
                logging.error(f"Web extraction failed: {error_msg}")
                progress_bar.progress(100)
                status_placeholder.error(f"Web extraction failed: {error_msg}")
                print(f"❌ Web extraction failed: {error_msg}")

        except Exception as e:
            # Catch and log any unexpected errors
            logging.exception(f"Exception during web extraction: {str(e)}")
            progress_bar.progress(100)
            status_placeholder.error(f"Web extraction failed: {str(e)}")
            print(f"❌ Critical error during web extraction: {str(e)}")

def view_data_page():
    """Display the data viewing and exploration page."""
    st.subheader("Explore Content Data")
    
    # Filtering options
    st.sidebar.header("Data Filters")
    
    # Get unique values for filters
    engine = get_sqlalchemy_engine()
    
    # Fetch unique themes
    themes_query = text("SELECT DISTINCT unnest(themes) AS theme FROM content_data WHERE themes IS NOT NULL ORDER BY theme")
    themes = [t[0] for t in pd.read_sql(themes_query, engine)['theme']]
    themes.insert(0, "All")  # Add "All" option at the beginning
    
    # Similar approach for organizations and sentiments
    orgs_query = text("SELECT DISTINCT organization FROM content_data WHERE organization IS NOT NULL ORDER BY organization")
    organizations = [o[0] for o in pd.read_sql(orgs_query, engine)['organization']]
    organizations.insert(0, "All")  # Add "All" option at the beginning
    
    sentiments_query = text("SELECT DISTINCT sentiment FROM content_data WHERE sentiment IS NOT NULL ORDER BY sentiment")
    sentiments = [s[0] for s in pd.read_sql(sentiments_query, engine)['sentiment']]
    sentiments.insert(0, "All")  # Add "All" option at the beginning
    
    # Filter dropdowns
    filter_theme = st.sidebar.selectbox("Theme", themes)
    filter_org = st.sidebar.selectbox("Organization", organizations)
    filter_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    
    # Date range filter
    st.sidebar.header("Date Range")
    min_date_query = text("SELECT MIN(date) FROM content_data WHERE date IS NOT NULL")
    max_date_query = text("SELECT MAX(date) FROM content_data WHERE date IS NOT NULL")
    
    with engine.connect() as connection:
        min_date = connection.execute(min_date_query).scalar_one_or_none()
        max_date = connection.execute(max_date_query).scalar_one_or_none()
    
    if min_date and max_date:
        date_range = st.sidebar.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = None
    
    # Prepare filters
    filters = {}
    if filter_theme != "All":
        filters['theme'] = filter_theme
    if filter_org != "All":
        filters['organization'] = filter_org
    if filter_sentiment != "All":
        filters['sentiment'] = filter_sentiment
    
    # Add date range to filters
    if date_range:
        filters['start_date'] = date_range[0]
        filters['end_date'] = date_range[1]
    
    # Fetch and display data
    display_limit = st.slider("Number of records to display", 100, 5000, 500)
    df = fetch_data(limit=display_limit, filters=filters)
    
    # Display data
    st.dataframe(df)
    
    # Detailed view of selected row
    if not df.empty:
        st.subheader("Detailed View")
        selected_id = st.selectbox("Select a record to view details", df['id'].tolist())
        
        # Find selected row
        selected_row = df[df['id'] == selected_id].iloc[0]
        
        # Display detailed information
        st.markdown(f"**Title:** {selected_row['title']}")
        st.markdown(f"**Link:** {selected_row['link']}")
        st.markdown(f"**Date:** {selected_row['date']}")
        st.markdown(f"**Themes:** {', '.join(selected_row['themes']) if selected_row['themes'] else 'N/A'}")
        st.markdown(f"**Organization:** {selected_row['organization'] if pd.notna(selected_row['organization']) else 'N/A'}")
        st.markdown(f"**Sentiment:** {selected_row['sentiment'] if pd.notna(selected_row['sentiment']) else 'N/A'}")
        
        # Summary section
        st.subheader("Summary")
        st.write(selected_row['summary'] if pd.notna(selected_row['summary']) else "No summary available")
        
        # Full Content section
        st.subheader("Full Content")
        st.write(selected_row['full_content'] if pd.notna(selected_row['full_content']) else "No full content available")
        
        # Benefits section
        if 'benefits_to_germany' in selected_row and pd.notna(selected_row['benefits_to_germany']):
            st.subheader("Benefits to Germany")
            st.write(selected_row['benefits_to_germany'])

def dashboard_page():
    """Display the main dashboard with key metrics and visualizations."""
    logger.info("Rendering Dashboard page")
    
    # Fetch data for the dashboard
    df = fetch_data(limit=1000)
    
    # Display key metrics
    st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(df))
    
    with col2:
        org_count = df['organization'].nunique() if 'organization' in df.columns else 0
        st.metric("Organizations", org_count)
    
    with col3:
        theme_count = 0
        if 'themes' in df.columns:
            # Count unique themes across all rows
            all_themes = set()
            for themes_list in df['themes'].dropna():
                if themes_list:
                    all_themes.update(themes_list)
            theme_count = len(all_themes)
        st.metric("Themes", theme_count)
    
    with col4:
        # Convert to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            date_range = (df['date'].max() - df['date'].min()).days if not df['date'].isna().all() else 0
        else:
            date_range = 0
        st.metric("Date Range (days)", date_range)

    # Create visualizations if we have sufficient data
    if len(df) > 0:
        # Create tabs for different visualizations
        tab1, tab2, tab3 = st.tabs(["Organizations", "Themes", "Timeline"])
        
        with tab1:
            if 'organization' in df.columns:
                # Organization distribution
                org_counts = df['organization'].value_counts().reset_index()
                org_counts.columns = ['Organization', 'Count']
                
                fig = px.bar(
                    org_counts, 
                    x='Organization', 
                    y='Count',
                    title='Content by Organization',
                    color='Count',
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'themes' in df.columns:
                # Theme distribution
                theme_counts = {}
                for themes_list in df['themes'].dropna():
                    if themes_list:
                        for theme in themes_list:
                            theme_counts[theme] = theme_counts.get(theme, 0) + 1
                
                theme_df = pd.DataFrame({
                    'Theme': list(theme_counts.keys()),
                    'Count': list(theme_counts.values())
                }).sort_values('Count', ascending=False)
                
                if not theme_df.empty:
                    fig = px.bar(
                        theme_df,
                        x='Count',
                        y='Theme',
                        title='Content by Theme',
                        orientation='h',
                        color='Count',
                        color_continuous_scale='greens'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No theme data available for visualization")
            else:
                st.info("Theme data not available in the dataset")
        
        with tab3:
            if 'date' in df.columns:
                # Timeline visualization
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                date_counts = df.dropna(subset=['date']).groupby(df['date'].dt.strftime('%Y-%m')).size().reset_index()
                date_counts.columns = ['Month', 'Count']
                
                if not date_counts.empty:
                    fig = px.line(
                        date_counts,
                        x='Month',
                        y='Count',
                        title='Content Timeline',
                        markers=True
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough date data for timeline visualization")
            else:
                st.info("Date data not available in the dataset")
    else:
        st.info("Not enough data available for visualizations. Extract more content first.")
        
    # Display a sample of recent content
    st.subheader("Recent Content")
    recent_df = df.head(5)
    if not recent_df.empty:
        for _, row in recent_df.iterrows():
            with st.expander(f"{row['title']}"):
                st.write(f"**Date:** {row['date']}")
                st.write(f"**Organization:** {row['organization'] if pd.notna(row['organization']) else 'N/A'}")
                st.write(f"**Summary:** {row['summary'] if pd.notna(row['summary']) else 'No summary available'}")
                st.write(f"**Link:** {row['link']}")
    else:
        st.info("No content available. Extract content first.")

# Sidebar for application navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Dashboard", "View Data", "Web Extraction"])
logger.info(f"User selected app mode: {app_mode}")
print(f"User navigated to: {app_mode}")

# Main application logic
if app_mode == "Dashboard":
    st.title("Germany Environmental Cooperation Analysis")
    dashboard_page()

elif app_mode == "View Data":
    st.title("Content Data Explorer")
    view_data_page()

elif app_mode == "Web Extraction":
    st.title("Web Content Extraction")
    st.write("Configure the extraction settings and click the button below to extract web content.")
    
    # Extraction options
    col1, col2 = st.columns(2)
    with col1:
        max_queries = st.slider("Number of search queries to process", 5, 40, 20)
        max_results_per_query = st.slider("Results per query", 3, 8, 6)
        logger.debug(f"User set max_queries to: {max_queries}, max_results_per_query to: {max_results_per_query}")
        print(f"User set extraction parameters: {max_queries} queries with {max_results_per_query} results per query")
    
    with col2:
        if not SERPER_API_KEY:
            st.warning("Search API key not configured. Please add SERPER_API_KEY to your .env file.")
            serper_help = st.expander("How to get a Serper API key")
            with serper_help:
                st.write("""
                1. Go to [Serper.dev](https://serper.dev)
                2. Sign up for an account
                3. Get your API key from the dashboard
                4. Add it to your .env file as SERPER_API_KEY=your_key_here
                """)
        
        total_potential_results = max_queries * max_results_per_query
        st.info(f"Maximum potential results: {total_potential_results}")
        st.caption("Note: Actual results will likely be fewer due to duplicate URLs and filtering")
    
    # Add a button to trigger the extraction
    if st.button("Start Web Extraction", key="extract_button", use_container_width=True):
        logger.info(f"User clicked Start Web Extraction button with params: max_queries={max_queries}, max_results_per_query={max_results_per_query}")
        print(f"Starting web extraction process with {max_queries} queries and {max_results_per_query} results per query...")
        run_web_extraction(max_queries=max_queries, max_results_per_query=max_results_per_query)

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
st.markdown("© 2025 Germany Environmental Cooperation Analysis")

# Logging
logger.info("Application rendering completed")

# Run the main app if this file is executed directly
if __name__ == "__main__":
    logger.info("Application main method executed")
    print("Application started via main method")