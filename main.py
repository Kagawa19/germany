import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import time
import os

# Web Extraction and Database Imports
from web_extractor import WebExtractor
from content_db import store_extract_data, get_all_content

# Logging and Environment
import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

# AI and Concurrency
import concurrent.futures
from openai import OpenAI

# Themes and Insights Functions
from insights_functions import (
    generate_insights_with_openai,
    generate_benefits_to_germany_with_openai,
    analyze_sentiment_with_openai,load_prompt
)
import os
# View Data page function
import streamlit as st
import pandas as pd
from sqlalchemy import text


import logging
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from web_extractor import WebExtractor
import concurrent.futures
from openai import OpenAI

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

# Load environment variables
load_dotenv()

# Get database connection from environment variable
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/appdb")
logger.info(f"Using database URL: {DATABASE_URL}")

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    logger.info("OpenAI client initialized successfully")
else:
    openai_client = None
    logger.warning("OpenAI API key not found, AI-enhanced features will be disabled")

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

# Function to fetch data from the database
def fetch_data(limit=100, filters=None):
    logger.info(f"Fetching data from database (limit: {limit}, filters: {filters})")
    try:
        engine = get_sqlalchemy_engine()
        
        # Prepare base query
        query_parts = []
        params = {}
        
        # Add filter conditions if provided
        if filters:
            if filters.get('theme'):
                query_parts.append(":theme = ANY(themes)")
                params['theme'] = filters['theme']
            
            if filters.get('organization'):
                query_parts.append("organization = :organization")
                params['organization'] = filters['organization']
            
            if filters.get('sentiment'):
                query_parts.append("sentiment = :sentiment")
                params['sentiment'] = filters['sentiment']
            
            if filters.get('start_date') and filters.get('end_date'):
                query_parts.append("date BETWEEN :start_date AND :end_date")
                params['start_date'] = filters['start_date']
                params['end_date'] = filters['end_date']
        
        # Construct WHERE clause
        where_clause = "WHERE " + " AND ".join(query_parts) if query_parts else ""
        
        # Full query
        query = text(f"""
        SELECT 
            id, link, title, date, summary, 
            full_content, information, themes, 
            organization, sentiment, 
            benefits_to_germany, insights, 
            created_at, updated_at
        FROM content_data 
        {where_clause}
        ORDER BY id DESC 
        LIMIT {limit}
        """)
        
        logger.debug(f"Executing query: {query}")
        
        # Execute query with or without parameters
        if params:
            df = pd.read_sql(query, engine, params=params)
        else:
            df = pd.read_sql(query, engine)
        
        logger.info(f"Fetched {len(df)} rows from database")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

def extract_benefits_with_openai(content, benefits_prompt):
    """
    Extract benefits using OpenAI based on the provided prompt.
    """
    if not openai_client or not content:
        return None
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": benefits_prompt},
                {"role": "user", "content": content}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        benefits_text = response.choices[0].message.content.strip()
        return benefits_text
    except Exception as e:
        logger.error(f"Error extracting benefits with OpenAI: {str(e)}")
        return None

def classify_themes_with_openai(content, themes_prompt):
    """
    Classify themes using OpenAI based on the provided prompt.
    """
    if not openai_client or not content:
        return []
    
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": themes_prompt},
                {"role": "user", "content": content}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        themes_str = response.choices[0].message.content.strip()
        themes_list = [theme.strip() for theme in themes_str.split(",")]
        return themes_list
    except Exception as e:
        logger.error(f"Error classifying themes with OpenAI: {str(e)}")
        return []

def load_prompts_from_folder(folder_path="prompts"):
    """
    Load prompts from the specified folder.
    """
    prompts = {}
    
    if not os.path.exists(folder_path):
        logger.error(f"Prompts folder not found: {folder_path}")
        return prompts
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            file_path = os.path.join(folder_path, filename)
            try:
                with open(file_path, "r", encoding="utf-8") as file:
                    prompts[filename] = file.read()
                logger.info(f"Loaded prompt from: {filename}")
            except Exception as e:
                logger.error(f"Error loading prompt from {filename}: {str(e)}")
    
    return prompts

# Function to generate AI summary using OpenAI
def generate_ai_summary(content, max_tokens=750):
    """Generate an extensive summary of content using OpenAI."""
    if not openai_client or not content:
        return None
    
    try:
        # Truncate very long content 
        content_preview = content[:15000] if len(content) > 15000 else content
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a detailed research assistant specializing in summarizing complex texts about international cooperation and environmental sustainability."},
                {"role": "user", "content": f"Create a detailed, informative summary of this content. Provide 6-7 sentences that comprehensively capture the key points, main arguments, and significant insights. Highlight specific initiatives, outcomes, and the broader context of international cooperation in environmental sustainability: {content_preview}"}
            ],
            max_tokens=max_tokens,
            temperature=0.3
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating AI summary: {str(e)}")
        return None





# Function to extract date using OpenAI
def extract_date_with_ai(content, url):
    """Extract publication date from content using OpenAI."""
    if not openai_client or not content:
        return None
    
    try:
        # Extract first 2000 characters where dates are typically found
        content_preview = content[:2000]
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that extracts publication dates from web content. Return only the date in YYYY-MM-DD format if found, or 'None' if no date is found."},
                {"role": "user", "content": f"Extract the publication date from this content from {url}. Return only the date in YYYY-MM-DD format or 'None' if you can't find a date: {content_preview}"}
            ],
            max_tokens=20,
            temperature=0.1
        )
        
        date_text = response.choices[0].message.content.strip()
        
        # Validate date format
        if date_text == "None" or date_text == "null" or date_text == "":
            return None
        
        try:
            # Try to parse the date to ensure it's valid
            parsed_date = datetime.strptime(date_text, "%Y-%m-%d")
            return date_text
        except ValueError:
            logger.warning(f"Invalid date format returned by AI: {date_text}")
            return None
            
    except Exception as e:
        logger.error(f"Error extracting date with AI: {str(e)}")
        return None


def view_data_page():
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
        
        # Information section
        st.subheader("Information")
        st.write(selected_row['information'] if pd.notna(selected_row['information']) else "No additional information available")
        
        # Benefits section
        st.subheader("Benefits to Germany")
        st.write(selected_row['benefits_to_germany'] if pd.notna(selected_row['benefits_to_germany']) else "No benefits information available")
        
        # Insights section
        st.subheader("Insights")
        st.write(selected_row['insights'] if pd.notna(selected_row['insights']) else "No insights available")

# Function to run web extraction
def run_web_extraction(max_queries=None, max_results_per_query=None, use_ai=True):
    """
    Run web extraction process with configurable parameters.
    
    Args:
        max_queries (int, optional): Maximum number of queries to run. Defaults to None.
        max_results_per_query (int, optional): Maximum results per query. Defaults to None.
        use_ai (bool, optional): Whether to use AI for content enhancement. Defaults to True.
    """
    # Log the start of the extraction process with detailed parameters
    logging.info(
        f"Starting web extraction process | "
        f"Max Queries: {max_queries or 'Unlimited'} | "
        f"Max Results Per Query: {max_results_per_query or 'Unlimited'} | "
        f"AI Enhancement: {use_ai}"
    )
    
    # Print user-friendly startup message
    print(f"🌐 Initiating web content extraction...")
    print(f"   Configuration:")
    print(f"   - Max Queries: {max_queries or 'Unlimited'}")
    print(f"   - Max Results Per Query: {max_results_per_query or 'Unlimited'}")
    print(f"   - AI Enhancement: {'Enabled' if use_ai else 'Disabled'}")

    # Use Streamlit spinner for visual feedback
    with st.spinner("Extracting web content... This may take a few minutes."):
        # Create progress bar
        progress_bar = st.progress(0)

        # Log extractor initialization
        logging.info("Initializing WebExtractor")
        print("🔍 Initializing web content extractor...")

        # Dynamically load prompts
        try:
            prompts = load_prompts_from_folder()
            themes_prompt = prompts.get('themes.txt', '')
            insights_prompt = prompts.get('insights.txt', '')
            benefits_prompt = prompts.get('benefits.txt', '')
            sentiment_prompt = prompts.get('sentiment.txt', '')
        except Exception as e:
            logging.error(f"Error loading prompts: {str(e)}")
            themes_prompt = insights_prompt = benefits_prompt = sentiment_prompt = ''

        # Initialize WebExtractor
        extractor = WebExtractor(
            use_ai=use_ai,
            openai_client=openai_client,
            generate_ai_summary_func=generate_ai_summary,
            extract_date_with_ai_func=extract_date_with_ai,
            prompt_path=os.path.join('prompts', 'extract.txt'),
            max_workers=10,
            serper_api_key=os.getenv('SERPER_API_KEY')
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

            # Check extraction results
            if results["status"] == "success" and results["results"]:
                result_count = len(results["results"])
                
                # Detailed logging of extraction results
                logging.info(f"Web extraction successful. Found {result_count} results")
                print(f"✅ Web extraction complete. {result_count} items retrieved.")
                
                status_placeholder.info(f"Enhancing {result_count} items with AI processing...")

                # AI-powered content enhancement
                if use_ai and openai_client and result_count > 0:
                    def enhance_item(item):
                        try:
                            if "content" in item and item["content"]:
                                # Generate AI summary if needed
                                summary = item.get("summary") or generate_ai_summary(item["content"])
                                
                                # Extract date if missing
                                date = item.get("date") or extract_date_with_ai(item["content"], item["link"])
                                
                                # Classify themes using OpenAI
                                themes = classify_themes_with_openai(
                                    item["content"], 
                                    themes_prompt
                                )
                                
                                # Generate insights and benefits
                                insights = generate_insights_with_openai(item["content"]) or None
                                benefits_to_germany = generate_benefits_to_germany_with_openai(item["content"]) or None
                                
                                # Determine sentiment
                                sentiment = analyze_sentiment_with_openai(
                                    item["content"], 
                                    sentiment_prompt
                                )
                                
                                # Update item with enhanced information
                                item.update({
                                    "summary": summary,
                                    "date": date,
                                    "themes": themes,
                                    "insights": insights,
                                    "benefits_to_germany": benefits_to_germany,
                                    "sentiment": sentiment
                                })
                            
                            return item
                        except Exception as e:
                            logging.error(f"Error enhancing item {item.get('link', 'unknown link')}: {str(e)}")
                            return item

                    # Concurrent enhancement of items
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        results["results"] = list(executor.map(enhance_item, results["results"]))
                    
                    print("🧠 AI enhancement completed.")

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
                    from content_db import store_extract_data
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
                # Handle extraction failure
                error_msg = results.get('error', 'Unknown error')
                logging.error(f"Web extraction failed: {error_msg}")
                progress_bar.progress(1.0)
                status_placeholder.error(f"Web extraction failed: {error_msg}")
                print(f"❌ Web extraction failed: {error_msg}")

        except Exception as e:
            # Catch and log any unexpected errors
            logging.exception(f"Exception during web extraction: {str(e)}")
            progress_bar.progress(1.0)
            status_placeholder.error(f"Web extraction failed: {str(e)}")
            print(f"❌ Critical error during web extraction: {str(e)}")


# Sidebar for application navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Dashboard", "View Data", "Web Extraction"])
logger.info(f"User selected app mode: {app_mode}")
print(f"User navigated to: {app_mode}")

# AI Enhancement options in sidebar
use_ai = False
if openai_client:
    ai_section = st.sidebar.expander("AI Enhancement Options", expanded=False)
    with ai_section:
        use_ai = st.checkbox("Use AI for enhanced processing", value=True)
        st.info("Using AI will improve summaries and date extraction")
else:
    st.sidebar.warning("OpenAI API key not configured. AI enhancements disabled.")

# Main application logic
if app_mode == "Dashboard":
    logger.info("Rendering Dashboard page")
    
    # Fetch data for the dashboard
    df = fetch_data(limit=1000)
    
    # Display key metrics
    st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Total Documents</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        org_count = df['organization'].nunique() if 'organization' in df.columns else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Organizations</div>
        </div>
        """.format(org_count), unsafe_allow_html=True)
    
    with col3:
        theme_count = df['theme'].nunique() if 'theme' in df.columns else 0
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Themes</div>
        </div>
        """.format(theme_count), unsafe_allow_html=True)
    
    with col4:
        # Convert to datetime if it's not
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            date_range = (df['date'].max() - df['date'].min()).days if not df['date'].isna().all() else 0
        else:
            date_range = 0
            
        st.markdown("""
        <div class="metric-card">
            <div class="metric-value">{}</div>
            <div class="metric-label">Date Range (days)</div>
        </div>
        """.format(date_range), unsafe_allow_html=True)

elif app_mode == "View Data":
    view_data_page()

elif app_mode == "Web Extraction":
    st.subheader("Web Content Extraction")
    st.write("Configure the extraction settings and click the button below to extract web content.")
    
    # Extraction options
    col1, col2 = st.columns(2)
    with col1:
        max_queries = st.slider("Number of search queries to process", 5, 40, 20)
        max_results_per_query = st.slider("Results per query", 3, 8, 6)
        logger.debug(f"User set max_queries to: {max_queries}, max_results_per_query to: {max_results_per_query}")
        print(f"User set extraction parameters: {max_queries} queries with {max_results_per_query} results per query")
    
    with col2:
        use_custom_prompt = st.checkbox("Use custom prompt file")
        logger.debug(f"Use custom prompt: {use_custom_prompt}")
        print(f"Use custom prompt option set to: {use_custom_prompt}")
        
        # AI enhancement options
        if openai_client:
            use_ai_extraction = st.checkbox("Use AI for enhanced extraction", value=use_ai)
            st.info("AI will improve summaries and date extraction")
        else:
            use_ai_extraction = False
            st.warning("OpenAI API not configured. AI enhancements disabled.")
        
        total_potential_results = max_queries * max_results_per_query
        st.info(f"Maximum potential results: {total_potential_results}")
        st.caption("Note: Actual results will likely be fewer due to duplicate URLs and filtering")
    
    if use_custom_prompt:
        prompt_file = st.text_input("Prompt file path (relative to project root)", value="prompts/extract.txt")
        logger.debug(f"Custom prompt file path: {prompt_file}")
        print(f"Custom prompt file path set to: {prompt_file}")
    
    # Add a button to trigger the extraction
    if st.button("Start Web Extraction", key="extract_button", use_container_width=True):
        logger.info(f"User clicked Start Web Extraction button with params: max_queries={max_queries}, max_results_per_query={max_results_per_query}, AI={use_ai_extraction}")
        print(f"Starting web extraction process with {max_queries} queries and {max_results_per_query} results per query, AI: {use_ai_extraction}...")
        run_web_extraction(max_queries=max_queries, max_results_per_query=max_results_per_query, use_ai=use_ai_extraction)

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