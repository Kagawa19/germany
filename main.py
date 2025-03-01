import streamlit as st
import pandas as pd
import requests
import os
import json
import logging
import time
from datetime import datetime
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from web_extractor import WebExtractor
from content_db import store_extract_data, analyze_content_for_benefits, get_db_connection
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

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
def fetch_data(limit=100):
    logger.info(f"Fetching data from database (limit: {limit})")
    try:
        engine = get_sqlalchemy_engine()
        query = text(f"""
        SELECT id, link, title, date, summary, theme, organization 
        FROM content_data 
        ORDER BY id DESC 
        LIMIT {limit}
        """)
        logger.debug(f"Executing query: {query}")
        df = pd.read_sql(query, engine)
        logger.info(f"Fetched {len(df)} rows from database")
        print(f"Fetched {len(df)} rows from content_data table")
        return df
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        print(f"ERROR: Failed to fetch data: {str(e)}")
        st.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()

# Function to generate AI summary using OpenAI
def generate_ai_summary(content, max_tokens=250):
    """Generate a concise summary of content using OpenAI."""
    if not openai_client or not content:
        return None
    
    try:
        # Truncate long content to avoid token limits
        content_preview = content[:10000] if len(content) > 10000 else content
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text about environmental sustainability and international cooperation. Create concise, factual summaries."},
                {"role": "user", "content": f"Summarize this content in 2-3 sentences focusing on the key information about international cooperation and environmental sustainability efforts: {content_preview}"}
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

# Function to analyze content with AI for benefits and insights
def analyze_content_with_ai(content_id, title, content, limit=10):
    """Analyze content for benefits and insights using OpenAI."""
    if not openai_client or not content:
        return None, None
    
    try:
        # Truncate long content
        content_preview = content[:8000] if len(content) > 8000 else content
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-16k",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that analyzes content about German international cooperation and environmental sustainability projects. Extract concrete benefits to Germany and key insights."},
                {"role": "user", "content": f"Analyze this content titled '{title}' and extract: 1) Specific benefits to Germany from this cooperation or project, and 2) Key insights about international cooperation and environmental sustainability. Format your response as JSON with two keys: 'benefits_to_germany' and 'insights'. Each should be an array of strings with 1-5 items.\n\nContent: {content_preview}"}
            ],
            max_tokens=1000,
            temperature=0.5,
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        benefits = result.get("benefits_to_germany", [])
        insights = result.get("insights", [])
        
        # Format as text for database storage
        benefits_text = "\n\n".join(benefits) if benefits else None
        insights_text = "\n\n".join(insights) if insights else None
        
        return benefits_text, insights_text
    except Exception as e:
        logger.error(f"Error analyzing content with AI: {str(e)}")
        return None, None

# Enhanced process_data function with AI capabilities
def process_data(limit=1000, use_ai=True):
    """
    Process content data to extract benefits and insights.
    
    Args:
        limit: Maximum number of content items to process
        use_ai: Whether to use AI for analysis
    """
    logger.info(f"Starting data processing with limit: {limit}, AI: {use_ai}")
    print(f"Beginning data processing routine (limit: {limit} items, AI: {use_ai})")
    
    with st.spinner("Processing data..."):
        # Show initial progress
        progress_bar = st.progress(0)
        status_placeholder = st.empty()
        status_placeholder.info(f"Analyzing content for benefits and insights (up to {limit} items)...")
        
        try:
            # Get database connection
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Get content items that haven't been analyzed yet
            query = """
            SELECT cd.id, cd.link, cd.title, cd.full_content
            FROM content_data cd
            LEFT JOIN content_benefits cb ON cd.id = cb.content_id
            WHERE cb.content_id IS NULL
            LIMIT %s;
            """
            
            cursor.execute(query, (limit,))
            content_items = cursor.fetchall()
            
            processed_count = 0
            total_items = len(content_items)
            
            logger.info(f"Found {total_items} content items to analyze")
            
            if total_items == 0:
                status_placeholder.info("No new content to analyze.")
                progress_bar.progress(100)
                return 0
            
            # Update status
            status_placeholder.info(f"Analyzing {total_items} content items...")
            
            # Process items
            for idx, item in enumerate(content_items):
                content_id, link, title, content = item
                
                # Skip if content is missing
                if not content:
                    logger.warning(f"Skipping content ID {content_id}: No content available")
                    continue
                    
                logger.info(f"Analyzing content ID {content_id}: {title}")
                
                # Extract benefits and insights
                if use_ai and openai_client:
                    # Use AI for analysis
                    benefits_text, insights_text = analyze_content_with_ai(content_id, title, content)
                else:
                    # Fallback to keyword-based analysis
                    benefits_text, insights_text = analyze_content_for_benefits_keyword(content)
                
                # Store results
                if benefits_text or insights_text:
                    # Insert into benefits table
                    benefit_query = """
                    INSERT INTO benefits (links, benefits_to_germany, insights)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                    """
                    
                    links_array = "{" + link + "}"
                    cursor.execute(benefit_query, (links_array, benefits_text, insights_text))
                    benefit_id = cursor.fetchone()[0]
                    
                    # Create relationship in content_benefits table
                    relation_query = """
                    INSERT INTO content_benefits (content_id, benefit_id)
                    VALUES (%s, %s);
                    """
                    
                    cursor.execute(relation_query, (content_id, benefit_id))
                    logger.info(f"Created benefit entry for content ID {content_id}")
                else:
                    # Create empty entry to mark as processed
                    benefit_query = """
                    INSERT INTO benefits (links, benefits_to_germany, insights)
                    VALUES (%s, %s, %s)
                    RETURNING id;
                    """
                    
                    links_array = "{" + link + "}"
                    cursor.execute(benefit_query, (links_array, None, None))
                    benefit_id = cursor.fetchone()[0]
                    
                    # Create relationship in content_benefits table
                    relation_query = """
                    INSERT INTO content_benefits (content_id, benefit_id)
                    VALUES (%s, %s);
                    """
                    
                    cursor.execute(relation_query, (content_id, benefit_id))
                    logger.info(f"Created empty benefit entry for content ID {content_id}")
                
                processed_count += 1
                # Update progress
                progress = min(0.1 + (idx + 1) / total_items * 0.9, 1.0)
                progress_bar.progress(progress)
                if idx % 5 == 0 or idx == total_items - 1:
                    status_placeholder.info(f"Analyzed {idx + 1} of {total_items} items ({int(progress * 100)}%)...")
            
            # Commit the transaction
            conn.commit()
            cursor.close()
            conn.close()
            
            # Update final status
            progress_bar.progress(1.0)
            if processed_count > 0:
                status_placeholder.success(f"Successfully analyzed {processed_count} content items for benefits and insights.")
            else:
                status_placeholder.info("No content was analyzed.")
                
            logger.info(f"Content analysis completed with {processed_count} items processed")
            print(f"Content analysis completed with {processed_count} items processed")
            
            return processed_count
            
        except Exception as e:
            logger.error(f"Error in content analysis: {str(e)}")
            print(f"ERROR: Content analysis failed: {str(e)}")
            status_placeholder.error(f"Error in content analysis: {str(e)}")
            return 0

# Fallback keyword-based analysis function
def analyze_content_for_benefits_keyword(content):
    """Extract benefits and insights using keyword matching."""
    benefits_to_germany = []
    insights = []
    
    # Keywords for benefits
    benefit_keywords = [
        "benefit", "advantage", "gain", "profit", "value", 
        "impact", "sustainable development", "cooperation",
        "economic", "trade", "partnership", "collaboration",
        "technology transfer", "innovation", "expertise"
    ]
    
    # Keywords for insights
    insight_keywords = [
        "insight", "lesson", "finding", "discovery", "conclusion",
        "learn", "knowledge", "understand", "approach",
        "strategy", "method", "success", "challenge", "solution"
    ]
    
    content_paragraphs = content.split('\n\n')
    
    # Process paragraphs to extract benefits and insights
    for paragraph in content_paragraphs:
        paragraph = paragraph.strip()
        if not paragraph or len(paragraph) < 40:  # Skip short paragraphs
            continue
            
        # Check for benefits
        if any(keyword in paragraph.lower() for keyword in benefit_keywords):
            if len(paragraph) > 30:  # Avoid very short matches
                benefits_to_germany.append(paragraph)
                
        # Check for insights
        if any(keyword in paragraph.lower() for keyword in insight_keywords):
            if len(paragraph) > 30:  # Avoid very short matches
                insights.append(paragraph)
    
    # Format as text
    benefits_text = "\n\n".join(benefits_to_germany) if benefits_to_germany else None
    insights_text = "\n\n".join(insights) if insights else None
    
    return benefits_text, insights_text

# Function to run web extraction with AI enhancements
def run_web_extraction(max_queries=None, max_results_per_query=None, use_ai=True):
    logger.info(f"Starting web extraction process with max_queries={max_queries}, max_results_per_query={max_results_per_query}, AI={use_ai}")
    print(f"Starting web content extraction (max queries: {max_queries}, max results per query: {max_results_per_query}, AI: {use_ai})")

    with st.spinner("Extracting web content... This may take a few minutes."):
        # Create progress bar
        progress_bar = st.progress(0)

        # Initialize WebExtractor
        logger.info("Initializing WebExtractor")
        extractor = WebExtractor()
        
        # Configure AI capabilities
        extractor.use_ai = use_ai
        extractor.openai_client = openai_client
        extractor.generate_ai_summary = generate_ai_summary
        extractor.extract_date_with_ai = extract_date_with_ai

        # Set up a placeholder for status updates
        status_placeholder = st.empty()
        status_placeholder.info("Initializing web extraction...")

        # Run the extractor
        logger.info("Running web extractor")
        print("Searching for relevant web content...")
        status_placeholder.info("Searching the web for relevant content...")
        progress_bar.progress(25)

        try:
            results = extractor.run(max_queries=max_queries, max_results_per_query=max_results_per_query)
            progress_bar.progress(75)

            # Check if we got results
            if results["status"] == "success" and results["results"]:
                result_count = len(results["results"])
                logger.info(f"Web extraction successful. Found {result_count} results")
                print(f"Found {result_count} relevant web pages")
                status_placeholder.info(f"Found {result_count} relevant web pages. Saving to database...")

                # Process content items to enhance with AI and prepare for database
                if use_ai and openai_client and result_count > 0:
                    status_placeholder.info(f"Enhancing {result_count} items with AI summaries...")
                    
                    # Process items with AI in parallel batches for efficiency
                    def enhance_item(item):
                        if "content" in item and item["content"]:
                            # Generate AI summary if no snippet or short snippet
                            if not item.get("snippet") or len(item.get("snippet", "")) < 50:
                                item["summary"] = generate_ai_summary(item["content"])
                            else:
                                item["summary"] = item.get("snippet")
                                
                            # Refine date with AI if not already detected
                            if not item.get("date") and use_ai:
                                item["date"] = extract_date_with_ai(item["content"], item["link"])
                        return item
                    
                    # Process in batches using ThreadPoolExecutor for parallel execution
                    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                        results["results"] = list(executor.map(enhance_item, results["results"]))
                
                try:
                    # Save results using our database storage method
                    stored_ids = store_extract_data(results["results"])
                    stored_count = len(stored_ids)
                    results["database_stored_count"] = stored_count
                    results["database_stored_ids"] = stored_ids

                    progress_bar.progress(100)
                    logger.info(f"Web extraction completed. Saved {stored_count}/{result_count} items to database")
                    print(f"Web extraction completed! Saved {stored_count}/{result_count} items")
                    status_placeholder.success(f"Web extraction completed! Saved {stored_count} items to database.")

                    # Display the saved items
                    if stored_ids:
                        try:
                            engine = get_sqlalchemy_engine()
                            ids_str = ','.join(str(id) for id in stored_ids)
                            query = text(f"SELECT id, link, title, date, summary FROM content_data WHERE id IN ({ids_str})")
                            saved_df = pd.read_sql(query, engine)
                            
                            if not saved_df.empty:
                                st.subheader("Newly Extracted Content")
                                st.dataframe(saved_df)
                        except Exception as e:
                            logger.error(f"Error displaying saved items: {str(e)}")
                            print(f"ERROR: Could not display saved items: {str(e)}")

                except Exception as e:
                    logger.error(f"Error handling extraction results: {str(e)}")
                    print(f"ERROR: Failed to handle extraction results: {str(e)}")
                    status_placeholder.error(f"Error handling extraction results: {str(e)}")

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

# Function to add new content manually
def add_new_content(link, title, date, summary, theme, organization, content):
    logger.info(f"Adding new content manually: {link}")
    
    try:
        engine = get_sqlalchemy_engine()
        conn = engine.connect()
        
        query = text("""
        INSERT INTO content_data 
        (link, title, date, summary, full_content, theme, organization) 
        VALUES (:link, :title, :date, :summary, :content, :theme, :organization)
        RETURNING id
        """)
        
        params = {
            "link": link,
            "title": title,
            "date": date,
            "summary": summary,
            "content": content,
            "theme": theme,
            "organization": organization
        }
        
        result = conn.execute(query, params)
        
        # Get the ID of the inserted record
        record_id = result.fetchone()[0]
        conn.commit()
        conn.close()
        
        logger.info(f"Successfully added content with ID: {record_id}")
        return record_id
        
    except Exception as e:
        logger.error(f"Error adding content: {str(e)}")
        raise

# Sidebar for application navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "View Data", "Process Data", "Web Extraction", "Add Content"])
logger.info(f"User selected app mode: {app_mode}")
print(f"User navigated to: {app_mode}")

# AI Enhancement options in sidebar
use_ai = False
if openai_client:
    ai_section = st.sidebar.expander("AI Enhancement Options", expanded=False)
    with ai_section:
        use_ai = st.checkbox("Use AI for enhanced processing", value=True)
        st.info("Using AI will improve summaries, date extraction, and content analysis")
else:
    st.sidebar.warning("OpenAI API key not configured. AI enhancements disabled.")

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
            process_data(use_ai=use_ai)
    with col2:
        if st.button("Run Web Extraction", key="home_extract", use_container_width=True):
            logger.info("User clicked Run Web Extraction button from Home page")
            print("Web Extraction button clicked from Home page")
            run_web_extraction(max_queries=25, max_results_per_query=10, use_ai=use_ai)

    # Display some statistics
    try:
        engine = get_sqlalchemy_engine()
        stats_query = text("""
        SELECT 
            (SELECT COUNT(*) FROM content_data) as content_count,
            (SELECT COUNT(*) FROM benefits) as benefits_count,
            (SELECT COUNT(*) FROM content_benefits) as processed_count
        """)
        stats_df = pd.read_sql(stats_query, engine)
        
        if not stats_df.empty:
            st.subheader("Database Statistics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Content Items", stats_df['content_count'].iloc[0])
            with col2:
                st.metric("Benefits Entries", stats_df['benefits_count'].iloc[0])
            with col3:
                st.metric("Processed Items", stats_df['processed_count'].iloc[0])
                
            # Add detailed statistics
            if stats_df['content_count'].iloc[0] > 0:
                detailed_stats_query = text("""
                SELECT 
                    CASE 
                        WHEN theme IS NULL OR theme = '' THEN 'Unclassified' 
                        ELSE theme 
                    END as theme_group,
                    COUNT(*) as count
                FROM content_data
                GROUP BY theme_group
                ORDER BY count DESC
                LIMIT 10
                """)
                theme_stats = pd.read_sql(detailed_stats_query, engine)
                
                st.subheader("Content by Theme")
                st.bar_chart(theme_stats.set_index('theme_group'))
    except Exception as e:
        logger.error(f"Error fetching statistics: {str(e)}")
        print(f"ERROR: Could not fetch statistics: {str(e)}")

# View Data page
elif app_mode == "View Data":
    logger.info("Rendering View Data page")
    st.subheader("Content Data")
    
    # Add data display options
    display_limit = st.slider("Number of records to display", 10, 1000, 100)
    
    try:
        data = fetch_data(limit=display_limit)
        st.dataframe(data)
        
        # Add filtering options
        if not data.empty:
            logger.info(f"Loaded data with {len(data)} rows for viewing")
            st.subheader("Filter Data")
            col1, col2 = st.columns(2)
            with col1:
                # Create a "All" option plus existing themes
                theme_options = ["All"]
                if not data["theme"].isna().all():  # Only add if there are non-null themes
                    theme_options.extend([t for t in data["theme"].dropna().unique().tolist() if t])
                selected_theme = st.selectbox("Filter by Theme", theme_options)
                logger.debug(f"User selected theme filter: {selected_theme}")
            with col2:
                # Create a "All" option plus existing organizations
                org_options = ["All"]
                if not data["organization"].isna().all():  # Only add if there are non-null organizations
                    org_options.extend([o for o in data["organization"].dropna().unique().tolist() if o])
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
            
            # Add option to view full content
            if not filtered_data.empty:
                selected_id = st.selectbox("Select content to view details", filtered_data["id"].tolist())
                if selected_id:
                    try:
                        engine = get_sqlalchemy_engine()
                        details_query = text(f"""
                        SELECT id, link, title, date, summary, full_content, theme, organization, created_at, updated_at
                        FROM content_data
                        WHERE id = {selected_id}
                        """)
                        details_df = pd.read_sql(details_query, engine)
                        
                        if not details_df.empty:
                            st.subheader(f"Content Details: {details_df['title'].iloc[0]}")
                            st.write(f"**Link:** {details_df['link'].iloc[0]}")
                            st.write(f"**Date:** {details_df['date'].iloc[0] if not pd.isna(details_df['date'].iloc[0]) else 'Not available'}")
                            st.write(f"**Theme:** {details_df['theme'].iloc[0] if not pd.isna(details_df['theme'].iloc[0]) else 'Not assigned'}")
                            st.write(f"**Organization:** {details_df['organization'].iloc[0] if not pd.isna(details_df['organization'].iloc[0]) else 'Not assigned'}")
                            
                            st.subheader("Summary")
                            st.write(details_df['summary'].iloc[0] if not pd.isna(details_df['summary'].iloc[0]) else "No summary available")
                            
                            st.subheader("Full Content")
                            st.text_area("", details_df['full_content'].iloc[0] if not pd.isna(details_df['full_content'].iloc[0]) else "No content available", height=300)
                            
                            # Check if this content has been analyzed
                            benefits_query = text(f"""
                            SELECT b.id, b.benefits_to_germany, b.insights
                            FROM benefits b
                            JOIN content_benefits cb ON b.id = cb.benefit_id
                            WHERE cb.content_id = {selected_id}
                            """)
                            
                            benefits_df = pd.read_sql(benefits_query, engine)
                            if not benefits_df.empty:
                                st.subheader("Benefits Analysis")
                                if not pd.isna(benefits_df['benefits_to_germany'].iloc[0]):
                                    st.markdown("**Benefits to Germany:**")
                                    st.write(benefits_df['benefits_to_germany'].iloc[0])
                                
                                if not pd.isna(benefits_df['insights'].iloc[0]):
                                    st.markdown("**Insights:**")
                                    st.write(benefits_df['insights'].iloc[0])
                                    
                                # Add option to regenerate analysis with AI
                                if openai_client and use_ai:
                                    if st.button("Regenerate Analysis with AI", key="regenerate_analysis"):
                                        with st.spinner("Regenerating analysis..."):
                                            content = details_df['full_content'].iloc[0]
                                            title = details_df['title'].iloc[0]
                                            link = details_df['link'].iloc[0]
                                            
                                            # Generate new analysis
                                            benefits_text, insights_text = analyze_content_with_ai(selected_id, title, content)
                                            
                                            # Update database
                                            conn = get_db_connection()
                                            cursor = conn.cursor()
                                            
                                            # Get existing benefit_id
                                            query = """
                                            SELECT benefit_id 
                                            FROM content_benefits 
                                            WHERE content_id = %s
                                            """
                                            cursor.execute(query, (selected_id,))
                                            result = cursor.fetchone()
                                            
                                            if result:
                                                benefit_id = result[0]
                                                
                                                # Update the benefits entry
                                                update_query = """
                                                UPDATE benefits
                                                SET benefits_to_germany = %s, insights = %s, updated_at = CURRENT_TIMESTAMP
                                                WHERE id = %s
                                                """
                                                cursor.execute(update_query, (benefits_text, insights_text, benefit_id))
                                                conn.commit()
                                                
                                                st.success("Analysis regenerated successfully!")
                                                st.markdown("**New Benefits to Germany:**")
                                                st.write(benefits_text if benefits_text else "None identified")
                                                st.markdown("**New Insights:**")
                                                st.write(insights_text if insights_text else "None identified")
                                            
                                            cursor.close()
                                            conn.close()
                            else:
                                st.info("This content has not been analyzed for benefits yet.")
                                
                                # Add option to analyze with AI
                                if openai_client and use_ai:
                                    if st.button("Analyze with AI", key="analyze_with_ai"):
                                        with st.spinner("Analyzing content..."):
                                            content = details_df['full_content'].iloc[0]
                                            title = details_df['title'].iloc[0]
                                            link = details_df['link'].iloc[0]
                                            
                                            # Generate analysis
                                            benefits_text, insights_text = analyze_content_with_ai(selected_id, title, content)
                                            
                                            # Store in database
                                            conn = get_db_connection()
                                            cursor = conn.cursor()
                                            
                                            # Insert into benefits table
                                            benefit_query = """
                                            INSERT INTO benefits (links, benefits_to_germany, insights)
                                            VALUES (%s, %s, %s)
                                            RETURNING id;
                                            """
                                            
                                            links_array = "{" + link + "}"
                                            cursor.execute(benefit_query, (links_array, benefits_text, insights_text))
                                            benefit_id = cursor.fetchone()[0]
                                            
                                            # Create relationship in content_benefits table
                                            relation_query = """
                                            INSERT INTO content_benefits (content_id, benefit_id)
                                            VALUES (%s, %s);
                                            """
                                            
                                            cursor.execute(relation_query, (selected_id, benefit_id))
                                            conn.commit()
                                            cursor.close()
                                            conn.close()
                                            
                                            st.success("Content analyzed successfully!")
                                            st.markdown("**Benefits to Germany:**")
                                            st.write(benefits_text if benefits_text else "None identified")
                                            st.markdown("**Insights:**")
                                            st.write(insights_text if insights_text else "None identified")
                    except Exception as e:
                        logger.error(f"Error fetching content details: {str(e)}")
                        st.error(f"Error loading content details: {str(e)}")
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
    
    # Add processing options
    processing_limit = st.slider("Number of items to process", 10, 1000, 100)
    
    # AI processing checkbox
    if openai_client:
        use_ai_process = st.checkbox("Use AI for enhanced analysis", value=use_ai)
        st.info("AI-powered analysis will extract more accurate benefits and insights")
    else:
        use_ai_process = False
        st.warning("OpenAI API not configured. Using keyword-based analysis only.")
    
    # Add a button to trigger the processing
    if st.button("Process Data", key="process_page_button", use_container_width=True):
        logger.info(f"User clicked Process Data button with limit: {processing_limit}, AI: {use_ai_process}")
        print(f"Process Data button clicked with limit: {processing_limit}, AI: {use_ai_process}")
        process_data(limit=processing_limit, use_ai=use_ai_process)
    
    # Show processed data
    st.subheader("Processed Benefits Data")
    try:
        engine = get_sqlalchemy_engine()
        
        # Get count of unprocessed content
        unprocessed_query = text("""
        SELECT COUNT(*) as unprocessed_count
        FROM content_data cd
        LEFT JOIN content_benefits cb ON cd.id = cb.content_id
        WHERE cb.content_id IS NULL
        """)
        
        unprocessed_df = pd.read_sql(unprocessed_query, engine)
        unprocessed_count = unprocessed_df['unprocessed_count'].iloc[0]
        
        st.info(f"There are {unprocessed_count} unprocessed content items in the database.")
        
        # Get already processed data
        benefits_query = text("""
        SELECT b.id, b.links, b.benefits_to_germany, b.insights, b.created_at
        FROM benefits b
        ORDER BY b.created_at DESC
        LIMIT 50
        """)
        
        benefits_df = pd.read_sql(benefits_query, engine)
        
        if not benefits_df.empty:
            st.dataframe(benefits_df)
            logger.info(f"Displayed {len(benefits_df)} processed benefits records")
            
            # Add download option
            csv = benefits_df.to_csv(index=False)
            st.download_button(
                label="Download Benefits Data as CSV",
                data=csv,
                file_name="benefits_analysis.csv",
                mime="text/csv",
            )
        else:
            st.info("No processed benefits data found.")
    except Exception as e:
        logger.error(f"Error displaying processed data: {str(e)}")
        st.error(f"Error loading processed data: {str(e)}")

# Web Extraction page
elif app_mode == "Web Extraction":
    logger.info("Rendering Web Extraction page")
    st.subheader("Web Content Extraction")
    st.write("Configure the extraction settings and click the button below to extract web content about Germany's international cooperation efforts in environmental sustainability.")
    
    # Add options for extraction
    st.subheader("Extraction Options")
    col1, col2 = st.columns(2)
    with col1:
        max_queries = st.slider("Number of search queries to process", 5, 100, 25)
        max_results_per_query = st.slider("Results per query", 3, 20, 10)
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
    
    # Show extraction history
    st.subheader("Previous Extractions")
    try:
        engine = get_sqlalchemy_engine()
        # Query to get extraction statistics by date
        stats_query = text("""
        SELECT 
            DATE(created_at) as extraction_date, 
            COUNT(*) as records_count,
            MIN(id) as min_id,
            MAX(id) as max_id
        FROM 
            content_data 
        GROUP BY 
            DATE(created_at)
        ORDER BY 
            extraction_date DESC
        LIMIT 10;
        """)
        
        stats_df = pd.read_sql(stats_query, engine)
        
        if not stats_df.empty:
            st.dataframe(stats_df)
            
            # Allow viewing records from a specific extraction date
            selected_date = st.selectbox(
                "View records from date:", 
                options=stats_df['extraction_date'].tolist()
            )
            
            if selected_date:
                date_records_query = text(f"""
                SELECT id, title, link, date, summary, theme, organization
                FROM content_data
                WHERE DATE(created_at) = '{selected_date}'
                LIMIT 100;
                """)
                
                date_records_df = pd.read_sql(date_records_query, engine)
                st.subheader(f"Records from {selected_date}")
                st.dataframe(date_records_df)
        else:
            st.info("No previous extractions found.")
            
    except Exception as e:
        logger.error(f"Error fetching extraction history: {str(e)}")
        print(f"ERROR: Could not fetch extraction history: {str(e)}")
        st.error(f"Error fetching extraction history: {str(e)}")
    
    # Add a button to process all content data
    st.subheader("Process Extracted Content")
    if st.button("Process All Content Data", key="process_all_content", use_container_width=True):
        logger.info("User clicked Process All Content Data button")
        print("Starting content processing...")
        process_data(limit=1000, use_ai=use_ai_extraction)

# Add Content page
elif app_mode == "Add Content":
    logger.info("Rendering Add Content page")
    st.subheader("Add New Content")
    st.write("Use this form to manually add new content to the database.")
    
    with st.form("add_content_form"):
        title = st.text_input("Title")
        link = st.text_input("Link (URL)")
        date = st.date_input("Date", value=None)
        theme = st.text_input("Theme")
        organization = st.text_input("Organization")
        summary = st.text_area("Summary")
        content = st.text_area("Full Content")
        
        # AI summary generation
        generate_summary = False
        if openai_client and use_ai:
            generate_summary = st.checkbox("Generate summary with AI")
        
        # Submit button
        submitted = st.form_submit_button("Add Content")
        if submitted:
            logger.info("User submitted Add Content form")
            print(f"Adding new content: {title}")
            
            if not title or not link:
                st.error("Title and Link are required fields.")
            else:
                try:
                    # Generate AI summary if requested
                    if generate_summary and content and openai_client:
                        with st.spinner("Generating AI summary..."):
                            generated_summary = generate_ai_summary(content)
                            if generated_summary:
                                summary = generated_summary
                    
                    # Convert date to string in ISO format if provided
                    date_str = date.isoformat() if date else None
                    
                    # Add content to database
                    record_id = add_new_content(
                        link=link,
                        title=title,
                        date=date_str,
                        summary=summary,
                        theme=theme,
                        organization=organization,
                        content=content
                    )
                    
                    st.success(f"Content added successfully with ID: {record_id}")
                    logger.info(f"Content added successfully with ID: {record_id}")
                    
                    # Show the added content
                    st.json({
                        "id": record_id,
                        "title": title,
                        "link": link,
                        "date": date_str,
                        "summary": summary,
                        "theme": theme,
                        "organization": organization
                    })
                    
                    # Process with AI immediately if available
                    if openai_client and use_ai and content:
                        if st.button("Analyze Content with AI"):
                            with st.spinner("Analyzing content..."):
                                benefits_text, insights_text = analyze_content_with_ai(record_id, title, content)
                                
                                # Store in database
                                conn = get_db_connection()
                                cursor = conn.cursor()
                                
                                # Insert into benefits table
                                benefit_query = """
                                INSERT INTO benefits (links, benefits_to_germany, insights)
                                VALUES (%s, %s, %s)
                                RETURNING id;
                                """
                                
                                links_array = "{" + link + "}"
                                cursor.execute(benefit_query, (links_array, benefits_text, insights_text))
                                benefit_id = cursor.fetchone()[0]
                                
                                # Create relationship in content_benefits table
                                relation_query = """
                                INSERT INTO content_benefits (content_id, benefit_id)
                                VALUES (%s, %s);
                                """
                                
                                cursor.execute(relation_query, (record_id, benefit_id))
                                conn.commit()
                                cursor.close()
                                conn.close()
                                
                                st.success("Content analyzed successfully!")
                                st.subheader("Analysis Results")
                                st.markdown("**Benefits to Germany:**")
                                st.write(benefits_text if benefits_text else "None identified")
                                st.markdown("**Insights:**")
                                st.write(insights_text if insights_text else "None identified")
                    
                except Exception as e:
                    logger.error(f"Error adding content: {str(e)}")
                    st.error(f"Error adding content: {str(e)}")
    
    # Add a separator
    st.markdown("---")
    
    # Bulk import section
    st.subheader("Bulk Import Content")
    st.write("Upload a CSV file to import multiple content items at once.")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read CSV file
            df = pd.read_csv(uploaded_file)
            st.write("CSV file preview:")
            st.dataframe(df.head())
            
            # Check for required columns
            required_columns = ["title", "link"]
            if not all(col in df.columns for col in required_columns):
                st.error(f"CSV must contain the following columns: {', '.join(required_columns)}")
            else:
                # AI enhancement for bulk import
                use_ai_for_bulk = False
                if openai_client and use_ai:
                    use_ai_for_bulk = st.checkbox("Use AI to enhance imported content")
                    if use_ai_for_bulk:
                        st.info("AI will generate summaries for entries and extract dates if missing")
                
                # Show import button
                if st.button("Import Data", use_container_width=True):
                    with st.spinner("Importing data..."):
                        # Connect to database
                        engine = get_sqlalchemy_engine()
                        conn = engine.connect()
                        
                        # Track import progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        # Import each row
                        success_count = 0
                        error_count = 0
                        
                        for index, row in df.iterrows():
                            try:
                                # Prepare data
                                title = row.get("title", "")
                                link = row.get("link", "")
                                date = row.get("date") if "date" in df.columns else None
                                summary = row.get("summary", "") if "summary" in df.columns else ""
                                theme = row.get("theme", "") if "theme" in df.columns else ""
                                organization = row.get("organization", "") if "organization" in df.columns else ""
                                content = row.get("content", "") if "content" in df.columns else ""
                                
                                # Skip if title or link is missing
                                if not title or not link:
                                    error_count += 1
                                    continue
                                
                                # Use AI to enhance content if requested
                                if use_ai_for_bulk and openai_client and content:
                                    # Generate summary if missing
                                    if not summary or len(summary) < 50:
                                        summary = generate_ai_summary(content) or summary
                                    
                                    # Extract date if missing
                                    if not date and content:
                                        date = extract_date_with_ai(content, link)
                                
                                # Insert data
                                query = text("""
                                INSERT INTO content_data 
                                (link, title, date, summary, full_content, theme, organization) 
                                VALUES (:link, :title, :date, :summary, :content, :theme, :organization)
                                """)
                                
                                params = {
                                    "link": link,
                                    "title": title,
                                    "date": date,
                                    "summary": summary,
                                    "content": content,
                                    "theme": theme,
                                    "organization": organization
                                }
                                
                                conn.execute(query, params)
                                success_count += 1
                                
                                # Update progress
                                progress = (index + 1) / len(df)
                                progress_bar.progress(progress)
                                status_text.text(f"Imported {index + 1} of {len(df)} rows...")
                                
                            except Exception as e:
                                logger.error(f"Error importing row {index + 1}: {str(e)}")
                                error_count += 1
                        
                        # Commit transaction
                        conn.commit()
                        conn.close()
                        
                        # Show final status
                        progress_bar.progress(1.0)
                        status_text.text(f"Import completed: {success_count} successful, {error_count} failed")
                        
                        st.success(f"Successfully imported {success_count} content items")
                        
                        # Show the latest data after import
                        st.subheader("Latest Content Data")
                        st.dataframe(fetch_data())
        except Exception as e:
            logger.error(f"Error processing CSV file: {str(e)}")
            st.error(f"Error processing CSV file: {str(e)}")

# Footer
st.markdown("---")
st.markdown("© 2025 Data Processing Application")
logger.info("Page rendering completed")

# Run the main app if this file is executed directly
if __name__ == "__main__":
    logger.info("Application main method executed")
    print("Application started via main method")