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
from web_extractor import WebExtractor
from content_db import store_extract_data, fetch_data, get_all_content, create_schema

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
    page_title="Initiative Analysis Dashboard",
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
                                query = text(f"SELECT id, link, title, date, summary, themes, organization, sentiment, initiative FROM content_data WHERE id IN ({ids_str})")
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
    
    # Add initiatives to filters
    initiatives_query = text("SELECT DISTINCT initiative FROM content_data WHERE initiative IS NOT NULL ORDER BY initiative")
    initiatives = [i[0] for i in pd.read_sql(initiatives_query, engine)['initiative']]
    initiatives.insert(0, "All")  # Add "All" option at the beginning
    
    # Filter dropdowns
    filter_theme = st.sidebar.selectbox("Theme", themes)
    filter_org = st.sidebar.selectbox("Organization", organizations)
    filter_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    filter_initiative = st.sidebar.selectbox("Initiative", initiatives)
    
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
    if filter_initiative != "All":
        filters['initiative'] = filter_initiative
    
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
        st.markdown(f"**Initiative:** {selected_row['initiative'] if pd.notna(selected_row['initiative']) else 'N/A'}")
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
            st.subheader("Benefits")
            st.write(selected_row['benefits_to_germany'])
            
        # Benefit Examples (if available)
        if 'benefit_examples' in selected_row and pd.notna(selected_row['benefit_examples']):
            st.subheader("Benefit Examples")
            if isinstance(selected_row['benefit_examples'], list):
                for example in selected_row['benefit_examples']:
                    st.markdown(f"**Category:** {example.get('category', '').replace('_', ' ').title()}")
                    st.markdown(f"**Text:** {example.get('text', '')}")
                    st.markdown("---")
            elif isinstance(selected_row['benefit_examples'], str):
                # Try to parse JSON string
                try:
                    examples = json.loads(selected_row['benefit_examples'])
                    for example in examples:
                        st.markdown(f"**Category:** {example.get('category', '').replace('_', ' ').title()}")
                        st.markdown(f"**Text:** {example.get('text', '')}")
                        st.markdown("---")
                except:
                    st.write(selected_row['benefit_examples'])

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
        # Count initiatives
        if 'initiative' in df.columns:
            initiative_count = df['initiative'].nunique()
            st.metric("Initiatives", initiative_count)
        else:
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
        tab1, tab2, tab3, tab4 = st.tabs(["Organizations", "Themes", "Initiatives", "Timeline"])
        
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
            if 'initiative' in df.columns:
                # Initiative distribution
                initiative_counts = df['initiative'].value_counts().reset_index()
                initiative_counts.columns = ['Initiative', 'Count']
                
                fig = px.bar(
                    initiative_counts, 
                    x='Initiative', 
                    y='Count',
                    title='Content by Initiative',
                    color='Count',
                    color_continuous_scale='oranges'
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Initiative data not available in the dataset")
        
        with tab4:
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
                if 'initiative' in row and pd.notna(row['initiative']):
                    st.write(f"**Initiative:** {row['initiative']}")
                st.write(f"**Organization:** {row['organization'] if pd.notna(row['organization']) else 'N/A'}")
                st.write(f"**Summary:** {row['summary'] if pd.notna(row['summary']) else 'No summary available'}")
                st.write(f"**Link:** {row['link']}")
    else:
        st.info("No content available. Extract content first.")

def initiative_dashboard():
    """Display initiative-specific dashboard with metrics and visualizations."""
    st.title("Initiative Analysis Dashboard")
    
    # Add initiative filter to sidebar
    st.sidebar.header("Initiative Selection")
    
    # Get list of initiatives
    engine = get_sqlalchemy_engine()
    initiatives_query = text("SELECT DISTINCT initiative FROM content_data WHERE initiative IS NOT NULL ORDER BY initiative")
    initiatives = ["All Initiatives"] + [i[0] for i in pd.read_sql(initiatives_query, engine)['initiative']]
    
    # Add initiative filter dropdown
    selected_initiative = st.sidebar.selectbox("Select Initiative", initiatives)
    
    # Fetch data filtered by initiative if selected
    filters = {}
    if selected_initiative != "All Initiatives":
        filters['initiative'] = selected_initiative
    
    df = fetch_data(limit=1000, filters=filters)
    
    # Initiative metrics header
    st.header("Initiative Metrics")
    
    # Create metrics for the selected initiative or all initiatives
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(df))
    
    with col2:
        # Count documents with benefits
        benefits_count = df['benefits_to_germany'].notna().sum() 
        st.metric("Documents with Benefits", benefits_count)
    
    with col3:
        # Count unique organizations
        org_count = df['organization'].nunique()
        st.metric("Organizations", org_count)
    
    with col4:
        # Count unique themes
        theme_count = 0
        if 'themes' in df.columns:
            all_themes = set()
            for themes_list in df['themes'].dropna():
                if themes_list:
                    all_themes.update(themes_list)
            theme_count = len(all_themes)
        st.metric("Themes", theme_count)
    
    # Initiative overview section
    st.subheader(f"{'Overview of All Initiatives' if selected_initiative == 'All Initiatives' else f'Overview of {selected_initiative}'}")
    
    # If showing all initiatives, display a comparison
    if selected_initiative == "All Initiatives" and 'initiative' in df.columns:
        initiative_counts = df['initiative'].value_counts().reset_index()
        initiative_counts.columns = ['Initiative', 'Document Count']
        
        # Create bar chart
        fig = px.bar(
            initiative_counts,
            x='Initiative',
            y='Document Count',
            title='Documents by Initiative',
            color='Document Count',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Benefit categories analysis
    st.subheader("Benefit Categories Analysis")
    
    # Extract benefit categories from JSON
    benefit_categories = []
    for _, row in df.iterrows():
        if isinstance(row.get('benefit_categories'), dict):
            for category, score in row['benefit_categories'].items():
                if score > 0.2:  # Only include significant categories
                    benefit_categories.append({
                        'category': category.replace('_', ' ').title(),
                        'score': score,
                        'initiative': row.get('initiative', 'Unknown')
                    })
        elif isinstance(row.get('benefit_categories'), str):
            try:
                categories_dict = json.loads(row['benefit_categories'])
                for category, score in categories_dict.items():
                    if score > 0.2:  # Only include significant categories
                        benefit_categories.append({
                            'category': category.replace('_', ' ').title(),
                            'score': score,
                            'initiative': row.get('initiative', 'Unknown')
                        })
            except:
                pass
    
    if benefit_categories:
        # Create DataFrame from benefit categories
        benefit_df = pd.DataFrame(benefit_categories)
        
        # Aggregate by category
        agg_benefits = benefit_df.groupby('category')['score'].mean().reset_index()
        agg_benefits = agg_benefits.sort_values('score', ascending=False)
        
        # Create horizontal bar chart
        fig = px.bar(
            agg_benefits,
            x='score',
            y='category',
            title='Average Benefit Category Scores',
            orientation='h',
            color='score',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No benefit category data available for visualization.")
    
    # Initiative timeline
    st.subheader("Initiative Timeline")
    
    if 'date' in df.columns and not df['date'].isna().all():
        # Convert to datetime if not already
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        
        # Group by month and count
        timeline = df.groupby(df['date'].dt.strftime('%Y-%m'))['id'].count().reset_index()
        timeline.columns = ['Month', 'Document Count']
        
        fig = px.line(
            timeline, 
            x='Month', 
            y='Document Count',
            title=f"{'All Initiatives' if selected_initiative == 'All Initiatives' else selected_initiative} - Timeline",
            markers=True
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Insufficient date data for timeline visualization.")
    
    # Display benefit examples
    st.subheader("Benefit Examples")
    
    # Create tabs for benefit categories
    benefit_tabs = st.tabs([
        "Environmental Benefits", 
        "Economic Benefits", 
        "Social Benefits", 
        "Strategic Benefits"
    ])
    
    # Map tab indices to category keys
    category_keys = [
        "environmental_benefits",
        "economic_benefits", 
        "social_benefits",
        "strategic_benefits"
    ]
    
    # Function to extract examples for a category
    def get_examples_for_category(category):
        examples = []
        for _, row in df.iterrows():
            # Handle examples in different formats
            if isinstance(row.get('benefit_examples'), list):
                for example in row['benefit_examples']:
                    if example.get('category') == category:
                        examples.append({
                            'text': example['text'],
                            'initiative': row.get('initiative', 'Unknown'),
                            'link': row['link'],
                            'title': row['title'],
                            'word_count': example.get('word_count', 0)
                        })
            elif isinstance(row.get('benefit_examples'), str):
                try:
                    examples_list = json.loads(row['benefit_examples'])
                    for example in examples_list:
                        if example.get('category') == category:
                            examples.append({
                                'text': example['text'],
                                'initiative': row.get('initiative', 'Unknown'),
                                'link': row['link'],
                                'title': row['title'],
                                'word_count': example.get('word_count', 0)
                            })
                except:
                    pass
        return examples
    
    # Populate each tab with examples
    for i, tab in enumerate(benefit_tabs):
        with tab:
            category = category_keys[i]
            examples = get_examples_for_category(category)
            
            if examples:
                for example in examples:
                    with st.expander(f"{example['title'][:80]}..."):
                        st.write(example['text'])
                        st.caption(f"Source: {example['link']}")
            else:
                st.info(f"No {category.replace('_', ' ').title()} examples found.")

def initiative_comparison():
    """Compare metrics between initiatives."""
    st.title("Initiative Comparison")
    
    # Fetch all data
    df = fetch_data(limit=1000)
    
    # Get list of initiatives
    if 'initiative' in df.columns:
        initiatives = df['initiative'].dropna().unique().tolist()
    else:
        st.warning("No initiative data available for comparison.")
        return
    
    if len(initiatives) < 2:
        st.warning("Not enough initiatives to compare. Please extract more data with initiative information.")
        return
    
    # Create comparison metrics
    st.subheader("Key Metrics Comparison")
    
    # Create DataFrame for comparison
    comparison_data = []
    
    for initiative in initiatives:
        initiative_df = df[df['initiative'] == initiative]
        
        # Calculate metrics
        doc_count = len(initiative_df)
        benefit_count = initiative_df['benefits_to_germany'].notna().sum()
        org_count = initiative_df['organization'].nunique()
        
        # Get average sentiment
        sentiment_map = {'Positive': 1, 'Neutral': 0, 'Negative': -1}
        sentiment_values = initiative_df['sentiment'].map(sentiment_map)
        avg_sentiment = sentiment_values.mean() if not sentiment_values.empty else 0
        
        # Add to comparison data
        comparison_data.append({
            'Initiative': initiative,
            'Documents': doc_count,
            'Benefit Examples': benefit_count,
            'Organizations': org_count,
            'Sentiment Score': avg_sentiment
        })
    
    # Create comparison DataFrame
    comp_df = pd.DataFrame(comparison_data)
    
    # Display as a styled table
    st.dataframe(comp_df.style.highlight_max(axis=0), use_container_width=True)
    
    # Create a radar chart for comparison
    st.subheader("Initiative Comparison - Radar Chart")
    
    # Normalize metrics for radar chart
    radar_df = comp_df.copy()
    for col in ['Documents', 'Benefit Examples', 'Organizations', 'Sentiment Score']:
        max_val = radar_df[col].max()
        if max_val > 0:  # Avoid division by zero
            radar_df[col] = radar_df[col] / max_val
    
    # Create radar chart
    categories = ['Documents', 'Benefit Examples', 'Organizations', 'Sentiment Score']
    
    fig = go.Figure()
    
    for _, row in radar_df.iterrows():
        fig.add_trace(go.Scatterpolar(
            r=[row[cat] for cat in categories],
            theta=categories,
            fill='toself',
            name=row['Initiative']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Compare themes between initiatives
    st.subheader("Theme Distribution by Initiative")
    
    # Extract theme data by initiative
    theme_by_initiative = {}
    
    for initiative in initiatives:
        initiative_df = df[df['initiative'] == initiative]
        theme_counts = {}
        
        for _, row in initiative_df.iterrows():
            if row.get('themes') and isinstance(row['themes'], list):
                for theme in row['themes']:
                    theme_counts[theme] = theme_counts.get(theme, 0) + 1
        
        theme_by_initiative[initiative] = theme_counts
    
    # Create a combined dataframe for visualization
    theme_comparison_data = []
    
    for initiative, themes in theme_by_initiative.items():
        for theme, count in themes.items():
            theme_comparison_data.append({
                'Initiative': initiative,
                'Theme': theme,
                'Count': count
            })
    
    if theme_comparison_data:
        theme_comp_df = pd.DataFrame(theme_comparison_data)
        
        # Create grouped bar chart
        fig = px.bar(
            theme_comp_df,
            x='Theme',
            y='Count',
            color='Initiative',
            title='Theme Distribution by Initiative',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No theme data available for comparison.")
    
    # Compare benefit categories between initiatives
    st.subheader("Benefit Categories by Initiative")
    
    # Extract benefit categories from JSON by initiative
    benefit_categories = []
    
    for _, row in df.iterrows():
        if 'initiative' in row and pd.notna(row['initiative']):
            if isinstance(row.get('benefit_categories'), dict):
                for category, score in row['benefit_categories'].items():
                    if score > 0.2:  # Only include significant categories
                        benefit_categories.append({
                            'category': category.replace('_', ' ').title(),
                            'score': score,
                            'initiative': row['initiative']
                        })
            elif isinstance(row.get('benefit_categories'), str):
                try:
                    categories_dict = json.loads(row['benefit_categories'])
                    for category, score in categories_dict.items():
                        if score > 0.2:  # Only include significant categories
                            benefit_categories.append({
                                'category': category.replace('_', ' ').title(),
                                'score': score,
                                'initiative': row['initiative']
                            })
                except:
                    pass
    
    if benefit_categories:
        # Create DataFrame from benefit categories
        benefit_df = pd.DataFrame(benefit_categories)
        
        # Aggregate by category and initiative
        agg_benefits = benefit_df.groupby(['initiative', 'category'])['score'].mean().reset_index()
        
        # Create grouped bar chart
        fig = px.bar(
            agg_benefits,
            x='category',
            y='score',
            color='initiative',
            title='Benefit Categories by Initiative',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No benefit category data available for comparison.")

def initialization_page():
    """Display database initialization and schema setup page."""
    st.title("Database Initialization")
    
    st.write("""
    This page allows you to initialize or update the database schema for storing initiative analysis data.
    Use this if you're setting up a new database or need to update an existing one.
    """)
    
    # Create a code block with the SQL schema
    sql_schema = """
    -- Add new columns to the content_data table if they don't exist
    DO $
    BEGIN
        -- Check if initiative column exists
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name='content_data' AND column_name='initiative') THEN
            ALTER TABLE content_data ADD COLUMN initiative VARCHAR(100);
        END IF;

        -- Check if initiative_key column exists
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name='content_data' AND column_name='initiative_key') THEN
            ALTER TABLE content_data ADD COLUMN initiative_key VARCHAR(50);
        END IF;

        -- Check if benefit_categories column exists
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name='content_data' AND column_name='benefit_categories') THEN
            ALTER TABLE content_data ADD COLUMN benefit_categories JSONB;
        END IF;

        -- Check if benefit_examples column exists
        IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                      WHERE table_name='content_data' AND column_name='benefit_examples') THEN
            ALTER TABLE content_data ADD COLUMN benefit_examples JSONB;
        END IF;
    END
    $;

    -- Create indexes for new columns
    CREATE INDEX IF NOT EXISTS idx_content_data_initiative ON content_data (initiative);
    CREATE INDEX IF NOT EXISTS idx_content_data_initiative_key ON content_data (initiative_key);
    """
    
    st.code(sql_schema, language="sql")
    
    # Button to initialize database
    if st.button("Initialize/Update Database Schema", type="primary"):
        try:
            with st.spinner("Creating database schema..."):
                success = create_schema()
            
            if success:
                st.success("Database schema created/updated successfully!")
                st.info("You can now use the web extraction to gather data about initiatives.")
            else:
                st.error("Failed to create/update database schema. Check the logs for details.")
        except Exception as e:
            st.error(f"Error initializing database: {str(e)}")
    
    # Additional information about the database
    st.subheader("Database Information")
    
    try:
        engine = get_sqlalchemy_engine()
        
        with engine.connect() as conn:
            # Check if content_data table exists
            table_exists_query = text("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_name = 'content_data'
                );
            """)
            table_exists = conn.execute(table_exists_query).scalar_one()
            
            if table_exists:
                st.write("✅ Content data table exists")
                
                # Check for initiative columns
                initiative_col_query = text("""
                    SELECT column_name FROM information_schema.columns 
                    WHERE table_name = 'content_data' AND 
                    column_name IN ('initiative', 'initiative_key', 'benefit_categories', 'benefit_examples');
                """)
                
                initiative_cols = [row[0] for row in conn.execute(initiative_col_query)]
                
                if 'initiative' in initiative_cols:
                    st.write("✅ Initiative column exists")
                else:
                    st.write("❌ Initiative column missing")
                    
                if 'initiative_key' in initiative_cols:
                    st.write("✅ Initiative key column exists")
                else:
                    st.write("❌ Initiative key column missing")
                    
                if 'benefit_categories' in initiative_cols:
                    st.write("✅ Benefit categories column exists")
                else:
                    st.write("❌ Benefit categories column missing")
                    
                if 'benefit_examples' in initiative_cols:
                    st.write("✅ Benefit examples column exists")
                else:
                    st.write("❌ Benefit examples column missing")
                
                # Count records
                count_query = text("SELECT COUNT(*) FROM content_data")
                record_count = conn.execute(count_query).scalar_one()
                st.write(f"📊 Total records: {record_count}")
                
                # Count initiatives
                initiative_count_query = text("""
                    SELECT COUNT(DISTINCT initiative) 
                    FROM content_data 
                    WHERE initiative IS NOT NULL
                """)
                initiative_count = conn.execute(initiative_count_query).scalar_one()
                st.write(f"📊 Distinct initiatives: {initiative_count}")
            else:
                st.warning("Content data table does not exist. Please initialize the database.")
    except Exception as e:
        st.error(f"Error checking database: {str(e)}")

# Sidebar for application navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode", 
    ["Dashboard", "Initiative Analysis", "Initiative Comparison", "View Data", "Web Extraction", "Database Initialization"]
)
logger.info(f"User selected app mode: {app_mode}")
print(f"User navigated to: {app_mode}")

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

# Logging
logger.info("Application rendering completed")

# Run the main app if this file is executed directly
if __name__ == "__main__":
    logger.info("Application main method executed")
    print("Application started via main method")