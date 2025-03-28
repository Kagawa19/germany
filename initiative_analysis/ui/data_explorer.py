import streamlit as st
import pandas as pd
from datetime import datetime
import logging
import json

from database.connection import get_sqlalchemy_engine
from database.operations import fetch_data
from sqlalchemy import text
from utils.text_processing import clean_text

logger = logging.getLogger("DataExplorer")

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

def export_content_by_language(language="All"):
    """
    Export content data to CSV files, separated by language.
    
    Args:
        language: Language to filter by, or "All" to export all languages separately
    """
    try:
        engine = get_sqlalchemy_engine()
        
        if language == "All":
            # Get list of available languages
            query = text("SELECT DISTINCT language FROM content_data WHERE language IS NOT NULL")
            languages_df = pd.read_sql(query, engine)
            
            if languages_df.empty:
                st.warning("No language data found in the database.")
                return
            
            languages = languages_df['language'].tolist()
            
            if not languages:
                st.warning("No language data found in the database.")
                return
                
            st.info(f"Exporting data for {len(languages)} languages: {', '.join(languages)}")
            
            # Export each language separately
            for lang in languages:
                export_single_language(engine, lang)
        else:
            # Export just the selected language
            export_single_language(engine, language)
            
    except Exception as e:
        st.error(f"Error exporting data: {str(e)}")
        logger.error(f"Error in export_content_by_language: {str(e)}")

def export_single_language(engine, language):
    """Helper function to export data for a single language"""
    try:
        logger.info(f"Exporting data for language: {language}")
        
        # Create query for the selected language
        query = text("""
        SELECT 
            id, link, title, date, summary, organization, sentiment, 
            initiative, language, themes
        FROM content_data 
        WHERE language = :language
        ORDER BY id DESC
        """)
        
        # Execute query with parameter
        df = pd.read_sql(query, engine, params={"language": language})
        
        logger.info(f"Query returned {len(df)} records for language {language}")
        
        if df.empty:
            st.warning(f"No data found for language: {language}")
            return
            
        # Convert themes array to string for CSV export if it's a list
        if 'themes' in df.columns:
            df['themes'] = df['themes'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
        
        # Clean any encoding issues in the dataframe
        for col in df.columns:
            if df[col].dtype == 'object':  # Only process string columns
                df[col] = df[col].astype(str).apply(clean_text)
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"content_export_{language}_{timestamp}.csv"
        
        # Convert to CSV
        csv = df.to_csv(index=False, encoding='utf-8')
        
        # Create download button
        st.download_button(
            label=f"Download {language} data ({len(df)} records)",
            data=csv,
            file_name=filename,
            mime="text/csv"
        )
        
        st.success(f"Prepared {len(df)} records for {language}")
        
    except Exception as e:
        st.error(f"Error exporting {language} data: {str(e)}")
        logger.error(f"Error in export_single_language for {language}: {str(e)}")