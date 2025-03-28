import streamlit as st
import pandas as pd
from datetime import datetime
import logging
import json

from database.connection import get_sqlalchemy_engine
from database.operations import fetch_data, fetch_comprehensive_data, semantic_search
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
    themes_query = text("SELECT DISTINCT theme FROM thematic_areas ORDER BY theme")
    themes = [t[0] for t in pd.read_sql(themes_query, engine)['theme']]
    themes.insert(0, "All")  # Add "All" option at the beginning
    
    # Get organizations
    orgs_query = text("SELECT DISTINCT name FROM organizations ORDER BY name")
    organizations = [o[0] for o in pd.read_sql(orgs_query, engine)['name']]
    organizations.insert(0, "All")
    
    # Get sentiments
    sentiments_query = text("SELECT DISTINCT overall_sentiment FROM sentiment_analysis WHERE overall_sentiment IS NOT NULL ORDER BY overall_sentiment")
    sentiments = [s[0] for s in pd.read_sql(sentiments_query, engine)['overall_sentiment']]
    sentiments.insert(0, "All")
    
    # Get countries
    countries_query = text("SELECT DISTINCT country FROM geographic_focus WHERE country IS NOT NULL AND country != '' ORDER BY country")
    countries = [c[0] for c in pd.read_sql(countries_query, engine)['country']]
    countries.insert(0, "All")
    
    # Get regions
    regions_query = text("SELECT DISTINCT region FROM geographic_focus WHERE region IS NOT NULL AND region != '' ORDER BY region")
    regions = [r[0] for r in pd.read_sql(regions_query, engine)['region']]
    regions.insert(0, "All")
    
    # Filter dropdowns
    filter_theme = st.sidebar.selectbox("Theme", themes)
    filter_org = st.sidebar.selectbox("Organization", organizations)
    filter_sentiment = st.sidebar.selectbox("Sentiment", sentiments)
    filter_country = st.sidebar.selectbox("Country", countries)
    filter_region = st.sidebar.selectbox("Region", regions)
    
    # Date range filter
    st.sidebar.header("Date Range")
    min_date_query = text("SELECT MIN(publication_date) FROM content_sources WHERE publication_date IS NOT NULL")
    max_date_query = text("SELECT MAX(publication_date) FROM content_sources WHERE publication_date IS NOT NULL")
    
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
    
    # Add semantic search option
    st.sidebar.header("Semantic Search")
    use_semantic = st.sidebar.checkbox("Use Semantic Search")
    
    if use_semantic:
        search_query = st.sidebar.text_input("Enter search query")
        search_results_count = st.sidebar.slider("Number of results", 5, 50, 10)
    
    # Prepare filters
    filters = {}
    if filter_theme != "All":
        filters['theme'] = filter_theme
    if filter_org != "All":
        filters['organization'] = filter_org
    if filter_sentiment != "All":
        filters['sentiment'] = filter_sentiment
    if filter_country != "All":
        filters['country'] = filter_country
    if filter_region != "All":
        filters['region'] = filter_region
    
    # Add date range to filters
    if date_range:
        filters['start_date'] = date_range[0]
        filters['end_date'] = date_range[1]
    
    # Handle search vs filter display
    if use_semantic and search_query:
        st.subheader(f"Semantic Search Results for: '{search_query}'")
        
        # Perform semantic search
        search_results = semantic_search(search_query, top_k=search_results_count)
        
        if search_results:
            # Convert search results to dataframe for display
            results_df = pd.DataFrame(search_results)
            results_df['similarity'] = results_df['similarity'].apply(lambda x: f"{x:.2f}")
            
            # Display search results
            st.dataframe(results_df[['id', 'title', 'date', 'summary', 'sentiment', 'similarity']])
            
            # Allow selecting a result for detailed view
            selected_id = st.selectbox("Select a result to view details", results_df['id'].tolist())
            
            # Fetch comprehensive data for the selected ID
            selected_data = fetch_comprehensive_data(selected_id)
            
            # Display detailed view
            display_detailed_view(selected_data)
        else:
            st.info("No semantic search results found. Try a different query.")
    else:
        # Fetch and display data using filters
        display_limit = st.slider("Number of records to display", 100, 5000, 500)
        df = fetch_data(limit=display_limit, filters=filters)
        
        # Display data
        st.dataframe(df)
        
        # Detailed view of selected row
        if not df.empty:
            st.subheader("Detailed View")
            selected_id = st.selectbox("Select a record to view details", df['source_id'].tolist())
            
            # Fetch comprehensive data for the selected ID
            selected_data = fetch_comprehensive_data(selected_id)
            
            # Display detailed view
            display_detailed_view(selected_data)
        else:
            st.warning("No data matches your filter criteria. Try adjusting your filters.")

def display_detailed_view(data):
    """
    Display a detailed view of a record with all its related information.
    
    Args:
        data: Dictionary with comprehensive data for the record
    """
    if not data or 'error' in data:
        st.error(f"Error retrieving data: {data.get('error', 'Unknown error')}")
        return
    
    # Main content information
    st.markdown(f"## {data.get('title', 'No Title')}")
    st.markdown(f"**URL:** {data.get('url', 'N/A')}")
    st.markdown(f"**Date:** {data.get('publication_date', 'N/A')}")
    st.markdown(f"**Source Type:** {data.get('source_type', 'N/A')}")
    st.markdown(f"**Language:** {data.get('language', 'N/A')}")
    
    # Create tabs for different types of information
    tabs = st.tabs(["Summary", "Mentions", "Geography", "Organizations", "Themes", "Projects", "Resources", "Full Content"])
    
    # Summary tab
    with tabs[0]:
        # Display sentiment if available
        if 'sentiment' in data:
            sentiment_color = {
                'Positive': 'green',
                'Neutral': 'gray',
                'Negative': 'red'
            }.get(data['sentiment'].get('overall_sentiment', 'Neutral'), 'gray')
            
            st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};'>{data['sentiment'].get('overall_sentiment', 'Neutral')}</span> (Score: {data['sentiment'].get('score', 0):.2f}, Confidence: {data['sentiment'].get('confidence', 0):.2f})", unsafe_allow_html=True)
        
        # Display summary
        st.markdown("### Summary")
        st.markdown(data.get('content_summary', 'No summary available'))
    
    # Mentions tab
    with tabs[1]:
        if 'abs_mentions' in data and data['abs_mentions']:
            st.markdown("### ABS Initiative Mentions")
            for mention in data['abs_mentions']:
                with st.expander(f"{mention.get('name_variant', 'Unknown Variant')} (Score: {mention.get('relevance_score', 0):.2f})"):
                    st.markdown(f"**Type:** {mention.get('mention_type', 'N/A')}")
                    st.markdown(f"**Context:** {mention.get('mention_context', 'N/A')}")
        else:
            st.info("No specific ABS Initiative mentions identified in this content.")
    
    # Geography tab
    with tabs[2]:
        if 'geographic_focus' in data and data['geographic_focus']:
            st.markdown("### Geographic Focus")
            
            # Create a DataFrame for better display
            geo_data = pd.DataFrame(data['geographic_focus'])
            
            # Map if we have country data
            if 'country' in geo_data.columns and not geo_data['country'].isna().all():
                try:
                    import plotly.express as px
                    
                    # Count occurrences of each country
                    country_counts = geo_data['country'].value_counts().reset_index()
                    country_counts.columns = ['country', 'count']
                    
                    # Create map
                    fig = px.choropleth(
                        country_counts,
                        locations='country',
                        locationmode='country names',
                        color='count',
                        title='Countries Mentioned in Document',
                        color_continuous_scale='blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create map visualization: {str(e)}")
            
            # Display geographic data table
            st.dataframe(geo_data)
        else:
            st.info("No geographic focus information available for this content.")
    
    # Organizations tab
    with tabs[3]:
        if 'organizations' in data and data['organizations']:
            st.markdown("### Organizations")
            
            # Create a DataFrame for better display
            org_data = pd.DataFrame(data['organizations'])
            
            # Display organizations as cards
            for i, org in enumerate(data['organizations']):
                with st.expander(f"{org.get('name', 'Unknown Organization')} ({org.get('organization_type', 'Unknown Type')})"):
                    st.markdown(f"**Role:** {org.get('relationship_type', 'N/A')}")
                    if org.get('website'):
                        st.markdown(f"**Website:** {org.get('website')}")
                    if org.get('description'):
                        st.markdown(f"**Description:** {org.get('description')}")
        else:
            st.info("No organization information available for this content.")
    
    # Themes tab
    with tabs[4]:
        if 'themes' in data and data['themes']:
            st.markdown("### Themes")
            
            # Create visual representation of themes
            try:
                import plotly.express as px
                
                # Create DataFrame for visualization
                theme_data = pd.DataFrame({'theme': data['themes']})
                
                # Create horizontal bar chart
                fig = px.bar(
                    theme_data['theme'].value_counts().reset_index(),
                    x='count',
                    y='index',
                    orientation='h',
                    title='Themes',
                    labels={'index': 'Theme', 'count': 'Occurrences'},
                    color='count',
                    color_continuous_scale='greens'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create theme visualization: {str(e)}")
                
            # List themes
            for theme in data['themes']:
                st.markdown(f"- {theme}")
        else:
            st.info("No theme information available for this content.")
    
    # Projects tab
    with tabs[5]:
        if 'projects' in data and data['projects']:
            st.markdown("### Projects")
            
            # Display projects as cards
            for project in data['projects']:
                with st.expander(f"{project.get('project_name', 'Unnamed Project')} ({project.get('project_type', 'Unknown Type')})"):
                    # Date information
                    start_date = project.get('start_date', 'N/A')
                    end_date = project.get('end_date', 'N/A')
                    st.markdown(f"**Period:** {start_date} to {end_date}")
                    
                    # Status
                    status_color = {
                        'completed': 'green',
                        'ongoing': 'blue',
                        'planned': 'orange'
                    }.get(project.get('status', '').lower(), 'gray')
                    
                    st.markdown(f"**Status:** <span style='color:{status_color};'>{project.get('status', 'Unknown')}</span>", unsafe_allow_html=True)
                    
                    # Description
                    if project.get('description'):
                        st.markdown(f"**Description:** {project.get('description')}")
        else:
            st.info("No project information available for this content.")
    
    # Resources tab
    with tabs[6]:
        if 'resources' in data and data['resources']:
            st.markdown("### Resources")
            
            # Group resources by type
            resource_types = {}
            for resource in data['resources']:
                resource_type = resource.get('resource_type', 'Other')
                if resource_type not in resource_types:
                    resource_types[resource_type] = []
                resource_types[resource_type].append(resource)
            
            # Display resources by type
            for resource_type, resources in resource_types.items():
                st.markdown(f"#### {resource_type}")
                
                for resource in resources:
                    with st.expander(f"{resource.get('resource_name', 'Unnamed Resource')}"):
                        if resource.get('resource_url'):
                            st.markdown(f"**URL:** {resource.get('resource_url')}")
                        
                        if resource.get('description'):
                            st.markdown(f"**Description:** {resource.get('description')}")
        else:
            st.info("No resource information available for this content.")
    
    # Full content tab
    with tabs[7]:
        st.markdown("### Full Content")
        st.markdown(data.get('full_content', 'No full content available'))

def export_content():
    """
    Export content data in various formats.
    """
    st.title("Export Content Data")
    
    # Get export options
    export_format = st.selectbox(
        "Export Format",
        ["CSV", "JSON", "Excel"]
    )
    
    export_scope = st.selectbox(
        "Export Scope",
        ["Basic Data", "Full Content", "Comprehensive Data"]
    )
    
    # Get filter options
    st.subheader("Export Filters")
    
    # Get unique values for filters
    engine = get_sqlalchemy_engine()
    
    # Fetch unique themes
    themes_query = text("SELECT DISTINCT theme FROM thematic_areas ORDER BY theme")
    themes = [t[0] for t in pd.read_sql(themes_query, engine)['theme']]
    themes.insert(0, "All")
    
    # Get languages
    languages_query = text("SELECT DISTINCT language FROM content_sources WHERE language IS NOT NULL ORDER BY language")
    languages = [l[0] for l in pd.read_sql(languages_query, engine)['language']]
    languages.insert(0, "All")
    
    # Get sentiments
    sentiments_query = text("SELECT DISTINCT overall_sentiment FROM sentiment_analysis WHERE overall_sentiment IS NOT NULL ORDER BY overall_sentiment")
    sentiments = [s[0] for s in pd.read_sql(sentiments_query, engine)['overall_sentiment']]
    sentiments.insert(0, "All")
    
    # Filter options
    filter_language = st.selectbox("Language", languages)
    filter_theme = st.selectbox("Theme", themes)
    filter_sentiment = st.selectbox("Sentiment", sentiments)
    
    # Date range
    min_date_query = text("SELECT MIN(publication_date) FROM content_sources WHERE publication_date IS NOT NULL")
    max_date_query = text("SELECT MAX(publication_date) FROM content_sources WHERE publication_date IS NOT NULL")
    
    with engine.connect() as connection:
        min_date = connection.execute(min_date_query).scalar_one_or_none()
        max_date = connection.execute(max_date_query).scalar_one_or_none()
    
    if min_date and max_date:
        date_range = st.date_input(
            "Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )
    else:
        date_range = None
    
    # Prepare filters
    filters = {}
    if filter_language != "All":
        filters['language'] = filter_language
    if filter_theme != "All":
        filters['theme'] = filter_theme
    if filter_sentiment != "All":
        filters['sentiment'] = filter_sentiment
    
    # Add date range to filters
    if date_range:
        filters['start_date'] = date_range[0]
        filters['end_date'] = date_range[1]
    
    # Get data based on scope and filters
    if st.button("Prepare Export"):
        with st.spinner("Preparing export data..."):
            try:
                if export_scope == "Basic Data":
                    # Use view to get minimal data
                    query_parts = ["SELECT source_id, url, title, publication_date, language, content_summary, overall_sentiment FROM v_abs_content_analysis"]
                    
                    # Add WHERE clause if needed
                    where_clauses = []
                    query_params = {}
                    
                    if filter_language != "All":
                        where_clauses.append("language = %(language)s")
                        query_params['language'] = filter_language
                    
                    if filter_theme != "All":
                        where_clauses.append("themes @> ARRAY[%(theme)s]")
                        query_params['theme'] = filter_theme
                    
                    if filter_sentiment != "All":
                        where_clauses.append("overall_sentiment = %(sentiment)s")
                        query_params['sentiment'] = filter_sentiment
                    
                    if date_range:
                        where_clauses.append("publication_date BETWEEN %(start_date)s AND %(end_date)s")
                        query_params['start_date'] = date_range[0]
                        query_params['end_date'] = date_range[1]
                    
                    if where_clauses:
                        query_parts.append("WHERE " + " AND ".join(where_clauses))
                    
                    # Add ORDER BY clause
                    query_parts.append("ORDER BY publication_date DESC")
                    
                    # Execute query
                    query = " ".join(query_parts)
                    df = pd.read_sql(query, engine, params=query_params)
                    
                elif export_scope == "Full Content":
                    # Use view with full content joined
                    query_parts = ["""
                    SELECT 
                        v.source_id, v.url, v.title, v.publication_date, 
                        v.language, v.content_summary, v.overall_sentiment,
                        v.themes, v.name_variants, v.countries, v.regions,
                        v.organizations, cs.full_content
                    FROM v_abs_content_analysis v
                    JOIN content_sources cs ON v.source_id = cs.id
                    """]
                    
                    # Add WHERE clause if needed
                    where_clauses = []
                    query_params = {}
                    
                    if filter_language != "All":
                        where_clauses.append("v.language = %(language)s")
                        query_params['language'] = filter_language
                    
                    if filter_theme != "All":
                        where_clauses.append("v.themes @> ARRAY[%(theme)s]")
                        query_params['theme'] = filter_theme
                    
                    if filter_sentiment != "All":
                        where_clauses.append("v.overall_sentiment = %(sentiment)s")
                        query_params['sentiment'] = filter_sentiment
                    
                    if date_range:
                        where_clauses.append("v.publication_date BETWEEN %(start_date)s AND %(end_date)s")
                        query_params['start_date'] = date_range[0]
                        query_params['end_date'] = date_range[1]
                    
                    if where_clauses:
                        query_parts.append("WHERE " + " AND ".join(where_clauses))
                    
                    # Add ORDER BY clause
                    query_parts.append("ORDER BY v.publication_date DESC")
                    
                    # Execute query
                    query = " ".join(query_parts)
                    df = pd.read_sql(query, engine, params=query_params)
                    
                else:  # Comprehensive Data
                    # Use fetch_data function to get filtered data
                    df = fetch_data(limit=10000, filters=filters)
                
                # Format date columns
                for col in df.columns:
                    if col.endswith('_date') and not df[col].empty:
                        df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
                
                # Prepare export based on format
                if export_format == "CSV":
                    csv_data = df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_data,
                        file_name=f"abs_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
                    
                elif export_format == "JSON":
                    # Handle non-serializable types by converting them
                    json_data = df.to_json(orient="records", date_format="iso")
                    st.download_button(
                        label="Download JSON",
                        data=json_data,
                        file_name=f"abs_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                    
                elif export_format == "Excel":
                    # Create Excel file in memory
                    import io
                    from xlsxwriter import Workbook
                    
                    buffer = io.BytesIO()
                    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
                        df.to_excel(writer, sheet_name='ABS Initiative Data', index=False)
                    
                    buffer.seek(0)
                    
                    st.download_button(
                        label="Download Excel",
                        data=buffer,
                        file_name=f"abs_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                
                # Show preview
                st.subheader("Data Preview")
                st.markdown(f"**Total Records:** {len(df)}")
                st.dataframe(df.head(10))
                
            except Exception as e:
                st.error(f"Error preparing export: {str(e)}")
                logger.error(f"Export error: {str(e)}")

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
            query = text("SELECT DISTINCT language FROM content_sources WHERE language IS NOT NULL")
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
            cs.id, cs.url, cs.title, cs.publication_date, 
            cs.content_summary, cs.language,
            sa.overall_sentiment
        FROM content_sources cs
        LEFT JOIN sentiment_analysis sa ON cs.id = sa.source_id
        WHERE cs.language = :language
        ORDER BY cs.id DESC
        """)
        
        # Execute query with parameter
        df = pd.read_sql(query, engine, params={"language": language})
        
        logger.info(f"Query returned {len(df)} records for language {language}")
        
        if df.empty:
            st.warning(f"No data found for language: {language}")
            return
        
        # Clean any encoding issues in the dataframe
        for col in df.columns:
            if df[col].dtype == 'object':  # Only process string columns
                df[col] = df[col].astype(str).apply(clean_text)
        
        # Format date columns
        for col in df.columns:
            if col.endswith('_date') and not df[col].empty:
                df[col] = pd.to_datetime(df[col]).dt.strftime('%Y-%m-%d')
        
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


def display_detailed_view(data):
    """
    Display a detailed view of a record with all its related information.
    
    Args:
        data: Dictionary with comprehensive data for the record
    """
    if not data or 'error' in data:
        st.error(f"Error retrieving data: {data.get('error', 'Unknown error')}")
        return
    
    # Main content information
    st.markdown(f"## {data.get('title', 'No Title')}")
    st.markdown(f"**URL:** {data.get('url', 'N/A')}")
    st.markdown(f"**Date:** {data.get('publication_date', 'N/A')}")
    st.markdown(f"**Source Type:** {data.get('source_type', 'N/A')}")
    st.markdown(f"**Language:** {data.get('language', 'N/A')}")
    
    # Create tabs for different types of information
    tabs = st.tabs(["Summary", "Mentions", "Geography", "Organizations", "Themes", "Projects", "Resources", "Full Content"])
    
    # Summary tab
    with tabs[0]:
        # Display sentiment if available
        if 'sentiment' in data:
            sentiment_color = {
                'Positive': 'green',
                'Neutral': 'gray',
                'Negative': 'red'
            }.get(data['sentiment'].get('overall_sentiment', 'Neutral'), 'gray')
            
            st.markdown(f"**Sentiment:** <span style='color:{sentiment_color};'>{data['sentiment'].get('overall_sentiment', 'Neutral')}</span> (Score: {data['sentiment'].get('score', 0):.2f}, Confidence: {data['sentiment'].get('confidence', 0):.2f})", unsafe_allow_html=True)
        
        # Display summary
        st.markdown("### Summary")
        st.markdown(data.get('content_summary', 'No summary available'))
    
    # Mentions tab
    with tabs[1]:
        if 'abs_mentions' in data and data['abs_mentions']:
            st.markdown("### ABS Initiative Mentions")
            for mention in data['abs_mentions']:
                with st.expander(f"{mention.get('name_variant', 'Unknown Variant')} (Score: {mention.get('relevance_score', 0):.2f})"):
                    st.markdown(f"**Type:** {mention.get('mention_type', 'N/A')}")
                    st.markdown(f"**Context:** {mention.get('mention_context', 'N/A')}")
        else:
            st.info("No specific ABS Initiative mentions identified in this content.")
    
    # Geography tab
    # Geography tab
    with tabs[2]:
        if 'geographic_focus' in data and data['geographic_focus']:
            st.markdown("### Geographic Focus")
            
            # Create a DataFrame for better display
            geo_data = pd.DataFrame(data['geographic_focus'])
            
            # Map if we have country data
            if 'country' in geo_data.columns and not geo_data['country'].isna().all():
                try:
                    import plotly.express as px
                    
                    # Count occurrences of each country
                    country_counts = geo_data['country'].value_counts().reset_index()
                    country_counts.columns = ['country', 'count']
                    
                    # Create map
                    fig = px.choropleth(
                        country_counts,
                        locations='country',
                        locationmode='country names',
                        color='count',
                        title='Countries Mentioned in Document',
                        color_continuous_scale='blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not create map visualization: {str(e)}")
            
            # Display geographic data table
            st.dataframe(geo_data)
        else:
            st.info("No geographic focus information available for this content.")

    # Organizations tab
    with tabs[3]:
        if 'organizations' in data and data['organizations']:
            st.markdown("### Organizations")
            
            # Create a DataFrame for better display
            org_data = pd.DataFrame(data['organizations'])
            
            # Display organizations as cards
            for i, org in enumerate(data['organizations']):
                with st.expander(f"{org.get('name', 'Unknown Organization')} ({org.get('organization_type', 'Unknown Type')})"):
                    st.markdown(f"**Role:** {org.get('relationship_type', 'N/A')}")
                    if org.get('website'):
                        st.markdown(f"**Website:** {org.get('website')}")
                    if org.get('description'):
                        st.markdown(f"**Description:** {org.get('description')}")
        else:
            st.info("No organization information available for this content.")

    # Themes tab
    with tabs[4]:
        if 'themes' in data and data['themes']:
            st.markdown("### Themes")
            
            # Create visual representation of themes
            try:
                import plotly.express as px
                
                # Create DataFrame for visualization
                theme_data = pd.DataFrame({'theme': data['themes']})
                
                # Create horizontal bar chart
                fig = px.bar(
                    theme_data['theme'].value_counts().reset_index(),
                    x='count',
                    y='index',
                    orientation='h',
                    title='Themes',
                    labels={'index': 'Theme', 'count': 'Occurrences'},
                    color='count',
                    color_continuous_scale='greens'
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.warning(f"Could not create theme visualization: {str(e)}")
                
            # List themes
            for theme in data['themes']:
                st.markdown(f"- {theme}")
        else:
            st.info("No theme information available for this content.")

    # Projects tab
    with tabs[5]:
        if 'projects' in data and data['projects']:
            st.markdown("### Projects")
            
            # Display projects as cards
            for project in data['projects']:
                with st.expander(f"{project.get('project_name', 'Unnamed Project')} ({project.get('project_type', 'Unknown Type')})"):
                    # Date information
                    start_date = project.get('start_date', 'N/A')
                    end_date = project.get('end_date', 'N/A')
                    st.markdown(f"**Period:** {start_date} to {end_date}")
                    
                    # Status
                    status_color = {
                        'completed': 'green',
                        'ongoing': 'blue',
                        'planned': 'orange'
                    }.get(project.get('status', '').lower(), 'gray')
                    
                    st.markdown(f"**Status:** <span style='color:{status_color};'>{project.get('status', 'Unknown')}</span>", unsafe_allow_html=True)
                    
                    # Description
                    if project.get('description'):
                        st.markdown(f"**Description:** {project.get('description')}")
        else:
            st.info("No project information available for this content.")

    # Resources tab
    with tabs[6]:
        if 'resources' in data and data['resources']:
            st.markdown("### Resources")
            
            # Group resources by type
            resource_types = {}
            for resource in data['resources']:
                resource_type = resource.get('resource_type', 'Other')
                if resource_type not in resource_types:
                    resource_types[resource_type] = []
                resource_types[resource_type].append(resource)
            
            # Display resources by type
            for resource_type, resources in resource_types.items():
                st.markdown(f"#### {resource_type}")
                
                for resource in resources:
                    with st.expander(f"{resource.get('resource_name', 'Unnamed Resource')}"):
                        if resource.get('resource_url'):
                            st.markdown(f"**URL:** {resource.get('resource_url')}")
                        
                        if resource.get('description'):
                            st.markdown(f"**Description:** {resource.get('description')}")
        else:
            st.info("No resource information available for this content.")

    # Full content tab
    with tabs[7]:
        st.markdown("### Full Content")
        st.markdown(data.get('full_content', 'No full content available'))