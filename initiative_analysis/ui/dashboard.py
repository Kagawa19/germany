import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import logging

from database.connection import get_sqlalchemy_engine
from database.operations import fetch_data, get_geographic_statistics, search_abs_mentions, get_abs_name_variants
from sqlalchemy import text



logger = logging.getLogger("Dashboard")

def safe_join(items):
    """
    Safely join a list of items, handling None and non-string values
    """
    if not items:
        return ""
    
    # Convert all items to strings, filter out None
    safe_items = [str(item) for item in items if item is not None]
    
    return ', '.join(safe_items) if safe_items else ""


def dashboard_page():
    """Display the main dashboard with key metrics and visualizations."""
    logger.info("Rendering Dashboard page")
    
    # Fetch data for the dashboard
    try:
        df = fetch_data(limit=1000)
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return
    
    # Display key metrics
    st.markdown("<h2 class='sub-header'>Key Metrics</h2>", unsafe_allow_html=True)
    
    # Create metric cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Documents", len(df))
    
    with col2:
        # Count unique organizations across all documents
        org_count = 0
        if 'organizations' in df.columns:
            all_orgs = set()
            for org_list in df['organizations'].dropna():
                try:
                    if org_list and isinstance(org_list, list):
                        all_orgs.update(org_list)
                except Exception as e:
                    logger.warning(f"Error processing organizations: {str(e)}")
            org_count = len(all_orgs)
        st.metric("Organizations", org_count)
    
    with col3:
        # Count unique themes across all rows
        theme_count = 0
        if 'themes' in df.columns:
            all_themes = set()
            for themes_list in df['themes'].dropna():
                try:
                    if themes_list and isinstance(themes_list, list):
                        all_themes.update(themes_list)
                except Exception as e:
                    logger.warning(f"Error processing themes: {str(e)}")
            theme_count = len(all_themes)
        st.metric("Themes", theme_count)
    
    with col4:
        # Count countries
        country_count = 0
        if 'countries' in df.columns:
            all_countries = set()
            for countries_list in df['countries'].dropna():
                try:
                    if countries_list and isinstance(countries_list, list):
                        all_countries.update(countries_list)
                except Exception as e:
                    logger.warning(f"Error processing countries: {str(e)}")
            country_count = len(all_countries)
        st.metric("Countries", country_count)

    # Create visualizations if we have sufficient data
    if len(df) > 0:
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Organizations", "Themes", "Geography", "Name Variants"])
        
        with tab1:
            if 'organizations' in df.columns:
                # Flatten the organizations lists to count occurrences
                all_orgs = []
                for org_list in df['organizations'].dropna():
                    if org_list:
                        all_orgs.extend(org_list)
                
                # Count occurrences
                org_counts = pd.Series(all_orgs).value_counts().reset_index()
                org_counts.columns = ['Organization', 'Count']
                
                # Only show top 15 for readability
                org_counts = org_counts.head(15)
                
                fig = px.bar(
                    org_counts, 
                    x='Organization', 
                    y='Count',
                    title='Top Organizations in Content',
                    color='Count',
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if 'themes' in df.columns:
                # Flatten the themes lists to count occurrences
                all_themes = []
                for themes_list in df['themes'].dropna():
                    if themes_list:
                        all_themes.extend(themes_list)
                
                # Count occurrences
                theme_counts = pd.Series(all_themes).value_counts().reset_index()
                theme_counts.columns = ['Theme', 'Count']
                
                # Only show top 15 for readability
                theme_counts = theme_counts.head(15)
                
                if not theme_counts.empty:
                    fig = px.bar(
                        theme_counts,
                        x='Count',
                        y='Theme',
                        title='Top Themes in Content',
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
            # Get geographic statistics
            geo_stats = get_geographic_statistics()
            
            # Create a dataframe for countries
            if geo_stats and 'countries' in geo_stats and geo_stats['countries']:
                countries_df = pd.DataFrame(geo_stats['countries'])
                
                fig = px.bar(
                    countries_df, 
                    x='country', 
                    y='count',
                    title='Countries Mentioned in Content',
                    color='count',
                    color_continuous_scale='blues'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a map visualization
                try:
                    # Convert country names to ISO codes if needed
                    fig_map = px.choropleth(
                        countries_df,
                        locations='country',
                        locationmode='country names',
                        color='count',
                        title='Global Distribution of Mentions',
                        color_continuous_scale='blues'
                    )
                    st.plotly_chart(fig_map, use_container_width=True)
                except Exception as e:
                    st.warning(f"Could not generate map visualization: {str(e)}")
            
            # Show regional distribution
            if geo_stats and 'regions' in geo_stats and geo_stats['regions']:
                regions_df = pd.DataFrame(geo_stats['regions'])
                
                fig = px.pie(
                    regions_df,
                    values='count',
                    names='region',
                    title='Regional Distribution of Focus'
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            # Get name variants
            name_variants = get_abs_name_variants()
            
            if name_variants:
                # Create a dataframe from name variants
                # Get frequency of each variant from database
                engine = get_sqlalchemy_engine()
                variant_counts = []
                
                try:
                    for variant in name_variants:
                        query = text("""
                        SELECT COUNT(*) 
                        FROM abs_mentions 
                        WHERE name_variant = :variant
                        """)
                        
                        with engine.connect() as conn:
                            count = conn.execute(query, {"variant": variant}).scalar()
                            variant_counts.append({"Variant": variant, "Count": count})
                    
                    # Create dataframe
                    variants_df = pd.DataFrame(variant_counts)
                    variants_df = variants_df.sort_values("Count", ascending=False)
                    
                    # Create visualization
                    fig = px.bar(
                        variants_df,
                        x='Variant',
                        y='Count',
                        title='Frequency of ABS Initiative Name Variants',
                        color='Count',
                        color_continuous_scale='oranges'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show table for reference
                    st.subheader("All ABS Initiative Name Variants")
                    st.dataframe(variants_df)
                    
                except Exception as e:
                    st.error(f"Error fetching name variant counts: {str(e)}")
                    st.write(name_variants)
            else:
                st.info("No name variants found in the database")
    
    # Display sentiment distribution
    st.subheader("Content Sentiment Analysis")
    
    # Create sentiment visualization
    try:
        engine = get_sqlalchemy_engine()
        sentiment_query = text("""
        SELECT overall_sentiment, COUNT(*) as count
        FROM sentiment_analysis
        GROUP BY overall_sentiment
        ORDER BY count DESC
        """)
        
        sentiment_df = pd.read_sql(sentiment_query, engine)
        
        if not sentiment_df.empty:
            # Create pie chart
            fig = px.pie(
                sentiment_df,
                values='count',
                names='overall_sentiment',
                title='Sentiment Distribution',
                color='overall_sentiment',
                color_discrete_map={
                    'Positive': 'green',
                    'Neutral': 'gray',
                    'Negative': 'red'
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data available")
            
    except Exception as e:
        st.error(f"Error fetching sentiment data: {str(e)}")
        
    # Display a sample of recent content
    st.subheader("Recent Content")
    recent_df = df.head(5)
    if not recent_df.empty:
        for i, row in recent_df.iterrows():
            with st.expander(f"{row.get('title', 'Untitled')}"):
                st.write(f"**Date:** {row.get('publication_date', 'N/A')}")
                st.write(f"**Sentiment:** {row.get('overall_sentiment', 'N/A')}")
                
                # Show themes if available
                themes = row.get('themes', [])
                if themes and isinstance(themes, list):
                    st.write(f"**Themes:** {safe_join(themes)}")
                
                # Show organizations if available
                orgs = row.get('organizations', [])
                if orgs and isinstance(orgs, list):
                    st.write(f"**Organizations:** {safe_join(orgs)}")
                
                # Show countries if available
                countries = row.get('countries', [])
                if countries and isinstance(countries, list):
                    st.write(f"**Countries:** {safe_join(countries)}")
                
                st.write(f"**Summary:** {row.get('content_summary', 'No summary')}")
                st.write(f"**URL:** {row.get('url', 'No URL')}")
    else:
        st.info("No content available. Extract content first.")

def abs_mentions_explorer():
    """
    Page for exploring specific mentions of ABS Initiative name variants.
    """
    st.title("ABS Initiative Mentions Explorer")
    
    # Get all name variants
    name_variants = get_abs_name_variants()
    
    if not name_variants:
        st.warning("No ABS Initiative name variants found in the database.")
        st.info("Please extract content first to populate the database with mentions.")
        return
    
    # Add custom search option
    search_options = ["All Variants"] + name_variants + ["Custom Search"]
    selected_option = st.selectbox("Select Name Variant", search_options)
    
    search_term = ""
    
    if selected_option == "Custom Search":
        search_term = st.text_input("Enter search term")
    elif selected_option != "All Variants":
        search_term = selected_option
    
    if search_term:
        # Search for mentions
        mentions = search_abs_mentions(search_term)
        
        if mentions:
            st.success(f"Found {len(mentions)} mentions of '{search_term}'")
            
            # Display mentions
            for i, mention in enumerate(mentions):
                with st.expander(f"{i+1}. {mention['title']} ({mention['date']})"):
                    st.markdown(f"**Context:** {mention['context']}")
                    st.markdown(f"**Name Variant:** {mention['name_variant']}")
                    st.markdown(f"**Relevance Score:** {mention['relevance_score']:.2f}")
                    st.markdown(f"**URL:** {mention['url']}")
        else:
            st.info(f"No mentions found for '{search_term}'")
    
    elif selected_option == "All Variants":
        # Show distribution of all variants
        try:
            engine = get_sqlalchemy_engine()
            query = text("""
            SELECT name_variant, COUNT(*) as count
            FROM abs_mentions
            GROUP BY name_variant
            ORDER BY count DESC
            """)
            
            df = pd.read_sql(query, engine)
            
            if not df.empty:
                # Create visualization
                fig = px.bar(
                    df,
                    x='name_variant',
                    y='count',
                    title='Distribution of ABS Initiative Name Variants',
                    color='count',
                    color_continuous_scale='viridis'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show table
                st.subheader("All Name Variants")
                st.dataframe(df)
            else:
                st.info("No variant data available")
                
        except Exception as e:
            st.error(f"Error fetching variant distribution: {str(e)}")

def geographic_analysis():
    """
    Page for geographic analysis of ABS Initiative mentions.
    """
    st.title("Geographic Analysis")
    
    # Get geographic statistics
    geo_stats = get_geographic_statistics()
    
    if not geo_stats or not geo_stats.get('countries'):
        st.warning("No geographic data found in the database.")
        st.info("Please extract content first to populate the database with geographic information.")
        return
    
    # Create tabs for different views
    tab1, tab2, tab3 = st.tabs(["Countries", "Regions", "Project Map"])
    
    with tab1:
        # Create a dataframe for countries
        countries_df = pd.DataFrame(geo_stats['countries'])
        
        # Top countries bar chart
        fig = px.bar(
            countries_df.head(15), 
            x='country', 
            y='count',
            title='Top Countries Mentioned',
            color='count',
            color_continuous_scale='blues'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # World map
        try:
            fig_map = px.choropleth(
                countries_df,
                locations='country',
                locationmode='country names',
                color='count',
                title='Global Distribution of Mentions',
                color_continuous_scale='blues'
            )
            st.plotly_chart(fig_map, use_container_width=True)
        except Exception as e:
            st.warning(f"Could not generate map visualization: {str(e)}")
        
        # Show full table of countries
        st.subheader("All Countries")
        st.dataframe(countries_df)
    
    with tab2:
        # Create a dataframe for regions
        if 'regions' in geo_stats and geo_stats['regions']:
            regions_df = pd.DataFrame(geo_stats['regions'])
            
            # Regions pie chart
            fig = px.pie(
                regions_df,
                values='count',
                names='region',
                title='Distribution by Region'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Regions bar chart
            fig = px.bar(
                regions_df, 
                x='region', 
                y='count',
                title='Mentions by Region',
                color='count',
                color_continuous_scale='greens'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Show full table of regions
            st.subheader("All Regions")
            st.dataframe(regions_df)
        else:
            st.info("No region data available")
    
    with tab3:
        # Create visualization of project locations
        st.subheader("Project Locations")
        
        try:
            engine = get_sqlalchemy_engine()
            query = text("""
            SELECT pd.project_name, gf.country, COUNT(*) as mentions
            FROM project_details pd
            JOIN geographic_focus gf ON pd.source_id = gf.source_id
            WHERE gf.country IS NOT NULL AND gf.country != ''
            GROUP BY pd.project_name, gf.country
            ORDER BY mentions DESC
            """)
            
            projects_df = pd.read_sql(query, engine)
            
            if not projects_df.empty:
                # Create bubble map
                fig = px.scatter_geo(
                    projects_df,
                    locations='country',
                    locationmode='country names',
                    size='mentions',
                    hover_name='project_name',
                    title='Project Locations',
                    size_max=25
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Show project-country table
                st.subheader("Projects by Country")
                st.dataframe(projects_df)
            else:
                st.info("No project location data available")
                
        except Exception as e:
            st.error(f"Error fetching project location data: {str(e)}")