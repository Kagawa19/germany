import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
from datetime import datetime
import logging

from database.connection import get_sqlalchemy_engine
from database.operations import fetch_data
from sqlalchemy import text

logger = logging.getLogger("Dashboard")

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