import streamlit as st
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"app_log_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ABS_Initiative_App")

# Import Langfuse monitoring
from monitoring.langfuse_client import get_langfuse_client
from monitoring.streamlit_integration import track_page_view, end_page_view, track_interaction
from monitoring.streamlit_integration import track_function, track_dashboard_load, end_dashboard_load
from monitoring.streamlit_integration import get_session_trace  # Import the missing function

# Initialize Langfuse client
langfuse = get_langfuse_client()

# Import UI components
from ui.dashboard import dashboard_page, abs_mentions_explorer, geographic_analysis
from ui.data_explorer import view_data_page, export_content, export_content_by_language
from ui.extraction_ui import run_web_extraction, initialization_page

# Set page config
st.set_page_config(
    page_title="ABS Initiative Metadata Explorer",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add tracking decorators to important functions
dashboard_page = track_function(name="dashboard_page")(dashboard_page)
view_data_page = track_function(name="view_data_page")(view_data_page)
abs_mentions_explorer = track_function(name="abs_mentions_explorer")(abs_mentions_explorer)
geographic_analysis = track_function(name="geographic_analysis")(geographic_analysis)
export_content = track_function(name="export_content", track_args=True)(export_content)

def main():
    """Main application entry point"""
    # Get session trace for tracking - using the imported function
    trace = get_session_trace()
    trace_id = trace.id if trace else None
    
    st.sidebar.title("ABS Initiative Metadata Explorer")
    
    # Check if the prompts directory exists
    prompts_dir = os.path.join(os.path.dirname(__file__), "prompts")
    if not os.path.exists(prompts_dir):
        try:
            os.makedirs(prompts_dir)
            logger.info(f"Created prompts directory at {prompts_dir}")
            st.sidebar.warning("Prompts directory was missing and has been created. You'll need to add prompt templates.")
        except Exception as e:
            logger.error(f"Failed to create prompts directory: {str(e)}")
            st.sidebar.error("Could not create prompts directory. Check permissions.")
    
    # Navigation
    app_mode = st.sidebar.selectbox(
        "Select Mode",
        ["Dashboard", "Data Explorer", "Mentions Explorer", "Geographic Analysis", 
         "Export Data", "Web Extraction", "Database Setup"]
    )
    
    # Track page selection interaction
    if langfuse and trace_id:
        track_interaction("page_selection", {"selected_page": app_mode})
    
    # Track the page view
    if langfuse and trace_id:
        track_page_view(app_mode)
    
    # Track dashboard load time
    if langfuse and trace_id:
        track_dashboard_load(app_mode)
    
    try:
        # Display the selected page
        if app_mode == "Dashboard":
            dashboard_page()
        
        elif app_mode == "Data Explorer":
            view_data_page()
        
        elif app_mode == "Mentions Explorer":
            abs_mentions_explorer()
        
        elif app_mode == "Geographic Analysis":
            geographic_analysis()
        
        elif app_mode == "Export Data":
            st.title("Export ABS Initiative Data")
            
            export_type = st.radio(
                "Export Type",
                ["Standard Export", "Language-Based Export"]
            )
            
            # Track user interaction with export type
            if langfuse and trace_id:
                track_interaction("export_type_selection", {"type": export_type})
            
            if export_type == "Standard Export":
                export_content()
            else:
                st.subheader("Export by Language")
                
                # Get available languages
                from database.connection import get_sqlalchemy_engine
                from sqlalchemy import text
                import pandas as pd
                
                engine = get_sqlalchemy_engine()
                languages_query = text("SELECT DISTINCT language FROM content_sources WHERE language IS NOT NULL ORDER BY language")
                
                try:
                    # Add "All" option for exporting all languages separately
                    languages = ["All"] + [l[0] for l in pd.read_sql(languages_query, engine)['language']]
                    selected_language = st.selectbox("Select Language to Export", languages)
                    
                    # Track language selection
                    if langfuse and trace_id:
                        track_interaction("language_selection", {"language": selected_language})
                    
                    if st.button("Prepare Language Export"):
                        # Track export operation
                        if langfuse and trace_id:
                            export_span = langfuse.create_span(
                                trace_id=trace_id,
                                name="language_export",
                                metadata={"language": selected_language}
                            )
                        
                        try:
                            export_content_by_language(selected_language)
                            
                            # Update span with success
                            if langfuse and trace_id and 'export_span' in locals():
                                export_span.update(output={"status": "success"})
                                export_span.end()
                                
                        except Exception as e:
                            # Update span with error
                            if langfuse and trace_id and 'export_span' in locals():
                                export_span.update(
                                    output={"status": "error", "error": str(e)},
                                    status="error"
                                )
                                export_span.end()
                            raise
                except Exception as e:
                    st.error(f"Error fetching languages: {str(e)}")
                    logger.error(f"Language query error: {str(e)}")
        
        elif app_mode == "Web Extraction":
            st.title("Web Content Extraction")
            
            # Language selection
            language = st.selectbox(
                "Select Language for Extraction",
                ["English", "German", "French", "Spanish", "Portuguese"]
            )
            
            # Track language selection
            if langfuse and trace_id:
                track_interaction("extraction_language_selection", {"language": language})
            
            # Configuration options
            max_queries = st.number_input("Maximum Number of Queries (0 = unlimited)", min_value=0, value=3)
            if max_queries == 0:
                max_queries = None
                
            max_results = st.number_input("Maximum Results per Query (0 = unlimited)", min_value=0, value=10)
            if max_results == 0:
                max_results = None
            
            # Start extraction button
            if st.button("Start Web Extraction", type="primary"):
                max_queries_label = max_queries if max_queries else "unlimited"
                max_results_label = max_results if max_results else "unlimited"
                
                # Create extraction trace
                extraction_trace = None
                if langfuse:
                    extraction_trace = langfuse.create_trace(
                        name="web_extraction",
                        metadata={
                            "language": language,
                            "max_queries": max_queries_label,
                            "max_results": max_results_label
                        },
                        tags=["extraction", language.lower()]
                    )
                
                logger.info(f"Starting extraction - Language: {language}, Max Queries: {max_queries_label}, Max Results: {max_results_label}")
                st.info(f"Starting extraction with the following settings: Language: {language}, Max Queries: {max_queries_label}, Max Results per Query: {max_results_label}")
                
                try:
                    # Run web extraction with trace ID
                    extraction_trace_id = extraction_trace.id if extraction_trace else None
                    run_web_extraction(max_queries, max_results, language, trace_id=extraction_trace_id)
                    
                    # Track successful extraction
                    if extraction_trace:
                        extraction_trace.update(
                            output={"status": "completed"},
                            status="success"
                        )
                        
                except Exception as e:
                    # Track failed extraction
                    if extraction_trace:
                        extraction_trace.update(
                            output={"status": "error", "error": str(e)},
                            status="error"
                        )
                    raise
        
        elif app_mode == "Database Setup":
            initialization_page()
        
        # End dashboard load time tracking with success
        if langfuse and trace_id:
            end_dashboard_load(app_mode, success=True)
            
    except Exception as e:
        # End dashboard load time tracking with failure
        if langfuse and trace_id:
            end_dashboard_load(app_mode, success=False, metadata={"error": str(e)})
        raise
    finally:
        # End the page view tracking
        if langfuse and trace_id:
            end_page_view(app_mode)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.info(
        "ABS Initiative Metadata Explorer allows you to analyze and visualize "
        "data related to the ABS Capacity Development Initiative across various "
        "sources, languages, and contexts."
    )

if __name__ == "__main__":
    main()