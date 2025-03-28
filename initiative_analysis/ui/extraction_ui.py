import streamlit as st
import pandas as pd
import time
import os
import logging
from dotenv import load_dotenv
from extraction.extractor import WebExtractor
from database.operations import store_extract_data, fetch_data, create_schema
from database.connection import get_sqlalchemy_engine
from config.settings import SERPER_API_KEY
from sqlalchemy import text

logger = logging.getLogger("ExtractionUI")

import logging
import pandas as pd
import streamlit as st
from sqlalchemy.sql import text

def run_web_extraction(max_queries=None, max_results_per_query=None, language="English"):
    """
    Run web extraction process with configurable parameters.
    
    Args:
        max_queries: Maximum number of queries to run
        max_results_per_query: Maximum results per query
        language: Language for search queries (English, German, French)
    """
    # Log the start of the extraction process with detailed parameters
    logging.info(
        f"Starting web extraction process | "
        f"Max Queries: {max_queries or 'Unlimited'} | "
        f"Max Results Per Query: {max_results_per_query or 'Unlimited'} | "
        f"Language: {language}"
    )
    
    # Print user-friendly startup message
    print(f"🌐 Initiating web content extraction...")
    print(f"   Configuration:")
    print(f"   - Max Queries: {max_queries or 'Unlimited'}")
    print(f"   - Max Results Per Query: {max_results_per_query or 'Unlimited'}")
    print(f"   - Language: {language}")

    # Use Streamlit spinner for visual feedback
    with st.spinner(f"Extracting web content in {language}... This may take a few minutes."):
        # Create progress bar
        progress_bar = st.progress(0)

        # Log extractor initialization
        logging.info(f"Initializing WebExtractor with language: {language}")
        print(f"🔍 Initializing web content extractor for {language} content...")

        # Initialize WebExtractor with language parameter
        extractor = WebExtractor(
            search_api_key=SERPER_API_KEY,
            max_workers=10,
            language=language
        )

        # Store in session state for reference
        st.session_state['extractor_instance'] = extractor

        # Status placeholders
        status_placeholder = st.empty()
        status_placeholder.info(f"Searching the web for relevant content in {language}...")
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
                if results["results"]:
                    # Case 1: Success with results
                    result_count = len(results["results"])
                    
                    # Detailed logging of extraction results
                    logging.info(f"Web extraction successful. Found {result_count} results in {language}")
                    print(f"✅ Web extraction complete. {result_count} {language} items retrieved.")
                    
                    status_placeholder.info(f"Processing {result_count} {language} items...")
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

                        # Add language to each result
                        for result in results["results"]:
                            result["language"] = language

                        # Store extracted data
                        stored_ids = store_extract_data(results["results"])
                        stored_count = len(stored_ids)

                        # Final progress and logging
                        progress_bar.progress(100)
                        logging.info(f"Web extraction completed. Saved {stored_count}/{result_count} {language} items to database")
                        print(f"💾 Saved {stored_count} new {language} items to database.")
                        
                        status_placeholder.success(f"Saved {stored_count} new {language} items to database.")

                        # Display saved items if any
                        if stored_ids:
                            try:
                                engine = get_sqlalchemy_engine()
                                ids_str = ','.join(str(id) for id in stored_ids)
                                query = text(f"SELECT id, link, title, date, summary, themes, organization, sentiment, language, initiative FROM content_data WHERE id IN ({ids_str})")
                                saved_df = pd.read_sql(query, engine)

                                if not saved_df.empty:
                                    st.subheader(f"Newly Extracted {language} Content")
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
                    message = results.get('message', 'No new content found')
                    skipped = results.get('skipped_urls', 0)
                    
                    logging.info(f"Web extraction successful: {message} in {language}")
                    print(f"✅ Web extraction complete. {message} in {language}")
                    
                    if skipped > 0:
                        print(f"⏭️ Skipped {skipped} already processed URLs")
                    
                    progress_bar.progress(100)
                    status_placeholder.success(f"Web extraction complete: {message} in {language}")
                    
                    # Still display latest content data
                    st.subheader("Latest Content Data")
                    st.dataframe(fetch_data())
                    
            elif results["status"] == "warning":
                # Case 3: Warning status - not an error but needs attention
                message = results.get('message', 'Warning during extraction')
                
                logging.warning(f"Web extraction warning: {message} in {language}")
                print(f"⚠️ Web extraction completed with warnings: {message} in {language}")
                
                progress_bar.progress(100)
                status_placeholder.warning(f"Web extraction warning: {message} in {language}")
                
                # Still display latest content data
                st.subheader("Latest Content Data")
                st.dataframe(fetch_data())
                
            else:
                # Case 4: True error
                error_msg = results.get('error', results.get('message', 'Unknown error'))
                logging.error(f"Web extraction failed: {error_msg} in {language}")
                progress_bar.progress(100)
                status_placeholder.error(f"Web extraction failed: {error_msg} in {language}")
                print(f"❌ Web extraction failed: {error_msg} in {language}")

        except Exception as e:
            # Catch and log any unexpected errors
            logging.exception(f"Exception during web extraction in {language}: {str(e)}")
            progress_bar.progress(100)
            status_placeholder.error(f"Web extraction failed: {str(e)}")
            print(f"❌ Critical error during web extraction in {language}: {str(e)}")


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