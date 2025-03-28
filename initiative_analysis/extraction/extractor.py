import os
import time
import logging
import traceback
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from urllib.parse import urlparse

from database.connection import get_db_connection
from database.operations import store_extract_data
from config.settings import SERPER_API_KEY, MAX_WORKERS
from extraction.search import Search  # Import the Search class
from extraction.content import Processing  # Import the Processing class

logger = logging.getLogger("WebExtractor")

class WebExtractor:
    """Class for extracting web content related to initiatives."""
    
    def __init__(self, 
                search_api_key=None,
                max_workers=5,
                language="English"):
        """Initialize the WebExtractor with language-specific settings."""
        self.search_api_key = search_api_key or SERPER_API_KEY
        self.max_workers = max_workers or MAX_WORKERS
        self.language = language
        
        # Initialize helper classes
        self.searcher = Search()  # Instance of Search class
        self.processor = Processing()  # Instance of Processing class
        
        # Configure initiative names and search parameters
        self.configure_initiatives()
        
        logger.info(f"Initialized WebExtractor for {language} with {max_workers} workers")

    def configure_initiatives(self):
        """Configure the specific ABS initiative names to search for in different languages."""
        # [Keep existing configure_initiatives() implementation exactly as is]
        
    def extract_web_content(self, max_queries=None, max_results_per_query=None) -> Dict:
        """
        Main method to extract web content based on search queries.
        Now uses methods from Search and Processing classes.
        """
        logger = logging.getLogger(__name__)
        
        # Create formatted header for console output
        header = f"\n{'='*60}\n{' '*20}WEB CONTENT EXTRACTION\n{'='*60}"
        logger.info("Starting web content extraction process")
        print(header)
        
        # Track timing for performance analysis
        start_time = time.time()
        
        try:
            # First, get all existing URLs from the database to avoid re-processing
            existing_urls = set()
            try:
                conn = get_db_connection()
                cursor = conn.cursor()
                cursor.execute("SELECT link FROM content_sources")
                rows = cursor.fetchall()
                for row in rows:
                    existing_urls.add(row[0])
                cursor.close()
                conn.close()
                
                if not existing_urls:
                    logger.warning("Database query succeeded but returned no URLs - treating all content as new")
                    print(f"\n⚠️ Warning: No existing URLs found in database - treating all content as new")
                else:
                    logger.info(f"Database: Loaded {len(existing_urls)} existing URLs")
                    print(f"\n📊 Database: Loaded {len(existing_urls)} existing URLs")
                    
            except Exception as e:
                logger.error(f"Database error: Failed to fetch existing URLs - {str(e)}")
                print(f"\n⚠️ Warning: Could not fetch existing URLs from database")
                print(f"   Error details: {str(e)}")
            
            # Generate search queries using Search class method
            queries = self.searcher.generate_search_queries(max_queries)
            
            # Search the web and collect results
            all_results = []
            processed_urls = set()
            skipped_total = 0
            logger.info("Starting with empty processed_urls set - no in-memory URL history")
            print("🔄 Starting with fresh URL tracking - no previous history carried over")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i, query in enumerate(queries):
                    query_preview = query[:50] + "..." if len(query) > 50 else query
                    logger.info(f"Query {i+1}/{len(queries)}: '{query_preview}'")
                    print(f"\n📝 QUERY {i+1}/{len(queries)}:")
                    print(f"   '{query_preview}'")
                    
                    # Use Search class method for web search
                    results = self.searcher.search_web(query, num_results=max_results_per_query or 20)
                    logger.info(f"Query {i+1} returned {len(results)} results")
                    print(f"   Found {len(results)} search results")
                    
                    future_to_result = {}
                    skipped_urls = 0
                    new_urls = 0
                    
                    for j, result in enumerate(results):
                        url = result.get("link")
                        
                        if not url:
                            logger.warning(f"Result {j+1} has no URL, skipping")
                            continue
                            
                        # Use Search class method for domain filtering
                        domain = urlparse(url).netloc.lower()
                        if self.searcher._is_junk_domain(domain):
                            logger.info(f"Skipping junk domain: {domain}")
                            skipped_urls += 1
                            continue
                            
                        if url in processed_urls or url in existing_urls:
                            logger.info(f"Skipping URL already processed: {url}")
                            skipped_urls += 1
                            skipped_total += 1
                            continue
                        
                        logger.info(f"Processing new URL: {url}")
                        processed_urls.add(url)
                        new_urls += 1
                        
                        # Use Processing class method for result processing
                        future = executor.submit(
                            self.processor.process_search_result, 
                            result, i, j, processed_urls
                        )
                        future_to_result[future] = result
                    
                    logger.info(f"Query {i+1}: Processing {new_urls} new URLs, skipped {skipped_urls}")
                    print(f"   Processing {new_urls} new URLs")
                    if skipped_urls > 0:
                        print(f"   Skipped {skipped_urls} already processed URLs")
                    
                    for future in as_completed(future_to_result):
                        result_data = future.result()
                        
                        if result_data:
                            # Use Processing class method for relevance check
                            if self.processor._contains_relevant_content(result_data):
                                store_extract_data([result_data])
                                all_results.append(result_data)
                            else:
                                logger.info(f"Skipping irrelevant content from {result_data.get('link')}")
                    
                    time.sleep(1)
                
                all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                duration = time.time() - start_time
                
                if len(all_results) > 0:
                    status = "success"
                    status_message = f"Successfully found {len(all_results)} new results"
                else:
                    if skipped_total > 0:
                        if len(existing_urls) == 0:
                            status = "warning"
                            status_message = f"No results found. Skipped {skipped_total} URLs but database appears empty."
                        else:
                            status = "success"
                            status_message = f"No new content found. All {skipped_total} URLs were already processed."
                    else:
                        status = "warning"
                        status_message = f"No results found in search. Try different search queries."
                
                logger.info(f"Web content extraction completed in {duration:.2f} seconds. Status: {status}. {status_message}")
                
                return {
                    "status": status,
                    "message": status_message,
                    "execution_time": f"{duration:.2f} seconds",
                    "results": all_results,
                    "skipped_urls": skipped_total,
                    "database_urls_count": len(existing_urls)
                }
                
        except Exception as e:
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            logger.error(f"Critical error in extraction process: {error_type}: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "execution_time": f"{time.time() - start_time:.2f} seconds",
                "results": [],
                "error": str(e),
                "error_type": error_type
            }

    def run(self, max_queries=None, max_results_per_query=None) -> Dict:
        """
        Run the web extraction process and return results in a structured format.
        """
        logger.info("Running web extractor")
        
        try:
            start_time = time.time()
            extraction_result = self.extract_web_content(max_queries, max_results_per_query)
            end_time = time.time()
            execution_time = end_time - start_time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            output = {
                "status": extraction_result.get("status", "error"),
                "message": extraction_result.get("message", "Unknown result"),
                "timestamp": timestamp,
                "execution_time": f"{execution_time:.2f} seconds",
                "result_count": len(extraction_result.get("results", [])),
                "skipped_urls": extraction_result.get("skipped_urls", 0),
                "results": extraction_result.get("results", [])
            }
            
            if output["status"] == "success":
                if output["result_count"] > 0:
                    logger.info(f"Web extractor completed successfully in {execution_time:.2f} seconds with {output['result_count']} results")
                    print(f"Web extraction completed successfully with {output['result_count']} new results")
                else:
                    logger.info(f"Web extractor completed successfully in {execution_time:.2f} seconds. No new content found.")
                    print(f"Web extraction completed successfully. {output['message']}")
            else:
                logger.warning(f"Web extractor completed with status '{output['status']}' in {execution_time:.2f} seconds: {output['message']}")
                print(f"Web extraction completed with status: {output['status']}. {output['message']}")
            
            return output
            
        except Exception as e:
            error_traceback = traceback.format_exc()
            logger.error(f"Error running web extractor: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            
            print("\n" + "!"*50)
            print(f"CRITICAL ERROR: {str(e)}")
            print(error_traceback)
            print("!"*50 + "\n")
            
            return {
                "status": "error",
                "message": f"Critical error: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": f"{time.time() - start_time:.2f} seconds",
                "error": str(e),
                "results": [],
                "result_count": 0
            }