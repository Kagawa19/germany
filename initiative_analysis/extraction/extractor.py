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

# Import Langfuse for monitoring (wrapped in try/except since it might not be available)
try:
    from monitoring.langfuse_client import get_langfuse_client
except ImportError:
    pass

logger = logging.getLogger("WebExtractor")

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
        
        # Initialize Langfuse client
        try:
            from monitoring.langfuse_client import get_langfuse_client
            self.langfuse = get_langfuse_client()
            logger.info("Langfuse client initialized for WebExtractor")
        except (ImportError, Exception) as e:
            logger.warning(f"Failed to initialize Langfuse client: {str(e)}")
            self.langfuse = None
        
        # Configure initiative names and search parameters
        self.configure_initiatives()
        
        logger.info(f"Initialized WebExtractor for {language} with {max_workers} workers")

    def configure_initiatives(self):
        """Configure the specific ABS initiative names to search for in different languages."""
        # [Keep existing configure_initiatives() implementation exactly as is]
        
    def extract_web_content(self, max_queries=None, max_results_per_query=None, trace_id=None, parent_span_id=None) -> Dict:
        """
        Main method to extract web content based on search queries.
        Now uses methods from Search and Processing classes.
        
        Args:
            max_queries: Maximum number of queries to run
            max_results_per_query: Maximum results per query  
            trace_id: Optional Langfuse trace ID for monitoring
            parent_span_id: Optional parent span ID for nested tracing
            
        Returns:
            Dictionary with extraction results
        """
        logger = logging.getLogger(__name__)
        
        # Get Langfuse client if trace_id is provided
        langfuse = getattr(self, 'langfuse', None)
        if not langfuse and trace_id:
            try:
                from monitoring.langfuse_client import get_langfuse_client
                langfuse = get_langfuse_client()
            except ImportError:
                logging.warning("Could not import Langfuse client - monitoring will be disabled")
        
        # Create formatted header for console output
        header = f"\n{'='*60}\n{' '*20}WEB CONTENT EXTRACTION\n{'='*60}"
        logger.info("Starting web content extraction process")
        print(header)
        
        # Track timing for performance analysis
        start_time = time.time()
        
        # Create a span for the extraction process
        content_extraction_span = None
        if langfuse and trace_id:
            content_extraction_span = langfuse.create_span(
                trace_id=trace_id,
                name="content_extraction",
                parent_span_id=parent_span_id,
                metadata={
                    "max_queries": max_queries,
                    "max_results_per_query": max_results_per_query,
                    "language": self.language
                }
            )
        
        try:
            # Database span for getting existing URLs
            db_span = None
            if langfuse and trace_id:
                db_span = langfuse.create_span(
                    trace_id=trace_id,
                    name="fetch_existing_urls",
                    parent_span_id=content_extraction_span.id if content_extraction_span else parent_span_id,
                    metadata={"operation": "database_query"}
                )
            
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
                    
                    # Update database span
                    if db_span:
                        db_span.update(
                            output={"status": "warning", "message": "No existing URLs found"}
                        )
                else:
                    logger.info(f"Database: Loaded {len(existing_urls)} existing URLs")
                    print(f"\n📊 Database: Loaded {len(existing_urls)} existing URLs")
                    
                    # Update database span
                    if db_span:
                        db_span.update(
                            output={"status": "success", "url_count": len(existing_urls)}
                        )
                    
            except Exception as e:
                logger.error(f"Database error: Failed to fetch existing URLs - {str(e)}")
                print(f"\n⚠️ Warning: Could not fetch existing URLs from database")
                print(f"   Error details: {str(e)}")
                
                # Update database span with error
                if db_span:
                    db_span.update(
                        output={"status": "error", "error": str(e)},
                        status="error"
                    )
            
            # End database span
            if db_span:
                db_span.end()
            
            # Search queries span
            queries_span = None
            if langfuse and trace_id:
                queries_span = langfuse.create_span(
                    trace_id=trace_id,
                    name="generate_search_queries",
                    parent_span_id=content_extraction_span.id if content_extraction_span else parent_span_id,
                    metadata={"max_queries": max_queries}
                )
            
            # Generate search queries using Search class method
            queries = self.searcher.generate_search_queries(max_queries)
            
            # Update queries span
            if queries_span:
                queries_span.update(
                    output={"query_count": len(queries), "queries": queries[:5]}  # Only log first 5 queries
                )
                queries_span.end()
            
            # Search and processing span
            search_span = None
            if langfuse and trace_id:
                search_span = langfuse.create_span(
                    trace_id=trace_id,
                    name="search_and_process",
                    parent_span_id=content_extraction_span.id if content_extraction_span else parent_span_id,
                    metadata={"query_count": len(queries), "max_workers": self.max_workers}
                )
            
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
                    
                    # Create span for individual query
                    query_span = None
                    if langfuse and trace_id:
                        query_span = langfuse.create_span(
                            trace_id=trace_id,
                            name=f"query_{i+1}",
                            parent_span_id=search_span.id if search_span else (content_extraction_span.id if content_extraction_span else parent_span_id),
                            metadata={"query": query, "query_index": i+1, "total_queries": len(queries)}
                        )
                    
                    # Use Search class method for web search
                    results = self.searcher.search_web(query, num_results=max_results_per_query or 20)
                    logger.info(f"Query {i+1} returned {len(results)} results")
                    print(f"   Found {len(results)} search results")
                    
                    # Update query span with results
                    if query_span:
                        query_span.update(
                            output={"result_count": len(results)}
                        )
                    
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
                        # Pass trace information for monitoring
                        future = executor.submit(
                            self.processor.process_search_result, 
                            result, i, j, processed_urls,
                            trace_id=trace_id,
                            parent_span_id=query_span.id if query_span else None
                        )
                        future_to_result[future] = result
                    
                    logger.info(f"Query {i+1}: Processing {new_urls} new URLs, skipped {skipped_urls}")
                    print(f"   Processing {new_urls} new URLs")
                    if skipped_urls > 0:
                        print(f"   Skipped {skipped_urls} already processed URLs")
                    
                    # Update query span with processing info
                    if query_span:
                        query_span.update(
                            output={
                                "new_urls": new_urls,
                                "skipped_urls": skipped_urls
                            }
                        )
                    
                    # Results processing span
                    results_span = None
                    if langfuse and trace_id:
                        results_span = langfuse.create_span(
                            trace_id=trace_id,
                            name=f"process_results_query_{i+1}",
                            parent_span_id=query_span.id if query_span else (search_span.id if search_span else None),
                            metadata={"future_count": len(future_to_result)}
                        )
                    
                    processed_count = 0
                    for future in as_completed(future_to_result):
                        result_data = future.result()
                        
                        if result_data:
                            # Use Processing class method for relevance check
                            if self.processor._contains_relevant_content(result_data):
                                processed_count += 1
                                all_results.append(result_data)
                            else:
                                logger.info(f"Skipping irrelevant content from {result_data.get('link')}")
                    
                    # Update results span
                    if results_span:
                        results_span.update(
                            output={"processed_count": processed_count}
                        )
                        results_span.end()
                    
                    # End query span
                    if query_span:
                        query_span.end()
                    
                    time.sleep(1)
                
                # End search span
                if search_span:
                    search_span.update(
                        output={
                            "total_results": len(all_results),
                            "skipped_total": skipped_total
                        }
                    )
                    search_span.end()
                
                # Sort results by relevance
                all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                duration = time.time() - start_time
                
                # Determine overall status
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
                
                # Update content extraction span
                if content_extraction_span:
                    content_extraction_span.update(
                        output={
                            "status": status,
                            "message": status_message,
                            "duration_seconds": duration,
                            "result_count": len(all_results),
                            "skipped_urls": skipped_total
                        },
                        status="success" if status != "error" else "error"
                    )
                    content_extraction_span.end()
                
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
            
            # Update content extraction span with error
            if content_extraction_span:
                content_extraction_span.update(
                    output={
                        "status": "error",
                        "error": str(e),
                        "error_type": error_type
                    },
                    status="error"
                )
                content_extraction_span.end()
            
            return {
                "status": "error",
                "message": f"Error: {str(e)}",
                "execution_time": f"{time.time() - start_time:.2f} seconds",
                "results": [],
                "error": str(e),
                "error_type": error_type
            }
        
        finally:
            # Make sure to end the content extraction span if it's still open
            if content_extraction_span and not getattr(content_extraction_span, 'ended', False):
                try:
                    content_extraction_span.end()
                except Exception:
                    # Ignore any errors in closing the span
                    pass

    def run(self, max_queries=None, max_results_per_query=None, trace_id=None) -> Dict:
        """
        Run the web extraction process and return results in a structured format.
        
        Args:
            max_queries: Maximum number of queries to run
            max_results_per_query: Maximum results per query
            trace_id: Optional Langfuse trace ID for monitoring
        
        Returns:
            Dictionary with extraction results
        """
        # Get Langfuse client if trace_id is provided
        langfuse = None
        if trace_id:
            try:
                from monitoring.langfuse_client import get_langfuse_client
                langfuse = get_langfuse_client()
                self.langfuse = langfuse  # Store for use in other methods
            except ImportError:
                logging.warning("Could not import Langfuse client - monitoring will be disabled")
        
        logger.info("Running web extractor")
        
        try:
            start_time = time.time()
            
            # Create a span for the extract_web_content method
            extract_span = None
            if langfuse and trace_id:
                extract_span = langfuse.create_span(
                    trace_id=trace_id,
                    name="extract_web_content",
                    metadata={
                        "max_queries": max_queries,
                        "max_results_per_query": max_results_per_query,
                        "language": self.language
                    }
                )
            
            # Pass trace_id to extract_web_content
            extraction_result = self.extract_web_content(
                max_queries=max_queries, 
                max_results_per_query=max_results_per_query,
                trace_id=trace_id,
                parent_span_id=extract_span.id if extract_span else None
            )
            
            # End the extract span
            if extract_span:
                extract_span.update(
                    output={
                        "status": extraction_result.get("status", "unknown"),
                        "result_count": len(extraction_result.get("results", [])),
                    }
                )
                extract_span.end()
            
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
            
            # Create a metrics span for performance statistics
            metrics_span = None
            if langfuse and trace_id:
                metrics_span = langfuse.create_span(
                    trace_id=trace_id,
                    name="performance_metrics",
                    metadata={
                        "execution_time_seconds": execution_time,
                        "result_count": output["result_count"],
                        "skipped_urls": output["skipped_urls"]
                    }
                )
                
                # End metrics span
                if metrics_span:
                    metrics_span.end()
            
            # Log quality score if results were found
            if output["result_count"] > 0 and langfuse and trace_id:
                langfuse.score(
                    trace_id=trace_id,
                    name="extraction_success_rate",
                    value=min(output["result_count"] / max(1, (output["result_count"] + output["skipped_urls"])) * 10, 10),
                    comment=f"Extraction success rate: {output['result_count']} results from {output['result_count'] + output['skipped_urls']} potential URLs"
                )
            
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
            
            # Log error in Langfuse
            if langfuse and trace_id:
                error_span = langfuse.create_span(
                    trace_id=trace_id,
                    name="extraction_error",
                    metadata={
                        "error": str(e),
                        "traceback": error_traceback[:1000]  # Truncate if too long
                    }
                )
                error_span.update(status="error")
                error_span.end()
            
            return {
                "status": "error",
                "message": f"Critical error: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "execution_time": f"{time.time() - start_time:.2f} seconds",
                "error": str(e),
                "results": [],
                "result_count": 0
            }