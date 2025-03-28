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

logger = logging.getLogger("WebExtractor")

class WebExtractor:
    """Class for extracting web content related to initiatives."""
    
    def __init__(self, 
                search_api_key=None,
                max_workers=5,
                language="English"):
        """Initialize the WebExtractor with language-specific settings."""
        # Implementation goes here
        pass
        
    def configure_initiatives(self):
        """Configure the specific ABS initiative names to search for in different languages."""
        
        # Define all ABS initiative names in different languages with expanded lists
        self.abs_names = {
            "English": [
                "ABS Capacity Development Initiative",
                "ABS CDI",
                "ABS Capacity Development Initiative for Africa",
                "ABS Initiative",
                "Access and Benefit Sharing Initiative",
                "Access and Benefit Sharing Capacity Development Initiative",
                "ABS Knowledge Hub",
                "African ABS Initiative",
                "International ABS Initiative",
                "ABS Implementation",
                "ABS for Development",
                "ABS Support",
                "ABS Resources",
                "Capacity Building in Access and Benefit Sharing",
                "Biodiversity Access and Benefit Sharing Program"
            ],
            "German": [
                "Initiative für Zugang und Vorteilsausgleich",
                "ABS-Kapazitätenentwicklungsinitiative für Afrika",
                "ABS-Initiative",
                "Initiative für biologische Vielfalt",
                "Zugangs- und Vorteilsausgleichsinitiative",
                "Kapazitätsentwicklung für ABS",
                "Afrikanische ABS-Initiative",
                "Internationale ABS-Initiative",
                "ABS-Implementierung",
                "Vorteilsausgleich Initiative",
                "ABS-Ressourcen",
                "ABS-Unterstützung",
                "Kapazitätsaufbau in Zugang und Vorteilsausgleich",
                "Biologische Vielfalt Zugang und Vorteilsausgleich Programm"
            ],
            "French": [
                "Initiative pour le renforcement des capacités en matière d'APA",
                "Initiative Accès et Partage des Avantages",
                "Initiative sur le développement des capacités pour l'APA",
                "Initiative de renforcement des capacités sur l'APA",
                "Initiative APA",
                "Initiative de développement des capacités en matière d'accès et de partage des avantages",
                "Pôle de connaissances APA",
                "Centre de renforcement des capacités sur l'APA",
                "Initiative APA Afrique",
                "Initiative Africaine APA",
                "APA pour le développement",
                "Mise en œuvre de l'APA",
                "Ressources APA",
                "Soutien APA",
                "Renforcement des capacités en accès et partage des avantages",
                "Programme de diversité biologique Accès et Partage des Avantages"
            ]
        }
        
        # Define common words to use for partial matching
        self.common_terms = {
            "English": [
                "ABS", "Initiative", "capacity", "development", "benefit", 
                "sharing", "access", "support", "resources", "implementation", 
                "genetic", "CBD", "GIZ", "knowledge", "program", "biodiversity"
            ],
            "German": [
                "ABS", "Initiative", "Kapazität", "Entwicklung", "Vorteil", 
                "Ausgleich", "Zugang", "Unterstützung", "Ressourcen", 
                "Implementierung", "genetisch", "CBD", "GIZ", "Wissen", 
                "Programm", "Biodiversität"
            ],
            "French": [
                "APA", "Initiative", "capacité", "développement", "avantage", 
                "partage", "accès", "soutien", "ressources", "mise en œuvre", 
                "génétique", "CBD", "GIZ", "connaissance", "programme", "biodiversité"
            ]
        }
        
        # Update related organizations
        self.related_orgs = [
            "GIZ", "BMZ", "SCBD", "CBD", "UNDP", "UNEP", 
            "African Union", "EU", "European Union", 
            "Swiss SECO", "Norwegian NORAD", 
            "COMIFAC", "SADC", "ECOWAS", "SIDA"
        ]
        
        # Expand context terms with more specific ABS-related terminology
        self.context_terms = {
            "English": [
                "biodiversity", "genetic resources", "traditional knowledge", 
                "Nagoya Protocol", "indigenous communities", "conservation", 
                "sustainable development", "bioprospecting", 
                "benefit-sharing mechanism", "biotrade", 
                "natural resources management", "capacity building", 
                "stakeholder engagement", "legal framework", 
                "access and benefit sharing", "biological diversity"
            ],
            "German": [
                "Biodiversität", "genetische Ressourcen", "traditionelles Wissen", 
                "Nagoya-Protokoll", "indigene Gemeinschaften", "Konservierung", 
                "nachhaltige Entwicklung", "Bioprospektierung", 
                "Vorteilsausgleichsmechanismus", "Biohandel", 
                "Naturressourcenmanagement", "Kapazitätsaufbau", 
                "Stakeholder-Engagement", "rechtlicher Rahmen", 
                "Zugang und Vorteilsausgleich", "biologische Vielfalt"
            ],
            "French": [
                "biodiversité", "ressources génétiques", "connaissances traditionnelles", 
                "Protocole de Nagoya", "communautés autochtones", "conservation", 
                "développement durable", "bioprospection", 
                "mécanisme de partage des avantages", "biocommerce", 
                "gestion des ressources naturelles", "renforcement des capacités", 
                "engagement des parties prenantes", "cadre juridique", 
                "accès et partage des avantages", "diversité biologique"
            ]
        }
        
        # Generate search queries
        self.search_queries = []
        
        # Get the names for the selected language
        language_names = self.abs_names.get(self.language, self.abs_names["English"])
        
        # Also include English names regardless of language to maximize results
        if self.language != "English":
            all_names = language_names + self.abs_names["English"]
        else:
            all_names = language_names
        
        # Create simple search queries for each initiative name
        for name in all_names:
            self.search_queries.append(name)
        
        logger.info(f"Generated {len(self.search_queries)} search queries for ABS initiatives in {self.language}")
        
    def extract_web_content(self, max_queries=None, max_results_per_query=None) -> Dict:
        """
        Main method to extract web content based on search queries.
        Enhanced with improved error handling and filtering of irrelevant content.
        
        Args:
            max_queries: Maximum number of queries to process (None for all)
            max_results_per_query: Maximum results to extract per query (None for default)
            
        Returns:
            Dictionary with status, results, and metadata
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
                # Get database connection
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Query all existing URLs
                cursor.execute("SELECT link FROM content_data")
                rows = cursor.fetchall()
                
                # Process all rows and add to set
                for row in rows:
                    existing_urls.add(row[0])
                    
                cursor.close()
                conn.close()
                
                # CRITICAL FIX: Verify we actually have data
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
            
            # Generate search queries
            queries = self.generate_search_queries(max_queries)
            
            # Search the web and collect results
            all_results = []
            processed_urls = set()  # IMPORTANT: Start with a fresh empty set each time
            skipped_total = 0      # Count of all skipped URLs
            
            # Add this line to explicitly log that we're starting with a fresh set
            logger.info("Starting with empty processed_urls set - no in-memory URL history")
            print("🔄 Starting with fresh URL tracking - no previous history carried over")
            
            # Use ThreadPoolExecutor for parallel processing of search results
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i, query in enumerate(queries):
                    query_preview = query[:50] + "..." if len(query) > 50 else query
                    logger.info(f"Query {i+1}/{len(queries)}: '{query_preview}'")
                    
                    print(f"\n📝 QUERY {i+1}/{len(queries)}:")
                    print(f"   '{query_preview}'")
                    
                    # Search the web
                    results = self.search_web(query, num_results=max_results_per_query or 20)
                    logger.info(f"Query {i+1} returned {len(results)} results")
                    print(f"   Found {len(results)} search results")
                    
                    # Submit all results for parallel processing, skipping already processed URLs
                    future_to_result = {}
                    skipped_urls = 0
                    new_urls = 0
                    
                    for j, result in enumerate(results):
                        url = result.get("link")
                        
                        # Skip if URL is invalid
                        if not url:
                            logger.warning(f"Result {j+1} has no URL, skipping")
                            continue
                            
                        # Enhanced URL filtering: Skip known junk domains
                        domain = urlparse(url).netloc.lower()
                        if self._is_junk_domain(domain):
                            logger.info(f"Skipping junk domain: {domain}")
                            skipped_urls += 1
                            continue
                            
                        # Note: We check if URL is in processed_urls BEFORE we add it to the set
                        if url in processed_urls or url in existing_urls:
                            logger.info(f"Skipping URL already processed: {url}")
                            skipped_urls += 1
                            skipped_total += 1
                            continue
                        
                        # Important: Only mark URL as processed AFTER we've decided to process it
                        logger.info(f"Processing new URL: {url}")
                        processed_urls.add(url)
                        new_urls += 1
                        
                        # Submit for processing - INCLUDE the processed_urls parameter
                        future = executor.submit(
                            self.process_search_result, 
                            result, i, j, processed_urls
                        )
                        future_to_result[future] = result
                    
                    logger.info(f"Query {i+1}: Processing {new_urls} new URLs, skipped {skipped_urls}")
                    print(f"   Processing {new_urls} new URLs")
                    if skipped_urls > 0:
                        print(f"   Skipped {skipped_urls} already processed URLs")
                    
                    # Immediately store results as they are processed
                    for future in as_completed(future_to_result):
                        result_data = future.result()
                        
                        if result_data:
                            # Filter out junk content before storing
                            if self._contains_relevant_content(result_data):
                                # Store results immediately after processing
                                store_extract_data([result_data])
                                all_results.append(result_data)
                            else:
                                logger.info(f"Skipping irrelevant content from {result_data.get('link')}")
                    
                    # Add small delay between queries to be respectful
                    time.sleep(1)
                
                # Sort results by relevance score (descending)
                all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
                
                duration = time.time() - start_time
                
                # Determine status based on results
                if len(all_results) > 0:
                    status = "success"
                    status_message = f"Successfully found {len(all_results)} new results"
                else:
                    # No results but all URLs were already processed - this is still a success
                    # But double check that the database has records if we skipped URLs
                    if skipped_total > 0:
                        if len(existing_urls) == 0:
                            status = "warning"
                            status_message = f"No results found. Skipped {skipped_total} URLs but database appears empty."
                        else:
                            status = "success"
                            status_message = f"No new content found. All {skipped_total} URLs were already processed."
                    else:
                        # No results and no skipped URLs - this might indicate a problem
                        status = "warning"
                        status_message = f"No results found in search. Try different search queries."
                
                logger.info(f"Web content extraction completed in {duration:.2f} seconds. Status: {status}. {status_message}")
                
                # Return a dictionary with status and results
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
            
            # Return error information
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
        
        Args:
            max_queries: Maximum number of queries to process
            max_results_per_query: Maximum results to extract per query
            
        Returns:
            Dictionary with extraction results and metadata
        """
        logger.info("Running web extractor")
        
        try:
            start_time = time.time()
            
            # Run the extraction process with the parameters
            extraction_result = self.extract_web_content(max_queries, max_results_per_query)
            
            # Calculate timing and prepare output
            end_time = time.time()
            execution_time = end_time - start_time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Build the output dictionary
            output = {
                "status": extraction_result.get("status", "error"),
                "message": extraction_result.get("message", "Unknown result"),
                "timestamp": timestamp,
                "execution_time": f"{execution_time:.2f} seconds",
                "result_count": len(extraction_result.get("results", [])),
                "skipped_urls": extraction_result.get("skipped_urls", 0),
                "results": extraction_result.get("results", [])
            }
            
            # Log completion with appropriate message
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