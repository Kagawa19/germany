import os
import time
import requests
import logging
from bs4 import BeautifulSoup
from dotenv import load_dotenv
# Import statement to add at the top of your file
from content_db import store_extract_data
from typing import Dict, List, Optional
import traceback
import json
from langchain_community.utilities import GoogleSerperAPIWrapper

# Configure logging with a custom formatter that includes more details
class CustomFormatter(logging.Formatter):
    """Custom formatter with color support for terminal"""
    grey = "\x1b[38;20m"
    blue = "\x1b[34;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: blue + format_str + reset,
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_extractor.log"),
    ]
)

# Add a separate stream handler with the custom formatter
console_handler = logging.StreamHandler()
console_handler.setFormatter(CustomFormatter())
console_handler.setLevel(logging.INFO)  # Set to DEBUG for even more verbose output

logger = logging.getLogger("WebExtractor")
logger.addHandler(console_handler)

class WebExtractor:
    """Class for extracting web content based on prompt files."""
    
    def __init__(self):
        """Initialize the WebExtractor with API keys from environment variables."""
        logger.info("Initializing WebExtractor")
        
        # Load environment variables
        load_dotenv()
        self.serper_api_key = os.getenv('SERPER_API_KEY')
        if not self.serper_api_key:
            logger.warning("SERPER_API_KEY not found in environment variables")
        else:
            logger.info("Successfully loaded SERPER_API_KEY")
            
        # Initialize other instance variables
        self.prompt_path = os.path.join('prompts', 'extract.txt')
        logger.info(f"Default prompt path set to: {self.prompt_path}")
        
        logger.info("WebExtractor initialized successfully")
        
    def read_prompt_file(self, file_path: Optional[str] = None) -> str:
        """Read the content of a prompt file."""
        path = file_path or self.prompt_path
        logger.info(f"Reading prompt file: {path}")
        
        try:
            with open(path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Successfully read prompt file ({len(content)} chars)")
            return content
        except FileNotFoundError:
            error_msg = f"Prompt file not found at path: {path}"
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        except Exception as e:
            error_msg = f"Error reading prompt file: {str(e)}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg)

    def extract_search_terms(self, prompt_content: str) -> Dict[str, List[str]]:
        """Extract search terms and categories from the prompt content."""
        logger.info("Extracting search terms from prompt content")
        
        search_terms = {
            "main_topic": [],
            "organizations": [],
            "focus_areas": []
        }
        
        # Extract main topic (simplified extraction)
        if "Germany's international cooperation efforts" in prompt_content:
            search_terms["main_topic"].append("Germany's international cooperation efforts environmental sustainability")
            logger.info("Extracted main topic: Germany's international cooperation efforts")
            
        # Extract organizations (looking for bullet points after organizations section)
        if "organizations and institutions" in prompt_content:
            logger.info("Found organizations section in prompt")
            org_section = prompt_content.split("organizations and institutions")[1].split("Return only")[0]
            for line in org_section.split("*"):
                if ":" in line:
                    org_text = line.split(":", 1)[1].strip()
                    if org_text:
                        # Split by comma to separate multiple organizations on a line
                        for org in org_text.split(","):
                            clean_org = org.strip()
                            if clean_org:
                                search_terms["organizations"].append(clean_org)
                                logger.info(f"Added organization: {clean_org}")
        
        # Extract focus areas
        logger.info("Extracting focus areas")
        focus_areas_section = prompt_content.split("Include results related to:")[1].split("Search for relevant information")[0]
        for line in focus_areas_section.split("*"):
            area = line.strip().strip("*").strip()
            if area and "(" not in area:  # Skip explanatory parts in parentheses
                search_terms["focus_areas"].append(area)
                logger.info(f"Added focus area: {area}")
        
        logger.info(f"Extracted {len(search_terms['main_topic'])} main topics, {len(search_terms['organizations'])} organizations, and {len(search_terms['focus_areas'])} focus areas")
        
        # Log complete terms for debugging
        for category, terms in search_terms.items():
            if terms:
                logger.info(f"{category}: {', '.join(terms)}")
                
        return search_terms

    def generate_search_queries(self, search_terms: Dict[str, List[str]]) -> List[str]:
        """Generate search queries based on extracted search terms."""
        logger.info("Generating search queries")
        queries = []
        
        main_topic = " ".join(search_terms["main_topic"])
        logger.info(f"Main topic for queries: {main_topic}")
        
        # Generate queries combining main topic with each focus area
        focus_queries_count = 0
        for focus in search_terms["focus_areas"]:
            query = f"{main_topic} {focus}"
            queries.append(query)
            focus_queries_count += 1
            logger.info(f"Added focus query: {query}")
        
        # Generate queries combining main topic with each organization
        org_queries_count = 0
        for org in search_terms["organizations"]:
            query = f"{org} environmental sustainability cooperation"
            queries.append(query)
            org_queries_count += 1
            logger.info(f"Added organization query: {query}")
            
            # Also combine organizations with focus areas (limited to prevent too many queries)
            combined_queries_count = 0
            for focus in search_terms["focus_areas"][:3]:  # Limit to first 3 focus areas
                query = f"{org} {focus}"
                queries.append(query)
                combined_queries_count += 1
                logger.info(f"Added combined query: {query}")
        
        logger.info(f"Generated {len(queries)} search queries: {focus_queries_count} focus-based, {org_queries_count} organization-based, {combined_queries_count} combined")
        
        return queries

    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """Search the web using the given query."""
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"Searching web for: '{query_preview}' (limit: {num_results} results)")
        
        if not self.serper_api_key:
            error_msg = "Cannot search web: SERPER_API_KEY not set"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        try:
            search_start_time = time.time()
            logger.info(f"Initializing GoogleSerperAPIWrapper with API key")
            serper = GoogleSerperAPIWrapper(
                serper_api_key=self.serper_api_key,
                k=num_results
            )
            logger.info(f"Sending search request to Serper API")
            results = serper.results(query)
            search_time = time.time() - search_start_time
            
            # Extract just the organic results
            organic_results = results.get("organic", [])
            logger.info(f"Received {len(organic_results)} results in {search_time:.2f} seconds")
            
            # Log result URLs for debugging
            for i, result in enumerate(organic_results):
                logger.info(f"Result {i+1}: {result.get('title', 'No title')} - {result.get('link', 'No link')}")
                
            return organic_results
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}", exc_info=True)
            return []

    def scrape_webpage(self, url: str) -> str:
        """Scrape content from a webpage."""
        logger.info(f"Scraping webpage: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        try:
            scrape_start_time = time.time()
            logger.info(f"Sending HTTP request to {url}")
            
            response = requests.get(url, headers=headers, timeout=15)
            request_time = time.time() - scrape_start_time
            logger.info(f"Received response in {request_time:.2f} seconds. Status code: {response.status_code}")
            
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            
            # Parse HTML
            logger.info("Parsing HTML content")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            logger.info("Removing script, style, and navigation elements")
            removed_elements = 0
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.extract()
                removed_elements += 1
            logger.info(f"Removed {removed_elements} elements from DOM")
            
            # Get text content
            content = soup.get_text(separator="\n")
            
            # Clean up the content (remove excessive whitespace)
            original_length = len(content)
            content = "\n".join(line.strip() for line in content.split("\n") if line.strip())
            cleaned_length = len(content)
            
            scrape_time = time.time() - scrape_start_time
            logger.info(f"Successfully scraped {len(content)} chars from {url} in {scrape_time:.2f} seconds (cleaned from {original_length} to {cleaned_length} chars)")
            
            return content
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error scraping {url}: {str(e)}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}", exc_info=True)
            return ""

    def extract_web_content(self) -> List[Dict]:
        """Main method to extract web content based on the prompt file."""
        logger.info("Starting web content extraction process")
        print("\n" + "="*50)
        print("STARTING WEB CONTENT EXTRACTION PROCESS")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        try:
            # Read the prompt file
            prompt_content = self.read_prompt_file()
            logger.info(f"Prompt content length: {len(prompt_content)} chars")
            
            # Extract search terms
            search_terms = self.extract_search_terms(prompt_content)
            
            # Generate search queries
            queries = self.generate_search_queries(search_terms)
            
            # Search the web and collect results
            all_results = []
            processed_urls = set()  # Track URLs to avoid duplicates
            logger.info(f"Processing {min(5, len(queries))} queries out of {len(queries)} generated")
            
            print("\n" + "-"*50)
            print(f"PROCESSING {min(5, len(queries))} SEARCH QUERIES")
            print("-"*50 + "\n")
            
            for i, query in enumerate(queries[:5]):  # Limit to first 5 queries for efficiency
                query_preview = query[:50] + "..." if len(query) > 50 else query
                logger.info(f"Processing query {i+1}/5: '{query_preview}'")
                
                print(f"\nQUERY {i+1}/5: '{query_preview}'")
                
                results = self.search_web(query, num_results=3)  # Limit to top 3 results per query
                logger.info(f"Query {i+1} returned {len(results)} results")
                
                for j, result in enumerate(results):
                    url = result.get("link")
                    if not url or url in processed_urls:
                        logger.info(f"Skipping already processed URL: {url}")
                        continue
                    
                    processed_urls.add(url)
                    logger.info(f"Processing result {j+1}/{len(results)} from query {i+1}: {url}")
                    
                    print(f"\n  URL {len(processed_urls)}: {url}")
                    
                    # Check if URL is from a relevant domain
                    is_org_domain = any(org.lower() in url.lower() for org in search_terms["organizations"])
                    if is_org_domain:
                        # Higher priority for organization websites
                        matching_orgs = [org for org in search_terms["organizations"] if org.lower() in url.lower()]
                        logger.info(f"URL from organization domain: {', '.join(matching_orgs)}")
                        print(f"  [HIGH PRIORITY] Organization domain: {matching_orgs[0]}")
                        scrape_priority = True
                    else:
                        # Skip if not directly related to Germany or development cooperation
                        relevant_terms = ["german", "germany", "deutsch", "giz", "bmz", "kfw", "cooperation"]
                        if not any(term.lower() in url.lower() for term in relevant_terms):
                            logger.info(f"Skipping less relevant URL: {url}")
                            print(f"  [SKIPPED] URL not relevant to search criteria")
                            continue
                        matching_terms = [term for term in relevant_terms if term.lower() in url.lower()]
                        logger.info(f"URL contains relevant terms: {', '.join(matching_terms)}")
                        print(f"  [RELEVANT] Contains terms: {', '.join(matching_terms)}")
                        scrape_priority = False
                    
                    # Scrape the webpage
                    print(f"  Scraping content... ", end="", flush=True)
                    content = self.scrape_webpage(url)
                    
                    if content:
                        result_data = {
                            "title": result.get("title", "No title"),
                            "link": url,
                            "snippet": result.get("snippet", ""),
                            "source": result.get("source", ""),
                            "content": content,
                            "priority": scrape_priority
                        }
                        
                        all_results.append(result_data)
                        logger.info(f"Added result {len(all_results)}: {result_data['title']} ({len(content)} chars)")
                        print(f"SUCCESS ({len(content)} chars)")
                        print(f"  Title: {result_data['title']}")
                    else:
                        logger.warning(f"No content extracted from {url}")
                        print(f"FAILED (No content extracted)")
                
                # Add small delay between queries to be respectful
                if i < len(queries[:5]) - 1:  # Don't delay after the last query
                    logger.info("Adding delay between queries (2 seconds)")
                    print("\n  Waiting 2 seconds before next query...")
                    time.sleep(2)
            
            # Sort results by priority
            logger.info("Sorting results by priority")
            all_results.sort(key=lambda x: x.get("priority", False), reverse=True)
            
            duration = time.time() - start_time
            logger.info(f"Web content extraction completed in {duration:.2f} seconds. Found {len(all_results)} results.")
            
            print("\n" + "="*50)
            print(f"EXTRACTION COMPLETED IN {duration:.2f} SECONDS")
            print(f"FOUND {len(all_results)} RESULTS")
            print("="*50 + "\n")
            
            # Log summary for each result
            if all_results:
                print("RESULTS SUMMARY:")
                for i, result in enumerate(all_results):
                    priority_status = "HIGH PRIORITY" if result.get("priority", False) else "Standard"
                    logger.info(f"Result {i+1}: {result['title']} - {priority_status} - Content length: {len(result['content'])} chars")
                    print(f"{i+1}. [{priority_status}] {result['title']}")
                    print(f"   URL: {result['link']}")
                    print(f"   Length: {len(result['content'])} chars")
                    print()
            
            return all_results
            
        except Exception as e:
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            logger.error(f"Error in extraction process: {error_type}: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            
            print("\n" + "!"*50)
            print(f"ERROR IN EXTRACTION PROCESS: {error_type}: {str(e)}")
            print("TRACEBACK:")
            print(error_traceback)
            print("!"*50 + "\n")
            
            return []
            
    def run(self) -> Dict:
        """Run the web extraction process, store results in database, and return results in a structured format."""
        logger.info("Running web extractor")
        
        try:
            start_time = time.time()
            
            # Run the extraction process
            results = self.extract_web_content()
            
            # Store results in database if available
            stored_ids = []
            if results:
                logger.info("Extraction successful, proceeding to database storage")
                try:
                    stored_ids = store_extract_data(results)
                    logger.info(f"Database storage completed with {len(stored_ids)} records")
                except Exception as db_error:
                    logger.error(f"Error storing data in database: {str(db_error)}", exc_info=True)
                    print(f"\nWARNING: Database storage failed: {str(db_error)}")
                    print("Continuing with extraction results only.")
            else:
                logger.warning("No extraction results available for database storage")
            
            # Calculate timing and prepare output
            end_time = time.time()
            execution_time = end_time - start_time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            output = {
                "status": "success" if results else "error",
                "timestamp": timestamp,
                "execution_time": f"{execution_time:.2f} seconds",
                "result_count": len(results),
                "database_stored_count": len(stored_ids),
                "database_stored_ids": stored_ids,
                "results": results
            }
            
            logger.info(f"Web extractor completed in {execution_time:.2f} seconds with {len(results)} results, {len(stored_ids)} stored in database")
            
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
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error": str(e),
                "results": [],
                "database_stored_count": 0,
                "database_stored_ids": []
            }

# Example usage:
if __name__ == "__main__":
    try:
        print("\n" + "="*50)
        print("STARTING WEB EXTRACTOR")
        print("="*50 + "\n")
        
        extractor = WebExtractor()
        results = extractor.run()
        
        print("\n" + "="*50)
        print("WEB EXTRACTOR COMPLETED")
        print(f"Status: {results.get('status')}")
        print(f"Total results: {results.get('result_count', 0)}")
        print(f"Execution time: {results.get('execution_time')}")
        print("="*50 + "\n")
        
    except Exception as e:
        print("\n" + "!"*50)
        print(f"UNHANDLED EXCEPTION: {str(e)}")
        traceback.print_exc()
        print("!"*50 + "\n")