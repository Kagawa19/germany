import os
import time
import requests
import logging
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Callable, Any
import traceback
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    
    def __init__(self, 
                use_ai=True, 
                openai_client=None, 
                generate_ai_summary_func=None, 
                extract_date_with_ai_func=None,
                prompt_path=None,
                max_workers=5,
                serper_api_key=None):
        """
        Initialize WebExtractor with optional AI capabilities.
        
        Args:
            use_ai: Whether to use AI features
            openai_client: OpenAI client instance
            generate_ai_summary_func: Function to generate AI summaries
            extract_date_with_ai_func: Function to extract dates with AI
            prompt_path: Path to the prompt file
            max_workers: Number of workers for ThreadPoolExecutor
            serper_api_key: API key for Serper web search
        """
        # Load environment variables
        load_dotenv()
        
        self.use_ai = use_ai and openai_client is not None
        self.openai_client = openai_client
        self.generate_ai_summary = generate_ai_summary_func
        self.extract_date_with_ai = extract_date_with_ai_func
        
        # Set prompt path, with a default if not provided
        self.prompt_path = prompt_path or os.path.join('prompts', 'extract.txt')
        
        # Set max_workers
        self.max_workers = max_workers
        
        # Set Serper API key
        self.serper_api_key = serper_api_key or os.getenv('SERPER_API_KEY') 
        
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

    def extract_date_from_content(self, html_content: str, url: str, soup: BeautifulSoup) -> Optional[str]:
        """Extract publication date from content using multiple strategies."""
        logger.info(f"Extracting date from webpage: {url}")
        
        # Try AI-based date extraction first if enabled
        if self.use_ai and self.extract_date_with_ai:
            try:
                ai_date = self.extract_date_with_ai(html_content, url)
                if ai_date:
                    logger.info(f"Found date using AI extraction: {ai_date}")
                    return ai_date
            except Exception as e:
                logger.warning(f"AI date extraction failed: {str(e)}")
        
        # Strategy 1: Look for meta tags with date information
        date_meta_tags = [
            "article:published_time", "og:published_time", "publication_date", 
            "date", "pubdate", "publishdate", "datePublished", "DC.date.issued",
            "published-date", "release_date", "created", "article.published", 
            "lastModified", "datemodified", "last-modified"
        ]
        
        for tag_name in date_meta_tags:
            meta_tag = soup.find("meta", property=tag_name) or soup.find("meta", attrs={"name": tag_name})
            if meta_tag and meta_tag.get("content"):
                date_str = meta_tag.get("content")
                logger.info(f"Found date in meta tag {tag_name}: {date_str}")
                return date_str
        
        # Strategy 2: Look for time elements
        time_elements = soup.find_all("time")
        for time_element in time_elements:
            if time_element.get("datetime"):
                date_str = time_element.get("datetime")
                logger.info(f"Found date in time element: {date_str}")
                return date_str
            elif time_element.text.strip():
                date_str = time_element.text.strip()
                logger.info(f"Found date in time element text: {date_str}")
                return date_str
        
        # Strategy 3: Check for data attributes
        date_attrs = ["data-date", "data-published", "data-timestamp"]
        for attr in date_attrs:
            elements = soup.find_all(attrs={attr: True})
            if elements:
                date_str = elements[0].get(attr)
                logger.info(f"Found date in {attr} attribute: {date_str}")
                return date_str
        
        # Strategy 4: Check for JSON-LD structured data
        script_tags = soup.find_all("script", {"type": "application/ld+json"})
        for script in script_tags:
            try:
                if script.string:
                    json_data = json.loads(script.string)
                    if isinstance(json_data, dict):
                        date_fields = ["datePublished", "dateCreated", "dateModified", "uploadDate"]
                        for field in date_fields:
                            if field in json_data:
                                date_str = json_data[field]
                                logger.info(f"Found date in JSON-LD data ({field}): {date_str}")
                                return date_str
                    elif isinstance(json_data, list):
                        for item in json_data:
                            if isinstance(item, dict):
                                for field in date_fields:
                                    if field in item:
                                        date_str = item[field]
                                        logger.info(f"Found date in JSON-LD data list ({field}): {date_str}")
                                        return date_str
            except Exception as e:
                logger.debug(f"Error parsing JSON-LD: {str(e)}")
                
        # Strategy 5: Look for common date patterns in the HTML
        date_patterns = [
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
            r'((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4})',
            r'(\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})',
            r'((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4})',
            r'Published\s*(?:on|date)?:?\s*([^<>]+\d{4})',
            r'Posted\s*(?:on)?:?\s*([^<>]+\d{4})',
            r'Date:?\s*([^<>]+\d{4})',
            r'(\d{4}-\d{2}-\d{2})',
            r'(\d{2}/\d{2}/\d{4})',
            r'(\d{2}\.\d{2}\.\d{4})',
            r'(\d{2}-\d{2}-\d{4})'
        ]
        
        # Check in the raw HTML
        for pattern in date_patterns:
            matches = re.search(pattern, html_content, re.IGNORECASE)
            if matches:
                date_str = matches.group(1)
                logger.info(f"Found date using regex pattern in HTML: {date_str}")
                return date_str
        
        # Strategy 6: Look for date classes
        date_classes = ["date", "published", "timestamp", "post-date", "article-date"]
        for cls in date_classes:
            elements = soup.find_all(class_=lambda c: c and cls in c.lower())
            if elements:
                for element in elements:
                    text = element.text.strip()
                    if text and re.search(r'\d{4}', text):  # Has a year
                        logger.info(f"Found date in element with class '{cls}': {text}")
                        return text
        
        logger.info("No date information found")
        return None

    def extract_title_from_content(self, soup: BeautifulSoup, url: str, search_result_title: str) -> str:
        """Extract the title from the webpage content."""
        logger.info(f"Extracting title from webpage: {url}")
        
        # Strategy 1: Look for title tag
        title_tag = soup.find("title")
        if title_tag and title_tag.text.strip():
            title_text = title_tag.text.strip()
            logger.info(f"Found title in title tag: {title_text}")
            return title_text
        
        # Strategy 2: Look for og:title meta tag
        og_title = soup.find("meta", property="og:title")
        if og_title and og_title.get("content"):
            title_text = og_title.get("content").strip()
            logger.info(f"Found title in og:title meta tag: {title_text}")
            return title_text
        
        # Strategy 3: Look for h1 tags
        h1_tags = soup.find_all("h1")
        if h1_tags and len(h1_tags) > 0:
            for h1 in h1_tags:
                if h1.text.strip():
                    title_text = h1.text.strip()
                    logger.info(f"Found title in h1 tag: {title_text}")
                    return title_text
        
        # Strategy 4: Use the search result title
        if search_result_title and search_result_title != "No title":
            logger.info(f"Using search result title: {search_result_title}")
            return search_result_title
        
        # Strategy 5: Use URL as last resort
        domain = url.split("//")[-1].split("/")[0]
        logger.info(f"Using domain as fallback title: {domain}")
        return domain

    def scrape_webpage(self, url: str, search_result_title: str) -> Tuple[str, str, str]:
        """Scrape content, title and date from a webpage."""
        logger.info(f"Scraping webpage: {url}")
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8'
        }
        
        try:
            scrape_start_time = time.time()
            logger.info(f"Sending HTTP request to {url}")
            
            # Set a longer timeout for PDFs and large documents
            is_pdf = url.lower().endswith('.pdf')
            timeout = 30 if is_pdf else 15
            
            response = requests.get(url, headers=headers, timeout=timeout)
            request_time = time.time() - scrape_start_time
            logger.info(f"Received response in {request_time:.2f} seconds. Status code: {response.status_code}")
            
            response.raise_for_status()  # Raise exception for 4XX/5XX status codes
            
            # Special handling for PDF documents
            if is_pdf:
                logger.info("PDF document detected, using title from URL")
                title = search_result_title or url.split('/')[-1].replace('-', ' ').replace('_', ' ')
                return f"PDF Document: {len(response.content)} bytes", title, None
            
            # Parse HTML
            logger.info("Parsing HTML content")
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title first before removing elements
            title = self.extract_title_from_content(soup, url, search_result_title)
            
            # Extract date before removing elements - pass the full HTML text for comprehensive search
            date = self.extract_date_from_content(response.text, url, soup)
            
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
            logger.info(f"Extracted title: {title}")
            logger.info(f"Extracted date: {date}")
            
            return content, title, date
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error scraping {url}: {str(e)}")
            return "", "", None
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}", exc_info=True)
            return "", "", None

    def process_search_result(self, result, query_index, result_index, search_terms, processed_urls):
        """Process a single search result in a thread-safe manner."""
        url = result.get("link")
        if not url or url in processed_urls:
            logger.info(f"Skipping already processed URL: {url}")
            return None
        
        # Skip certain file types
        if url.lower().endswith(('.jpg', '.jpeg', '.png', '.gif', '.svg', '.zip', '.exe', '.mp3', '.mp4')):
            logger.info(f"Skipping unsupported file type: {url}")
            return None
        
        processed_urls.add(url)
        logger.info(f"Processing result {result_index+1} from query {query_index+1}: {url}")
        
        print(f"\n  URL {len(processed_urls)}: {url}")
        
        # Check if URL is from a relevant domain
        is_org_domain = any(org.lower() in url.lower() for org in search_terms["organizations"])
        if is_org_domain:
            # Higher priority for organization websites
            matching_orgs = [org for org in search_terms["organizations"] if org.lower() in url.lower()]
            logger.info(f"URL from organization domain: {', '.join(matching_orgs)}")
            print(f"  [HIGH PRIORITY] Organization domain: {matching_orgs[0]}")
            scrape_priority = True
            organization = matching_orgs[0] if matching_orgs else None
        else:
            # Skip if not directly related to Germany or development cooperation
            relevant_terms = ["german", "germany", "deutsch", "giz", "bmz", "kfw", "cooperation"]
            if not any(term.lower() in url.lower() for term in relevant_terms):
                logger.info(f"Skipping less relevant URL: {url}")
                print(f"  [SKIPPED] URL not relevant to search criteria")
                return None
            matching_terms = [term for term in relevant_terms if term.lower() in url.lower()]
            logger.info(f"URL contains relevant terms: {', '.join(matching_terms)}")
            print(f"  [RELEVANT] Contains terms: {', '.join(matching_terms)}")
            scrape_priority = False
            organization = None
        
        # Scrape the webpage
        print(f"  Scraping content... ", end="", flush=True)
        content, title, date = self.scrape_webpage(url, result.get("title", "No title"))
        
        if content:
            # Use AI to generate summary if enabled
            summary = None
            if self.use_ai and self.generate_ai_summary and len(content) > 100:
                try:
                    summary = self.generate_ai_summary(content)
                    if summary:
                        logger.info(f"Generated AI summary ({len(summary)} chars)")
                except Exception as e:
                    logger.warning(f"Failed to generate AI summary: {str(e)}")
            
            # Use snippet as summary if no AI summary
            if not summary:
                summary = result.get("snippet", "")
            
            # Set a default theme
            theme = "Environmental Sustainability"
            
            result_data = {
                "title": title,
                "link": url,
                "date": date,
                "summary": summary,
                "snippet": result.get("snippet", ""),
                "source": result.get("source", ""),
                "content": content,
                "theme": theme,
                "organization": organization,
                "priority": scrape_priority
            }
            
            logger.info(f"Added result for {result_data['title']} ({len(content)} chars)")
            print(f"SUCCESS ({len(content)} chars)")
            print(f"  Title: {result_data['title']}")
            print(f"  Date: {result_data['date'] if result_data['date'] else 'Not found'}")
            return result_data
        else:
            logger.warning(f"No content extracted from {url}")
            print(f"FAILED (No content extracted)")
            return None

    def extract_web_content(self, max_queries=None, max_results_per_query=None) -> List[Dict]:
        """Main method to extract web content based on the prompt file.
        
        Args:
            max_queries: Maximum number of queries to process (None for all)
            max_results_per_query: Maximum results to extract per query (None for default)
        """
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
            
            # Limit number of queries if specified
            if max_queries is not None:
                queries = queries[:max_queries]
            
            # Search the web and collect results
            all_results = []
            processed_urls = set()  # Track URLs to avoid duplicates
            logger.info(f"Processing {len(queries)} queries")
            
            print("\n" + "-"*50)
            print(f"PROCESSING {len(queries)} SEARCH QUERIES")
            print("-"*50 + "\n")
            
            results_per_query = 3 if max_results_per_query is None else max_results_per_query
            
            # Use ThreadPoolExecutor for parallel processing of search results
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i, query in enumerate(queries):
                    query_preview = query[:50] + "..." if len(query) > 50 else query
                    logger.info(f"Processing query {i+1}/{len(queries)}: '{query_preview}'")
                    
                    print(f"\nQUERY {i+1}/{len(queries)}: '{query_preview}'")
                    
                    results = self.search_web(query, num_results=results_per_query)
                    logger.info(f"Query {i+1} returned {len(results)} results")
                    
                    # Submit all results for parallel processing
                    future_to_result = {}
                    for j, result in enumerate(results):
                        # Clone the processed_urls set to avoid race conditions
                        # Each thread will check against the current state but only the main thread will update it
                        if result.get("link") not in processed_urls:
                            future = executor.submit(
                                self.process_search_result, 
                                result, i, j, search_terms, processed_urls
                            )
                            future_to_result[future] = result
                    
                    # Collect results as they complete
                    for future in as_completed(future_to_result):
                        result_data = future.result()
                        if result_data:
                            all_results.append(result_data)
                    
                    # Add small delay between queries to be respectful
                    if i < len(queries) - 1:  # Don't delay after the last query
                        logger.info("Adding delay between queries (1 second)")
                        print("\n  Waiting 1 second before next query...")
                        time.sleep(1)
            
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
                    print(f"   Date: {result['date'] if result['date'] else 'Not found'}")
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

    def run(self, max_queries=None, max_results_per_query=None) -> Dict:
        """
        Run the web extraction process, store results in database, and return results in a structured format.
        
        Args:
            max_queries: Maximum number of queries to process (None for all)
            max_results_per_query: Maximum results to extract per query (None for default)
        """
        logger.info("Running web extractor")
        
        try:
            start_time = time.time()
            
            # Run the extraction process with the parameters
            results = self.extract_web_content(max_queries, max_results_per_query)
        
            # Store results in database if available
            stored_ids = []
            
            from content_db import store_extract_data
            
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