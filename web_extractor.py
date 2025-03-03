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
import numpy as np
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
    """Class for extracting web content based on prompt files with embedding support."""
    
    def __init__(self, 
                use_ai=True, 
                openai_client=None, 
                generate_ai_summary_func=None, 
                extract_date_with_ai_func=None,
                create_embedding_func=None,
                semantic_search_func=None,
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
            create_embedding_func: Function to create embeddings for semantic understanding
            semantic_search_func: Function to search the database using embeddings
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
        self.create_embedding = create_embedding_func
        self.semantic_search_func = semantic_search_func
        
        # Set prompt path, with a default if not provided
        self.prompt_path = prompt_path or os.path.join('prompts', 'extract.txt')
        
        # Set max_workers
        self.max_workers = max_workers
        
        # Set Serper API key
        self.serper_api_key = serper_api_key or os.getenv('SERPER_API_KEY')
        
        # Check if embeddings are enabled
        self.use_embeddings = use_ai and create_embedding_func is not None
        
        # Print initialization status
        if self.use_embeddings:
            logger.info("Embeddings are enabled for enhanced semantic understanding")
            print("EMBEDDINGS: Enabled for enhanced semantic understanding")
        else:
            logger.info("Embeddings are disabled - embedding creation function not provided")
            print("EMBEDDINGS: Disabled")
            
        if self.semantic_search_func is not None:
            logger.info("Semantic search in database is enabled")
            print("SEMANTIC SEARCH: Enabled for database content")
        else:
            logger.info("Semantic search in database is disabled")
            print("SEMANTIC SEARCH: Disabled")
            
    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        """
        Calculate the cosine similarity between two vectors.
        
        Args:
            vec_a: First vector
            vec_b: Second vector
            
        Returns:
            Cosine similarity score (0-1)
        """
        if not vec_a or not vec_b:
            return 0.0
            
        try:
            vec_a = np.array(vec_a)
            vec_b = np.array(vec_b)
            
            dot_product = np.dot(vec_a, vec_b)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            
            if norm_a == 0 or norm_b == 0:
                return 0.0
                
            return dot_product / (norm_a * norm_b)
        except Exception as e:
            logger.error(f"Error calculating cosine similarity: {str(e)}")
            return 0.0
            
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
    
    def extract_prompt_context(self, prompt_content: str) -> dict:
        """
        Extract context from prompt content and generate embedding.
        
        Args:
            prompt_content: Content of the prompt file
            
        Returns:
            Dictionary with prompt content and its embedding
        """
        logger.info("Processing prompt and generating embedding")
        print("Processing prompt and generating embedding...")
        
        # Create basic context with the prompt text
        context = {
            "prompt_text": prompt_content,
            "embedding": None,
        }
        
        # Generate embedding for the entire prompt if enabled
        if self.use_embeddings and self.create_embedding:
            try:
                logger.info("Generating embedding for prompt")
                
                # Use the entire prompt text for embedding
                context["embedding"] = self.create_embedding(prompt_content)
                
                logger.info(f"Generated prompt embedding with {len(context['embedding'])} dimensions")
                print(f"Generated prompt embedding with {len(context['embedding'])} dimensions")
            except Exception as e:
                logger.error(f"Error generating prompt embedding: {str(e)}")
                print(f"Error generating prompt embedding: {str(e)}")
        
        return context

    def generate_search_queries(self, context: dict) -> List[Dict]:
        """
        Generate search queries from context dictionary with full embedding support.
        
        Args:
            context: Dictionary with prompt content and its embedding
            
        Returns:
            List of query dictionaries
        """
        logger.info("Generating search queries with embedding support")
        print("Using full prompt embedding for semantic search...")
        
        # Get the prompt text and embedding from the context
        prompt_text = context.get("prompt_text", "")
        prompt_embedding = context.get("embedding")
        
        # Create a single semantic query using the full prompt embedding
        semantic_query = {
            "query": "Extract environmental sustainability information",  # Human-readable query text
            "type": "semantic_prompt",
            "embedding": prompt_embedding  # Use the full embedding we generated
        }
        
        # Get entities if available for fallback queries
        entities = context.get("entities", [])
        
        # Create a list of queries with the semantic query first
        queries = [semantic_query]
        
        # Add a few entity-based queries as fallback
        if entities:
            for entity in entities[:3]:  # Limit to 3 entities
                entity_query = {
                    "query": f"{entity} sustainability",
                    "type": "entity_fallback",
                    "embedding": None  # We won't generate separate embeddings for these
                }
                queries.append(entity_query)
                logger.info(f"Added fallback query: {entity_query['query']}")
        
        logger.info(f"Generated {len(queries)} queries using prompt embedding")
        print(f"Generated {len(queries)} search queries")
        
        return queries

    def search_web(self, query_obj: Dict, num_results: int = 5) -> List[Dict]:
        """
        Search the web using the given query.
        Enhanced with semantic search capabilities for improved results.
        
        Args:
            query_obj: Dictionary containing query string and optional embedding
            num_results: Maximum number of results to retrieve
            
        Returns:
            List of dictionaries containing search results
        """
        query = query_obj["query"]
        embedding = query_obj.get("embedding")
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"Searching web for: '{query_preview}' (limit: {num_results} results)")
        
        # Combined results will hold both semantic and web search results
        combined_results = []
        
        # Try semantic search first if enabled and embedding is available
        if self.use_embeddings and embedding and self.semantic_search_func:
            try:
                logger.info(f"Performing semantic search using embedding")
                print(f"  Performing semantic search in database...")
                semantic_results = self.semantic_search_func(embedding, limit=num_results)
                
                if semantic_results and len(semantic_results) > 0:
                    logger.info(f"Found {len(semantic_results)} semantic search results")
                    print(f"  Found {len(semantic_results)} semantic search results")
                    
                    # Convert semantic results to web search format for consistency
                    for result in semantic_results:
                        combined_results.append({
                            "title": result.get("title", ""),
                            "link": result.get("link", ""),
                            "snippet": result.get("summary", ""),
                            "source": "database",
                            "semantic_score": result.get("similarity", 0),
                            "has_embedding": True
                        })
                    
                    # If we have enough semantic results, we can reduce web search
                    num_web_results = max(1, num_results - len(semantic_results))
                    logger.info(f"Will fetch {num_web_results} additional results from web search")
                else:
                    logger.info("No semantic search results found")
                    print("  No semantic search results found")
                    num_web_results = num_results
            except Exception as e:
                logger.error(f"Error in semantic search: {str(e)}")
                print(f"  Error in semantic search: {str(e)}")
                num_web_results = num_results
        else:
            # No semantic search possible
            num_web_results = num_results
            
        # Perform web search if needed
        if not self.serper_api_key:
            logger.error("Cannot search web: SERPER_API_KEY not set")
            print("  ERROR: SERPER_API_KEY not set")
            return combined_results  # Return any semantic results we might have
        
        try:
            search_start_time = time.time()
            print(f"  Searching web via Serper API...")
            
            serper = GoogleSerperAPIWrapper(
                serper_api_key=self.serper_api_key,
                k=num_web_results
            )
            
            web_results = serper.results(query)
            search_time = time.time() - search_start_time
            
            # Extract just the organic results
            organic_results = web_results.get("organic", [])
            logger.info(f"Received {len(organic_results)} web search results in {search_time:.2f} seconds")
            print(f"  Found {len(organic_results)} web search results in {search_time:.2f} seconds")
            
            # For web results, mark them accordingly
            for result in organic_results:
                result["source"] = "web"
                result["has_embedding"] = False
                combined_results.append(result)
            
            # Log result URLs for debugging
            for i, result in enumerate(organic_results):
                logger.info(f"Result {i+1}: {result.get('title', 'No title')} - {result.get('link', 'No link')}")
            
            # Sort results by relevance (semantic results first, then web results)
            combined_results.sort(key=lambda x: (
                0 if x.get("source") == "database" else 1,  # Database results first
                -x.get("semantic_score", 0)  # Then by semantic score (descending)
            ))
            
            return combined_results[:num_results]  # Return only the top results
                
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}", exc_info=True)
            print(f"  ERROR searching web: {str(e)}")
            return combined_results  # Return any semantic results we might have

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

    def process_search_result(self, result, query_index, result_index, context, processed_urls):
        """Process a single search result with embedding-based semantic relevance."""
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
        
        # Default values
        relevance_score = 0.5
        organization = None
        
        # Scrape the webpage
        print(f"  Scraping content... ", end="", flush=True)
        content, title, date = self.scrape_webpage(url, result.get("title", "No title"))
        
        if not content:
            logger.warning(f"No content extracted from {url}")
            print(f"FAILED (No content extracted)")
            return None
        
        # Generate embedding for the content if enabled
        content_embedding = None
        semantic_relevance = None
        
        # Use embeddings to calculate semantic relevance if available
        if self.use_embeddings and self.create_embedding and context.get("embedding"):
            try:
                logger.info(f"Generating embedding for content from {url}")
                
                # Create embedding from title and content
                embedding_text = f"{title}\n\n{content[:4000]}"  # First 4000 chars for embedding
                content_embedding = self.create_embedding(embedding_text)
                
                logger.info(f"Generated content embedding with {len(content_embedding)} dimensions")
                
                # Calculate semantic similarity to prompt context
                semantic_relevance = self.cosine_similarity(context["embedding"], content_embedding)
                logger.info(f"Semantic relevance to prompt: {semantic_relevance}")
                
                # Use semantic relevance as the main relevance score
                relevance_score = semantic_relevance
                
                if semantic_relevance > 0.7:
                    print(f"  [HIGH RELEVANCE] Semantic similarity: {semantic_relevance:.2f}")
                elif semantic_relevance > 0.5:
                    print(f"  [MEDIUM RELEVANCE] Semantic similarity: {semantic_relevance:.2f}")
                else:
                    print(f"  [LOW RELEVANCE] Semantic similarity: {semantic_relevance:.2f}")
                    
            except Exception as e:
                logger.warning(f"Failed to generate content embedding: {str(e)}")
                content_embedding = None
                print(f"  [WARNING] Failed to generate embedding: {str(e)}")
        else:
            # Fallback when embeddings are disabled
            # Check if URL is from a known organization
            entities = context.get("entities", [])
            is_org_domain = any(entity.lower() in url.lower() for entity in entities)
            
            if is_org_domain:
                matching_orgs = [org for org in entities if org.lower() in url.lower()]
                logger.info(f"URL from known entity domain: {', '.join(matching_orgs)}")
                print(f"  [HIGH PRIORITY] Known entity domain: {matching_orgs[0]}")
                relevance_score = 0.8
                organization = matching_orgs[0] if matching_orgs else None
            else:
                print(f"  [STANDARD] Processing with standard relevance (embeddings disabled)")
        
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
            "source": result.get("source", "web"),
            "content": content,
            "theme": theme,
            "organization": organization,
            "relevance_score": relevance_score,
            "semantic_relevance": semantic_relevance,
            "embedding": content_embedding
        }
        
        logger.info(f"Added result for {result_data['title']} ({len(content)} chars, relevance: {relevance_score:.2f})")
        print(f"SUCCESS ({len(content)} chars)")
        print(f"  Title: {result_data['title']}")
        print(f"  Date: {result_data['date'] if result_data['date'] else 'Not found'}")
        if semantic_relevance:
            print(f"  Semantic relevance: {semantic_relevance:.2f}")
        print(f"  Overall relevance: {relevance_score:.2f}")
        
        return result_data
        
    def extract_web_content(self, max_queries=None, max_results_per_query=None) -> List[Dict]:
        """
        Main method to extract web content based on the prompt file.
        Enhanced with embedding-based semantic understanding.
        
        Args:
            max_queries: Maximum number of queries to process (None for all)
            max_results_per_query: Maximum results to extract per query (None for default)
            
        Returns:
            List of dictionaries containing the extracted content
        """
        logger.info("Starting web content extraction process with embeddings")
        print("\n" + "="*50)
        print("STARTING WEB CONTENT EXTRACTION PROCESS")
        if self.use_embeddings:
            print("EMBEDDINGS ENABLED: Using semantic understanding")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        try:
            # Read the prompt file
            prompt_content = self.read_prompt_file()
            logger.info(f"Prompt content length: {len(prompt_content)} chars")
            
            # Extract context from prompt with embeddings
            context = self.extract_prompt_context(prompt_content)
            
            # Generate search queries
            queries = self.generate_search_queries(context)
            
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
                for i, query_obj in enumerate(queries):
                    query = query_obj["query"]
                    query_preview = query[:50] + "..." if len(query) > 50 else query
                    logger.info(f"Processing query {i+1}/{len(queries)}: '{query_preview}'")
                    
                    print(f"\nQUERY {i+1}/{len(queries)}: '{query_preview}'")
                    
                    results = self.search_web(query_obj, num_results=results_per_query)
                    logger.info(f"Query {i+1} returned {len(results)} results")
                    
                    # Submit all results for parallel processing
                    future_to_result = {}
                    for j, result in enumerate(results):
                        # Each thread will check against the current state but only the main thread will update it
                        if result.get("link") not in processed_urls:
                            future = executor.submit(
                                self.process_search_result, 
                                result, i, j, context, processed_urls
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
            
            # Sort results by relevance score (descending)
            logger.info("Sorting results by relevance score")
            all_results.sort(key=lambda x: x.get("relevance_score", 0), reverse=True)
            
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
                    relevance = result.get("relevance_score", 0)
                    relevance_status = "HIGH RELEVANCE" if relevance > 0.7 else "MEDIUM RELEVANCE" if relevance > 0.4 else "LOW RELEVANCE"
                    source = result.get("source", "web").upper()
                    
                    logger.info(f"Result {i+1}: {result['title']} - {relevance_status} ({relevance:.2f}) - Source: {source}")
                    print(f"{i+1}. [{relevance_status}] [{source}] {result['title']}")
                    print(f"   URL: {result['link']}")
                    print(f"   Date: {result['date'] if result['date'] else 'Not found'}")
                    print(f"   Relevance: {relevance:.2f}")
                    if result.get("semantic_relevance"):
                        print(f"   Semantic Relevance: {result['semantic_relevance']:.2f}")
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
        Run the web extraction process and return results in a structured format.
        Enhanced with embedding support for better semantic understanding.
        
        Args:
            max_queries: Maximum number of queries to process
            max_results_per_query: Maximum results to extract per query
            
        Returns:
            Dictionary with extraction results and metadata
        """
        logger.info("Running web extractor with embedding enhancement")
        
        try:
            start_time = time.time()
            
            # Run the extraction process with the parameters
            results = self.extract_web_content(max_queries, max_results_per_query)
        
            # Calculate timing and prepare output
            end_time = time.time()
            execution_time = end_time - start_time
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add embeddings status
            embeddings_status = {
                "enabled": self.use_embeddings,
                "embedding_count": sum(1 for r in results if r.get("embedding") is not None),
                "database_search_enabled": self.semantic_search_func is not None,
                "database_results": sum(1 for r in results if r.get("source") == "database")
            }
            
            output = {
                "status": "success" if results else "error",
                "timestamp": timestamp,
                "execution_time": f"{execution_time:.2f} seconds",
                "result_count": len(results),
                "embeddings": embeddings_status,
                "results": results
            }
            
            logger.info(f"Web extractor completed in {execution_time:.2f} seconds with {len(results)} results")
            if self.use_embeddings:
                logger.info(f"Generated {embeddings_status['embedding_count']} embeddings")
                logger.info(f"Found {embeddings_status['database_results']} results from database")
            
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
                "embeddings": {"enabled": self.use_embeddings, "embedding_count": 0}
            }

# Example usage:
if __name__ == "__main__":
    # This is just an example of how to use the enhanced extractor
    # You'll need to provide the actual embedding functions
    try:
        print("\n" + "="*50)
        print("STARTING WEB EXTRACTOR WITH EMBEDDINGS")
        print("="*50 + "\n")
        
        # Example dummy embedding function
        def dummy_embedding(text):
            import numpy as np
            return list(np.random.random(384))  # Simulate a 384-dim embedding vector
        
        # Example dummy semantic search function
        def dummy_semantic_search(embedding, limit=5):
            # This would normally query a database
            return []
        
        extractor = WebExtractor(
            create_embedding_func=dummy_embedding,
            semantic_search_func=dummy_semantic_search
        )
        
        results = extractor.run(max_queries=3)
        
        print("\n" + "="*50)
        print("WEB EXTRACTOR COMPLETED")
        print(f"Status: {results.get('status')}")
        print(f"Total results: {results.get('result_count', 0)}")
        print(f"Execution time: {results.get('execution_time')}")
        if results.get("embeddings", {}).get("enabled"):
            print(f"Embeddings generated: {results.get('embeddings', {}).get('embedding_count', 0)}")
            print(f"Database results: {results.get('embeddings', {}).get('database_results', 0)}")
        print("="*50 + "\n")
        
    except Exception as e:
        print("\n" + "!"*50)
        print(f"UNHANDLED EXCEPTION: {str(e)}")
        traceback.print_exc()
        print("!"*50 + "\n")