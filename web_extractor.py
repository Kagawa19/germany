import os
import time
import requests
import logging
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from typing import Dict, List, Optional, Tuple, Any
import traceback
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from content_db import get_db_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("web_extractor.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("WebExtractor")

class WebExtractor:
    """Class for extracting web content related to ABS CDI and Bio-innovation Africa initiatives."""
    
    def __init__(self, 
                search_api_key=None,
                max_workers=5,
                language="English"):
        """
        Initialize WebExtractor with configuration options.
        
        Args:
            search_api_key: API key for search engine (e.g., Serper)
            max_workers: Number of workers for ThreadPoolExecutor
            language: Language for search queries (English, German, French)
        """
        # Load environment variables
        load_dotenv()
        
        # Set API key
        self.search_api_key = search_api_key or os.getenv('SERPER_API_KEY')
        
        # Set max_workers
        self.max_workers = max_workers
        
        # Set language
        self.language = language
        
        # Target organizations and domains to prioritize
        self.priority_domains = []
        
        # Configure the specific initiatives to track
        self.configure_initiatives()
        
    def configure_initiatives(self):
        """Configure the specific ABS initiative names to search for in different languages."""
        
        # Define all ABS initiative names in different languages
        self.abs_names = {
            "English": [
                "ABS Capacity Development Initiative",
                "ABS CDI",
                "ABS Capacity Development Initiative for Africa",
                "ABS Initiative"
            ],
            "German": [
                "Initiative für Zugang und Vorteilsausgleich",
                "ABS-Kapazitätenentwicklungsinitiative für Afrika",
                "ABS-Initiative"
            ],
            "French": [
                "Initiative pour le renforcement des capacités en matière d'APA",
                "Initiative Accès et Partage des Avantages",
                "Initiative sur le développement des capacités pour l'APA",
                "Initiative de renforcement des capacités sur l'APA",
                "Initiative APA",
                "Initiative de développement des capacités en matière d'accès et de partage des avantages"
            ]
        }
        
        # Get the names for the selected language
        language_names = self.abs_names.get(self.language, self.abs_names["English"])
        
        # Also include English names regardless of language to maximize results
        if self.language != "English":
            all_names = language_names + self.abs_names["English"]
        else:
            all_names = language_names
        
        # Generate search queries
        self.search_queries = []
        
        # Create simple search queries for each initiative name
        for name in all_names:
            self.search_queries.append(name)
        
        logger.info(f"Generated {len(self.search_queries)} search queries for ABS initiatives in {self.language}")
    
    def generate_search_queries(self, max_queries: Optional[int] = None) -> List[str]:
        """
        Generate a list of search queries based on configured initiatives.
        
        Args:
            max_queries: Maximum number of queries to generate (None for all)
                
        Returns:
            List of search query strings
        """
        # Start with basic queries - exact initiative names
        queries = list(self.search_queries)
        
        # Add exact match queries with quotes for more precise results
        for name in self.abs_names.get(self.language, self.abs_names["English"]):
            # Add quoted version for exact match
            queries.append(f'"{name}"')
        
        # Add filetype searches to find documents about the ABS Initiative
        main_names = ["ABS Initiative", "ABS Capacity Development Initiative"]
        if self.language == "German":
            main_names.append("ABS-Initiative")
        elif self.language == "French":
            main_names.append("Initiative APA")
            
        for name in main_names:
            queries.append(f'filetype:pdf "{name}"')
            queries.append(f'filetype:doc "{name}"')
        
        # Limit the number of queries if specified
        if max_queries:
            queries = queries[:max_queries]
                
        logger.info(f"Generated {len(queries)} search queries for {self.language}")
        return queries
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict]:
        """
        Search the web using the given query via Serper API.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to retrieve
            
        Returns:
            List of dictionaries containing search results
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"Searching web for: '{query_preview}' (limit: {num_results} results)")
        print(f"  Searching for: '{query_preview}'...")
        
        if not self.search_api_key:
            error_msg = "Cannot search web: API_KEY not set"
            logger.error(error_msg)
            print(f"  ERROR: Search API_KEY not set")
            return []
        
        try:
            search_start_time = time.time()
            
            # Use Serper API for searching
            url = "https://google.serper.dev/search"
            
            payload = json.dumps({
                "q": query,
                "num": num_results
            })
            
            headers = {
                'X-API-KEY': self.search_api_key,
                'Content-Type': 'application/json'
            }
            
            response = requests.request("POST", url, headers=headers, data=payload, timeout=30)
            response.raise_for_status()
            
            search_time = time.time() - search_start_time
            
            # Parse response
            search_results = response.json()
            
            # Extract organic results
            organic_results = search_results.get("organic", [])
            logger.info(f"Received {len(organic_results)} search results in {search_time:.2f} seconds")
            print(f"  Found {len(organic_results)} results in {search_time:.2f} seconds")
            
            # Log result URLs for debugging
            for i, result in enumerate(organic_results):
                logger.info(f"Result {i+1}: {result.get('title', 'No title')} - {result.get('link', 'No link')}")
            
            return organic_results
                
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}", exc_info=True)
            print(f"  ERROR searching web: {str(e)}")
            return []
    
    def extract_date_from_content(self, html_content: str, url: str, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract publication date from content using multiple strategies.
        Uses regex patterns instead of AI to save costs.
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            soup: BeautifulSoup object
            
        Returns:
            Date string or None if not found
        """
        logger.info(f"Extracting date from webpage: {url}")
        
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
        """
        Extract the title from the webpage content.
        
        Args:
            soup: BeautifulSoup object
            url: URL of the webpage
            search_result_title: Title from search results
            
        Returns:
            Extracted title
        """
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
    
    def scrape_webpage(self, url: str, search_result_title: str = "") -> Tuple[str, str, str, str]:
        """
        Scrape content, title, date, and clean summary from a webpage.
        
        Args:
            url: URL to scrape
            search_result_title: Title from search results
            
        Returns:
            Tuple of (content, title, date, clean_summary)
        """
        logger.info(f"Scraping webpage: {url}")
        
        try:
            # Check if this is a PDF document
            if url.lower().endswith('.pdf'):
                logger.info("PDF document detected, using specialized handling")
                return self.handle_pdf_document(url, search_result_title)
            
            scrape_start_time = time.time()
            logger.info(f"Sending HTTP request to {url}")
            
            # Use our enhanced method to get the page content
            html_content, success = self.get_page_content(url, timeout=15)
            
            if not success:
                logger.error(f"Failed to retrieve content from {url}")
                return "", "", None, ""
            
            request_time = time.time() - scrape_start_time
            logger.info(f"Received response in {request_time:.2f} seconds.")
            
            # Parse HTML
            logger.info("Parsing HTML content")
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title first before removing elements
            title = self.extract_title_from_content(soup, url, search_result_title)
            
            # Extract date before removing elements - pass the full HTML text for comprehensive search
            date = self.extract_date_from_content(html_content, url, soup)
            
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
            
            # Create clean summary using OpenAI
            logger.info("Generating clean summary from content")
            clean_summary = clean_and_enhance_summary(content, title, url)
            
            scrape_time = time.time() - scrape_start_time
            logger.info(f"Successfully scraped {len(content)} chars from {url} in {scrape_time:.2f} seconds (cleaned from {original_length} to {cleaned_length} chars)")
            logger.info(f"Extracted title: {title}")
            logger.info(f"Extracted date: {date}")
            logger.info(f"Generated clean summary: {len(clean_summary)} chars")
            
            return content, title, date, clean_summary
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}", exc_info=True)
            return "", "", None, ""
        
    def get_page_content(self, url, timeout=15, max_retries=2):
        """
        Get page content with retry logic and rotating user agents.
        This makes web scraping more robust without changing the existing methods.
        
        Args:
            url: URL to retrieve
            timeout: Request timeout in seconds
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (response_text, success_boolean)
        """
        # List of common user agents to rotate through
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:94.0) Gecko/20100101 Firefox/94.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/15.1 Safari/605.1.15',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/96.0.1054.43',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'
        ]
        
        # Try different user agents if we need to retry
        attempts = 0
        while attempts <= max_retries:
            try:
                # Select a random user agent
                import random
                user_agent = random.choice(user_agents)
                
                headers = {
                    'User-Agent': user_agent,
                    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                    'Accept-Language': 'en-US,en;q=0.9',
                    'Referer': 'https://www.google.com/',
                    'DNT': '1'  # Do Not Track
                }
                
                # Add a small delay between attempts to be respectful
                if attempts > 0:
                    import time
                    time.sleep(2)
                
                # Log attempt
                logger.info(f"Request attempt {attempts+1}/{max_retries+1} for {url}")
                
                # Make the request
                response = requests.get(url, headers=headers, timeout=timeout)
                
                # Check if successful
                response.raise_for_status()
                
                # Return the response text and success flag
                return response.text, True
                
            except requests.exceptions.RequestException as e:
                attempts += 1
                logger.warning(f"Request attempt {attempts} failed for {url}: {str(e)}")
                
                # If we've exhausted our retries, give up
                if attempts > max_retries:
                    logger.error(f"All {max_retries+1} attempts failed for {url}")
                    return "", False
                    
        # We should never reach here, but just in case
        return "", False


    # Helper function for clean_and_enhance_summary that should be added to the WebExtractor class
    def clean_and_enhance_summary(content: str, title: str = "", url: str = "") -> str:
        """
        Clean and enhance the summary text using OpenAI.
        This can be integrated into the scrape_webpage method to clean summaries from the start.
        
        Args:
            content: The original content text
            title: The page title for context
            url: The URL for context
            
        Returns:
            Cleaned and enhanced summary
        """
        # Skip if no content or very short content
        if not content or len(content) < 100:
            return content
            
        # Clean HTML entities and encoding issues first
        content = clean_html_entities(content)
        
        # Get OpenAI client
        client = get_openai_client()
        if not client:
            logger.warning("OpenAI client not available. Using basic summary extraction.")
            # Fall back to basic summary extraction - first 500 chars
            return content[:500] + "..." if len(content) > 500 else content
        
        try:
            # Create a simplified prompt
            prompt = f"""
    Create a clear, concise summary of this content about the ABS Initiative. Fix any encoding issues.
    Only include factual information that is explicitly mentioned in the content.

    Title: {title}
    URL: {url}

    Content: {content[:3000]}  # Limit content to first 3000 chars to save tokens

    IMPORTANT: Create a coherent, well-formatted summary. Don't mention encoding issues.
    """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            enhanced_summary = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated enhanced summary with OpenAI ({len(enhanced_summary)} chars)")
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"Error using OpenAI for summary enhancement: {str(e)}")
            # Fall back to basic summary extraction
            return content[:500] + "..." if len(content) > 500 else content
    
    def identify_initiative(self, content: str) -> Tuple[str, float]:
        """
        Check if any specific ABS initiative names appear in the content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Tuple of (initiative_key, confidence_score)
        """
        content_lower = content.lower()
        
        # Flatten all initiative names from all languages into one list
        all_initiative_names = []
        for language, names in self.abs_names.items():
            for name in names:
                all_initiative_names.append(name.lower())
        
        # Check if any initiative name is found in the content
        for name in all_initiative_names:
            if name.lower() in content_lower:
                # If found, return "abs_initiative" with a high confidence score
                return "abs_initiative", 1.0
        
        # If no names found, return "unknown" with zero confidence
        return "unknown", 0.0
    
    def identify_benefit_categories(self, content: str) -> Dict[str, float]:
        """
        Identify which benefit categories are mentioned in the content and their relevance.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary of benefit categories and their scores
        """
        content_lower = content.lower()
        content_length = max(1, len(content_lower))
        
        # Track scores for each benefit category
        category_scores = {}
        
        for category, keywords in self.benefit_categories.items():
            keyword_count = 0
            for keyword in keywords:
                keyword_count += content_lower.count(keyword.lower())
            
            # Normalize score based on content length
            normalized_score = min(1.0, (keyword_count * 500) / content_length)
            category_scores[category] = normalized_score
        
        return category_scores
    
    def clean_and_enhance_summary(content: str, title: str = "", url: str = "") -> str:
        """
        Clean and enhance the summary text using OpenAI.
        This can be integrated into the scrape_webpage method to clean summaries from the start.
        
        Args:
            content: The original content text
            title: The page title for context
            url: The URL for context
            
        Returns:
            Cleaned and enhanced summary
        """
        # Skip if no content or very short content
        if not content or len(content) < 100:
            return content
            
        # Clean HTML entities and encoding issues first
        content = clean_html_entities(content)
        
        # Get OpenAI client
        client = get_openai_client()
        if not client:
            logger.warning("OpenAI client not available. Using basic summary extraction.")
            # Fall back to basic summary extraction - first 500 chars
            return content[:500] + "..." if len(content) > 500 else content
        
        try:
            # Create a simplified prompt
            prompt = f"""
    Create a clear, concise summary of this content about the ABS Initiative. Fix any encoding issues.
    Only include factual information that is explicitly mentioned in the content.

    Title: {title}
    URL: {url}

    Content: {content[:3000]}  # Limit content to first 3000 chars to save tokens

    IMPORTANT: Create a coherent, well-formatted summary. Don't mention encoding issues.
    """

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            enhanced_summary = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated enhanced summary with OpenAI ({len(enhanced_summary)} chars)")
            return enhanced_summary
            
        except Exception as e:
            logger.error(f"Error using OpenAI for summary enhancement: {str(e)}")
            # Fall back to basic summary extraction
            return content[:500] + "..." if len(content) > 500 else content

    def clean_html_entities(text):
        """Clean HTML entities and common encoding issues in text"""
        if not isinstance(text, str):
            return text
            
        # First decode HTML entities
        try:
            import html
            text = html.unescape(text)
        except:
            pass
        
        # Replace common UTF-8 encoding issues
        replacements = {
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'Â': ' ',
            'â€¦': '...',
            'â€"': '—',
            'â€"': '-',
            'â€˜': "'",
            'Ã©': 'é',
            'Ã¨': 'è',
            'Ã¢': 'â',
            'Ã»': 'û',
            'Ã´': 'ô',
            'Ã®': 'î',
            'Ã¯': 'ï',
            'Ã': 'à',
            'Ã§': 'ç',
            'Ãª': 'ê',
            'Ã¹': 'ù',
            'Ã³': 'ó',
            'Ã±': 'ñ',
            'ï»¿': '',  # BOM
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Clean up excessive whitespace
        import re
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text

    def get_openai_client():
        """Get or initialize OpenAI client."""
        try:
            from openai import OpenAI
            from dotenv import load_dotenv
            import os
            
            load_dotenv()
            openai_api_key = os.getenv("OPENAI_API_KEY")
            if not openai_api_key:
                return None
            
            return OpenAI(api_key=openai_api_key)
        except Exception as e:
            logger.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    
    def extract_benefits_examples(self, content: str, initiative: str) -> List[Dict[str, Any]]:
        """
        Extract examples of benefits from the content related to ABS Initiative
        
        Args:
            content: Content text to analyze
            initiative: Initiative key (e.g., "abs_initiative")
            
        Returns:
            List of extracted benefit examples
        """
        if not content or len(content) < 100:
            return []
                
        content_lower = content.lower()
        
        # Flatten all initiative names from all languages into one list
        all_initiative_names = []
        for language, names in self.abs_names.items():
            for name in names:
                all_initiative_names.append(name.lower())
        
        # Find paragraphs that mention benefits
        paragraphs = content.split('\n\n')
        benefit_paragraphs = []
        
        # Use simplified approach - just find paragraphs that mention the initiative
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Check if the paragraph mentions any ABS initiative name
            initiative_mentioned = any(name in paragraph_lower for name in all_initiative_names)
            
            # If the paragraph mentions an initiative, include it
            if initiative_mentioned:
                # Only include paragraphs of reasonable length to avoid fragments
                if len(paragraph.split()) >= 10:
                    benefit_paragraphs.append(paragraph)
        
        # Extract structured benefit examples - simplified approach
        benefit_examples = []
        
        for paragraph in benefit_paragraphs:
            # Create benefit example with minimal structure
            benefit_example = {
                "text": paragraph.strip(),
                "category": "general",
                "initiative": "ABS Initiative",
                "word_count": len(paragraph.split())
            }
            
            benefit_examples.append(benefit_example)
        
        return benefit_examples
    
    def handle_pdf_document(self, url: str, search_result_title: str = "") -> Tuple[str, str, str, str]:
        """
        Create confident summaries for PDF documents based on metadata analysis.
        
        Args:
            url: URL to the PDF document
            search_result_title: Title from search results
            
        Returns:
            Tuple of (content, title, date, clean_summary)
        """
        logger.info(f"Handling PDF document: {url}")
        
        try:
            # Get a title for the PDF
            title = search_result_title
            if not title or title == "No title":
                # Extract a title from the URL
                filename = url.split('/')[-1]
                # Remove file extension and replace separators with spaces
                title = filename.rsplit('.', 1)[0].replace('-', ' ').replace('_', ' ').replace('%20', ' ')
                # Capitalize title properly
                title = ' '.join(word.capitalize() for word in title.split())
            
            # Extract organization from URL
            org_name = "Unknown"
            domain = urlparse(url).netloc
            if "." in domain:
                parts = domain.split(".")
                if len(parts) >= 2:
                    # Get organization from domain
                    if parts[-2] in ['org', 'com', 'net', 'edu', 'gov', 'int']:
                        org_name = parts[-3].upper()
                    else:
                        org_name = parts[-2].upper()
                        
            # If we have specific known organizations, use their full names
            org_mapping = {
                'cbd': 'Convention on Biological Diversity',
                'giz': 'Deutsche Gesellschaft für Internationale Zusammenarbeit (GIZ)',
                'bgci': 'Botanic Gardens Conservation International',
                'abs-initiative': 'ABS Initiative',
                'undp': 'United Nations Development Programme',
                'bmz': 'German Federal Ministry for Economic Cooperation and Development'
            }
            
            if org_name.lower() in org_mapping:
                org_name = org_mapping[org_name.lower()]
            
            # Create a descriptive content text
            content = f"PDF Document published by {org_name}. Title: {title}. Source: {url}"
            
            # Try to use OpenAI for a better summary
            try:
                from openai import OpenAI
                import os
                from dotenv import load_dotenv
                
                # Load environment variables and get API key
                load_dotenv()
                api_key = os.getenv("OPENAI_API_KEY")
                
                # If API key is available, use OpenAI
                if api_key:
                    client = OpenAI(api_key=api_key)
                    
                    # Create a prompt for PDF description
                    prompt = f"""
    Create a confident, factual summary for this PDF document about the ABS Initiative. The summary should be definitive without using phrases like "likely" or "may contain".

    PDF Title: {title}
    Publisher: {org_name}
    URL: {url}

    Create a 2-3 sentence summary describing what this document contains about the ABS Initiative. Make definitive statements as if you have read the document.
    """

                    # Make API call
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.3,
                        max_tokens=150
                    )
                    
                    # Extract summary from response
                    ai_summary = response.choices[0].message.content.strip()
                    
                    if ai_summary:
                        summary = f"{ai_summary}\n\nPublisher: {org_name}\nSource: {url}"
                    else:
                        summary = f"This document from {org_name} titled '{title}' provides information about the ABS Initiative and its implementation framework.\n\nSource: {url}"
                else:
                    summary = f"This document from {org_name} titled '{title}' provides information about the ABS Initiative and its implementation framework.\n\nSource: {url}"
            except Exception as e:
                logger.warning(f"Error generating PDF summary: {str(e)}")
                summary = f"This document from {org_name} titled '{title}' provides information about the ABS Initiative and its implementation framework.\n\nSource: {url}"
            
            return content, title, None, summary
            
        except Exception as e:
            logger.error(f"Error handling PDF document: {str(e)}")
            return f"PDF Document: {url}", "PDF Document", None, f"ABS Initiative document available at {url}"
        
    def identify_themes(self, content: str) -> List[str]:
        """
        Identify diverse themes in content using OpenAI with a data-driven approach.
        Does not include "ABS Initiative" as a default theme.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of identified themes
        """
        # Skip if content is too short
        if not content or len(content) < 100:
            return ["Content Analysis", "Documentation"]
        
        # Try to use OpenAI for theme extraction
        try:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv
            
            # Load environment variables and get API key
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            
            # If API key is available, use OpenAI
            if api_key:
                client = OpenAI(api_key=api_key)
                
                # Prepare content - limit to first 3000 chars to save tokens
                excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
                
                # Create a prompt for theme extraction without suggesting ABS Initiative
                prompt = f"""
    Analyze this text and identify the main substantive themes it discusses. Focus on the actual subject matter.

    Text excerpt:
    {excerpt}

    Extract exactly 5 specific, substantive themes from this content. Do NOT use generic themes like "ABS Initiative" or "Capacity Development" unless they're discussed in detail as topics themselves. Focus on the actual subjects being discussed such as "Biodiversity Conservation", "Traditional Knowledge", "Genetic Resources", etc.

    Return ONLY a comma-separated list of identified themes without explanations or additional text.
    """

                # Make API call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,  # Slightly higher temperature for more diversity
                    max_tokens=100
                )
                
                # Extract themes from response
                ai_themes_text = response.choices[0].message.content.strip()
                
                # Convert to list and clean up each theme
                ai_themes = [theme.strip() for theme in ai_themes_text.split(',')]
                
                # Remove any empty themes
                ai_themes = [theme for theme in ai_themes if theme]
                
                # Ensure we have at least some themes
                if ai_themes and len(ai_themes) >= 2:
                    return ai_themes
        
        except Exception as e:
            # Log the error but continue to fallback method
            logger.warning(f"Error using OpenAI for theme extraction: {str(e)}. Falling back to simple content analysis.")
        
        # Fallback approach without using "ABS Initiative" as default
        import re
        from collections import Counter
        
        # Define some substantive topics related to biodiversity and conservation
        potential_topics = [
            "Biodiversity", "Conservation", "Sustainable Development", "Genetic Resources",
            "Traditional Knowledge", "Indigenous Rights", "Policy Development", 
            "Legal Framework", "Compliance", "Implementation", "Benefit Sharing",
            "Sustainable Use", "Ecosystem Services", "Stakeholder Engagement",
            "Technology Transfer", "Capacity Building", "International Cooperation",
            "Research", "Innovation", "Monitoring", "Evaluation", "Governance"
        ]
        
        # Check which topics are present in the content
        found_topics = []
        content_lower = content.lower()
        
        for topic in potential_topics:
            if topic.lower() in content_lower:
                found_topics.append(topic)
                # Stop once we have 5 topics
                if len(found_topics) >= 5:
                    break
        
        # If we found specific topics, return them
        if found_topics:
            return found_topics
        
        # Otherwise use a more general approach - extract key terms
        # Extract all words and simple phrases
        text = re.sub(r'[^\w\s]', ' ', content_lower)  # Remove punctuation
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)  # Find words of 4+ characters
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Remove common stopwords
        stopwords = {"this", "that", "these", "those", "with", "from", "their", "would", "could", "should", 
                    "about", "which", "there", "where", "when", "what", "have", "will", "they", 
                    "them", "then", "than", "were", "been", "being", "other", "initiative", "development",
                    "capacity", "through", "between", "information", "because", "system", "process"}
        
        # Filter out stop words
        potential_themes = {word: count for word, count in word_counts.items() 
                        if word not in stopwords and count > 1}
        
        # Extract 5 most common potential theme words
        top_words = [word.capitalize() for word, _ in sorted(potential_themes.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # If we couldn't find good topic words, return generic research categories
        if not top_words:
            return ["Policy Analysis", "Research Findings", "Environmental Studies", "International Relations", "Resource Management"]
        
        return top_words
        


    def analyze_sentiment(self, content: str) -> str:
        """
        Analyze sentiment using simple keyword-based approach.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Sentiment (Positive, Negative, or Neutral)
        """
        content_lower = content.lower()
        
        # Define positive and negative keywords
        positive_keywords = [
            "success", "successful", "beneficial", "benefit", "positive", "improve", "improvement",
            "advantage", "effective", "efficiently", "progress", "achievement", "sustainable",
            "solution", "opportunity", "promising", "innovative", "advanced", "partnership"
        ]
        
        negative_keywords = [
            "failure", "failed", "problem", "challenge", "difficult", "negative", "risk",
            "threat", "damage", "harmful", "pollution", "degradation", "unsustainable",
            "danger", "crisis", "emergency", "concern", "alarming", "devastating"
        ]
        
        # Count occurrences
        positive_count = sum(content_lower.count(keyword) for keyword in positive_keywords)
        negative_count = sum(content_lower.count(keyword) for keyword in negative_keywords)
        
        # Determine sentiment
        if positive_count > negative_count * 1.5:
            return "Positive"
        elif negative_count > positive_count * 1.5:
            return "Negative"
        else:
            return "Neutral"
    
    def extract_organization_from_url(self, url: str) -> str:
        """
        Extract organization name from URL.
        
        Args:
            url: The URL to extract organization from
            
        Returns:
            Organization name based on domain
        """
        if not url:
            return "Unknown"
            
        try:
            # Parse the URL to extract domain
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            
            # Remove 'www.' prefix if present
            if domain.startswith('www.'):
                domain = domain[4:]
            
            # Special case handling for known domains
            domain_map = {
                "abs-initiative.info": "ABS Initiative",
                "cbd.int": "CBD",
                "absch.cbd.int": "ABS Clearing-House",
                "giz.de": "GIZ",
                "bmz.de": "BMZ",
                "kfw.de": "KfW",
                "unctad.org": "UNCTAD",
                "uebt.org": "UEBT",
                "ethicalbiotrade.org": "UEBT"
            }
            
            # Check if domain matches any in the map
            for key, value in domain_map.items():
                if key in domain:
                    return value
            
            # Extract main domain (before first dot)
            main_domain = domain.split('.')[0].upper()
            
            return main_domain
        except:
            return "Unknown"
            
    def process_search_result(self, result, query_index, result_index, processed_urls):
        """
        Process a single search result to extract and analyze content.
        
        Args:
            result: The search result dict with link, title, etc.
            query_index: Index of the query that produced this result
            result_index: Index of this result within the query results
            processed_urls: Set of already processed URLs (passed by reference)
            
        Returns:
            Dict containing processed content or None if extraction failed
        """
        logger = logging.getLogger(__name__)
        url = result.get("link")
        title = result.get("title", "Untitled")
        
        # Log processing start
        logger.info(f"Processing result {result_index+1} from query {query_index+1}: {title}")
        
        try:
            # Extract content from URL using scrape_webpage method - now returns clean summary
            content, extracted_title, date, clean_summary = self.scrape_webpage(url, title)
            
            if not content or len(content) < 100:
                logger.warning(f"Insufficient content from {url} (length: {len(content) if content else 0})")
                return None
            
            # Identify which initiative is mentioned
            initiative_key, initiative_score = self.identify_initiative(content)
            
            # Skip if no ABS initiative is mentioned
            if initiative_key == "unknown" or initiative_score < 0.1:
                logger.info(f"No ABS initiative mentioned in content from {url}")
                return None
            
            # Extract organization from URL domain
            organization = self.extract_organization_from_url(url)
            
            # Format the result
            result_data = {
                "title": extracted_title or title,
                "link": url,
                "date": date,
                "content": content,
                "summary": clean_summary,  # Use our enhanced clean summary
                "themes": ["ABS Initiative"],  # Simplified theme
                "organization": organization,
                "sentiment": "Neutral",  # Default sentiment
                "language": self.language,  # Add language field
                "initiative": "ABS Initiative",  # Simplified initiative name
                "initiative_key": initiative_key,
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed {url} (initiative: {initiative_key}, language: {self.language})")
            return result_data
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None


    def extract_web_content(self, max_queries=None, max_results_per_query=None) -> Dict:
        """
        Main method to extract web content based on search queries.
        Skips URLs that are already in the database to save time.
        
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
                # Continue with empty set if database query fails
            
            # Generate search queries
            queries = self.generate_search_queries(max_queries)
            
            # Search the web and collect results
            all_results = []
            processed_urls = set()  # IMPORTANT: Start with a fresh empty set each time
            skipped_total = 0      # Count of all skipped URLs
            
            # Add this line to explicitly log that we're starting with a fresh set
            logger.info("Starting with empty processed_urls set - no in-memory URL history")
            print("🔄 Starting with fresh URL tracking - no previous history carried over")
            
            logger.info(f"Search plan: Processing {len(queries)} queries with {self.max_workers} workers")
            
            query_header = f"\n{'-'*60}\n🔍 SEARCH QUERIES: Processing {len(queries)} queries\n{'-'*60}"
            print(query_header)
            
            results_per_query = 5 if max_results_per_query is None else max_results_per_query
            
            # Use ThreadPoolExecutor for parallel processing of search results
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                for i, query in enumerate(queries):
                    query_preview = query[:50] + "..." if len(query) > 50 else query
                    logger.info(f"Query {i+1}/{len(queries)}: '{query_preview}'")
                    
                    print(f"\n📝 QUERY {i+1}/{len(queries)}:")
                    print(f"   '{query_preview}'")
                    
                    # Search the web
                    results = self.search_web(query, num_results=results_per_query)
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
                            
                        # Note: We check if URL is in processed_urls BEFORE we add it to the set
                        if url in processed_urls:
                            logger.info(f"Skipping URL already processed in this run: {url}")
                            skipped_urls += 1
                            skipped_total += 1
                            continue
                            
                        if url in existing_urls:
                            logger.info(f"Skipping URL already in database: {url}")
                            skipped_urls += 1
                            skipped_total += 1
                            continue
                        
                        # Important: Only mark URL as processed AFTER we've decided to process it
                        # This prevents the bug where URLs are logged as "already processed" but were never processed
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
                    
                    # Collect results as they complete
                    completed = 0
                    for future in as_completed(future_to_result):
                        result_data = future.result()
                        completed += 1
                        
                        if result_data:
                            all_results.append(result_data)
                            logger.info(f"Processed result: '{result_data.get('title', 'Untitled')}' with relevance {result_data.get('relevance_score', 0):.2f}")
                        else:
                            logger.warning("Received empty result from processing")
                        
                        # Show progress
                        if completed % 5 == 0 or completed == len(future_to_result):
                            print(f"   Progress: {completed}/{len(future_to_result)} URLs processed")
                    
                    # Add small delay between queries to be respectful
                    if i < len(queries) - 1:  # Don't delay after the last query
                        logger.info("Adding delay between queries (1 second)")
                        print("\n   ⏱️ Waiting 1 second before next query...")
                        time.sleep(1)
            
            # Sort results by relevance score (descending)
            logger.info("Sorting results by relevance score")
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
            
            # Format summary header based on status
            if status == "success":
                status_emoji = "✅"
            elif status == "warning":
                status_emoji = "⚠️"
            else:
                status_emoji = "❌"
                
            summary_header = f"\n{'='*60}\n{' '*15}EXTRACTION SUMMARY {status_emoji}\n{'='*60}"
            print(summary_header)
            print(f"⏱️ Time: {duration:.2f} seconds")
            print(f"📊 Status: {status.upper()}")
            print(f"📝 Details: {status_message}")
            if len(all_results) > 0:
                print(f"🔍 Found {len(all_results)} new results")
            if skipped_total > 0:
                print(f"⏭️ Skipped {skipped_total} already processed URLs")
                if len(existing_urls) == 0:
                    print(f"⚠️ WARNING: Database integrity issue - URLs were skipped but database appears empty")
            print(f"{'-'*60}")
            
            # Log summary for each result
            if all_results:
                print("\n📋 RESULTS SUMMARY:")
                for i, result in enumerate(all_results):
                    relevance = result.get("relevance_score", 0)
                    initiative = result.get("initiative", "Unknown Initiative")
                    
                    # Determine relevance category and emoji
                    if relevance > 0.7:
                        relevance_status = "HIGH RELEVANCE"
                        rel_emoji = "🌟"
                    elif relevance > 0.4:
                        relevance_status = "MEDIUM RELEVANCE"
                        rel_emoji = "⭐"
                    else:
                        relevance_status = "LOW RELEVANCE"
                        rel_emoji = "⚪"
                    
                    logger.info(f"Result {i+1}: {result['title']} - {relevance_status} ({relevance:.2f})")
                    
                    # Format result information with emojis and structure
                    print(f"\n{i+1}. {rel_emoji} [{relevance_status}] {result['title']}")
                    print(f"   🔗 URL: {result['link']}")
                    print(f"   📅 Date: {result['date'] if result['date'] else 'Not found'}")
                    print(f"   🏢 Initiative: {initiative}")
                    print(f"   📊 Relevance: {relevance:.2f}")
                    print(f"   🏷️ Themes: {', '.join(result['themes']) if result['themes'] else 'None'}")
                    
                    # Show benefit categories
                    if result.get('benefit_categories'):
                        top_benefits = sorted(result['benefit_categories'].items(), key=lambda x: x[1], reverse=True)[:3]
                        benefit_str = ", ".join([f"{k.replace('_', ' ').title()} ({v:.2f})" for k, v in top_benefits])
                        print(f"   💼 Top Benefits: {benefit_str}")
                    
                    # Show number of benefit examples
                    if result.get('benefit_examples'):
                        print(f"   📝 Benefit Examples: {len(result['benefit_examples'])}")
                    
                    print(f"   📏 Length: {len(result['content'])} chars")
            
            # Return a dictionary with status and results
            return {
                "status": status,
                "message": status_message,
                "execution_time": f"{duration:.2f} seconds",
                "results": all_results,
                "skipped_urls": skipped_total,
                "database_urls_count": len(existing_urls)  # Add this for better debugging
            }
            
        except Exception as e:
            error_type = type(e).__name__
            error_traceback = traceback.format_exc()
            logger.error(f"Critical error in extraction process: {error_type}: {str(e)}")
            logger.error(f"Traceback: {error_traceback}")
            
            # Format error message with visual separator
            error_header = f"\n{'!'*60}\n{' '*15}EXTRACTION ERROR ❌\n{'!'*60}"
            print(error_header)
            print(f"❌ Error Type: {error_type}")
            print(f"❌ Error Message: {str(e)}")
            print(f"\n⚙️ Traceback:")
            print(error_traceback)
            print(f"{'!'*60}")
            
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

# Example usage:
if __name__ == "__main__":
    # This is just an example of how to use the extractor directly
    try:
        print("\n" + "="*50)
        print("STARTING WEB EXTRACTOR")
        print("="*50 + "\n")
        
        # Load the API key from environment
        load_dotenv()
        api_key = os.getenv('SERPER_API_KEY')
        
        if not api_key:
            print("ERROR: SERPER_API_KEY not found in environment variables.")
            print("Please set the SERPER_API_KEY environment variable and try again.")
            exit(1)
        
        # Create and run extractor
        extractor = WebExtractor(search_api_key=api_key)
        results = extractor.run(max_queries=3, max_results_per_query=3)
        
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