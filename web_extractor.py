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
import numpy as np
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
    """Class for extracting web content related to environmental sustainability."""
    
    def __init__(self, 
                search_api_key=None,
                max_workers=5):
        """
        Initialize WebExtractor with configuration options.
        
        Args:
            search_api_key: API key for search engine (e.g., Serper)
            max_workers: Number of workers for ThreadPoolExecutor
        """
        # Load environment variables
        load_dotenv()
        
        # Set API key
        self.search_api_key = search_api_key or os.getenv('SERPER_API_KEY')
        
        # Set max_workers
        self.max_workers = max_workers
        
        # Target organizations to prioritize
        self.priority_domains = [
            "giz.de", 
            "bmz.de", 
            "kfw.de"
        ]
        
        # Configure environmental topics
        self.configure_topics()
    
    def configure_topics(self):
        """Configure the environmental topics and associated keywords."""
        self.topics = {
            "indigenous_peoples": [
                "indigenous", "native communities", "indigenous rights", 
                "traditional knowledge", "aboriginal", "local communities"
            ],
            "protected_areas": [
                "national park", "conservation area", "marine reserve", "protected area",
                "wildlife refuge", "conservation", "nature reserve"
            ],
            "forest_restoration": [
                "forest restoration", "land restoration", "reforestation", "afforestation",
                "rewilding", "forest rehabilitation", "ecological restoration"
            ],
            "marine_conservation": [
                "marine conservation", "marine protection", "ocean conservation", 
                "coral reef", "coastal protection", "blue economy"
            ],
            "ecosystem_services": [
                "ecosystem services", "carbon sequestration", "Amazon basin", 
                "Congo basin", "rainforest", "biodiversity", "watershed services"
            ],
            "sustainable_agriculture": [
                "sustainable agriculture", "agroforestry", "organic farming",
                "permaculture", "sustainable farming", "crop rotation"
            ],
            "sustainable_forestry": [
                "sustainable forestry", "sustainable timber", "forest management",
                "reduced impact logging", "FSC certification", "sustainable logging"
            ],
            "aquaculture": [
                "aquaculture", "fish farming", "sustainable aquaculture",
                "biodiversity in aquaculture", "responsible aquaculture"
            ]
        }
        
        # Generate search queries based on topics and organizations
        self.search_queries = []
        
        # Base organization keywords
        organizations = ["GIZ", "BMZ", "KfW", "Germany international cooperation"]
        
        # Generate combination of queries
        for org in organizations:
            # Add general environmental query for each organization
            self.search_queries.append(f"{org} environmental sustainability")
            
            # Add specific topic queries
            for topic_name, keywords in self.topics.items():
                # Use the first two keywords from each topic
                for keyword in keywords[:2]:
                    self.search_queries.append(f"{org} {keyword}")
                    
        logger.info(f"Generated {len(self.search_queries)} search queries")
    
    def generate_search_queries(self, max_queries: Optional[int] = None) -> List[str]:
        """
        Generate a list of search queries based on configured topics and organizations.
        
        Args:
            max_queries: Maximum number of queries to generate (None for all)
            
        Returns:
            List of search query strings
        """
        queries = self.search_queries
        if max_queries:
            queries = queries[:max_queries]
            
        logger.info(f"Returning {len(queries)} search queries")
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
    
    def scrape_webpage(self, url: str, search_result_title: str = "") -> Tuple[str, str, str]:
        """
        Scrape content, title and date from a webpage.
        
        Args:
            url: URL to scrape
            search_result_title: Title from search results
            
        Returns:
            Tuple of (content, title, date)
        """
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
    
    def estimate_relevance(self, title: str, content: str) -> Tuple[float, Dict[str, float]]:
        """
        Estimate relevance of content to environmental topics using keyword matching.
        This function replaces AI-based relevance scoring with simple keyword matching.
        
        Args:
            title: Title of the content
            content: Main content text
            
        Returns:
            Tuple of (overall_score, topic_scores_dict)
        """
        combined_text = f"{title.lower()} {content.lower()}"
        
        # Count keywords for each topic
        topic_scores = {}
        for topic, keywords in self.topics.items():
            topic_count = 0
            for keyword in keywords:
                # Count occurrences of each keyword
                count = combined_text.count(keyword.lower())
                topic_count += count
            
            # Normalize score based on content length
            content_length = len(combined_text)
            normalized_score = min(1.0, (topic_count * 500) / max(1, content_length))
            topic_scores[topic] = normalized_score
        
        # Calculate overall relevance as average of top 3 topic scores
        top_scores = sorted(topic_scores.values(), reverse=True)[:3]
        overall_score = sum(top_scores) / len(top_scores) if top_scores else 0
        
        # Boost score for priority domains
        domain_boost = 0
        try:
            url_domain = urlparse(url).netloc.lower()
            if any(domain in url_domain for domain in self.priority_domains):
                domain_boost = 0.3
                overall_score = min(1.0, overall_score + domain_boost)
        except:
            pass
        
        return overall_score, topic_scores
    
    def identify_themes(self, content: str) -> List[str]:
        """
        Identify themes in content using keyword matching.
        This function replaces AI-based theme identification.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of identified themes
        """
        content_lower = content.lower()
        identified_themes = []
        
        for topic, keywords in self.topics.items():
            for keyword in keywords:
                if keyword.lower() in content_lower:
                    # Convert topic from snake_case to Title Case
                    theme = " ".join(word.capitalize() for word in topic.split("_"))
                    identified_themes.append(theme)
                    break  # Only add the theme once
        
        return identified_themes
    
    def analyze_sentiment(self, content: str) -> str:
        """
        Analyze sentiment using simple keyword-based approach.
        This function replaces AI-based sentiment analysis.
        
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
    
    def extract_benefits_to_germany(self, content: str) -> Optional[str]:
        """
        Extract benefits to Germany using keyword proximity.
        This function replaces AI-based benefits extraction.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Extracted benefits or None
        """
        if "germany" not in content.lower() and "german" not in content.lower():
            return None
        
        # Find sentences that mention Germany and benefit keywords
        sentences = re.split(r'(?<=[.!?])\s+', content)
        benefit_sentences = []
        
        benefit_keywords = [
            "benefit", "advantage", "gain", "profit", "value", "opportunity",
            "improvement", "enhanced", "strengthen", "contribute", "partnership",
            "cooperation", "support", "funding", "investment", "expertise"
        ]
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if ("germany" in sentence_lower or "german" in sentence_lower):
                if any(keyword in sentence_lower for keyword in benefit_keywords):
                    benefit_sentences.append(sentence.strip())
        
        if benefit_sentences:
            return " ".join(benefit_sentences)
        
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
            processed_urls = set()  # Track URLs to avoid duplicates within current run
            skipped_total = 0      # Count of all skipped URLs
            
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
                    print(f"   📊 Relevance: {relevance:.2f}")
                    print(f"   🏷️ Themes: {', '.join(result['themes']) if result['themes'] else 'None'}")
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
            # Extract content from URL using scrape_webpage method
            content, extracted_title, date = self.scrape_webpage(url, title)
            
            if not content or len(content) < 100:
                logger.warning(f"Insufficient content from {url} (length: {len(content) if content else 0})")
                return None
            
            # Create a simple summary (first 500 chars)
            summary = content[:500] + "..." if len(content) > 500 else content
            
            # Identify themes using the class method
            themes = self.identify_themes(content)
            
            # Extract organization from URL domain
            # Get domain from URL - simple implementation inside the class
            parsed_url = urlparse(url)
            domain = parsed_url.netloc.lower()
            if domain.startswith('www.'):
                domain = domain[4:]
            organization = domain.split('.')[0].upper()
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(content)
            
            # Calculate relevance score
            relevance_score, topic_scores = self.estimate_relevance(extracted_title or title, content)
            
            # Extract benefits to Germany if the method exists
            benefits_to_germany = None
            if hasattr(self, 'extract_benefits_to_germany'):
                benefits_to_germany = self.extract_benefits_to_germany(content)
            
            # Create result dictionary
            result_data = {
                "title": extracted_title or title,
                "link": url,
                "date": date,
                "content": content,
                "summary": summary,
                "themes": themes,
                "organization": organization,
                "sentiment": sentiment,
                "relevance_score": relevance_score,
                "query_index": query_index,
                "result_index": result_index,
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed {url} (relevance: {relevance_score:.2f})")
            return result_data
            
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
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