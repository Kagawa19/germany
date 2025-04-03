import os
import time
import traceback
import requests
import logging
import re
import json
from bs4 import BeautifulSoup
from datetime import datetime
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# Import the Analysis class
from extraction.analysis import Analysis, get_openai_client

logger = logging.getLogger("Processing")

class Processing:
    """Extension of WebExtractor with content extraction methods."""
    
    def __init__(self, language="English"):
        """
        Initialize the Processing class.
        
        Args:
            language: Content language (English, German, French)
        """
        # Set language - validate and default to English if invalid
        valid_languages = ["English", "German", "French"]
        self.language = language if language in valid_languages else "English"
        
        # Create an instance of the Analysis class to use its methods
        self.analyzer = Analysis()
        
        # Load environment variables
        load_dotenv()
        
        # Initialize OpenAI API key for analysis functions
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            logger.info("OpenAI API key loaded from environment variables")
        else:
            logger.warning("OpenAI API key not found in environment variables")
    
    def process_search_result(self, result, query_index, result_index, processed_urls, trace_id=None, parent_span_id=None):
        """
        Process a single search result to extract and analyze content.
        Enhanced with comprehensive metadata extraction using the updated Analysis class.
        
        Args:
            result: The search result dict with link, title, etc.
            query_index: Index of the query that produced this result
            result_index: Index of this result within the query results
            processed_urls: Set of already processed URLs (passed by reference)
            trace_id: Optional trace ID for monitoring
            parent_span_id: Optional parent span ID for nested tracing
                
        Returns:
            Dict containing processed content or None if extraction failed
        """
        # Configure logging
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)
        
        # Initialize Langfuse client if tracing is enabled
        langfuse = None
        if trace_id:
            try:
                from monitoring.langfuse_client import get_langfuse_client
                langfuse = get_langfuse_client()
            except (ImportError, Exception) as e:
                logger.warning(f"Could not import Langfuse client: {str(e)}")
        
        # Create a span for this result processing
        result_span = None
        if langfuse and trace_id:
            result_span = langfuse.create_span(
                trace_id=trace_id,
                name=f"process_result_{query_index}_{result_index}",
                parent_span_id=parent_span_id,
                metadata={
                    "url": result.get("link"),
                    "query_index": query_index,
                    "result_index": result_index
                }
            )
        
        try:
            # Print start of processing
            print(f"🔍 Starting to process search result {result_index+1} from query {query_index+1}")
            
            url = result.get("link")
            title = result.get("title", "Untitled")
            
            # Log processing start with more details
            logger.info(f"Processing result {result_index+1} from query {query_index+1}: {title}")
            logger.debug(f"URL: {url}")
            
            # Print before content extraction
            print(f"📥 Attempting to extract content from: {url}")
            
            # Extract content from URL using scrape_webpage method
            content, extracted_title, date, clean_summary = self.scrape_webpage(url, title)
            
            # Log content extraction
            if content:
                logger.info(f"Successfully extracted content from {url}")
                print(f"✅ Content extracted successfully. Length: {len(content)} characters")
            else:
                logger.warning(f"No content extracted from {url}")
                print(f"❌ No content extracted from {url}")
                
                # Update result span with failure
                if result_span:
                    result_span.update(
                        output={"status": "no_content"},
                        status="warning"
                    )
                
                return None
            
            # Initialize result_data dictionary to store all extracted information
            result_data = {
                "url": url,
                "title": extracted_title or title,
                "date": date,
                "content": content,
                "summary": clean_summary or ""
            }
            
            # Extract organization from URL domain
            organization = self.extract_organization_from_url(url)
            logger.info(f"Extracted organization: {organization}")
            print(f"🏢 Organization extracted: {organization}")
            
            # Add organization to result data
            result_data["organization"] = organization
            
            # Use the comprehensive analyze_content method from the Analysis class
            logger.info(f"Running comprehensive content analysis for {url}")
            print(f"🔬 Starting comprehensive content analysis")
            
            analysis_result = self.analyzer.analyze_content(
                content=content,
                title=extracted_title or title,
                url=url, 
                language=self.language
            )
            
            if not analysis_result:
                logger.warning(f"Comprehensive analysis failed for {url}")
                print(f"⚠️ Comprehensive analysis failed for {url}")
                analysis_result = {}  # Fallback to empty dict
            else:
                logger.info(f"Comprehensive analysis completed successfully for {url}")
                print(f"✅ Comprehensive analysis completed")
                
                # Merge analysis results into result_data
                result_data.update(analysis_result)
            
            # Add source type and language 
            result_data["source_type"] = "web"
            result_data["language"] = self.language
            
            # Ensure sentiment information is properly structured for storage
            if "overall_sentiment" in result_data:
                # Create sentiment_info structure if it doesn't exist
                if "sentiment_info" not in result_data:
                    result_data["sentiment_info"] = {
                        "overall_sentiment": result_data.get("overall_sentiment", "Neutral"),
                        "sentiment_score": result_data.get("sentiment_score", 0.0),
                        "sentiment_confidence": result_data.get("sentiment_confidence", 0.0)
                    }
            
            # Add a relevance score based on the content analysis if the method exists
            if hasattr(self.analyzer, '_calculate_relevance_score'):
                result_data["initiative_score"] = self.analyzer._calculate_relevance_score(result_data)
            
            # Add a final print statement for successful processing
            print(f"🎉 Successfully processed {url}")
            
            # Update result span with success
            if result_span:
                result_span.update(
                    output={
                        "status": "success",
                        "url": url,
                        "content_length": len(content),
                        "organization": organization
                    },
                    status="success"
                )
            
            return result_data
                
        except Exception as e:
            # Comprehensive error logging
            logger.error(f"Error processing {url}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Print error details
            print(f"❌ Error processing {url}")
            print(f"Error details: {str(e)}")
            print(f"Traceback: {traceback.format_exc()}")
            
            # Update result span with error
            if result_span:
                result_span.update(
                    output={
                        "status": "error",
                        "error": str(e),
                        "url": url
                    },
                    status="error"
                )
            
            return None

    def scrape_webpage(self, url: str, search_result_title: str = "") -> Tuple[str, str, str, str]:
        """
        Scrape content, title, and date from a webpage (no summary generation).
        
        Args:
            url: URL to scrape
            search_result_title: Title from search results
                
        Returns:
            Tuple of (content, title, date, None)
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
                return "", "", None, None
            
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
            
            # Log stats about the content
            scrape_time = time.time() - scrape_start_time
            logger.info(f"Successfully scraped {len(content)} chars from {url} in {scrape_time:.2f} seconds (cleaned from {original_length} to {cleaned_length} chars)")
            logger.info(f"Extracted title: {title}")
            logger.info(f"Extracted date: {date}")
            
            # No summary generation here - return None for the summary
            # This will be handled in process_search_result after embedding generation
            return content, title, date, None
            
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}", exc_info=True)
            return "", "", None, None
    
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
                
                # Get API key from class
                api_key = self.openai_api_key
                
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
    
    def extract_date_from_content(self, html_content: str, url: str, soup: BeautifulSoup) -> Optional[str]:
        """
        Extract publication date from content using multiple strategies.
        Uses regex patterns and fallback options to maximize date extraction.
        
        Args:
            html_content: Raw HTML content
            url: Source URL
            soup: BeautifulSoup object
            
        Returns:
            Date string or None if not found
        """
        logger.info(f"Extracting date from webpage: {url}")
        
        # Strategy 1: Check for common meta tags with date information
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
        
        # Strategy 2: Look for time elements with datetime attribute or text content
        time_elements = soup.find_all("time")
        for time_element in time_elements:
            if time_element.get("datetime"):
                date_str = time_element.get("datetime")
                logger.info(f"Found date in time element datetime attribute: {date_str}")
                return date_str
            elif time_element.text.strip():
                date_str = time_element.text.strip()
                logger.info(f"Found date in time element text content: {date_str}")
                return date_str
        
        # Strategy 3: Check for common data attributes that may contain date
        date_attrs = ["data-date", "data-published", "data-timestamp", "data-publishdate", "data-pubdate"]
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
                
        # Strategy 5: Look for common date patterns in the HTML using regex
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
        
        for pattern in date_patterns:
            matches = re.search(pattern, html_content, re.IGNORECASE)
            if matches:
                date_str = matches.group(1)
                logger.info(f"Found date using regex pattern in HTML: {date_str}")
                return date_str
        
        # Strategy 6: Look for common date-related class names
        date_classes = ["date", "published", "timestamp", "post-date", "article-date", "pubdate", "publishdate"]
        for cls in date_classes:
            elements = soup.find_all(class_=lambda c: c and cls in c.lower())
            if elements:
                for element in elements:
                    text = element.text.strip()
                    if text and re.search(r'\d{4}', text):  # Ensure it has a year
                        logger.info(f"Found date in element with class '{cls}': {text}")
                        return text
        
        # Fallback 1: Extract year from URL if available
        url_year_match = re.search(r'/(\d{4})/', url)
        if url_year_match:
            year = url_year_match.group(1)
            logger.info(f"Extracted year from URL: {year}")
            return f"{year}-01-01"  # Assume January 1st if only year is available
        
        # Fallback 2: Check for copyright year in the footer
        footer_elements = soup.select("footer, div.footer, span.copyright")
        for footer in footer_elements:
            text = footer.text.strip()
            year_match = re.search(r'(?:©|Copyright)\s*(\d{4})', text)
            if year_match:
                year = year_match.group(1)
                logger.info(f"Found copyright year in footer: {year}")
                return f"{year}-01-01"  # Assume January 1st if only year is available
        
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

    def _contains_relevant_content(self, result_data):
        """
        Check if result data contains relevant ABS content.
        
        Args:
            result_data: Result data dictionary
            
        Returns:
            Boolean indicating if content is relevant
        """
        if not result_data.get('content'):
            return False
                
        content = result_data.get('content', '').lower()
        title = result_data.get('title', '').lower()
        
        # Get language-appropriate ABS terms
        abs_terms = {
            "English": ["abs initiative", "capacity development", "benefit sharing", "genetic resources"],
            "German": ["abs-initiative", "kapazitätsentwicklung", "vorteilsausgleich", "genetische ressourcen"],
            "French": ["initiative apa", "développement des capacités", "partage des avantages", "ressources génétiques"]
        }
        
        # Use terms for the current language
        terms = abs_terms.get(self.language, abs_terms["English"])
        
        # Check for relevant terms
        return any(term in content or term in title for term in terms)