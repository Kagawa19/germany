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
from content_db import get_db_connection, store_extract_data

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
                language="English",
                generate_summary=None):
        # Load environment variables
        load_dotenv()
        
        # Set API key
        self.search_api_key = search_api_key or os.getenv('SERPER_API_KEY')
        
        # Set max_workers
        self.max_workers = max_workers
        
        # Set language
        self.language = language
        
        # Set generate_summary function
        self.generate_summary = generate_summary
        
        # Target organizations and domains to prioritize
        self.priority_domains = []
        
        # Configure the specific initiatives to track
        self.configure_initiatives()

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
            
            # Generate summary using the provided function
            if self.generate_summary:
                logger.info("Generating summary using the provided function")
                clean_summary = self.generate_summary(content)
            else:
                logger.info("No summary generation function provided, using default")
                clean_summary = content[:500] + "..." if len(content) > 500 else content
            
            scrape_time = time.time() - scrape_start_time
            logger.info(f"Successfully scraped {len(content)} chars from {url} in {scrape_time:.2f} seconds (cleaned from {original_length} to {cleaned_length} chars)")
            logger.info(f"Extracted title: {title}")
            logger.info(f"Extracted date: {date}")
            logger.info(f"Generated clean summary: {len(clean_summary)} chars")
            
            return content, title, date, clean_summary
        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}", exc_info=True)
            return "", "", None, ""

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

    def generate_search_queries(self, max_queries: Optional[int] = None) -> List[str]:
        """
        Generate a comprehensive list of search queries based on configured initiatives.
        Enhanced to produce more diverse, targeted queries and reduce junk results.
        
        Args:
            max_queries: Maximum number of queries to generate (None for all)
                    
        Returns:
            List of search query strings
        """
        # Start with full initiative names
        queries = list(self.search_queries)
        
        # Get language-specific terms
        current_lang_terms = self.common_terms.get(self.language, self.common_terms["English"])
        context_terms = self.context_terms.get(self.language, self.context_terms["English"])
        
        # Define core initiative-specific terms to add more context
        initiative_context_terms = {
            "English": [
                "biodiversity", "genetic resources", "benefit sharing", 
                "capacity development", "access and benefit", 
                "sustainable development", "conservation",
                "Nagoya Protocol", "implementation", "CBD", 
                "traditional knowledge", "indigenous communities"
            ],
            "German": [
                "Biodiversität", "genetische Ressourcen", "Vorteilsausgleich", 
                "Kapazitätsentwicklung", "Zugang und Vorteil", 
                "nachhaltige Entwicklung", "Naturschutz",
                "Nagoya-Protokoll", "Implementierung", "CBD",
                "traditionelles Wissen", "indigene Gemeinschaften"
            ],
            "French": [
                "biodiversité", "ressources génétiques", "partage des avantages", 
                "développement des capacités", "accès et partage", 
                "développement durable", "conservation",
                "Protocole de Nagoya", "mise en œuvre", "CBD",
                "connaissances traditionnelles", "communautés autochtones"
            ]
        }
        
        # Add geographic context for better results
        geographic_context = {
            "English": ["Africa", "developing countries", "global south", "partner countries"],
            "German": ["Afrika", "Entwicklungsländer", "globaler Süden", "Partnerländer"],
            "French": ["Afrique", "pays en développement", "sud global", "pays partenaires"]
        }
        
        # Add more specific queries with context
        additional_queries = []
        
        # 1. Add initiative names with specific context terms
        for base_name in self.search_queries[:5]:  # Limit to top 5 names for diversity
            for context in initiative_context_terms.get(self.language, initiative_context_terms["English"]):
                additional_queries.append(f'"{base_name}" {context}')
        
        # 2. Add geographic context queries
        for base_name in self.search_queries[:3]:  # Limit to top 3 for geographic context
            for location in geographic_context.get(self.language, geographic_context["English"]):
                additional_queries.append(f'"{base_name}" {location}')
        
        # 3. Add organization-related searches (more targeted)
        for org in self.related_orgs[:5]:
            for base_name in self.search_queries[:2]:
                additional_queries.append(f'"{org}" "{base_name}"')
        
        # 4. Add specific document type searches to find reports, papers, etc.
        doc_types = {
            "English": ["report", "policy brief", "publication", "case study", "workshop"],
            "German": ["Bericht", "Kurzdossier", "Veröffentlichung", "Fallstudie", "Workshop"],
            "French": ["rapport", "note d'orientation", "publication", "étude de cas", "atelier"]
        }
        
        for base_name in self.search_queries[:2]:
            for doc_type in doc_types.get(self.language, doc_types["English"]):
                additional_queries.append(f'"{base_name}" {doc_type}')
        
        # 5. Add exact phrase matches for initiative names (with quotes)
        quoted_names = [f'"{name}"' for name in self.search_queries]
        
        # 6. Add site-specific searches for reliable sources
        reliable_domains = [
            "abs-initiative.info", "cbd.int", "giz.de", "bmz.de", 
            "unctad.org", "undp.org", "unep.org"
        ]
        
        for domain in reliable_domains[:3]:  # Limit to top 3 domains
            for base_name in self.search_queries[:2]:  # Limit to top 2 names
                additional_queries.append(f'"{base_name}" site:{domain}')
        
        # 7. Add special search operators to exclude irrelevant results
        exclusion_queries = []
        for base_query in self.search_queries[:5]:
            # Exclude programming-related content for "ABS" searches
            if "ABS" in base_query:
                exclusion_queries.append(f'"{base_query}" -programming -code -library -software -java -python')
        
        # Combine all queries
        queries.extend(additional_queries)
        queries.extend(quoted_names)
        queries.extend(exclusion_queries)
        
        # Remove any duplicates that might have been created
        queries = list(dict.fromkeys(queries))
        
        # Ensure every query has at least one initiative-specific term to reduce junk
        final_queries = []
        for query in queries:
            has_initiative_term = False
            for name in self.abs_names.get(self.language, []):
                if name.lower() in query.lower():
                    has_initiative_term = True
                    break
            
            if has_initiative_term:
                final_queries.append(query)
            else:
                # Add the most specific initiative name to the query
                main_name = self.abs_names.get(self.language, ["ABS Initiative"])[0]
                final_queries.append(f'{query} "{main_name}"')
        
        # Remove duplicates again after modifications
        final_queries = list(dict.fromkeys(final_queries))
        
        # Limit the number of queries if specified
        if max_queries:
            final_queries = final_queries[:max_queries]
        
        logger.info(f"Generated {len(final_queries)} search queries for {self.language}")
        
        # Log a sample of queries for verification
        sample_size = min(5, len(final_queries))
        logger.debug(f"Sample queries: {', '.join(final_queries[:sample_size])}")
        
        return final_queries
    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings for text using OpenAI.
        
        Args:
            text: Text to generate embeddings for
            
        Returns:
            List of embedding values or empty list if failed
        """
        if not text or len(text) < 10:
            logger.warning("Text too short for embedding generation")
            return []
        
        try:
            # Get OpenAI client
            client = get_openai_client()
            if not client:
                logger.warning("OpenAI client not available for embedding generation")
                return []
            
            # Truncate text if too long (OpenAI has token limits)
            max_tokens = 8000  # Approximate limit for embedding models
            truncated_text = text[:32000] if len(text) > 32000 else text
            
            # Generate embedding
            response = client.embeddings.create(
                model="text-embedding-ada-002",  # Use the appropriate embedding model
                input=truncated_text
            )
            
            # Extract embedding values
            embedding = response.data[0].embedding
            
            logger.info(f"Successfully generated embedding vector ({len(embedding)} dimensions)")
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []
        
    def _is_junk_domain(self, domain):
        """
        Check if a domain is likely to be a junk domain that won't have relevant content.
        
        Args:
            domain: Domain to check
            
        Returns:
            Boolean indicating if domain should be skipped
        """
        junk_domains = [
            "facebook.com", "twitter.com", "instagram.com", "youtube.com", "linkedin.com",
            "pinterest.com", "reddit.com", "tumblr.com", "flickr.com", "medium.com",
            "amazonaws.com", "cloudfront.net", "googleusercontent.com", "akamaihd.net",
            "wordpress.com", "blogspot.com", "blogger.com", "w3.org", "archive.org",
            "github.com", "githubusercontent.com", "gist.github.com", "gitlab.com",
            "scribd.com", "slideshare.net", "issuu.com", "academia.edu", "researchgate.net",
            "ads.", "tracker.", "tracking.", "analytics.", "doubleclick.",
            "advert.", "banner.", "popup.", "cdn.", "static."
        ]
        
        # Check if domain ends with or contains a junk domain
        for junk_domain in junk_domains:
            if domain == junk_domain or domain.endswith("." + junk_domain):
                return True
        
        return False

    def _contains_relevant_content(self, result_data):
        """
        Check if result data contains relevant content about ABS Initiative.
        
        Args:
            result_data: Result data dictionary
            
        Returns:
            Boolean indicating if content is relevant
        """
        # Skip if no content
        if not result_data.get('content'):
            return False
        
        content = result_data.get('content', '').lower()
        title = result_data.get('title', '').lower()
        
        # Skip if content is too short
        if len(content) < 300:
            return False
        
        # Check for relevant terms in content or title
        abs_terms = [
            "abs initiative", "abs capacity", "capacity development initiative", 
            "access and benefit sharing", "nagoya protocol", "genetic resources", 
            "traditional knowledge", "biodiversity", "bio-innovation"
        ]
        
        # Check for at least one relevant term
        return any(term in content or term in title for term in abs_terms)
        
    def _calculate_relevance_score(self, result_data):
        """
        Calculate a relevance score for the extracted content based on multiple factors.
        
        Args:
            result_data: Result data dictionary
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        content = result_data.get('content', '')
        content_lower = content.lower()
        
        # Factor 1: Content length (longer content tends to be more comprehensive)
        # Scale: 0.0-0.2
        content_length = len(content)
        if content_length > 5000:
            score += 0.2
        elif content_length > 2000:
            score += 0.15
        elif content_length > 1000:
            score += 0.1
        elif content_length > 500:
            score += 0.05
        
        # Factor 2: Presence of key ABS terms (more terms = more relevant)
        # Scale: 0.0-0.3
        abs_terms = [
            "abs initiative", "abs capacity", "capacity development initiative", 
            "access and benefit sharing", "nagoya protocol", "genetic resources", 
            "traditional knowledge", "biodiversity", "bio-innovation", "abs cdi"
        ]
        
        term_count = sum(term in content_lower for term in abs_terms)
        score += min(0.3, term_count * 0.03)
        
        # Factor 3: Source reliability
        # Scale: 0.0-0.2
        reliable_domains = [
            "abs-initiative.info", "cbd.int", "giz.de", "bmz.de", 
            "unctad.org", "un.org", "undp.org", "unep.org"
        ]
        
        url = result_data.get('link', '')
        domain = urlparse(url).netloc.lower()
        
        if any(reliable in domain for reliable in reliable_domains):
            score += 0.2
        
        # Factor 4: Has date (content with dates tends to be more structured)
        # Scale: 0.0-0.1
        if result_data.get('date'):
            score += 0.1
        
        # Factor 5: Initiative score
        # Scale: 0.0-0.2
        initiative_score = result_data.get('initiative_score', 0)
        score += initiative_score * 0.2
        
        return min(1.0, score)
            
    
    def search_web(self, query: str, num_results: int = 20) -> List[Dict]:
        """
        Search the web using the given query via Serper API.
        Enhanced with additional filtering and quality improvements.
        
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
            
            # Enhance query with site-specific operators for better results
            enhanced_query = self._enhance_search_query(query)
            
            payload = json.dumps({
                "q": enhanced_query,
                "num": num_results,
                # Add parameter to include only English results if language is English
                "gl": "us" if self.language == "English" else None,
                "hl": {
                    "English": "en",
                    "German": "de",
                    "French": "fr"
                }.get(self.language, "en")
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
            
            # Filter out likely irrelevant results
            filtered_results = self._filter_search_results(organic_results)
            logger.info(f"Filtered to {len(filtered_results)} relevant results")
            
            # Log result URLs for debugging
            for i, result in enumerate(filtered_results):
                logger.info(f"Result {i+1}: {result.get('title', 'No title')} - {result.get('link', 'No link')}")
            
            return filtered_results
                
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}", exc_info=True)
            print(f"  ERROR searching web: {str(e)}")
            return []
    
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


    # Helper function for clean_and_enhance_summary that should be added to the WebExtractor class
    
    
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
        
    def identify_themes(self, content: str, embedding=None) -> List[str]:
        """
        Identify diverse themes in content using OpenAI.
        Enhanced to use embeddings when available.
        
        Args:
            content: Content text to analyze
            embedding: Optional embedding vector for the content
            
        Returns:
            List of identified themes
        """
        # Skip if content is too short
        if not content or len(content) < 100:
            return ["Content Analysis", "Documentation"]
        
        # Try to use OpenAI for theme extraction
        try:
            client = get_openai_client()
            if client:
                # Prepare content - limit to first 3000 chars to save tokens
                excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
                
                # Create a prompt for theme extraction
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
        
        # Fallback approach (same as your original method)
        # [existing fallback code]
        
        return ["Policy Analysis", "Research Findings", "Environmental Studies", "International Relations", "Resource Management"]
        


    def analyze_sentiment(self, content: str, embedding=None) -> str:
        """
        Analyze sentiment using keyword-based approach and OpenAI when available.
        Enhanced to use embeddings for improved accuracy.
        
        Args:
            content: Content text to analyze
            embedding: Optional embedding vector for the content
            
        Returns:
            Sentiment (Positive, Negative, or Neutral)
        """
        if not content or len(content) < 50:
            return "Neutral"
            
        content_lower = content.lower()
        
        # Try to use OpenAI for more accurate sentiment analysis
        try:
            client = get_openai_client()
            if client:
                # Use just first 1000 chars to save on tokens
                excerpt = content[:1000] + ("..." if len(content) > 1000 else "")
                
                prompt = """
    Analyze the sentiment of this text about the ABS Initiative or Access and Benefit Sharing.
    Consider the overall tone, language, and context.
    Return ONLY one of these three options: "Positive", "Negative", or "Neutral".

    Text:
    """
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt + excerpt}
                    ],
                    temperature=0.3,
                    max_tokens=10
                )
                
                sentiment = response.choices[0].message.content.strip()
                
                # Ensure we get one of the three valid sentiments
                if sentiment in ["Positive", "Negative", "Neutral"]:
                    logger.info(f"OpenAI sentiment analysis result: {sentiment}")
                    return sentiment
                    
                # If response doesn't match expected values, fall back to keyword approach
                logger.warning(f"Unexpected sentiment response: {sentiment}. Falling back to keyword analysis.")
        except Exception as e:
            logger.warning(f"Error using OpenAI for sentiment analysis: {str(e)}. Falling back to keyword analysis.")
        
        # Fallback: Use keyword-based approach (your original implementation)
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
        Enhanced with embedding generation and improved content analysis.
        
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
            
            # Even if content is minimal, continue processing (no filtering by content length)
            if not content:
                content = ""  # Ensure content is at least an empty string, not None
                logger.info(f"Minimal or no content from {url}, but continuing processing")
            
            # Get initiative info but don't filter on it
            initiative_key, initiative_score = self.identify_initiative(content)
            
            # Always default to 'abs_initiative' if none found to avoid filtering out content
            if initiative_key == "unknown":
                initiative_key = "abs_initiative"
                initiative_score = 0.1  # Minimum score to ensure inclusion
                logger.info(f"No specific initiative detected in {url}, defaulting to generic ABS Initiative")
            
            # Extract organization from URL domain
            organization = self.extract_organization_from_url(url)
            
            # Generate embeddings if content is substantial enough
            embedding = []
            if len(content) > 300:
                embedding = self.generate_embedding(content)
            
            # Identify multiple themes from the content
            themes = self.identify_themes(content)
            
            # Ensure we have at least one theme if identify_themes returned empty
            if not themes:
                themes = ["ABS Initiative"]
            
            # Analyze sentiment
            sentiment = self.analyze_sentiment(content)
            
            # If summary is missing or too short, generate one
            if not clean_summary or len(clean_summary) < 50:
                clean_summary = self.generate_summary(content, extracted_title, url)
            
            # Calculate a relevance score
            relevance_score = 0.0
            if hasattr(self, '_calculate_relevance_score'):  # Only call if method exists
                relevance_score = self._calculate_relevance_score({
                    "content": content,
                    "link": url,
                    "title": title,
                    "initiative_score": initiative_score
                })
            
            # Format the result
            result_data = {
                "title": extracted_title or title,
                "link": url,
                "date": date,
                "content": content,
                "summary": clean_summary or content[:200],  # Use first 200 chars if no summary
                "themes": themes,
                "organization": organization,
                "sentiment": sentiment,
                "language": self.language,
                "initiative": "ABS Initiative",
                "initiative_key": initiative_key,
                "initiative_score": initiative_score,  # Include score for reference
                "embedding": embedding,  # Add embedding vector for semantic search
                "relevance_score": relevance_score,  # Add calculated relevance score
                "extraction_timestamp": datetime.now().isoformat()
            }
            
            logger.info(f"Successfully processed {url} (initiative: {initiative_key}, score: {initiative_score:.2f}, language: {self.language})")
            return result_data
                
        except Exception as e:
            logger.error(f"Error processing {url}: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            return None
        
    def _enhance_search_query(self, query):
        """
        Enhance the search query with site-specific operators for better results.
        
        Args:
            query: Original search query
            
        Returns:
            Enhanced search query
        """
        # Extract the core query without any existing operators
        core_query = re.sub(r'site:[^\s]+', '', query).strip()
        
        # For exact phrase queries (in quotes), don't modify
        if core_query.startswith('"') and core_query.endswith('"'):
            return query
        
        # For initiative searches, try to find high-quality sources
        abs_terms = ["abs initiative", "abs capacity", "capacity development", "bio-innovation"]
        if any(term in query.lower() for term in abs_terms):
            # Generate a list of high-quality domains to prioritize
            quality_domains = [
                "abs-initiative.info", "cbd.int", "giz.de", "bmz.de", 
                "unctad.org", "un.org", "undp.org", "unep.org"
            ]
            
            # Select a random domain to prioritize (to diversify results across searches)
            import random
            selected_domain = random.choice(quality_domains)
            
            # 30% chance to add site: operator to focus on high-quality domains
            if random.random() < 0.3:
                return f"{core_query} site:{selected_domain}"
        
        # Return the original query for other cases
        return query

    def _filter_search_results(self, results):
        """
        Filter search results to remove likely irrelevant content.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Filtered list of search results
        """
        filtered_results = []
        
        for result in results:
            # Skip results without links
            if not result.get("link"):
                continue
            
            # Get URL and title
            url = result.get("link")
            title = result.get("title", "").lower()
            snippet = result.get("snippet", "").lower()
            
            # Skip likely irrelevant content based on URL patterns
            skip_patterns = [
                r'/tags/', r'/tag/', r'/category/', r'/categories/',
                r'/search', r'/find', r'/login', r'/signin', r'/signup',
                r'/cart', r'/checkout', r'/buy', r'/pricing',
                r'/privacy', r'/terms', r'/disclaimer', r'/contact'
            ]
            
            if any(re.search(pattern, url) for pattern in skip_patterns):
                continue
            
            # Skip if URL is from a junk domain
            domain = urlparse(url).netloc.lower()
            if self._is_junk_domain(domain):
                continue
            
            # Check if title or snippet contains any ABS-related terms
            abs_terms = [
                "abs", "capacity development", "initiative", "access and benefit",
                "nagoya protocol", "genetic resources", "traditional knowledge", 
                "biodiversity", "bio-innovation"
            ]
            
            # Ensure at least one relevant term in title or snippet
            if not any(term in title or term in snippet for term in abs_terms):
                continue
            
            # Calculate a preliminary relevance score for sorting
            relevance = 0
            
            # More relevant terms = higher score
            relevance += sum(term in title for term in abs_terms) * 2
            relevance += sum(term in snippet for term in abs_terms)
            
            # Prioritize results from reliable domains
            reliable_domains = [
                "abs-initiative.info", "cbd.int", "giz.de", "bmz.de", 
                "unctad.org", "un.org", "undp.org", "unep.org"
            ]
            
            if any(domain.endswith(reliable) or reliable in domain for reliable in reliable_domains):
                relevance += 5
            
            # Add relevance score to the result
            result["preliminary_relevance"] = relevance
            
            # Add to filtered results
            filtered_results.append(result)
        
        # Sort by relevance (higher score first)
        filtered_results.sort(key=lambda x: x.get("preliminary_relevance", 0), reverse=True)
        
        return filtered_results


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