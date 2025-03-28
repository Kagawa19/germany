import os
import time
import requests
import logging
import re
import json
import random
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple, Any

class WebExtractor:
    """Extension of WebExtractor with search-related methods."""
    

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
        Modified to consider language-specific patterns.
        
        Args:
            results: List of search result dictionaries
            
        Returns:
            Filtered list of search results
        """
        filtered_results = []
        
        # Language-specific keywords for filtering
        language_keywords = {
            "English": ["abs", "capacity development", "initiative", "access and benefit", 
                        "nagoya protocol", "genetic resources", "traditional knowledge"],
            "German": ["abs", "kapazitätsentwicklung", "initiative", "zugang und vorteilsausgleich", 
                    "nagoya-protokoll", "genetische ressourcen", "traditionelles wissen"],
            "French": ["apa", "développement des capacités", "initiative", "accès et partage", 
                    "protocole de nagoya", "ressources génétiques", "connaissances traditionnelles"]
        }
        
        # Get keywords for current language
        current_keywords = language_keywords.get(self.language, language_keywords["English"])
        
        # Language specific domains
        preferred_domains = self.lang_settings["domains"]
        
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
            
            # Check if title or snippet contains any language-specific keywords
            # Ensure at least one relevant term in title or snippet
            if not any(term in title or term in snippet for term in current_keywords):
                continue
            
            # Calculate a preliminary relevance score for sorting
            relevance = 0
            
            # More relevant terms = higher score
            relevance += sum(term in title for term in current_keywords) * 2
            relevance += sum(term in snippet for term in current_keywords)
            
            # Bonus for language-appropriate domains
            if any(domain.endswith(tld) for tld in preferred_domains):
                relevance += 2
            
            # Add relevance score to the result
            result["preliminary_relevance"] = relevance
            
            # Add to filtered results
            filtered_results.append(result)
        
        # Sort by relevance (higher score first)
        filtered_results.sort(key=lambda x: x.get("preliminary_relevance", 0), reverse=True)
        
        return filtered_results

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

    
            
    
    def search_web(self, query: str, num_results: int = 20) -> List[Dict]:
        """
        Search the web using the given query via Serper API.
        Enhanced with language-specific parameters for better results.
        
        Args:
            query: Search query string
            num_results: Maximum number of results to retrieve
            
        Returns:
            List of dictionaries containing search results
        """
        query_preview = query[:50] + "..." if len(query) > 50 else query
        logger.info(f"Searching web for: '{query_preview}' (limit: {num_results} results, language: {self.language})")
        print(f"  Searching for: '{query_preview}' in {self.language}...")
        
        if not self.search_api_key:
            error_msg = "Cannot search web: API_KEY not set"
            logger.error(error_msg)
            print(f"  ERROR: Search API_KEY not set")
            return []
        
        try:
            search_start_time = time.time()
            
            # Use Serper API for searching
            url = "https://google.serper.dev/search"
            
            # Language-specific search parameters
            gl_params = {
                "English": "us",
                "German": "de", 
                "French": "fr"
            }
            
            hl_params = {
                "English": "en",
                "German": "de",
                "French": "fr"
            }
            
            # Enhanced query with language-specific location filters
            if self.language in ["German", "French"]:
                # For non-English languages, prioritize regional results
                enhanced_query = self._enhance_search_query(query)
            else:
                # For English, use the original query
                enhanced_query = query
            
            # Create payload with language-specific parameters
            payload = json.dumps({
                "q": enhanced_query,
                "num": num_results,
                "gl": gl_params.get(self.language, "us"),  # Geographic location
                "hl": hl_params.get(self.language, "en")   # Interface language
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
            
            # Filter results to remove junk and irrelevant content
            filtered_results = self._filter_search_results(organic_results)
            logger.info(f"Filtered to {len(filtered_results)} relevant {self.language} results")
            
            # Log result URLs for debugging
            for i, result in enumerate(filtered_results):
                logger.info(f"Result {i+1}: {result.get('title', 'No title')} - {result.get('link', 'No link')}")
            
            return filtered_results
                
        except Exception as e:
            logger.error(f"Error searching web: {str(e)}", exc_info=True)
            print(f"  ERROR searching web: {str(e)}")
            return []
