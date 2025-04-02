import os
import logging
import json
import re
from datetime import datetime
from urllib.parse import urlparse
import nltk
from nltk.tokenize import sent_tokenize
from typing import Dict, List, Optional, Tuple, Any

from dotenv import load_dotenv
from openai import OpenAI

logger = logging.getLogger("analysis")

class Analysis:
    """Extension of WebExtractor with comprehensive content analysis methods."""
    
    def __init__(self):
        """Initialize Analysis class"""
        # Load environment variables
        load_dotenv()
        
        # Load OpenAI API key
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if self.openai_api_key:
            logger.info("OpenAI API key loaded from environment variables")
        else:
            logger.warning("OpenAI API key not found in environment variables")
        
        # Initialize OpenAI client
        self.openai_client = None
        if self.openai_api_key:
            try:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
                logger.info("OpenAI client initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize OpenAI client: {str(e)}")
        
        # Initialize prompt loader - does not create or load prompts here
        # They will be loaded on demand from the prompts folder
        self.prompts_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "prompts")
    
    def _load_prompt(self, prompt_name: str) -> str:
        """
        Load a prompt from the prompts directory.
        
        Args:
            prompt_name: Name of the prompt file (without .txt extension)
            
        Returns:
            Prompt content or empty string if not found
        """
        prompt_path = os.path.join(self.prompts_dir, f"{prompt_name}.txt")
        try:
            if os.path.exists(prompt_path):
                with open(prompt_path, "r", encoding="utf-8") as f:
                    return f.read().strip()
            else:
                logger.warning(f"Prompt file {prompt_name}.txt not found")
                return ""
        except Exception as e:
            logger.error(f"Error loading prompt {prompt_name}: {str(e)}")
            return ""
        
    def _extract_with_ai(self, prompt_name: str, content: str, expected_key: str, 
                  default_value=None, truncate_length: int = 3000) -> Any:
        """
        Generic method to extract information using OpenAI with robust error handling.
        
        Args:
            prompt_name: Name of the prompt to use
            content: Content to analyze
            expected_key: The expected key in the API response
            default_value: Default value to return if extraction fails
            truncate_length: Maximum content length to send
            
        Returns:
            Extracted information or default value if failed
        """
        if not content or len(content) < 100:
            return default_value if default_value is not None else []
        
        if not self.openai_client:
            logger.error(f"OpenAI client not available for {prompt_name} extraction")
            return default_value if default_value is not None else []
        
        try:
            # Load prompt
            prompt = self._load_prompt(prompt_name)
            if not prompt:
                logger.error(f"{prompt_name} prompt not available")
                return default_value if default_value is not None else []
            
            # Truncate content to save tokens
            excerpt = content[:truncate_length] + ("..." if len(content) > truncate_length else "")
            
            # Format the prompt - handle both with content and excerpt
            if "{content}" in prompt:
                formatted_prompt = prompt.format(content=excerpt)
            elif "{excerpt}" in prompt:
                formatted_prompt = prompt.format(excerpt=excerpt)
            else:
                # Default to using excerpt
                formatted_prompt = prompt + f"\n\nHere's the content to analyze:\n{excerpt}"
            
            # Make API call with explicit JSON format
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that always responds with valid JSON."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            
            # Get the response content
            result_text = response.choices[0].message.content.strip()
            
            # Debug log the raw response
            logger.debug(f"Raw API response for {prompt_name}: {result_text}")
            
            # Try to parse the JSON with multiple fallbacks
            result = None
            
            # First attempt - direct parsing
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                logger.warning(f"Initial JSON parsing failed for {prompt_name}")
                
                try:
                    # Second attempt - fix common formatting issues
                    clean_text = result_text.replace('\n', ' ').replace('\r', ' ')
                    result = json.loads(clean_text)
                except json.JSONDecodeError:
                    logger.warning(f"Second JSON parsing attempt failed for {prompt_name}")
                    
                    try:
                        # Third attempt - try to extract JSON part if there's text before/after
                        if '{' in result_text and '}' in result_text:
                            json_part = result_text[result_text.find('{'):result_text.rfind('}')+1]
                            result = json.loads(json_part)
                    except json.JSONDecodeError:
                        logger.error(f"All JSON parsing attempts failed for {prompt_name}")
                        return default_value if default_value is not None else []
            
            # If we got a result but it doesn't have the expected key
            if result and expected_key not in result:
                logger.warning(f"Response doesn't contain expected key '{expected_key}' for {prompt_name}")
                
                # Check if the whole response is the intended value (without the expected key wrapper)
                if isinstance(result, list) and default_value is None:
                    logger.info(f"Using direct list response for {prompt_name}")
                    return result
                elif isinstance(result, dict) and len(result) > 0:
                    # If there's only one key, try using that instead
                    if len(result) == 1:
                        only_key = list(result.keys())[0]
                        logger.info(f"Using alternative key '{only_key}' instead of '{expected_key}' for {prompt_name}")
                        return result[only_key]
                
                # Return default as fallback
                return default_value if default_value is not None else []
            
            # If all goes well, return the expected data
            if result and expected_key in result:
                return result[expected_key]
            
            # Final fallback
            return default_value if default_value is not None else []
            
        except Exception as e:
            logger.error(f"Error in {prompt_name} extraction: {str(e)}")
            return default_value if default_value is not None else []
        
    
    def extract_geographic_focus(self, content: str) -> List[Dict[str, str]]:
        """
        Extract geographic focus information from content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with country, region, and scope information
        """
        if not content or len(content) < 100:
            return []
        
        if not self.openai_client:
            logger.error("OpenAI client not available for geographic focus extraction")
            return []
        
        try:
            # Load geographic focus prompt
            prompt = self._load_prompt("geographic_focus")
            if not prompt:
                logger.error("Geographic focus prompt not available")
                return []
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(excerpt=excerpt)
            
            # Make API call with explicit JSON response format
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Get the response content
            result_text = response.choices[0].message.content.strip()
            
            # Ensure proper JSON format before parsing
            if not result_text.startswith('{'):
                result_text = '{' + result_text.split('{', 1)[1]
            
            # Parse the result, handling potential JSON errors
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing error: {str(json_err)}")
                logger.debug(f"Problematic JSON: {result_text}")
                # Try to fix common JSON issues
                result_text = result_text.replace('\n', ' ').replace('\r', '')
                # Try again with cleaned text
                try:
                    result = json.loads(result_text)
                except:
                    # Last resort fallback
                    return []
            
            # Convert to expected format
            geographic_data = []
            if "locations" in result and isinstance(result["locations"], list):
                for location in result["locations"]:
                    geo_item = {
                        "country": location.get("country", ""),
                        "region": location.get("region", ""),
                        "scope": location.get("scope", "")
                    }
                    geographic_data.append(geo_item)
            
            return geographic_data
            
        except Exception as e:
            logger.error(f"Error extracting geographic focus: {str(e)}")
            return []

    
    def extract_organizations(self, content: str) -> List[Dict[str, str]]:
        """
        Extract organization information from content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with organization information
        """
        if not content or len(content) < 100:
            return []
        
        if not self.openai_client:
            logger.error("OpenAI client not available for organization extraction")
            return []
        
        try:
            # Load organizations prompt
            prompt = self._load_prompt("organizations")
            if not prompt:
                logger.error("Organizations prompt not available")
                return []
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(excerpt=excerpt)
            
            # Make API call with explicit JSON format
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Get the response content
            result_text = response.choices[0].message.content.strip()
            
            # Ensure proper JSON format before parsing
            if not result_text.startswith('{'):
                result_text = '{' + result_text.split('{', 1)[1]
            
            # Parse the result with error handling
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError as json_err:
                logger.error(f"JSON parsing error: {str(json_err)}")
                logger.debug(f"Problematic JSON: {result_text}")
                # Try to fix common JSON issues
                result_text = result_text.replace('\n', ' ').replace('\r', '')
                # Try again with cleaned text
                try:
                    result = json.loads(result_text)
                except:
                    # Last resort fallback
                    return []
            
            # Convert to expected format
            organizations = []
            if "organizations" in result and isinstance(result["organizations"], list):
                for org in result["organizations"]:
                    org_item = {
                        "name": org.get("name", ""),
                        "organization_type": org.get("type", ""),
                        "relationship": org.get("relationship", ""),
                        "website": org.get("website", ""),
                        "description": org.get("description", "")
                    }
                    organizations.append(org_item)
            
            return organizations
            
        except Exception as e:
            logger.error(f"Error extracting organizations: {str(e)}")
            return []

    

    def extract_abs_mentions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract specific mentions of ABS Initiative from content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with mention information
        """
        mentions_data = self._extract_with_ai("abs_mentions", content, "mentions", [])
        
        # Process mentions data if successfully extracted
        mentions = []
        if isinstance(mentions_data, list):
            for idx, mention in enumerate(mentions_data):
                if not isinstance(mention, dict):
                    continue
                    
                # Handle potential type conversion issues safely
                try:
                    relevance = mention.get("relevance", 0.5)
                    if isinstance(relevance, str):
                        relevance = float(relevance)
                except:
                    relevance = 0.5
                    
                mention_item = {
                    "name_variant": mention.get("variant", ""),
                    "mention_context": mention.get("context", ""),
                    "mention_type": mention.get("type", ""),
                    "relevance_score": relevance,
                    "mention_position": idx + 1
                }
                mentions.append(mention_item)
        
        return mentions

    def extract_geographic_focus(self, content: str) -> List[Dict[str, str]]:
        """
        Extract geographic focus information from content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with country, region, and scope information
        """
        locations_data = self._extract_with_ai("geographic_focus", content, "locations", [])
        
        # Process geographic data if successfully extracted
        geographic_data = []
        if isinstance(locations_data, list):
            for location in locations_data:
                if not isinstance(location, dict):
                    continue
                    
                geo_item = {
                    "country": location.get("country", ""),
                    "region": location.get("region", ""),
                    "scope": location.get("scope", "")
                }
                geographic_data.append(geo_item)
        
        return geographic_data

    def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenAI API with prompt from file.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary with sentiment information
        """
        default_sentiment = {"overall_sentiment": "Neutral", "sentiment_score": 0.0, "sentiment_confidence": 0.0}
        
        # Get sentiment data with a special handler for direct sentiment
        sentiment_data = self._extract_with_ai("sentiment", content, "sentiment", None)
        
        # If we got back a string (direct sentiment value), construct a result
        if isinstance(sentiment_data, str):
            return {
                "overall_sentiment": sentiment_data,
                "sentiment_score": 0.0,
                "sentiment_confidence": 0.5
            }
        
        # If we got a dictionary, it might be the full result or might have nested data
        if isinstance(sentiment_data, dict):
            # Check if it already has the expected fields
            if "overall_sentiment" in sentiment_data:
                return sentiment_data
                
            # Otherwise, try to extract values from whatever keys are present
            result = {}
            
            if "sentiment" in sentiment_data:
                result["overall_sentiment"] = sentiment_data["sentiment"]
            else:
                result["overall_sentiment"] = "Neutral"
                
            try:
                if "score" in sentiment_data:
                    score = sentiment_data["score"]
                    result["sentiment_score"] = float(score) if score is not None else 0.0
                else:
                    result["sentiment_score"] = 0.0
            except:
                result["sentiment_score"] = 0.0
                
            try:
                if "confidence" in sentiment_data:
                    confidence = sentiment_data["confidence"]
                    result["sentiment_confidence"] = float(confidence) if confidence is not None else 0.5
                else:
                    result["sentiment_confidence"] = 0.5
            except:
                result["sentiment_confidence"] = 0.5
                
            return result
        
        # If result was None or some other unexpected type
        return default_sentiment

    def extract_organizations(self, content: str) -> List[Dict[str, str]]:
        """
        Extract organization information from content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with organization information
        """
        orgs_data = self._extract_with_ai("organizations", content, "organizations", [])
        
        # Process organization data if successfully extracted
        organizations = []
        if isinstance(orgs_data, list):
            for org in orgs_data:
                if not isinstance(org, dict):
                    continue
                    
                org_item = {
                    "name": org.get("name", ""),
                    "organization_type": org.get("type", ""),
                    "relationship": org.get("relationship", ""),
                    "website": org.get("website", ""),
                    "description": org.get("description", "")
                }
                organizations.append(org_item)
        
        return organizations

    def extract_resources(self, content: str) -> List[Dict[str, str]]:
        """
        Extract resources mentioned in the content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with resource information
        """
        resources_data = self._extract_with_ai("resources", content, "resources", [])
        
        # Process resource data if successfully extracted
        resources = []
        if isinstance(resources_data, list):
            for resource in resources_data:
                if not isinstance(resource, dict):
                    continue
                    
                resource_item = {
                    "resource_type": resource.get("type", ""),
                    "resource_name": resource.get("name", ""),
                    "resource_url": resource.get("url", ""),
                    "description": resource.get("description", "")
                }
                resources.append(resource_item)
        
        return resources

    def extract_project_details(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract project details from content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with project information
        """
        projects_data = self._extract_with_ai("project_details", content, "projects", [])
        
        # Process project data if successfully extracted
        projects = []
        if isinstance(projects_data, list):
            for project in projects_data:
                if not isinstance(project, dict):
                    continue
                    
                # Convert dates to proper format if they exist
                start_date = None
                if "start_date" in project and project["start_date"]:
                    try:
                        start_date = datetime.strptime(project["start_date"], "%Y-%m-%d").date()
                    except:
                        pass
                
                end_date = None
                if "end_date" in project and project["end_date"]:
                    try:
                        end_date = datetime.strptime(project["end_date"], "%Y-%m-%d").date()
                    except:
                        pass
                
                project_item = {
                    "project_name": project.get("name", ""),
                    "project_type": project.get("type", ""),
                    "start_date": start_date,
                    "end_date": end_date,
                    "status": project.get("status", ""),
                    "description": project.get("description", "")
                }
                projects.append(project_item)
        
        return projects
    def identify_themes(self, content: str) -> List[str]:
        """
        Identify themes in content using OpenAI with a prompt from file.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of identified themes
        """
        # Skip if content is too short
        if not content or len(content) < 100:
            return []
        
        if not self.openai_client:
            logger.error("OpenAI client not available for theme identification")
            return []
        
        try:
            # Load themes prompt
            prompt = self._load_prompt("themes")
            if not prompt:
                logger.error("Themes prompt not available")
                return []
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(excerpt=excerpt)
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.7,  # Slightly higher temperature for more diversity
                max_tokens=150
            )
            
            # Extract themes from response
            themes_text = response.choices[0].message.content.strip()
            
            # Convert to list and clean up each theme
            themes = [theme.strip() for theme in themes_text.split(',')]
            
            # Remove any empty themes
            return [theme for theme in themes if theme]
            
        except Exception as e:
            logger.error(f"Error identifying themes: {str(e)}")
            return []
    
    
    
    
    
    
    
    
    def extract_target_audiences(self, content: str) -> List[str]:
        """
        Extract target audiences mentioned in the content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of target audience types
        """
        if not content or len(content) < 100:
            return []
        
        if not self.openai_client:
            logger.error("OpenAI client not available for target audience extraction")
            return []
        
        try:
            # Load target audiences prompt
            prompt = self._load_prompt("target_audiences")
            if not prompt:
                logger.error("Target audiences prompt not available")
                return []
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(excerpt=excerpt)
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            # Extract audience types from response
            audience_text = response.choices[0].message.content.strip()
            
            # Parse as JSON if possible
            try:
                audience_data = json.loads(audience_text)
                if "audiences" in audience_data and isinstance(audience_data["audiences"], list):
                    return audience_data["audiences"]
            except:
                # Fall back to parsing comma-separated text
                audiences = [audience.strip() for audience in audience_text.split(',')]
                return [audience for audience in audiences if audience]
            
        except Exception as e:
            logger.error(f"Error extracting target audiences: {str(e)}")
            return []
    
    
    
    def generate_summary(self, content: str, title: str = "", url: str = "", language: str = "English") -> str:
        """
        Generate a high-quality summary using OpenAI API with a prompt from file.
        
        Args:
            content: Content text to summarize
            title: Content title for context
            url: Source URL for context
            language: Content language
            
        Returns:
            AI-generated summary text
        """
        # Skip if no content or too short
        if not content or len(content) < 100:
            logger.info("Content too short for summarization")
            return "No content available for summarization."

        if not self.openai_client:
            logger.warning("OpenAI client not available for summary generation")
            return "Summary generation not available."
        
        try:
            # Load summary prompt
            prompt = self._load_prompt("summary")
            if not prompt:
                logger.error("Summary prompt not available")
                return "Summary generation failed: prompt not available."
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(
                title=title,
                url=url,
                language=language,
                excerpt=excerpt
            )
            
            # Generate summary
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, accurate summaries."},
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,  # Lower temperature for more deterministic output
                max_tokens=300
            )
            
            # Extract and clean summary
            summary = response.choices[0].message.content.strip()
            
            if len(summary) > 50:
                logger.info(f"Generated summary ({len(summary)} chars)")
                return summary
            else:
                logger.warning("Generated summary too short")
                return "Generated summary was insufficient."
            
        except Exception as e:
            logger.error(f"Summary generation error: {str(e)}")
            return f"Unable to generate summary due to technical issues."
    
    def generate_embedding(self, text: str, language: str = "English") -> List[float]:
        """
        Generate embeddings for text using OpenAI.
        Support multilingual content.
        
        Args:
            text: Text to generate embeddings for
            language: Content language for logging
            
        Returns:
            List of embedding values or empty list if failed
        """
        if not text or len(text) < 10:
            logger.warning(f"Text too short for {language} embedding generation")
            return []
        
        try:
            # Get OpenAI client from instance
            client = self.openai_client
            if not client:
                logger.warning(f"OpenAI client not available for {language} embedding generation")
                return []
            
            # Truncate text if too long (OpenAI has token limits)
            max_tokens = 8000  # Approximate limit for embedding models
            truncated_text = text[:32000] if len(text) > 32000 else text
            
            # Generate embedding
            response = client.embeddings.create(
                model="text-embedding-ada-002",  # This model supports multilingual text
                input=truncated_text
            )
            
            # Extract embedding values
            embedding = response.data[0].embedding
            
            logger.info(f"Successfully generated {language} embedding vector ({len(embedding)} dimensions)")
            return embedding
        
        except Exception as e:
            logger.error(f"Error generating {language} embedding: {str(e)}")
            return []
    
    def extract_benefits(self, content: str) -> Dict[str, Any]:
        """
        Extract benefits and benefit categories using OpenAI.
        
        Args:
            content: Text content to analyze
            
        Returns:
            Dictionary with benefits summary and categorized benefits
        """
        if not content or len(content) < 100:
            return {"benefits_summary": "", "benefit_categories": {}, "benefit_examples": []}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for benefits extraction")
            return {"benefits_summary": "", "benefit_categories": {}, "benefit_examples": []}
        
        # First, extract the benefit summary
        benefits_summary = ""
        try:
            # Load benefits prompt
            prompt = self._load_prompt("benefits")
            if prompt:
                # Truncate content to save tokens
                excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
                
                # Format the prompt
                formatted_prompt = prompt.format(excerpt=excerpt)
                
                # Make API call
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.3,
                    max_tokens=300
                )
                
                benefits_summary = response.choices[0].message.content.strip()
                logger.info(f"Extracted benefits summary ({len(benefits_summary)} chars)")
            else:
                logger.error("Benefits prompt not available")
        except Exception as e:
            logger.error(f"Error extracting benefits summary: {str(e)}")
        
        # Next, extract benefit categories
        benefit_categories = {}
        try:
            # Load category extraction prompt
            prompt = self._load_prompt("category_extraction")
            if prompt:
                # Format the prompt
                formatted_prompt = prompt.format(excerpt=excerpt if 'excerpt' in locals() else content[:3000])
                
                # Make API call
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": formatted_prompt}
                    ],
                    temperature=0.3,
                    response_format={"type": "json_object"},
                    max_tokens=300
                )
                
                # Parse the result
                benefit_categories = json.loads(response.choices[0].message.content)
                logger.info(f"Extracted benefit categories: {list(benefit_categories.keys())}")
            else:
                logger.error("Category extraction prompt not available")
        except Exception as e:
            logger.error(f"Error extracting benefit categories: {str(e)}")
        
        # Finally, extract specific benefit examples
        benefit_examples = []
        try:
            # Split content into paragraphs
            paragraphs = content.split('\n\n')
            
            # Define the main benefit categories
            main_categories = [
                "environmental_benefits",
                "economic_benefits",
                "social_benefits",
                "strategic_benefits"
            ]
            
            # Analyze each paragraph for benefits
            for paragraph in paragraphs:
                if len(paragraph) < 50:  # Skip very short paragraphs
                    continue
                
                # Check which category this paragraph best fits
                best_category = "general"
                best_score = 0.0
                
                # Use the category scores from benefit_categories if available
                if benefit_categories:
                    # Load paragraph categorization prompt
                    prompt = self._load_prompt("paragraph_categorization")
                    if prompt:
                        # Format the prompt
                        formatted_prompt = prompt.format(
                            paragraph=paragraph,
                            categories=", ".join(main_categories)
                        )
                        
                        # Make API call
                        response = self.openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[
                                {"role": "user", "content": formatted_prompt}
                            ],
                            temperature=0.3,
                            response_format={"type": "json_object"},
                            max_tokens=200
                        )
                        
                        # Parse the result
                        result = json.loads(response.choices[0].message.content)
                        if "category" in result and "score" in result:
                            best_category = result["category"]
                            best_score = float(result["score"])
                
                # Only add if it's a meaningful benefit example
                if best_score > 0.3:
                    benefit_example = {
                        "text": paragraph.strip(),
                        "category": best_category,
                        "word_count": len(paragraph.split())
                    }
                    benefit_examples.append(benefit_example)
        except Exception as e:
            logger.error(f"Error extracting benefit examples: {str(e)}")
        
        return {
            "benefits_summary": benefits_summary,
            "benefit_categories": benefit_categories,
            "benefit_examples": benefit_examples
        }
    
    def analyze_content(self, content: str, title: str = "", url: str = "", language: str = "English") -> Dict[str, Any]:
        """
        Comprehensive content analysis method that calls all other methods.
        
        Args:
            content: Content text to analyze
            title: Content title for context
            url: Source URL for context
            language: Content language
            
        Returns:
            Dictionary with all extracted information
        """
        # Skip if content is too short
        if not content or len(content) < 100:
            logger.info("Content too short for comprehensive analysis")
            return {"error": "Content too short for analysis"}
        
        # Initialize result dictionary
        result = {
            "title": title,
            "url": url,
            "language": language,
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Generate embedding
        embedding = self.generate_embedding(content, language)
        if embedding:
            result["embedding"] = embedding
        
        # Generate summary
        summary = self.generate_summary(content, title, url, language)
        if summary:
            result["summary"] = summary
        
        # Analyze sentiment
        sentiment_info = self.analyze_sentiment(content)
        if sentiment_info:
            # Structure sentiment information specifically for storage
            result["sentiment_info"] = sentiment_info
            # Add direct field access for convenience
            result["overall_sentiment"] = sentiment_info.get("overall_sentiment", "Neutral")
            result["sentiment_score"] = sentiment_info.get("sentiment_score", 0.0)
            result["sentiment_confidence"] = sentiment_info.get("sentiment_confidence", 0.0)
        
        # Identify themes
        themes = self.identify_themes(content)
        if themes:
            result["themes"] = themes
        
        # Extract ABS mentions
        mentions = self.extract_abs_mentions(content)
        if mentions:
            result["abs_mentions"] = mentions
        
        # Extract geographic focus
        geo_focus = self.extract_geographic_focus(content)
        if geo_focus:
            result["geographic_focus"] = geo_focus
        
        # Extract project details
        projects = self.extract_project_details(content)
        if projects:
            result["projects"] = projects
        
        # Extract organizations
        organizations = self.extract_organizations(content)
        if organizations:
            result["organizations"] = organizations
        
        # Extract resources
        resources = self.extract_resources(content)
        if resources:
            result["resources"] = resources
        
        # Extract target audiences
        audiences = self.extract_target_audiences(content)
        if audiences:
            result["target_audiences"] = audiences
        
        # Extract benefits
        benefits_info = self.extract_benefits(content)
        if benefits_info:
            # Ensure benefit categories and examples are properly structured
            if "benefit_categories" in benefits_info:
                result["benefit_categories"] = benefits_info["benefit_categories"]
            
            if "benefit_examples" in benefits_info:
                result["benefit_examples"] = benefits_info["benefit_examples"]
            
            if "benefits_summary" in benefits_info:
                result["benefits_summary"] = benefits_info["benefits_summary"]
        
        # Generate additional benefit examples if not already present
        if "benefit_examples" not in result:
            benefit_examples = self.extract_benefit_examples(content)
            if benefit_examples:
                result["benefit_examples"] = benefit_examples
        
        return result
    
    def validate_summary(self, summary: str, original_content: str) -> Dict[str, Any]:
        """
        Validate that a summary contains only information present in the original content.
        
        Args:
            summary: Generated summary to validate
            original_content: Original content to check against
            
        Returns:
            Dictionary with validation results
        """
        if not self.openai_client:
            # If we can't validate, assume it's valid but log a warning
            logger.warning("Cannot validate summary: OpenAI client not available")
            return {"valid": True, "issues": []}
        
        try:
            # Load validation prompt
            prompt = self._load_prompt("summary_validation")
            if not prompt:
                # Use a default prompt
                prompt = """
    Your task is to verify if the summary below contains ONLY information that is explicitly stated in the original content.

    Summary to validate:
    {summary}

    Original content:
    {content}

    Check for these issues:
    1. Hallucinations - information in the summary not present in the original
    2. Misrepresentations - information that distorts what's in the original
    3. Omissions of critical context that change meaning

    Return a JSON object with the following structure:
    {
    "valid": true/false,
    "issues": ["specific issue 1", "specific issue 2", ...],
    "explanation": "Brief explanation of validation result"
    }
    """
            
            # Format the prompt
            formatted_prompt = prompt.format(summary=summary, content=original_content[:3000])
            
            # Make API call for validation
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                response_format={"type": "json_object"},
                temperature=0.2,
                max_tokens=300
            )
            
            # Parse response
            validation_result = json.loads(response.choices[0].message.content)
            logger.info(f"Summary validation result: {validation_result['valid']}")
            
            if not validation_result['valid']:
                issues = validation_result.get('issues', [])
                logger.warning(f"Summary validation issues: {issues}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Error validating summary: {str(e)}")
            # If validation fails, assume the summary is valid but log the error
            return {"valid": True, "issues": []}

    def _calculate_relevance_score(self, result_data: Dict[str, Any]) -> float:
        """
        Calculate a relevance score for the extracted content based on multiple factors.
        
        Args:
            result_data: Result data dictionary
            
        Returns:
            Relevance score (0-1)
        """
        score = 0.0
        content = result_data.get('content', '')
        title = result_data.get('title', '')
        url = result_data.get('link', '')
        content_lower = content.lower()
        title_lower = title.lower()
        
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
        
        # Adjust terms based on language
        if hasattr(self, 'language') and self.language == "German":
            abs_terms = [
                "abs-initiative", "abs kapazität", "kapazitätsentwicklungsinitiative",
                "zugang und vorteilsausgleich", "nagoya-protokoll", "genetische ressourcen",
                "traditionelles wissen", "biodiversität", "bio-innovation", "abs cdi"
            ]
        elif hasattr(self, 'language') and self.language == "French":
            abs_terms = [
                "initiative apa", "capacité apa", "initiative de développement des capacités",
                "accès et partage des avantages", "protocole de nagoya", "ressources génétiques",
                "connaissances traditionnelles", "biodiversité", "bio-innovation", "apa cdi"
            ]
        
        # Check title (more importance)
        title_term_count = sum(term in title_lower for term in abs_terms)
        score += min(0.15, title_term_count * 0.05)
        
        # Check content
        content_term_count = sum(term in content_lower for term in abs_terms)
        score += min(0.15, content_term_count * 0.03)
        
        # Factor 3: Source reliability
        # Scale: 0.0-0.2
        reliable_domains = [
            "abs-initiative.info", "cbd.int", "giz.de", "bmz.de", 
            "unctad.org", "un.org", "undp.org", "unep.org"
        ]
        
        domain = urlparse(url).netloc.lower()
        
        if any(reliable in domain for reliable in reliable_domains):
            score += 0.2
        
        # Factor 4: Has date (content with dates tends to be more structured)
        # Scale: 0.0-0.1
        if result_data.get('date'):
            score += 0.1
        
        # Factor 5: Initiative score if available
        # Scale: 0.0-0.2
        initiative_score = result_data.get('initiative_score', 0)
        score += initiative_score * 0.2
        
        # Factor 6: Has ABS mentions if available
        # Scale: 0.0-0.1
        abs_mentions = result_data.get('abs_mentions', [])
        if abs_mentions and len(abs_mentions) > 0:
            score += min(0.1, len(abs_mentions) * 0.02)
        
        # Factor 7: Geographic focus if available
        # Scale: 0.0-0.05
        geographic_focus = result_data.get('geographic_focus', [])
        if geographic_focus and len(geographic_focus) > 0:
            score += min(0.05, len(geographic_focus) * 0.01)
        
        # Factor 8: Has organizational relationships if available
        # Scale: 0.0-0.05
        organizations = result_data.get('organizations', [])
        if organizations and len(organizations) > 0:
            score += min(0.05, len(organizations) * 0.01)
        
        return min(1.0, score)

    def is_high_quality_content(self, content: str, title: str, url: str) -> bool:
        """
        Determine if content is high quality enough for AI processing.
        
        Args:
            content: The full content text
            title: The content title
            url: The source URL
            
        Returns:
            Boolean indicating if content is high quality
        """
        # Skip if content is too short
        if len(content) < 500:
            logger.info(f"Content too short for quality AI processing: {len(content)} chars")
            return False
        
        # Skip if content doesn't mention relevant terms
        relevant_terms = ["abs", "capacity", "bio-innovation", "nagoya", "africa", "initiative"]
        
        # Adjust terms based on language
        if hasattr(self, 'language'):
            if self.language == "German":
                relevant_terms = ["abs", "kapazität", "bio-innovation", "nagoya", "afrika", "initiative"]
            elif self.language == "French":
                relevant_terms = ["apa", "capacité", "bio-innovation", "nagoya", "afrique", "initiative"]
        
        content_lower = content.lower()
        if not any(term in content_lower for term in relevant_terms):
            logger.info(f"Content doesn't mention relevant terms, skipping AI processing")
            return False
        
        # Check if the content is from a reliable domain
        reliable_domains = ["giz.de", "bmz.de", "kfw.de", "europa.eu", "un.org", "abs-initiative.info", "cbd.int"]
        is_reliable_source = any(domain in url.lower() for domain in reliable_domains)
        
        # Look for quality indicators in the content
        quality_keywords = [
            "cooperation", "sustainable", "development", "partnership", "initiative",
            "project", "bilateral", "agreement", "funding", "investment", 
            "climate", "conservation", "biodiversity", "renewable", "forest",
            "abs", "nagoya", "capacity", "benefit sharing"
        ]
        
        # Calculate a quality score based on keyword presence and other factors
        quality_score = 0
        
        # Add points for keywords
        for keyword in quality_keywords:
            if keyword in content_lower:
                quality_score += 1
        
        # Add points for longer content which tends to be more substantive
        if len(content) > 1000:
            quality_score += 2
        if len(content) > 3000:
            quality_score += 2
        
        # Add points for reliable sources
        if is_reliable_source:
            quality_score += 5
        
        # Add points for structured content (likely more organized information)
        if content.count('\n\n') > 5:
            quality_score += 2
            
        # Calculate ratio of keywords to content length (density of relevant info)
        keyword_density = quality_score / (len(content) / 500)  # Normalize for content length
        
        # Log quality assessment
        logger.info(f"Content quality assessment - Score: {quality_score}, Keyword density: {keyword_density:.2f}")
        
        # Return True if content meets quality thresholds
        return quality_score >= 5 or (is_reliable_source and quality_score >= 3) or keyword_density > 0.5

    def extract_benefit_examples(self, content: str, initiative: str = "ABS Initiative") -> List[Dict[str, Any]]:
        """
        Extract examples of benefits from the content.
        
        Args:
            content: Content text to analyze
            initiative: Initiative key
            
        Returns:
            List of extracted benefit examples
        """
        if not content or len(content) < 100:
            return []
                
        content_lower = content.lower()
        
        # Define benefit categories
        benefit_categories = {
            "environmental_benefits": [
                "biodiversity conservation", "ecosystem restoration", "sustainable use",
                "habitat protection", "ecological integrity", "conservation", "protected areas",
                "species protection", "environmental sustainability"
            ],
            "economic_benefits": [
                "poverty alleviation", "private sector", "technology transfer", 
                "sustainable development", "job creation", "employment", "income generation",
                "public-private partnerships", "market access", "trade", "investment",
                "economic growth", "livelihoods", "business opportunities", "value chains"
            ],
            "social_benefits": [
                "indigenous peoples", "local communities", "iplcs", "capacity building",
                "empowerment", "gender equality", "education", "training", "skills development",
                "participatory approach", "inclusion", "community development", "knowledge sharing"
            ],
            "strategic_benefits": [
                "global governance", "policy development", "legislation", "regulations",
                "institutional frameworks", "international cooperation", "partnerships",
                "stakeholder engagement", "compliance", "legal framework", "policy implementation"
            ]
        }
        
        # Find paragraphs that mention benefits
        paragraphs = content.split('\n\n')
        benefit_paragraphs = []
        
        for paragraph in paragraphs:
            paragraph_lower = paragraph.lower()
            
            # Skip very short paragraphs
            if len(paragraph) < 50:
                continue
            
            # Determine benefit category through keyword matching
            category = "general"
            max_score = 0
            
            for cat_key, cat_terms in benefit_categories.items():
                score = sum(paragraph_lower.count(term) for term in cat_terms)
                if score > max_score:
                    max_score = score
                    category = cat_key
                    
            # Create benefit example
            if max_score > 0:
                benefit_example = {
                    "text": paragraph.strip(),
                    "category": category,
                    "initiative": initiative,
                    "word_count": len(paragraph.split())
                }
                
                benefit_paragraphs.append(benefit_example)
        
        return benefit_paragraphs

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