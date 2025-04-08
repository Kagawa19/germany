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
    
    def analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """
        Analyze sentiment using OpenAI API with prompt from file.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary with sentiment information
        """
        if not content or len(content) < 100:
            return {"overall_sentiment": "Neutral", "sentiment_score": 0.0, "sentiment_confidence": 0.0}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for sentiment analysis")
            return {"overall_sentiment": "Neutral", "sentiment_score": 0.0, "sentiment_confidence": 0.0}
        
        try:
            # Load sentiment prompt
            prompt = self._load_prompt("sentiment")
            if not prompt:
                logger.error("Sentiment prompt not available")
                return {"overall_sentiment": "Neutral", "sentiment_score": 0.0, "sentiment_confidence": 0.0}
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(content=excerpt)
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=150
            )
            
            # Parse the result
            result = json.loads(response.choices[0].message.content)
            
            # Return sentiment analysis with standard fields
            return {
                "overall_sentiment": result.get("sentiment", "Neutral"),
                "sentiment_score": result.get("score", 0.0),
                "sentiment_confidence": result.get("confidence", 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {str(e)}")
            return {"overall_sentiment": "Neutral", "sentiment_score": 0.0, "sentiment_confidence": 0.0}
    
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
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Parse the result
            result = json.loads(response.choices[0].message.content)
            
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
    
    def extract_project_details(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract project details from content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with project information
        """
        if not content or len(content) < 100:
            return []
        
        if not self.openai_client:
            logger.error("OpenAI client not available for project details extraction")
            return []
        
        try:
            # Load project details prompt
            prompt = self._load_prompt("project_details")
            if not prompt:
                logger.error("Project details prompt not available")
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
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Parse the result
            result = json.loads(response.choices[0].message.content)
            
            # Convert to expected format
            projects = []
            if "projects" in result and isinstance(result["projects"], list):
                for project in result["projects"]:
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
            
        except Exception as e:
            logger.error(f"Error extracting project details: {str(e)}")
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
            
            # Make API call
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Parse the result
            result = json.loads(response.choices[0].message.content)
            
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
    
    def extract_resources(self, content: str) -> List[Dict[str, str]]:
        """
        Extract resources mentioned in the content.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with resource information
        """
        if not content or len(content) < 100:
            return []
        
        if not self.openai_client:
            logger.error("OpenAI client not available for resource extraction")
            return []
        
        try:
            # Load resources prompt
            prompt = self._load_prompt("resources")
            if not prompt:
                logger.error("Resources prompt not available")
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
                response_format={"type": "json_object"},
                max_tokens=1000
            )
            
            # Parse the result
            result = json.loads(response.choices[0].message.content)
            
            # Convert to expected format
            resources = []
            if "resources" in result and isinstance(result["resources"], list):
                for resource in result["resources"]:
                    resource_item = {
                        "resource_type": resource.get("type", ""),
                        "resource_name": resource.get("name", ""),
                        "resource_url": resource.get("url", ""),
                        "description": resource.get("description", "")
                    }
                    resources.append(resource_item)
            
            return resources
            
        except Exception as e:
            logger.error(f"Error extracting resources: {str(e)}")
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
    

    def extract_abs_mentions(self, content: str) -> List[Dict[str, Any]]:
        """
        Extract specific mentions of ABS Initiative from content.
            
        Args:
            content: Content text to analyze
            
        Returns:
            List of dictionaries with mention information
        """
        if not content or len(content) < 100:
            logger.info("Content too short for ABS mentions extraction")
            print("Skipping ABS mentions extraction: content too short")
            return []
            
        if not self.openai_client:
            logger.error("OpenAI client not available for ABS mentions extraction")
            print("Error: OpenAI client not available for ABS mentions extraction")
            return []
            
        try:
            # Load abs mentions prompt
            logger.info("Loading ABS mentions prompt...")
            prompt = self._load_prompt("abs_mentions")
            if not prompt:
                logger.error("ABS mentions prompt not available")
                print("Error: ABS mentions prompt not available")
                return []
            
            print("ABS mentions prompt loaded successfully")
                    
            # Format the prompt
            logger.info("Formatting ABS mentions prompt...")
            formatted_prompt = prompt.format(content=content)
            print("ABS mentions prompt formatted")
                    
            # Make API call
            logger.info("Making OpenAI API call for ABS mentions extraction...")
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": formatted_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"},
                max_tokens=1500
            )
            print("OpenAI API call completed")
                    
            # Parse the result
            logger.info("Parsing OpenAI response...")
            result = json.loads(response.choices[0].message.content)
            print("OpenAI response parsed")
                    
            # Convert to expected format
            mentions = []
            if "mentions" in result and isinstance(result["mentions"], list):
                for idx, mention in enumerate(result["mentions"]):
                    mention_item = {
                        "name_variant": mention.get("variant", ""),
                        "mention_context": mention.get("context", ""),
                        "mention_type": mention.get("type", ""),
                        "relevance_score": float(mention.get("relevance", 0.5)),
                        "mention_position": idx + 1
                    }
                    mentions.append(mention_item)
            
            if mentions:
                logger.info("ABS mentions extracted successfully")
                print(f"Extracted {len(mentions)} ABS mentions")
            else:
                logger.info("No ABS mentions found")
                print("No ABS mentions found in the content")
            
            return mentions
            
        except Exception as e:
            logger.exception(f"Error extracting ABS mentions: {str(e)}")
            print(f"Error extracting ABS mentions: {str(e)}")
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

    def analyze_context_relevance(self, content: str, title: str = "", url: str = "") -> Dict[str, Any]:
        """
        Analyze context relevance using OpenAI API.
        
        Args:
            content: Content text to analyze
            title: Content title for context
            url: Source URL for context
            
        Returns:
            Dictionary with context relevance information
        """
        if not content or len(content) < 100:
            return {"context_relevance_score": 0.0, "context_relevance_confidence": 0.0}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for context relevance analysis")
            return {"context_relevance_score": 0.0, "context_relevance_confidence": 0.0}
        
        try:
            # Load context relevance prompt
            prompt = self._load_prompt("context_relevance")
            if not prompt:
                # Create a default prompt if file not found
                prompt = """
                Evaluate how relevant the following content is to ABS (Access and Benefit Sharing) 
                and the Nagoya Protocol. Consider:
                - Whether it directly discusses ABS topics
                - How central ABS issues are to the content
                - Whether it relates to genetic resources, traditional knowledge, or biodiversity
                
                Title: {title}
                URL: {url}
                Content:
                {content}
                
                Return your analysis as a JSON object with:
                - context_relevance_score: A float from 0.0 (not relevant) to 1.0 (highly relevant)
                - context_relevance_confidence: Your confidence in this assessment (0.0-1.0)
                - issues: Array of notes about relevance (optional)
                """
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(content=excerpt, title=title, url=url)
            
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
            result = json.loads(response.choices[0].message.content)
            
            # Return context relevance analysis with standard fields
            return {
                "context_relevance_score": result.get("context_relevance_score", 0.0),
                "context_relevance_confidence": result.get("context_relevance_confidence", 0.0),
                "context_relevance_issues": result.get("issues", [])
            }
            
        except Exception as e:
            logger.error(f"Error in context relevance analysis: {str(e)}")
            return {"context_relevance_score": 0.0, "context_relevance_confidence": 0.0}

    def analyze_conciseness(self, content: str) -> Dict[str, Any]:
        """
        Analyze conciseness using OpenAI API.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary with conciseness information
        """
        if not content or len(content) < 100:
            return {"conciseness_score": 0.0, "conciseness_confidence": 0.0}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for conciseness analysis")
            return {"conciseness_score": 0.0, "conciseness_confidence": 0.0}
        
        try:
            # Load conciseness prompt
            prompt = self._load_prompt("conciseness")
            if not prompt:
                # Create a default prompt if file not found
                prompt = """
                Evaluate the conciseness of the following content.
                Consider:
                - Whether the content is direct and to the point
                - Whether it avoids unnecessary repetition or verbosity
                - Whether key information is expressed efficiently
                
                Content:
                {content}
                
                Return your analysis as a JSON object with:
                - conciseness_score: A float from 0.0 (extremely verbose) to 1.0 (perfectly concise)
                - conciseness_confidence: Your confidence in this assessment (0.0-1.0)
                - issues: Array of specific areas that could be more concise (optional)
                """
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(content=excerpt)
            
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
            result = json.loads(response.choices[0].message.content)
            
            # Return conciseness analysis with standard fields
            return {
                "conciseness_score": result.get("conciseness_score", 0.0),
                "conciseness_confidence": result.get("conciseness_confidence", 0.0),
                "conciseness_issues": result.get("issues", [])
            }
            
        except Exception as e:
            logger.error(f"Error in conciseness analysis: {str(e)}")
            return {"conciseness_score": 0.0, "conciseness_confidence": 0.0}

    def analyze_correctness(self, content: str) -> Dict[str, Any]:
        """
        Analyze factual correctness using OpenAI API.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary with correctness information
        """
        if not content or len(content) < 100:
            return {"correctness_score": 0.0, "correctness_confidence": 0.0}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for correctness analysis")
            return {"correctness_score": 0.0, "correctness_confidence": 0.0}
        
        try:
            # Load correctness prompt
            prompt = self._load_prompt("correctness")
            if not prompt:
                # Create a default prompt if file not found
                prompt = """
                Evaluate the factual correctness of the following content.
                Consider whether the statements are accurate, supported by evidence,
                and consistent with general knowledge about Access and Benefit Sharing (ABS),
                the Nagoya Protocol, and related topics.
                
                Content:
                {content}
                
                Return your analysis as a JSON object with:
                - correctness_score: A float from 0.0 (completely incorrect) to 1.0 (completely correct)
                - correctness_confidence: Your confidence in this assessment (0.0-1.0)
                - issues: Array of potential factual errors (if any)
                """
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(content=excerpt)
            
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
            result = json.loads(response.choices[0].message.content)
            
            # Return correctness analysis with standard fields
            return {
                "correctness_score": result.get("correctness_score", 0.0),
                "correctness_confidence": result.get("correctness_confidence", 0.0),
                "correctness_issues": result.get("issues", [])
            }
            
        except Exception as e:
            logger.error(f"Error in correctness analysis: {str(e)}")
            return {"correctness_score": 0.0, "correctness_confidence": 0.0}

    def analyze_hallucination(self, content: str) -> Dict[str, Any]:
        """
        Analyze for potential hallucinations/fabrications using OpenAI API.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary with hallucination information
        """
        if not content or len(content) < 100:
            return {"hallucination_score": 0.0, "hallucination_confidence": 0.0}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for hallucination analysis")
            return {"hallucination_score": 0.0, "hallucination_confidence": 0.0}
        
        try:
            # Load hallucination prompt
            prompt = self._load_prompt("hallucination")
            if not prompt:
                # Create a default prompt if file not found
                prompt = """
                Evaluate whether the following content contains hallucinations or fabricated information.
                Focus specifically on statements about:
                - Organizations, projects, or initiatives that don't exist
                - Events or dates that seem fabricated
                - Quotes or statistics without proper attribution
                - Claims that contradict established facts about Access and Benefit Sharing
                
                Content:
                {content}
                
                Return your analysis as a JSON object with:
                - hallucination_score: A float from 0.0 (no hallucinations) to 1.0 (completely fabricated)
                - hallucination_confidence: Your confidence in this assessment (0.0-1.0)
                - issues: Array of specific suspected hallucinations (if any)
                """
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(content=excerpt)
            
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
            result = json.loads(response.choices[0].message.content)
            
            # Return hallucination analysis with standard fields
            return {
                "hallucination_score": result.get("hallucination_score", 0.0),
                "hallucination_confidence": result.get("hallucination_confidence", 0.0),
                "hallucination_issues": result.get("issues", [])
            }
            
        except Exception as e:
            logger.error(f"Error in hallucination analysis: {str(e)}")
            return {"hallucination_score": 0.0, "hallucination_confidence": 0.0}

    def analyze_helpfulness(self, content: str) -> Dict[str, Any]:
        """
        Analyze helpfulness of the content using OpenAI API.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary with helpfulness information
        """
        if not content or len(content) < 100:
            return {"helpfulness_score": 0.0, "helpfulness_confidence": 0.0}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for helpfulness analysis")
            return {"helpfulness_score": 0.0, "helpfulness_confidence": 0.0}
        
        try:
            # Load helpfulness prompt
            prompt = self._load_prompt("helpfulness")
            if not prompt:
                # Create a default prompt if file not found
                prompt = """
                Evaluate how helpful the following content would be for stakeholders interested in
                Access and Benefit Sharing (ABS), the Nagoya Protocol, or related topics.
                Consider whether the content:
                - Provides actionable information
                - Addresses key questions or challenges
                - Explains complex concepts clearly
                - Offers resources or guidance
                
                Content:
                {content}
                
                Return your analysis as a JSON object with:
                - helpfulness_score: A float from 0.0 (not helpful) to 1.0 (extremely helpful)
                - helpfulness_confidence: Your confidence in this assessment (0.0-1.0)
                - notes: Array of ways the content is helpful or could be more helpful
                """
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(content=excerpt)
            
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
            result = json.loads(response.choices[0].message.content)
            
            # Return helpfulness analysis with standard fields
            return {
                "helpfulness_score": result.get("helpfulness_score", 0.0),
                "helpfulness_confidence": result.get("helpfulness_confidence", 0.0),
                "helpfulness_notes": result.get("notes", [])
            }
            
        except Exception as e:
            logger.error(f"Error in helpfulness analysis: {str(e)}")
            return {"helpfulness_score": 0.0, "helpfulness_confidence": 0.0}
        
    def analyze_toxicity(self, content: str) -> Dict[str, Any]:
        """
        Analyze content for toxic or harmful language using OpenAI API.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary with toxicity information
        """
        if not content or len(content) < 100:
            return {"toxicity_score": 0.0, "toxicity_confidence": 0.0}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for toxicity analysis")
            return {"toxicity_score": 0.0, "toxicity_confidence": 0.0}
        
        try:
            # Load toxicity prompt
            prompt = self._load_prompt("toxicity")
            if not prompt:
                # Create a default prompt if file not found
                prompt = """
                Evaluate the following content for potentially toxic, harmful, biased, or offensive language.
                Consider:
                - Presence of hate speech, discrimination, or prejudice
                - Aggressive or hostile tone
                - Misinformation that could cause harm
                - Offensive stereotypes or characterizations
                
                Content:
                {content}
                
                Return your analysis as a JSON object with:
                - toxicity_score: A float from 0.0 (not toxic) to 1.0 (extremely toxic)
                - toxicity_confidence: Your confidence in this assessment (0.0-1.0)
                - concerns: Array of specific concerns (if any)
                """
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(content=excerpt)
            
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
            result = json.loads(response.choices[0].message.content)
            
            # Return toxicity analysis with standard fields
            return {
                "toxicity_score": result.get("toxicity_score", 0.0),
                "toxicity_confidence": result.get("toxicity_confidence", 0.0),
                "toxicity_concerns": result.get("concerns", [])
            }
            
        except Exception as e:
            logger.error(f"Error in toxicity analysis: {str(e)}")
            return {"toxicity_score": 0.0, "toxicity_confidence": 0.0}

    def analyze_relevance(self, content: str) -> Dict[str, Any]:
        """
        Analyze general relevance of the content using OpenAI API.
        
        Args:
            content: Content text to analyze
            
        Returns:
            Dictionary with relevance information
        """
        if not content or len(content) < 100:
            return {"relevance_score": 0.0, "relevance_confidence": 0.0}
        
        if not self.openai_client:
            logger.error("OpenAI client not available for relevance analysis")
            return {"relevance_score": 0.0, "relevance_confidence": 0.0}
        
        try:
            # Load relevance prompt
            prompt = self._load_prompt("relevance")
            if not prompt:
                # Create a default prompt if file not found
                prompt = """
                Evaluate the overall relevance of the following content to stakeholders in the 
                ABS Initiative (Access and Benefit Sharing). 
                Consider:
                - Current relevance to policy or practice
                - Importance to various stakeholder groups
                - Whether it addresses significant issues in the field
                
                Content:
                {content}
                
                Return your analysis as a JSON object with:
                - relevance_score: A float from 0.0 (completely irrelevant) to 1.0 (highly relevant)
                - relevance_confidence: Your confidence in this assessment (0.0-1.0)
                - key_points: Array of key points that establish relevance
                """
            
            # Truncate content to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Format the prompt
            formatted_prompt = prompt.format(content=excerpt)
            
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
            result = json.loads(response.choices[0].message.content)
            
            # Return relevance analysis with standard fields
            return {
                "relevance_score": result.get("relevance_score", 0.0),
                "relevance_confidence": result.get("relevance_confidence", 0.0),
                "relevance_key_points": result.get("key_points", [])
            }
            
        except Exception as e:
            logger.error(f"Error in relevance analysis: {str(e)}")
            return {"relevance_score": 0.0, "relevance_confidence": 0.0}
    
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
            logger.warning("Content too short for comprehensive analysis")
            print("Warning: Content too short for analysis")
            return {"error": "Content too short for analysis"}
            
        # Initialize result dictionary
        result = {
            "title": title,
            "url": url,
            "language": language, 
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Generate embedding
        logger.info("Generating embedding...")
        embedding = self.generate_embedding(content, language)
        if embedding:
            result["embedding"] = embedding
            print("Embedding generated successfully")
        else:
            logger.warning("Failed to generate embedding")
            print("Warning: Embedding generation failed")
            
        # Generate summary 
        logger.info("Generating summary...")
        summary = self.generate_summary(content, title, url, language)
        if summary:
            result["summary"] = summary
            print("Summary generated successfully")
        else:
            logger.warning("Failed to generate summary")
            print("Warning: Summary generation failed")
            
        # Analyze sentiment
        logger.info("Analyzing sentiment...")
        sentiment_info = self.analyze_sentiment(content)
        if sentiment_info:
            result.update(sentiment_info)
            print("Sentiment analysis completed")
        else:
            logger.warning("Failed to analyze sentiment")
            print("Warning: Sentiment analysis failed")
        
        # NEW ANALYSES
        
        # Analyze context relevance
        logger.info("Analyzing context relevance...")
        context_relevance_info = self.analyze_context_relevance(content, title, url)
        if context_relevance_info:
            result.update(context_relevance_info)
            print("Context relevance analysis completed")
        else:
            logger.warning("Failed to analyze context relevance")
            print("Warning: Context relevance analysis failed")
        
        # Analyze conciseness
        logger.info("Analyzing conciseness...")
        conciseness_info = self.analyze_conciseness(content)
        if conciseness_info:
            result.update(conciseness_info)
            print("Conciseness analysis completed")
        else:
            logger.warning("Failed to analyze conciseness")
            print("Warning: Conciseness analysis failed")
        
        # Analyze correctness
        logger.info("Analyzing correctness...")
        correctness_info = self.analyze_correctness(content)
        if correctness_info:
            result.update(correctness_info)
            print("Correctness analysis completed")
        else:
            logger.warning("Failed to analyze correctness")
            print("Warning: Correctness analysis failed")
        
        # Analyze hallucination
        logger.info("Analyzing hallucination...")
        hallucination_info = self.analyze_hallucination(content)
        if hallucination_info:
            result.update(hallucination_info)
            print("Hallucination analysis completed")
        else:
            logger.warning("Failed to analyze hallucination")
            print("Warning: Hallucination analysis failed")
        
        # Analyze helpfulness
        logger.info("Analyzing helpfulness...")
        helpfulness_info = self.analyze_helpfulness(content)
        if helpfulness_info:
            result.update(helpfulness_info)
            print("Helpfulness analysis completed")
        else:
            logger.warning("Failed to analyze helpfulness")
            print("Warning: Helpfulness analysis failed")
        
        # Analyze relevance
        logger.info("Analyzing relevance...")
        relevance_info = self.analyze_relevance(content)
        if relevance_info:
            result.update(relevance_info)
            print("Relevance analysis completed")
        else:
            logger.warning("Failed to analyze relevance")
            print("Warning: Relevance analysis failed")
        
        # Analyze toxicity
        logger.info("Analyzing toxicity...")
        toxicity_info = self.analyze_toxicity(content)
        if toxicity_info:
            result.update(toxicity_info)
            print("Toxicity analysis completed")
        else:
            logger.warning("Failed to analyze toxicity")
            print("Warning: Toxicity analysis failed")
        
        # Continue with existing analyses
        
        # Identify themes
        logger.info("Identifying themes...")
        themes = self.identify_themes(content)
        if themes:
            result["themes"] = themes 
            print(f"Identified {len(themes)} themes")
        else:
            logger.warning("Failed to identify themes")
            print("Warning: Theme identification failed")

        # Extract ABS mentions
        logger.info("Extracting ABS mentions...")
        mentions = self.extract_abs_mentions(content)
        if mentions: 
            result["abs_mentions"] = mentions
            print(f"Extracted {len(mentions)} ABS mentions")
        else:
            logger.info("No ABS mentions found")
            
        # Extract geographic focus
        logger.info("Extracting geographic focus...")
        geo_focus = self.extract_geographic_focus(content)
        if geo_focus:
            result["geographic_focus"] = geo_focus
            print(f"Identified geographic focus: {len(geo_focus)} locations")
        else:
            logger.info("No clear geographic focus identified")

        # Extract project details
        logger.info("Extracting project details...") 
        projects = self.extract_project_details(content)
        if projects:
            result["projects"] = projects
            print(f"Extracted {len(projects)} projects")
        else:
            logger.info("No project details found")

        # Extract organizations
        logger.info("Extracting organizations...")
        organizations = self.extract_organizations(content)
        if organizations:
            result["organizations"] = organizations
            print(f"Extracted {len(organizations)} organizations") 
        else:
            logger.info("No organizations found")

        # Extract resources  
        logger.info("Extracting resources...")
        resources = self.extract_resources(content)
        if resources:
            result["resources"] = resources
            print(f"Extracted {len(resources)} resources")
        else: 
            logger.info("No resources found")
            
        # Extract target audiences
        logger.info("Extracting target audiences...")
        audiences = self.extract_target_audiences(content)
        if audiences:
            result["target_audiences"] = audiences
            print(f"Extracted {len(audiences)} target audiences")
        else:
            logger.info("No target audiences found")
        
        # Collect all issues from different analyses into a single array
        all_issues = []
        for issue_key in [
            "correctness_issues", "context_relevance_issues", "conciseness_issues", 
            "hallucination_issues", "toxicity_concerns", "helpfulness_notes"
        ]:
            if issue_key in result and result[issue_key]:
                all_issues.extend([{
                    "category": issue_key.replace("_issues", "").replace("_concerns", "").replace("_notes", ""),
                    "issue": issue
                } for issue in result[issue_key]])
        
        if all_issues:
            result["all_issues"] = all_issues

        print("Comprehensive content analysis completed")
        logger.info("Content analysis completed with all metrics")

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
        
        # Factor 5: Has ABS mentions if available
        # Scale: 0.0-0.1
        abs_mentions = result_data.get('abs_mentions', [])
        if abs_mentions and len(abs_mentions) > 0:
            score += min(0.1, len(abs_mentions) * 0.02)
        
        # Factor 6: Geographic focus if available
        # Scale: 0.0-0.05
        geographic_focus = result_data.get('geographic_focus', [])
        if geographic_focus and len(geographic_focus) > 0:
            score += min(0.05, len(geographic_focus) * 0.01)
        
        # Factor 7: Has organizational relationships if available
        # Scale: 0.0-0.05
        organizations = result_data.get('organizations', [])
        if organizations and len(organizations) > 0:
            score += min(0.05, len(organizations) * 0.01)
        
        return min(1.0, score)

    

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