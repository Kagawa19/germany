import os
import logging
import json
import re
from datetime import datetime
import time
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
    
    


    # Add these imports at the top of extraction/analysis.py
# from monitoring.langfuse_client import get_langfuse_client

# Then update the analyze_content method in the Analysis class:

    def analyze_content(self, content: str, title: str = "", url: str = "", language: str = "English", 
                    trace_id: str = None, parent_span_id: str = None) -> Dict[str, Any]:
        """
        Comprehensive content analysis method that calls all other methods.
        Enhanced with Langfuse tracing.
            
        Args:
            content: Content text to analyze
            title: Content title for context
            url: Source URL for context 
            language: Content language
            trace_id: Optional Langfuse trace ID for monitoring
            parent_span_id: Optional parent span ID for nested tracing
            
        Returns:
            Dictionary with all extracted information
        """
        # Get Langfuse client for tracing
        langfuse = None
        try:
            from monitoring.langfuse_client import get_langfuse_client
            langfuse = get_langfuse_client()
        except ImportError:
            logger.warning("Langfuse client not available for tracing analysis")
        
        # Create a span for the overall analysis process
        analysis_span = None
        if langfuse and trace_id:
            analysis_span = langfuse.create_span(
                trace_id=trace_id,
                name="comprehensive_analysis",
                parent_span_id=parent_span_id,
                metadata={
                    "content_length": len(content) if content else 0,
                    "title": title,
                    "url": url,
                    "language": language
                }
            )
        
        # Skip if content is too short
        if not content or len(content) < 100:
            logger.warning("Content too short for comprehensive analysis")
            print("Warning: Content too short for analysis")
            
            # Update span with error
            if analysis_span:
                analysis_span.update(
                    output={"error": "Content too short for analysis"},
                    status="error"
                )
                analysis_span.end()
                
            return {"error": "Content too short for analysis"}
            
        # Initialize result dictionary
        result = {
            "title": title,
            "url": url,
            "language": language, 
            "analysis_timestamp": datetime.now().isoformat()
        }
        
        # Track start time for performance monitoring
        start_time = time.time()
        
        # Generate embedding
        embedding_span = None
        if langfuse and trace_id:
            embedding_span = langfuse.create_span(
                trace_id=trace_id,
                name="generate_embedding",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id,
                metadata={"language": language}
            )
        
        logger.info("Generating embedding...")
        embedding = self.generate_embedding(content, language)
        
        # Update embedding span
        if embedding_span:
            embedding_span.update(
                output={
                    "success": bool(embedding),
                    "vector_size": len(embedding) if embedding else 0
                }
            )
            embedding_span.end()
        
        if embedding:
            result["embedding"] = embedding
            print("Embedding generated successfully")
        else:
            logger.warning("Failed to generate embedding")
            print("Warning: Embedding generation failed")
            
        # Generate summary 
        summary_span = None
        if langfuse and trace_id:
            summary_span = langfuse.create_span(
                trace_id=trace_id,
                name="generate_summary",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id,
                metadata={"title": title, "url": url}
            )
        
        logger.info("Generating summary...")
        summary_start = time.time()
        summary = self.generate_summary(content, title, url, language)
        summary_duration = time.time() - summary_start
        
        # Log the LLM generation for the summary
        if langfuse and trace_id and self.openai_client:
            # If we have the original prompt and response, log them
            prompt_text = f"Summarize content: {title}"
            langfuse.log_generation(
                trace_id=trace_id,
                name="summary_generation",
                model="gpt-3.5-turbo",
                prompt=prompt_text,
                completion=summary if summary else "Failed to generate summary",
                metadata={
                    "duration": summary_duration,
                    "content_length": len(content),
                    "summary_length": len(summary) if summary else 0
                },
                parent_span_id=summary_span.id if summary_span else (analysis_span.id if analysis_span else parent_span_id)
            )
        
        # Update summary span
        if summary_span:
            summary_span.update(
                output={
                    "success": bool(summary),
                    "summary_length": len(summary) if summary else 0,
                    "duration_seconds": summary_duration
                }
            )
            summary_span.end()
        
        if summary:
            result["summary"] = summary
            print("Summary generated successfully")
        else:
            logger.warning("Failed to generate summary")
            print("Warning: Summary generation failed")
            
        # Analyze sentiment
        sentiment_span = None
        if langfuse and trace_id:
            sentiment_span = langfuse.create_span(
                trace_id=trace_id,
                name="analyze_sentiment",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id
            )
        
        logger.info("Analyzing sentiment...")
        sentiment_start = time.time()
        sentiment_info = self.analyze_sentiment(content)
        sentiment_duration = time.time() - sentiment_start
        
        # Log sentiment analysis score
        if langfuse and trace_id and sentiment_info:
            sentiment_value = {
                "Positive": 1.0,
                "Neutral": 0.5,
                "Negative": 0.0
            }.get(sentiment_info.get("overall_sentiment", "Neutral"), 0.5)
            
            langfuse.score(
                trace_id=trace_id,
                name="content_sentiment",
                value=sentiment_value * 10, # Scale to 0-10
                comment=f"Content sentiment: {sentiment_info.get('overall_sentiment', 'Neutral')}"
            )
        
        # Update sentiment span
        if sentiment_span:
            sentiment_span.update(
                output={
                    "sentiment": sentiment_info.get("overall_sentiment", "Neutral"),
                    "score": sentiment_info.get("sentiment_score", 0),
                    "duration_seconds": sentiment_duration
                }
            )
            sentiment_span.end()
        
        if sentiment_info:
            result.update(sentiment_info)
            print("Sentiment analysis completed")
        else:
            logger.warning("Failed to analyze sentiment")
            print("Warning: Sentiment analysis failed")

        # Identify themes
        themes_span = None
        if langfuse and trace_id:
            themes_span = langfuse.create_span(
                trace_id=trace_id,
                name="identify_themes",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id
            )
        
        logger.info("Identifying themes...")
        themes_start = time.time()
        themes = self.identify_themes(content)
        themes_duration = time.time() - themes_start
        
        # Update themes span
        if themes_span:
            themes_span.update(
                output={
                    "theme_count": len(themes),
                    "themes": themes,
                    "duration_seconds": themes_duration
                }
            )
            themes_span.end()
        
        if themes:
            result["themes"] = themes 
            print(f"Identified {len(themes)} themes")
        else:
            logger.warning("Failed to identify themes")
            print("Warning: Theme identification failed")

        # Extract ABS mentions
        mentions_span = None
        if langfuse and trace_id:
            mentions_span = langfuse.create_span(
                trace_id=trace_id,
                name="extract_abs_mentions",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id
            )
        
        logger.info("Extracting ABS mentions...")
        mentions_start = time.time()
        mentions = self.extract_abs_mentions(content)
        mentions_duration = time.time() - mentions_start
        
        # Update mentions span
        if mentions_span:
            mentions_span.update(
                output={
                    "mention_count": len(mentions),
                    "duration_seconds": mentions_duration
                }
            )
            mentions_span.end()
        
        if mentions: 
            result["abs_mentions"] = mentions
            print(f"Extracted {len(mentions)} ABS mentions")
        else:
            logger.info("No ABS mentions found")
            
        # Extract geographic focus
        geo_span = None
        if langfuse and trace_id:
            geo_span = langfuse.create_span(
                trace_id=trace_id,
                name="extract_geographic_focus",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id
            )
        
        logger.info("Extracting geographic focus...")
        geo_start = time.time()
        geo_focus = self.extract_geographic_focus(content)
        geo_duration = time.time() - geo_start
        
        # Update geographic span
        if geo_span:
            geo_span.update(
                output={
                    "location_count": len(geo_focus),
                    "duration_seconds": geo_duration
                }
            )
            geo_span.end()
        
        if geo_focus:
            result["geographic_focus"] = geo_focus
            print(f"Identified geographic focus: {geo_focus}")
        else:
            logger.info("No clear geographic focus identified")

        # Extract project details
        projects_span = None
        if langfuse and trace_id:
            projects_span = langfuse.create_span(
                trace_id=trace_id,
                name="extract_project_details",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id
            )
        
        logger.info("Extracting project details...") 
        projects_start = time.time()
        projects = self.extract_project_details(content)
        projects_duration = time.time() - projects_start
        
        # Update projects span
        if projects_span:
            projects_span.update(
                output={
                    "project_count": len(projects),
                    "duration_seconds": projects_duration
                }
            )
            projects_span.end()
        
        if projects:
            result["projects"] = projects
            print(f"Extracted {len(projects)} projects")
        else:
            logger.info("No project details found")

        # Extract organizations
        orgs_span = None
        if langfuse and trace_id:
            orgs_span = langfuse.create_span(
                trace_id=trace_id,
                name="extract_organizations",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id
            )
        
        logger.info("Extracting organizations...")
        orgs_start = time.time()
        organizations = self.extract_organizations(content)
        orgs_duration = time.time() - orgs_start
        
        # Update organizations span
        if orgs_span:
            orgs_span.update(
                output={
                    "organization_count": len(organizations),
                    "duration_seconds": orgs_duration
                }
            )
            orgs_span.end()
        
        if organizations:
            result["organizations"] = organizations
            print(f"Extracted {len(organizations)} organizations") 
        else:
            logger.info("No organizations found")

        # Extract resources  
        resources_span = None
        if langfuse and trace_id:
            resources_span = langfuse.create_span(
                trace_id=trace_id,
                name="extract_resources",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id
            )
        
        logger.info("Extracting resources...")
        resources_start = time.time()
        resources = self.extract_resources(content)
        resources_duration = time.time() - resources_start
        
        # Update resources span
        if resources_span:
            resources_span.update(
                output={
                    "resource_count": len(resources),
                    "duration_seconds": resources_duration
                }
            )
            resources_span.end()
        
        if resources:
            result["resources"] = resources
            print(f"Extracted {len(resources)} resources")
        else: 
            logger.info("No resources found")
            
        # Extract target audiences
        audiences_span = None
        if langfuse and trace_id:
            audiences_span = langfuse.create_span(
                trace_id=trace_id,
                name="extract_target_audiences",
                parent_span_id=analysis_span.id if analysis_span else parent_span_id
            )
        
        logger.info("Extracting target audiences...")
        audiences_start = time.time()
        audiences = self.extract_target_audiences(content)
        audiences_duration = time.time() - audiences_start
        
        # Update audiences span
        if audiences_span:
            audiences_span.update(
                output={
                    "audience_count": len(audiences),
                    "duration_seconds": audiences_duration
                }
            )
            audiences_span.end()
        
        if audiences:
            result["target_audiences"] = audiences
            print(f"Extracted {len(audiences)} target audiences")
        else:
            logger.info("No target audiences found")

        # Calculate total analysis time
        total_duration = time.time() - start_time
        
        # Add completion metrics
        completion_percentage = 0
        critical_fields = ["summary", "sentiment_info", "themes", "abs_mentions"]
        completed_fields = sum(1 for field in critical_fields if field in result or f"{field}" in result)
        completion_percentage = (completed_fields / len(critical_fields)) * 100
        
        # Log analysis quality score in Langfuse
        if langfuse and trace_id:
            langfuse.score(
                trace_id=trace_id,
                name="analysis_completeness",
                value=min(completion_percentage / 10, 10),  # Scale to 0-10
                comment=f"Analysis completeness: {completion_percentage:.1f}% of critical fields extracted"
            )
        
        print("Comprehensive content analysis completed")
        logger.info(f"Content analysis completed in {total_duration:.2f} seconds with {len(result)} extracted components")
        
        # Update the main analysis span with final results
        if analysis_span:
            analysis_span.update(
                output={
                    "status": "success",
                    "duration_seconds": total_duration,
                    "extracted_fields": list(result.keys()),
                    "completion_percentage": completion_percentage
                }
            )
            analysis_span.end()

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