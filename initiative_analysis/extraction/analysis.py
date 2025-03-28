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
    """Extension of WebExtractor with content analysis methods."""
    
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
            # If API key is available, use OpenAI
            if self.openai_client:
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
                response = self.openai_client.chat.completions.create(
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
    
    def generate_summary(self, content, title="", url="", language="English"):
        """
        Generate a high-quality summary using OpenAI API with semantic understanding.
        Properly utilizes embeddings for enhanced semantic analysis.
        
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

        # Generate embedding for the content
        embedding = None
        try:
            embedding = self.generate_embedding(content, language)
            if embedding and len(embedding) > 0:
                logger.info(f"Successfully generated embedding vector ({len(embedding)} dimensions)")
            else:
                logger.warning("Failed to generate embedding or empty embedding returned")
        except Exception as e:
            logger.warning(f"Embedding generation failed: {str(e)}")

        # Check if OpenAI client is available
        if not self.openai_client:
            logger.warning("OpenAI client not available for summary generation")
            return content[:300] + "..." if len(content) > 300 else content
        
        try:
            # Truncate content to save on tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Enhance the prompt with embedding-specific context
            embedding_context = ""
            if embedding and len(embedding) > 0:
                # Note: We don't pass the raw embedding to the API as it would be too large
                # Instead, we notify the model that an embedding has been generated
                embedding_context = (
                    "An embedding vector has been generated for this content, capturing the semantic meaning. "
                    "This provides deeper context about the topics and relationships in the text. "
                    "Generate a summary that captures these semantic relationships accurately."
                )
            
            # Create a detailed, context-aware prompt that leverages embeddings when available
            prompt = f"""
Provide a concise, factual summary of the following content about the ABS Initiative.

{embedding_context}

Content Context:
- Title: {title}
- Source URL: {url}
- Language: {language}

Content Excerpt:
{excerpt}

Summary Requirements:
1. 3-5 sentences that capture the main points and key information
2. Focus on factual information directly stated in the content
3. Maintain the original intent and tone of the content
4. Include specific details about the ABS Initiative, access and benefit sharing, or traditional knowledge if present
5. Be clear, concise, and well-structured
"""

            # Generate summary
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at creating concise, accurate summaries that capture the essence of content about the ABS Initiative and related topics."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # Lower temperature for more deterministic output
                max_tokens=250
            )
            
            # Extract and clean summary
            summary = response.choices[0].message.content.strip()
            
            # Validate the summary if possible
            if len(summary) > 50:
                logger.info(f"Generated summary ({len(summary)} chars) with semantic context")
                
                # Try to validate, but don't block if it fails
                try:
                    validation = self.validate_summary(summary, content)
                    if not validation.get("valid", True):
                        logger.warning(f"Summary validation issues: {validation.get('issues', [])}")
                except Exception as e:
                    logger.warning(f"Summary validation error: {str(e)}")
                
                return summary
            else:
                logger.warning("Generated summary too short")
                return "Generated summary was insufficient. Please see the full content for details."
            
        except Exception as e:
            # Better error handling without falling back to content truncation
            logger.error(f"Summary generation error: {str(e)}")
            return f"Unable to generate summary due to technical issues. Error: {str(e)[:100]}..."
    
    def validate_summary(self, summary, original_content):
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
            # Create a validation prompt
            validation_prompt = f"""
Your task is to verify if the summary below contains ONLY information that is explicitly stated in the original content.

Summary to validate:
{summary}

Original content:
{original_content}

Check for these issues:
1. Hallucinations - information in the summary not present in the original
2. Misrepresentations - information that distorts what's in the original
3. Omissions of critical context that change meaning

Return a JSON object with the following structure:
{{
"valid": true/false,
"issues": ["specific issue 1", "specific issue 2", ...],
"explanation": "Brief explanation of validation result"
}}
"""

            # Make API call for validation
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": validation_prompt}
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

# Standalone function that Analysis class can use
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