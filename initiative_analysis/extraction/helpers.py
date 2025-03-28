import os
import logging
import json
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any


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
        
def load_prompt_file(filename):
    """
    Load prompt from file in the prompts directory.
    
    Args:
        filename: Name of the prompt file
        
    Returns:
        Content of the prompt file or empty string if file not found
    """
    prompt_path = os.path.join("prompts", filename)
    try:
        with open(prompt_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        logger.warning(f"Could not load prompt file {prompt_path}: {str(e)}")
        return ""

