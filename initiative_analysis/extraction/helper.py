import os
import time
import requests
import logging
import re
import json
import random
from urllib.parse import urlparse
from typing import Dict, List, Optional, Tuple, Any
from dotenv import load_dotenv

# Import the specific functions from their modules
from utils.text_processing import clean_html_entities
from extraction.analysis import get_openai_client

logger = logging.getLogger("helper")


    

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

    