import re
from datetime import datetime
import html
from typing import Optional

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
def clean_text(text):
        """Clean encoding issues from text"""
        if not isinstance(text, str):
            return text
        
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
            'ï»¿': ''
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text


def format_date(date_str: Optional[str]) -> Optional[str]:
    """
    Robustly parse and validate dates from various sources.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        Standardized ISO date string (YYYY-MM-DD) or None if invalid
    """
    if not date_str or not isinstance(date_str, str):
        return None
    
    # Remove any leading/trailing whitespace and common prefixes
    date_str = date_str.strip()
    date_str = re.sub(r'^.*?(?:date|on):\s*', '', date_str, flags=re.IGNORECASE)
    
    # Clean up malformed strings
    date_str = re.sub(r'[{}();"]', '', date_str)
    
    # Special handling for common date formats and metadata
    def clean_and_parse_date(input_str):
        # Try standard date-like formats
        formats_to_try = [
            '%Y-%m-%d',      # YYYY-MM-DD
            '%d-%m-%Y',      # DD-MM-YYYY
            '%m-%d-%Y',      # MM-DD-YYYY
            '%Y/%m/%d',      # YYYY/MM/DD
            '%d/%m/%Y',      # DD/MM/YYYY
            '%m/%d/%Y',      # MM/DD/YYYY
            '%d.%m.%Y',      # DD.MM.YYYY
            '%B %d, %Y',     # Month DD, YYYY
            '%d %B %Y',      # DD Month YYYY
            '%b %d, %Y',     # Mon DD, YYYY
            '%d %b %Y',      # DD Mon YYYY
        ]
        
        for fmt in formats_to_try:
            try:
                parsed_date = datetime.strptime(input_str, fmt)
                
                # Validate year is reasonable
                current_year = datetime.now().year
                if 1900 <= parsed_date.year <= (current_year + 10):
                    return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    # First, try direct parsing
    parsed_result = clean_and_parse_date(date_str)
    if parsed_result:
        return parsed_result
    
    # If direct parsing fails, try extracting date-like substrings
    date_patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',     # YYYY-MM-DD
        r'\b(\d{1,2}/\d{1,2}/\d{4})\b', # DD/MM/YYYY or MM/DD/YYYY
        r'\b(\d{1,2}-\d{1,2}-\d{4})\b', # DD-MM-YYYY or MM-DD-YYYY
        r'\b(\d{4}/\d{2}/\d{2})\b',     # YYYY/MM/DD
        # Add month name variations
        r'\b([A-Za-z]+ \d{1,2}, \d{4})\b',  # Month DD, YYYY
        r'\b(\d{1,2} [A-Za-z]+ \d{4})\b'   # DD Month YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            potential_date = match.group(1)
            parsed_result = clean_and_parse_date(potential_date)
            if parsed_result:
                return parsed_result
    
    # Fallback: extract year if nothing else works
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if year_match:
        year = int(year_match.group(1))
        if 1900 <= year <= datetime.now().year:
            return f"{year}-01-01"  # Use January 1st as a fallback
    
    # Final fallback
    logger.warning(f"Could not parse date string: {date_str}")
    return None
