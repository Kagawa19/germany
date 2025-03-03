import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import json
from openai import OpenAI
import os
from typing import Optional, List, Dict, Any
import nltk
from nltk.tokenize import sent_tokenize
import re
import re
from datetime import datetime
import nltk
from nltk.tokenize import sent_tokenize
from urllib.parse import urlparse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("content_db.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ContentDB")

# Download NLTK data if needed (uncomment this if you want to use NLTK)
# try:
#     nltk.download('punkt', quiet=True)
# except:
#     logger.warning("Failed to download NLTK data, sentence tokenization may be less accurate")

def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get database connection parameters from environment variables
        db_host = os.getenv("DB_HOST", "postgres")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME", "appdb")
        db_user = os.getenv("DB_USER", "postgres")
        db_password = os.getenv("DB_PASSWORD", "postgres")
        
        # Connect to the database
        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        
        logger.info(f"Successfully connected to database: {db_name} on {db_host}")
        return conn
    except Exception as e:
        logger.error(f"Error connecting to database: {str(e)}")
        raise

# Initialize OpenAI client
def get_openai_client():
    """Get or initialize OpenAI client."""
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        return None
    
    try:
        return OpenAI(api_key=openai_api_key)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        return None

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

def is_high_quality_content(content, title, url):
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
    
    # Skip if content doesn't mention Germany (for relevance)
    if "germany" not in content.lower() and "german" not in content.lower():
        logger.info(f"Content doesn't mention Germany, skipping AI processing")
        return False
    
    # Check if the content is from a reliable domain
    reliable_domains = ["giz.de", "bmz.de", "kfw.de", "europa.eu", "un.org", "worldbank.org"]
    is_reliable_source = any(domain in url.lower() for domain in reliable_domains)
    
    # Look for quality indicators in the content
    quality_keywords = [
        "cooperation", "sustainable", "development", "partnership", "initiative",
        "project", "bilateral", "agreement", "funding", "investment", 
        "climate", "conservation", "biodiversity", "renewable", "forest"
    ]
    
    # Calculate a quality score based on keyword presence and other factors
    quality_score = 0
    
    # Add points for keywords
    for keyword in quality_keywords:
        if keyword in content.lower():
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

def generate_summary(content, max_sentences=5):
    """
    Generate a summary from content using OpenAI.
    Only processes high-quality content.
    
    Args:
        content: Content text to summarize
        max_sentences: Maximum number of sentences (not used)
        
    Returns:
        Summarized text
    """
    if not content or len(content) < 100:
        return content
    
    # Check title and URL if available from context
    title = ""
    url = ""
    # These could be passed as additional parameters or retrieved from thread-local storage
    
    # Check content quality
    if not is_high_quality_content(content, title, url):
        logger.info("Content didn't pass quality check for summary generation")
        return content
    
    client = get_openai_client()
    if client:
        try:
            # Extract sections most relevant to Germany
            germany_paragraphs = []
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                if "germany" in para.lower() or "german" in para.lower():
                    germany_paragraphs.append(para)
            
            # Use either Germany-focused paragraphs or first part of content
            if germany_paragraphs and len(' '.join(germany_paragraphs)) >= 300:
                content_to_summarize = ' '.join(germany_paragraphs[:3])  # Top 3 most relevant paragraphs
                logger.info(f"Using {len(germany_paragraphs)} Germany-specific paragraphs for summary")
            else:
                # Use only first 3000 chars to save on token costs
                content_to_summarize = content[:3000] + ("..." if len(content) > 3000 else "")
            
            logger.info("Generating summary using OpenAI")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Summarize this content, focusing particularly on Germany's role, contributions, and any benefits mentioned: {content_to_summarize}"}
                ],
                temperature=0.3,
                max_tokens=150
            )
            
            summary = response.choices[0].message.content.strip()
            logger.info(f"Successfully generated summary with OpenAI ({len(summary)} chars)")
            return summary
        
        except Exception as e:
            logger.error(f"Error using OpenAI for summary: {str(e)}")
    
    return content

# FIXED: Removed the 'self' parameter from this function since it's not a class method
def analyze_sentiment(content: str) -> str:
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
    

def extract_benefits(content: str) -> Optional[str]:
    """
    Extract potential benefits to Germany using OpenAI with prompt from benefits.txt file.
    Only processes high-quality content.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Extracted benefits text or None
    """
    # Skip if no mention of Germany
    if "germany" not in content.lower() and "german" not in content.lower():
        return None
    
    # Check title and URL if available from context
    title = ""
    url = ""
    # These could be passed as additional parameters or retrieved from thread-local storage
    
    # Check content quality
    if not is_high_quality_content(content, title, url):
        logger.info("Content didn't pass quality check for benefits extraction")
        return None
    
    client = get_openai_client()
    if client:
        try:
            # Extract sections most relevant to Germany and benefits
            benefit_keywords = [
                "benefit", "advantage", "gain", "profit", "value", "opportunity",
                "improvement", "enhanced", "strengthen", "contribute", "partnership",
                "cooperation", "support", "funding", "investment", "expertise"
            ]
            
            # Try to use NLTK for better sentence segmentation
            try:
                sentences = sent_tokenize(content)
            except:
                # Fallback to simple sentence splitting
                sentences = re.split(r'(?<=[.!?])\s+', content)
            
            # Find sentences that mention both Germany and potential benefits
            benefit_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if ("germany" in sentence_lower or "german" in sentence_lower):
                    if any(keyword in sentence_lower for keyword in benefit_keywords):
                        benefit_sentences.append(sentence.strip())
            
            # Combine selected sentences or use content snippet
            if benefit_sentences and len(' '.join(benefit_sentences)) >= 200:
                content_to_analyze = ' '.join(benefit_sentences)
                logger.info(f"Using {len(benefit_sentences)} benefit-related sentences for analysis")
            else:
                # Use only first 3000 chars to save costs
                content_to_analyze = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Load prompt from benefits.txt file
            system_prompt = load_prompt_file("benefits.txt")
            if not system_prompt:
                logger.error("Benefits prompt file not found or empty")
                return None
            
            logger.info("Extracting benefits to Germany using OpenAI with benefits.txt prompt")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract any specific benefits to Germany from this text, focusing on concrete advantages, opportunities, or gains: {content_to_analyze}"}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            benefits = response.choices[0].message.content.strip()
            
            # Check if no benefits were found
            if "no specific benefits" in benefits.lower() or "no benefits" in benefits.lower():
                logger.info("OpenAI found no benefits to Germany")
                return None
                
            logger.info(f"Successfully extracted benefits with OpenAI ({len(benefits)} chars)")
            return benefits
        
        except Exception as e:
            logger.error(f"Error using OpenAI for benefits extraction: {str(e)}")
    
    return None

# Modified store_extract_data function to use the updated functions
def store_extract_data(extracted_data: List[Dict[str, Any]]) -> List[int]:
    """
    Store extracted data into the database using a batch transaction.
    Uses OpenAI for summary and benefits extraction when available.
    Improved error handling while maintaining batch efficiency.
    
    Args:
        extracted_data: List of dictionaries containing extracted web content
        
    Returns:
        List of database IDs for the stored records
    """
    if not extracted_data:
        logger.warning("No data to store")
        print("WARNING: No data to store in database")
        return []
    
    logger.info(f"Storing {len(extracted_data)} results in database")
    print(f"INFO: Attempting to store {len(extracted_data)} records in database")
    
    # List to store inserted record IDs
    inserted_ids = []
    success_count = 0
    error_count = 0
    
    conn = None
    cursor = None
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Process each item in a prepared batch
        valid_items = []
        
        # Pre-process items to filter out obviously invalid ones
        for i, item in enumerate(extracted_data):
            try:
                # Extract item data
                title = item.get("title", "")
                link = item.get("link", "")
                date_str = item.get("date")
                content = item.get("content", "")
                snippet = item.get("snippet", "")
                
                # Skip items with empty/invalid URLs
                if not link or len(link) < 5:
                    logger.warning(f"Skipping item {i+1} with invalid URL: {link}")
                    error_count += 1
                    continue
                    
                # Use existing summary or generate one with OpenAI
                summary = item.get("summary", snippet)
                if content and len(content) > 200:
                    generated_summary = generate_summary(content)
                    if generated_summary:
                        summary = generated_summary
                
                # Get themes (either from item or identify them)
                themes = item.get("themes", [])
                if not themes:
                    themes = identify_themes(content)
                
                # Get organization
                organization = item.get("organization", extract_organization_from_url(link))
                
                # Get sentiment (either from item or analyze it)
                sentiment = item.get("sentiment")
                if not sentiment and content:
                    sentiment = analyze_sentiment(content)
                
                # Extract benefits to Germany using OpenAI with benefits.txt prompt
                benefits_to_germany = item.get("benefits_to_germany")
                if content and ("germany" in content.lower() or "german" in content.lower()):
                    extracted_benefits = extract_benefits(content)
                    if extracted_benefits:
                        benefits_to_germany = extracted_benefits
                
                # Format and validate date
                date_value = None
                if date_str:
                    date_value = format_date(date_str)
                
                # Add to valid items list
                valid_items.append({
                    "link": link,
                    "title": title,
                    "date_value": date_value,
                    "summary": summary, 
                    "content": content,
                    "themes": themes,
                    "organization": organization,
                    "sentiment": sentiment,
                    "benefits_to_germany": benefits_to_germany
                })
                
            except Exception as prep_error:
                error_msg = f"Error preparing item {i+1}: {str(prep_error)}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                error_count += 1
        
        # Now insert all valid items in a single transaction
        for i, item in enumerate(valid_items):
            try:
                # Insert into content_data table
                query = """
                INSERT INTO content_data 
                (link, title, date, summary, full_content, information, themes, organization, sentiment, benefits_to_germany, insights)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                
                cursor.execute(
                    query, 
                    (item["link"], item["title"], item["date_value"], item["summary"], 
                     item["content"], item["summary"], item["themes"], 
                     item["organization"], item["sentiment"], item["benefits_to_germany"], None)
                )
                
                # Get the ID of the inserted record
                record_id = cursor.fetchone()[0]
                inserted_ids.append(record_id)
                success_count += 1
                
                logger.info(f"Inserted record with ID {record_id} for URL: {item['link']}")
                print(f"SUCCESS: Inserted record ID {record_id} | {item['title'][:50]}")
                
            except Exception as item_error:
                error_msg = f"Error storing item with URL {item.get('link', 'unknown')}: {str(item_error)}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                
                # IMPORTANT: Individual insert failures don't abort the whole transaction
                # We continue with the next item
                error_count += 1
                
                # Check if the error is transaction-related
                if "current transaction is aborted" in str(item_error):
                    logger.error("Transaction is aborted, rolling back and retrying with individual transactions")
                    raise  # This will cause a rollback and fall through to the individual insert retry
        
        # Commit the transaction if we got here
        conn.commit()
        logger.info(f"Transaction committed successfully with {success_count} records")
        
    except Exception as e:
        # If any error happens in the batch process, roll back
        error_msg = f"Error during batch insertion: {str(e)}"
        logger.error(error_msg)
        print(f"BATCH ERROR: {error_msg}")
        
        if conn:
            conn.rollback()
            logger.info("Transaction rolled back due to error")
        
        # FALLBACK: If batch mode failed, try individual inserts as a recovery
        if len(valid_items) > 0 and success_count == 0:
            logger.info("Retrying with individual inserts as fallback")
            print("Retrying failed items individually...")
            
            # Clear the IDs list since we're starting over
            inserted_ids = []
            success_count = 0
            
            # Try each item individually
            for item in valid_items:
                item_conn = None
                item_cursor = None
                
                try:
                    item_conn = get_db_connection()
                    item_cursor = item_conn.cursor()
                    
                    query = """
                    INSERT INTO content_data 
                    (link, title, date, summary, full_content, information, themes, organization, sentiment, benefits_to_germany, insights)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    RETURNING id;
                    """
                    
                    item_cursor.execute(
                        query, 
                        (item["link"], item["title"], item["date_value"], item["summary"], 
                         item["content"], item["summary"], item["themes"], 
                         item["organization"], item["sentiment"], item["benefits_to_germany"], None)
                    )
                    
                    record_id = item_cursor.fetchone()[0]
                    item_conn.commit()
                    
                    inserted_ids.append(record_id)
                    success_count += 1
                    
                    logger.info(f"Individual insert succeeded for URL: {item['link']}")
                    print(f"RECOVERY SUCCESS: Inserted record ID {record_id} | {item['title'][:50]}")
                    
                except Exception as item_error:
                    logger.error(f"Individual insert failed for URL {item['link']}: {str(item_error)}")
                    print(f"RECOVERY ERROR: {str(item_error)}")
                    error_count += 1
                    
                    if item_conn:
                        item_conn.rollback()
                
                finally:
                    if item_cursor:
                        item_cursor.close()
                    if item_conn:
                        item_conn.close()
    
    finally:
        # Always close cursor and connection
        if cursor:
            cursor.close()
        if conn:
            conn.close()
    
    # Summary after all items are processed
    logger.info(f"Successfully stored {success_count} records in database")
    print(f"\nDATABASE SUMMARY:")
    print(f"- Total records processed: {len(extracted_data)}")
    print(f"- Successfully stored: {success_count}")
    print(f"- Failed: {error_count}")
    
    if len(extracted_data) > 0:
        success_rate = (success_count/len(extracted_data))*100
        print(f"- Success rate: {success_rate:.1f}%")
    
    return inserted_ids

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
    
    # Extensive date parsing patterns
    date_patterns = [
        # ISO and standard formats
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{4}/\d{2}/\d{2})',  # YYYY/MM/DD
        r'(\d{2}-\d{2}-\d{4})',  # DD-MM-YYYY
        r'(\d{2}/\d{2}/\d{4})',  # DD/MM/YYYY
        
        # Verbose date formats
        r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4})',
        r'(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+(\d{4})',
        
        # ISO 8601 with time
        r'(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})',
        
        # Localized formats
        r'(\d{2}\.\d{2}\.\d{4})',  # German format DD.MM.YYYY
    ]
    
    # Preferred date parsing formats
    parse_formats = [
        '%Y-%m-%d',
        '%Y/%m/%d', 
        '%d-%m-%Y', 
        '%d/%m/%Y',
        '%d.%m.%Y',
        '%Y-%m-%dT%H:%M:%S',
        '%B %d, %Y',
        '%d %B %Y',
        '%b %d, %Y',
        '%d %b %Y'
    ]
    
    # First, try regex extraction
    for pattern in date_patterns:
        match = re.search(pattern, date_str, re.IGNORECASE)
        if match:
            date_str = match.group(0)  # Use the entire matched string
            break
    
    # Try parsing with different formats
    for fmt in parse_formats:
        try:
            parsed_date = datetime.strptime(date_str, fmt)
            
            # Additional validation
            current_year = datetime.now().year
            if parsed_date.year < 1900 or parsed_date.year > (current_year + 10):
                continue
            
            return parsed_date.strftime('%Y-%m-%d')
        except ValueError:
            continue
    
    # Fallback: attempt to extract year, month, day
    year_match = re.search(r'\b(19\d{2}|20\d{2})\b', date_str)
    if year_match:
        year = int(year_match.group(1))
        if 1900 <= year <= (datetime.now().year + 10):
            # Try to find month and day
            month_match = re.search(r'\b(0?[1-9]|1[0-2])\b', date_str)
            day_match = re.search(r'\b(0?[1-9]|[12]\d|3[01])\b', date_str)
            
            if month_match and day_match:
                month = int(month_match.group(1))
                day = int(day_match.group(1))
                
                try:
                    return datetime(year, month, day).strftime('%Y-%m-%d')
                except ValueError:
                    pass
    
    # Final fallback: log and return None
    logger.warning(f"Could not parse date string: {date_str}")
    return None

def identify_themes(content: str) -> List[str]:
    """
    Identify themes in content using keyword matching instead of AI.
    
    Args:
        content: Text content to analyze
        
    Returns:
        List of identified themes
    """
    content_lower = content.lower()
    themes = []
    
    # Define theme keywords dictionary
    theme_keywords = {
        "Indigenous Peoples": [
            "indigenous", "native communities", "indigenous rights", 
            "traditional knowledge", "aboriginal", "local communities"
        ],
        "Protected Areas": [
            "national park", "conservation area", "marine reserve", "protected area",
            "wildlife refuge", "conservation", "nature reserve"
        ],
        "Forest Restoration": [
            "forest restoration", "land restoration", "reforestation", "afforestation",
            "rewilding", "forest rehabilitation", "ecological restoration"
        ],
        "Marine Conservation": [
            "marine conservation", "marine protection", "ocean conservation", 
            "coral reef", "coastal protection", "blue economy"
        ],
        "Ecosystem Services": [
            "ecosystem services", "carbon sequestration", "Amazon basin", 
            "Congo basin", "rainforest", "biodiversity", "watershed services"
        ],
        "Sustainable Agriculture": [
            "sustainable agriculture", "agroforestry", "organic farming",
            "permaculture", "sustainable farming", "crop rotation"
        ],
        "Sustainable Forestry": [
            "sustainable forestry", "sustainable timber", "forest management",
            "reduced impact logging", "FSC certification", "sustainable logging"
        ],
        "Aquaculture": [
            "aquaculture", "fish farming", "sustainable aquaculture",
            "biodiversity in aquaculture", "responsible aquaculture"
        ]
    }
    
    # Check for each theme's keywords in the content
    for theme, keywords in theme_keywords.items():
        for keyword in keywords:
            if keyword in content_lower:
                if theme not in themes:
                    themes.append(theme)
                break  # Found one keyword for this theme, move to next theme
    
    # Add Environmental Sustainability as default theme if nothing else matched
    if not themes:
        themes.append("Environmental Sustainability")
    
    return themes

def fetch_data(limit=100, filters=None):
    """
    Fetch data from the database with optional filtering.
    
    Args:
        limit: Maximum number of records to retrieve
        filters: Dictionary of filter conditions
        
    Returns:
        Pandas DataFrame with the requested data
    """
    logger.info(f"Fetching data from database (limit: {limit}, filters: {filters})")
    
    try:
        import pandas as pd
        conn = get_db_connection()
        
        # Prepare base query
        query_parts = []
        params = {}
        
        # Add filter conditions if provided
        if filters:
            if filters.get('theme'):
                query_parts.append("%s = ANY(themes)")
                params['theme'] = filters['theme']
            
            if filters.get('organization'):
                query_parts.append("organization = %s")
                params['organization'] = filters['organization']
            
            if filters.get('sentiment'):
                query_parts.append("sentiment = %s")
                params['sentiment'] = filters['sentiment']
            
            if filters.get('start_date') and filters.get('end_date'):
                query_parts.append("date BETWEEN %s AND %s")
                params['start_date'] = filters['start_date']
                params['end_date'] = filters['end_date']
        
        # Construct WHERE clause
        where_clause = "WHERE " + " AND ".join(query_parts) if query_parts else ""
        
        # Full query
        query = f"""
        SELECT 
            id, link, title, date, summary, 
            full_content, information, themes, 
            organization, sentiment, 
            benefits_to_germany, insights, 
            created_at, updated_at
        FROM content_data 
        {where_clause}
        ORDER BY id DESC 
        LIMIT {limit}
        """
        
        # Create parameter list in correct order for the query
        param_values = []
        if filters:
            if filters.get('theme'):
                param_values.append(filters['theme'])
            
            if filters.get('organization'):
                param_values.append(filters['organization'])
            
            if filters.get('sentiment'):
                param_values.append(filters['sentiment'])
            
            if filters.get('start_date') and filters.get('end_date'):
                param_values.append(filters['start_date'])
                param_values.append(filters['end_date'])
        
        # Execute query
        cursor = conn.cursor()
        cursor.execute(query, param_values)
        
        # Fetch column names
        column_names = [desc[0] for desc in cursor.description]
        
        # Fetch all results
        results = cursor.fetchall()
        
        # Create DataFrame
        df = pd.DataFrame(results, columns=column_names)
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        logger.info(f"Fetched {len(df)} rows from database")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        # Return empty DataFrame in case of error
        import pandas as pd
        return pd.DataFrame()

def extract_organization_from_url(url):
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
        
        # Extract main domain (before first dot)
        main_domain = domain.split('.')[0]
        
        # Return capitalized domain
        return main_domain.upper()
    except:
        return "Unknown"

# The rest of the code remains unchanged

def get_content_by_id(content_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve content data by ID.
    
    Args:
        content_id: The ID of the content to retrieve
        
    Returns:
        Dictionary containing the content data or None if not found
    """
    logger.info(f"Retrieving content with ID: {content_id}")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT id, link, title, date, summary, full_content, information, themes, organization,
               sentiment, benefits_to_germany, insights, created_at, updated_at
        FROM content_data
        WHERE id = %s;
        """
        
        cursor.execute(query, (content_id,))
        record = cursor.fetchone()
        
        if record:
            # Get column names
            column_names = [desc[0] for desc in cursor.description]
            
            # Create dictionary from record
            content = dict(zip(column_names, record))
            
            # Convert date to string if needed
            if content['date'] and isinstance(content['date'], datetime):
                content['date'] = content['date'].isoformat()
            
            # Convert timestamps to strings
            if content['created_at'] and isinstance(content['created_at'], datetime):
                content['created_at'] = content['created_at'].isoformat()
            
            if content['updated_at'] and isinstance(content['updated_at'], datetime):
                content['updated_at'] = content['updated_at'].isoformat()
                
            logger.info(f"Found content: {content['title']}")
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            return content
        else:
            logger.warning(f"No content found with ID: {content_id}")
            
            # Close cursor and connection
            cursor.close()
            conn.close()
            
            return None
            
    except Exception as e:
        logger.error(f"Error retrieving content: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        raise

def get_all_content(limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieve multiple content entries with pagination.
    
    Args:
        limit: Maximum number of records to retrieve
        offset: Number of records to skip
        
    Returns:
        List of dictionaries containing content data
    """
    logger.info(f"Retrieving content (limit: {limit}, offset: {offset})")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT id, link, title, date, summary, themes, organization, sentiment, benefits_to_germany
        FROM content_data
        ORDER BY id DESC
        LIMIT %s OFFSET %s;
        """
        
        cursor.execute(query, (limit, offset))
        records = cursor.fetchall()
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        content_list = []
        for record in records:
            # Create dictionary from record
            content = dict(zip(column_names, record))
            
            # Convert date to string if needed
            if content['date'] and isinstance(content['date'], datetime):
                content['date'] = content['date'].isoformat()
                
            content_list.append(content)
        
        logger.info(f"Retrieved {len(content_list)} content records")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return content_list
            
    except Exception as e:
        logger.error(f"Error retrieving content list: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        raise

def update_content(content_id: int, data: Dict[str, Any]) -> bool:
    """
    Update content data by ID.
    
    Args:
        content_id: The ID of the content to update
        data: Dictionary containing the fields to update
        
    Returns:
        True if update was successful, False otherwise
    """
    logger.info(f"Updating content with ID: {content_id}")
    
    # Fields that can be updated
    allowed_fields = [
        "title", "date", "summary", "full_content", 
        "information", "themes", "organization", "sentiment",
        "benefits_to_germany", "insights"
    ]
    
    # Filter out any fields that are not allowed
    update_data = {k: v for k, v in data.items() if k in allowed_fields}
    
    if not update_data:
        logger.warning("No valid fields to update")
        return False
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Format date if present
        if "date" in update_data and update_data["date"]:
            try:
                update_data["date"] = format_date(update_data["date"])
            except Exception as e:
                logger.warning(f"Error formatting date for update: {str(e)}")
        
        # Build the SET part of the query dynamically
        set_clauses = []
        params = []
        
        for field, value in update_data.items():
            set_clauses.append(f"{field} = %s")
            params.append(value)
        
        set_clause = ", ".join(set_clauses)
        
        query = f"""
        UPDATE content_data
        SET {set_clause}
        WHERE id = %s
        RETURNING id;
        """
        
        # Add the content_id as the last parameter
        params.append(content_id)
        
        cursor.execute(query, params)
        
        # Check if a row was affected
        updated = cursor.fetchone() is not None
        
        # Commit the transaction
        conn.commit()
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        if updated:
            logger.info(f"Successfully updated content with ID: {content_id}")
        else:
            logger.warning(f"No content found with ID: {content_id}")
        
        return updated
        
    except Exception as e:
        logger.error(f"Error updating content: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        raise

def delete_content(content_id: int) -> bool:
    """
    Delete content data by ID.
    
    Args:
        content_id: The ID of the content to delete
        
    Returns:
        True if deletion was successful, False otherwise
    """
    logger.info(f"Deleting content with ID: {content_id}")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        DELETE FROM content_data
        WHERE id = %s
        RETURNING id;
        """
        
        cursor.execute(query, (content_id,))
        
        # Check if a row was affected
        deleted = cursor.fetchone() is not None
        
        # Commit the transaction
        conn.commit()
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        if deleted:
            logger.info(f"Successfully deleted content with ID: {content_id}")
        else:
            logger.warning(f"No content found with ID: {content_id}")
        
        return deleted
        
    except Exception as e:
        logger.error(f"Error deleting content: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        raise

def search_content(query_terms: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Search for content matching the given query terms.
    Uses PostgreSQL full-text search capabilities.
    
    Args:
        query_terms: String containing search terms
        limit: Maximum number of records to retrieve
        
    Returns:
        List of dictionaries containing matching content
    """
    logger.info(f"Searching for content with terms: {query_terms}")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create a tsquery from the search terms
        query = """
        SELECT id, link, title, date, summary, themes, organization, sentiment
        FROM content_data
        WHERE 
            to_tsvector('english', COALESCE(title, '')) @@ plainto_tsquery('english', %s) OR
            to_tsvector('english', COALESCE(summary, '')) @@ plainto_tsquery('english', %s) OR
            to_tsvector('english', COALESCE(full_content, '')) @@ plainto_tsquery('english', %s)
        ORDER BY 
            ts_rank(to_tsvector('english', COALESCE(title, '')), plainto_tsquery('english', %s)) +
            ts_rank(to_tsvector('english', COALESCE(summary, '')), plainto_tsquery('english', %s)) DESC
        LIMIT %s;
        """
        
        cursor.execute(query, (query_terms, query_terms, query_terms, query_terms, query_terms, limit))
        records = cursor.fetchall()
        
        # Get column names
        column_names = [desc[0] for desc in cursor.description]
        
        content_list = []
        for record in records:
            # Create dictionary from record
            content = dict(zip(column_names, record))
            
            # Convert date to string if needed
            if content['date'] and isinstance(content['date'], datetime):
                content['date'] = content['date'].isoformat()
                
            content_list.append(content)
        
        logger.info(f"Found {len(content_list)} content records matching the search terms")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return content_list
            
    except Exception as e:
        logger.error(f"Error searching content: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        raise

def get_stats():
    """
    Get statistics about the content in the database.
    
    Returns:
        Dictionary containing statistics
    """
    logger.info("Getting content statistics")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        stats = {}
        
        # Total content count
        cursor.execute("SELECT COUNT(*) FROM content_data")
        stats['total_content'] = cursor.fetchone()[0]
        
        # Organization count
        cursor.execute("SELECT COUNT(DISTINCT organization) FROM content_data WHERE organization IS NOT NULL")
        stats['organization_count'] = cursor.fetchone()[0]
        
        # Theme count (requires unnesting the array)
        cursor.execute("SELECT COUNT(DISTINCT unnest(themes)) FROM content_data WHERE themes IS NOT NULL")
        stats['theme_count'] = cursor.fetchone()[0]
        
        # Sentiment distribution
        cursor.execute("""
            SELECT sentiment, COUNT(*) 
            FROM content_data 
            WHERE sentiment IS NOT NULL 
            GROUP BY sentiment
        """)
        stats['sentiment_distribution'] = dict(cursor.fetchall())
        
        # Date range
        cursor.execute("SELECT MIN(date), MAX(date) FROM content_data WHERE date IS NOT NULL")
        min_date, max_date = cursor.fetchone()
        stats['date_range'] = {
            'min_date': min_date.isoformat() if min_date else None,
            'max_date': max_date.isoformat() if max_date else None,
        }
        
        # Top organizations
        cursor.execute("""
            SELECT organization, COUNT(*) as count
            FROM content_data
            WHERE organization IS NOT NULL
            GROUP BY organization
            ORDER BY count DESC
            LIMIT 5
        """)
        stats['top_organizations'] = dict(cursor.fetchall())
        
        # Top themes
        cursor.execute("""
            SELECT theme, COUNT(*) as count
            FROM (
                SELECT unnest(themes) as theme
                FROM content_data
                WHERE themes IS NOT NULL
            ) t
            GROUP BY theme
            ORDER BY count DESC
            LIMIT 5
        """)
        stats['top_themes'] = dict(cursor.fetchall())
        
        # Content with benefits to Germany
        cursor.execute("SELECT COUNT(*) FROM content_data WHERE benefits_to_germany IS NOT NULL")
        stats['content_with_benefits'] = cursor.fetchone()[0]
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        logger.info("Successfully retrieved content statistics")
        return stats
        
    except Exception as e:
        logger.error(f"Error getting content statistics: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        # Return empty stats in case of error
        return {}

def clean_text(text):
    """
    Clean text by removing extra whitespace, normalizing quotes, etc.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n\s*\n', '\n\n', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r' +', ' ', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_organization_from_url(url):
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
        
        # Extract main domain (before first dot)
        main_domain = domain.split('.')[0]
        
        # Return capitalized domain
        return main_domain.upper()
    except:
        return "Unknown"

def create_schema():
    """
    Create the database schema if it doesn't exist.
    This is useful for initializing a new database.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Creating database schema")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create content_data table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_data (
            id SERIAL PRIMARY KEY,
            link VARCHAR(255) NOT NULL UNIQUE,
            title VARCHAR(255),
            date DATE,
            summary TEXT,
            full_content TEXT,
            information TEXT,
            themes TEXT[],
            organization VARCHAR(100),
            sentiment VARCHAR(50),
            benefits_to_germany TEXT,
            insights TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_link ON content_data (link)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_themes ON content_data USING GIN (themes)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_organization ON content_data (organization)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_date ON content_data (date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_sentiment ON content_data (sentiment)")
        
        # Create trigger for updated_at
        cursor.execute("""
        CREATE OR REPLACE FUNCTION update_modified_column() RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = now();
            RETURN NEW;
        END;
        $$ LANGUAGE 'plpgsql'
        """)
        
        cursor.execute("""
        DROP TRIGGER IF EXISTS update_content_timestamp ON content_data;
        CREATE TRIGGER update_content_timestamp
        BEFORE UPDATE ON content_data
        FOR EACH ROW
        EXECUTE FUNCTION update_modified_column()
        """)
        
        # Commit changes
        conn.commit()
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        logger.info("Database schema created successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error creating database schema: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        return False

def filter_urls_by_keywords(urls, keywords):
    """
    Filter a list of URLs to keep only those containing keywords.
    
    Args:
        urls: List of URLs to filter
        keywords: List of keywords to check for
        
    Returns:
        Filtered list of URLs
    """
    if not urls or not keywords:
        return urls
        
    filtered_urls = []
    
    for url in urls:
        url_lower = url.lower()
        if any(keyword.lower() in url_lower for keyword in keywords):
            filtered_urls.append(url)
    
    return filtered_urls

# Example usage
if __name__ == "__main__":
    # This is an example of how to use the module directly
    try:
        print("\n" + "="*50)
        print("CONTENT DATABASE OPERATIONS")
        print("="*50 + "\n")
        
        # Create schema if needed
        create_schema()
        
        # Get statistics
        stats = get_stats()
        print(f"Database Statistics:")
        print(f"- Total Content: {stats.get('total_content', 0)}")
        print(f"- Organizations: {stats.get('organization_count', 0)}")
        print(f"- Themes: {stats.get('theme_count', 0)}")
        print(f"- Date Range: {stats.get('date_range', {})}")
        
        print("\nTesting theme identification...")
        test_content = """
        Germany's GIZ has been working with indigenous communities in the Amazon Basin
        to implement sustainable forestry practices that protect biodiversity while
        supporting local economies through carefully managed forest resources.
        """
        themes = identify_themes(test_content)
        print(f"Identified themes: {themes}")
        
        print("\nTesting sentiment analysis...")
        sentiment = analyze_sentiment(test_content)
        print(f"Sentiment: {sentiment}")
        
        print("\nTesting benefit extraction...")
        test_benefit_content = """
        This partnership between Germany and Brazil provides significant advantages
        for German companies seeking to develop sustainable forestry technologies
        while benefiting from the expertise of local communities.
        """
        benefits = extract_benefits(test_benefit_content)
        print(f"Benefits to Germany: {benefits}")
        
    except Exception as e:
        print(f"Error: {str(e)}")