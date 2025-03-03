import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
from openai import OpenAI

import re
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

# Configure logging
logger = logging.getLogger("ContentDB")

def get_db_connection():
    """Create a connection to the PostgreSQL database."""
    try:
        # Load environment variables
        load_dotenv()
        
        # Get database connection parameters from environment variables
        db_host = os.getenv("DB_HOST", "postgres")  # Use service name in Docker
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
            date_str = match.group(1)
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

def generate_themes_with_openai(content: str) -> List[str]:
    """
    Generate themes from content using OpenAI API with improved error handling.
    
    Args:
        content: The content to analyze
        
    Returns:
        List of themes
    """
    try:
        # Truncate content if it's too long for API limits
        max_content_length = 4000  # Adjust based on model token limits
        truncated_content = content[:max_content_length] if len(content) > max_content_length else content
        
        logger.debug(f"Sending content to OpenAI API (length: {len(truncated_content)})")
        print(f"Sending content to OpenAI API (length: {len(truncated_content)} chars)")
        
        # Call OpenAI API using the openai_client variable
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",  # Adjust based on available models
            messages=[
                {"role": "system", "content": "Extract 3-5 main themes from the following text. Return only a JSON array of strings."},
                {"role": "user", "content": truncated_content}
            ],
            temperature=0.3
        )
        
        # Debug the raw response format
        logger.debug(f"OpenAI API response type: {type(response)}")
        logger.debug(f"OpenAI API response structure: {dir(response)}")
        
        # Handle response
        themes = []
        
        # Access the content from the response
        if hasattr(response, 'choices') and len(response.choices) > 0:
            if hasattr(response.choices[0], 'message') and hasattr(response.choices[0].message, 'content'):
                content = response.choices[0].message.content
            else:
                content = str(response.choices[0])
        else:
            # Last resort: try to extract something useful
            content = str(response)
        
        # Process content to extract themes
        if content:
            # Try parsing as JSON if it looks like a JSON array
            if content.strip().startswith('[') and content.strip().endswith(']'):
                try:
                    import json
                    themes = json.loads(content)
                    logger.debug(f"Successfully parsed JSON themes: {themes}")
                except json.JSONDecodeError as json_error:
                    logger.warning(f"Failed to parse JSON themes: {str(json_error)}")
                    # Fall back to simple text parsing
                    themes = [theme.strip() for theme in content.strip('[]').split(',')]
            else:
                # Look for line breaks or commas
                if '\n' in content:
                    themes = [theme.strip() for theme in content.split('\n') if theme.strip()]
                elif ',' in content:
                    themes = [theme.strip() for theme in content.split(',') if theme.strip()]
                else:
                    themes = [content.strip()]
        
        # Clean up and validate themes
        validated_themes = []
        for theme in themes:
            # Remove quotes if present
            theme = theme.strip('"\'')
            if theme and len(theme) > 2:  # Minimum length check
                validated_themes.append(theme)
        
        logger.info(f"Generated {len(validated_themes)} themes")
        print(f"Generated {len(validated_themes)} themes")
        
        return validated_themes
    
    except Exception as e:
        logger.error(f"Error in OpenAI theme generation: {str(e)}")
        print(f"ERROR in theme generation: {str(e)}")
        return []

def store_extract_data(extracted_data: List[Dict[str, Any]]) -> List[int]:
    """
    Store extracted data into the database using the new consolidated schema.
    
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
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Import re module for date validation
        import re
        
        # List to store inserted record IDs
        inserted_ids = []
        success_count = 0
        error_count = 0
        
        logger.debug("Beginning processing of individual data items")
        print(f"Processing {len(extracted_data)} items...")
        
        for i, item in enumerate(extracted_data):
            try:
                # Log progress for larger datasets
                if i % 10 == 0 and i > 0:
                    logger.debug(f"Processed {i}/{len(extracted_data)} items so far")
                    print(f"Progress: {i}/{len(extracted_data)} items processed")
                
                # Extract item data
                title = item.get("title", "")
                link = item.get("link", "")
                date_str = item.get("date")
                content = item.get("content", "")
                snippet = item.get("snippet", "")
                summary = item.get("summary", snippet)
                
                logger.debug(f"Processing item: {title[:50]}... | URL: {link}")
                
                # Extract main domain from URL for organization identification
                import re
                from urllib.parse import urlparse
                
                # Parse the URL to extract domain
                try:
                    parsed_url = urlparse(link)
                    domain = parsed_url.netloc
                    
                    # Remove 'www.' prefix if present
                    if domain.startswith('www.'):
                        domain = domain[4:]
                    
                    # Extract main part of domain (before first dot)
                    main_domain = domain.split('.')[0]
                    
                    logger.debug(f"Extracted domain: {domain}, main part: {main_domain}")
                    print(f"Domain: {domain}")
                    
                    # Assign organization based on specific domains
                    if domain == "kfw.de" or main_domain == "kfw":
                        organization = "KfW"
                    elif domain == "giz.de" or main_domain == "giz":
                        organization = "GIZ"
                    elif domain == "bmz.de" or main_domain == "bmz":
                        organization = "BMZ"
                    else:
                        # Use the main domain as the organization
                        organization = main_domain.capitalize()
                        
                    logger.info(f"Identified organization: {organization} for domain: {domain}")
                    print(f"Organization: {organization} | URL: {link[:50]}...")
                    
                except Exception as url_error:
                    organization = None
                    logger.warning(f"Error extracting domain from URL: {link}: {str(url_error)}")
                    print(f"WARNING: Could not extract domain from URL: {link}")
                
                # Generate themes using OpenAI
                if openai_api_key:
                    try:
                        logger.info(f"Generating themes with OpenAI for: {title[:50]}...")
                        print(f"Generating themes for item {i+1}: {title[:30]}...")
                        item_themes = generate_themes_with_openai(content)
                        # Fix for the 'ChatCompletion' object is not subscriptable error
                        if hasattr(item_themes, 'choices') and len(item_themes.choices) > 0:
                            # Extract content from the ChatCompletion object
                            themes_content = item_themes.choices[0].message.content
                            # Convert string representation to list if needed
                            if isinstance(themes_content, str):
                                # Try to convert string to list if it looks like a list
                                if themes_content.strip().startswith('[') and themes_content.strip().endswith(']'):
                                    try:
                                        import json
                                        item_themes = json.loads(themes_content)
                                    except:
                                        # If JSON parsing fails, use simple string split
                                        item_themes = [theme.strip() for theme in themes_content.strip('[]').split(',')]
                                else:
                                    # Just use the content as a single theme
                                    item_themes = [themes_content]
                            else:
                                item_themes = themes_content
                        logger.debug(f"Generated themes: {item_themes}")
                    except Exception as openai_error:
                        error_msg = f"Error generating themes with OpenAI for URL {link}: {str(openai_error)}"
                        logger.warning(error_msg)
                        print(f"WARNING: {error_msg}")
                        item_themes = []
                else:
                    logger.warning("OpenAI API key not found. Skipping theme generation.")
                    print("WARNING: OpenAI API key not found. Skipping theme generation.")
                    item_themes = []
                
                # Format and validate date
                date_value = None
                if date_str:
                    try:
                        date_value = format_date(date_str)
                        logger.debug(f"Formatted date '{date_str}' to '{date_value}'")
                    except Exception as date_error:
                        error_msg = f"Error processing date '{date_str}' for URL {link}: {str(date_error)}"
                        logger.warning(error_msg)
                        print(f"WARNING: {error_msg}")
                else:
                    logger.debug(f"No date provided for URL: {link}")
                
                # Basic sentiment analysis on content
                sentiment = "Neutral"
                positive_keywords = ["success", "benefit", "advantage", "improve", "growth", "achievement", "positive", 
                                   "sustainable", "opportunity", "progress", "development", "cooperation"]
                negative_keywords = ["challenge", "risk", "problem", "difficulty", "issue", "concern", "negative", 
                                   "obstacle", "failure", "crisis", "conflict", "dispute"]
                
                # Simple sentiment analysis
                content_lower = content.lower()
                positive_count = sum(1 for word in positive_keywords if word in content_lower)
                negative_count = sum(1 for word in negative_keywords if word in content_lower)
                
                if positive_count > negative_count * 1.5:
                    sentiment = "Positive"
                elif negative_count > positive_count * 1.5:
                    sentiment = "Negative"
                
                logger.debug(f"Sentiment analysis: {sentiment} (Positive: {positive_count}, Negative: {negative_count})")
                print(f"Sentiment analysis: {sentiment}")
                
                # Extract potential benefits to Germany
                benefits_to_germany = None
                if "germany" in content_lower and any(benefit in content_lower for benefit in 
                                                    ["benefit", "advantage", "partnership", "cooperation", 
                                                     "collaboration", "economic", "trade", "investment"]):
                    # Extract paragraph containing both Germany and benefit terms
                    paragraphs = content.split('\n')
                    for para in paragraphs:
                        para_lower = para.lower()
                        if "germany" in para_lower and any(benefit in para_lower for benefit in 
                                                         ["benefit", "advantage", "partnership", "cooperation", 
                                                          "collaboration", "economic", "trade", "investment"]):
                            benefits_to_germany = para
                            break
                
                if benefits_to_germany:
                    logger.info(f"Found potential benefits to Germany: {benefits_to_germany[:100]}...")
                    print(f"Benefits to Germany: Found potential content")
                else:
                    print(f"Benefits to Germany: None identified")
                
                # Insert into content_data table with the new structure
                query = """
                INSERT INTO content_data 
                (link, title, date, summary, full_content, information, themes, organization, sentiment, benefits_to_germany, insights)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                
                logger.debug(f"Executing SQL insert for URL: {link}")
                cursor.execute(
                    query, 
                    (link, title, date_value, summary, content, summary, item_themes, organization, sentiment, benefits_to_germany, None)
                )
                
                # Get the ID of the inserted record
                record_id = cursor.fetchone()[0]
                inserted_ids.append(record_id)
                success_count += 1
                
                logger.info(f"Inserted record with ID {record_id} for URL: {link}")
                print(f"SUCCESS: Inserted record ID {record_id} | {title[:50]}")
                
            except Exception as item_error:
                error_msg = f"Error storing item with URL {item.get('link', 'unknown')}: {str(item_error)}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                error_count += 1
                # Continue with next item
                continue
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Successfully stored {len(inserted_ids)} records in database")
        print(f"\nDATABASE SUMMARY:")
        print(f"- Total records processed: {len(extracted_data)}")
        print(f"- Successfully stored: {success_count}")
        print(f"- Failed: {error_count}")
        print(f"- Success rate: {(success_count/len(extracted_data))*100:.1f}%")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        logger.debug("Database connection closed")
        
        return inserted_ids
    
    except Exception as e:
        error_msg = f"Error storing data in database: {str(e)}"
        logger.error(error_msg)
        print(f"CRITICAL ERROR: {error_msg}")
        if 'conn' in locals() and conn:
            logger.info("Rolling back database transaction")
            print("Rolling back database transaction...")
            conn.rollback()
            conn.close()
            logger.debug("Database connection closed after rollback")
        raise

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
        SELECT id, link, title, date, summary, full_content, information, theme, organization,
               created_at, updated_at
        FROM content_data
        WHERE id = %s;
        """
        
        cursor.execute(query, (content_id,))
        record = cursor.fetchone()
        
        if record:
            # Convert record to dictionary
            content = {
                "id": record[0],
                "link": record[1],
                "title": record[2],
                "date": record[3].isoformat() if record[3] else None,
                "summary": record[4],
                "full_content": record[5],
                "information": record[6],
                "theme": record[7],
                "organization": record[8],
                "created_at": record[9].isoformat() if record[9] else None,
                "updated_at": record[10].isoformat() if record[10] else None
            }
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
        SELECT id, link, title, date, summary, theme, organization
        FROM content_data
        ORDER BY id DESC
        LIMIT %s OFFSET %s;
        """
        
        cursor.execute(query, (limit, offset))
        records = cursor.fetchall()
        
        content_list = []
        for record in records:
            content = {
                "id": record[0],
                "link": record[1],
                "title": record[2],
                "date": record[3].isoformat() if record[3] else None,
                "summary": record[4],
                "theme": record[5],
                "organization": record[6]
            }
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
        "information", "theme", "organization"
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
        set_clause = ", ".join([f"{field} = %s" for field in update_data.keys()])
        set_clause += ", updated_at = CURRENT_TIMESTAMP"
        
        query = f"""
        UPDATE content_data
        SET {set_clause}
        WHERE id = %s
        RETURNING id;
        """
        
        # Build parameters list
        params = list(update_data.values())
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
        
        # Simple search query across multiple fields
        query = """
        SELECT id, link, title, date, summary, theme, organization
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
        
        content_list = []
        for record in records:
            content = {
                "id": record[0],
                "link": record[1],
                "title": record[2],
                "date": record[3].isoformat() if record[3] else None,
                "summary": record[4],
                "theme": record[5],
                "organization": record[6]
            }
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

def analyze_content_for_benefits(limit=1000):
    """
    Analyze content data to extract benefits and insights.
    This is a legacy function that uses keyword-based analysis.
    For AI-powered analysis, use the function in main.py.
    
    Args:
        limit: Maximum number of content items to analyze (default: 1000)
        
    Returns:
        Number of content items processed
    """
    logger.info(f"Starting content analysis with limit: {limit}")
    
    try:
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get content items that haven't been analyzed yet
        # We'll do this by finding content that doesn't exist in the content_benefits relationship table
        query = """
        SELECT cd.id, cd.link, cd.title, cd.full_content
        FROM content_data cd
        LEFT JOIN content_benefits cb ON cd.id = cb.content_id
        WHERE cb.content_id IS NULL
        LIMIT %s;
        """
        
        cursor.execute(query, (limit,))
        content_items = cursor.fetchall()
        
        processed_count = 0
        
        logger.info(f"Found {len(content_items)} content items to analyze")
        
        for item in content_items:
            content_id, link, title, content = item
            
            # Skip if content is missing
            if not content:
                logger.warning(f"Skipping content ID {content_id}: No content available")
                continue
                
            logger.info(f"Analyzing content ID {content_id}: {title}")
            
            # Extract benefits from content
            # This could be done with a more sophisticated approach like using an LLM
            # For now, we'll use a simple keyword-based approach
            
            benefits_to_germany = []
            insights = []
            
            # Example keywords for benefits (replace with actual logic)
            benefit_keywords = [
                "benefit", "advantage", "gain", "profit", "value", 
                "impact", "sustainable development", "cooperation",
                "economic", "trade", "partnership", "collaboration",
                "technology transfer", "innovation", "expertise"
            ]
            
            # Example keywords for insights (replace with actual logic)
            insight_keywords = [
                "insight", "lesson", "finding", "discovery", "conclusion",
                "learn", "knowledge", "understand", "approach",
                "strategy", "method", "success", "challenge", "solution"
            ]
            
            content_paragraphs = content.split('\n\n')
            
            # Process paragraphs to extract benefits and insights
            for paragraph in content_paragraphs:
                paragraph = paragraph.strip()
                if not paragraph or len(paragraph) < 40:  # Skip short paragraphs
                    continue
                    
                # Check for benefits
                if any(keyword in paragraph.lower() for keyword in benefit_keywords):
                    benefits_to_germany.append(paragraph)
                        
                # Check for insights
                if any(keyword in paragraph.lower() for keyword in insight_keywords):
                    insights.append(paragraph)
            
            # Only proceed if we found something
            if benefits_to_germany or insights:
                # Insert into benefits table
                benefit_query = """
                INSERT INTO benefits (links, benefits_to_germany, insights)
                VALUES (%s, %s, %s)
                RETURNING id;
                """
                
                links_array = "{" + link + "}"
                benefits_text = "\n\n".join(benefits_to_germany) if benefits_to_germany else None
                insights_text = "\n\n".join(insights) if insights else None
                
                cursor.execute(benefit_query, (links_array, benefits_text, insights_text))
                benefit_id = cursor.fetchone()[0]
                
                # Create relationship in content_benefits table
                relation_query = """
                INSERT INTO content_benefits (content_id, benefit_id)
                VALUES (%s, %s);
                """
                
                cursor.execute(relation_query, (content_id, benefit_id))
                
                logger.info(f"Created benefit entry for content ID {content_id} with {len(benefits_to_germany)} benefits and {len(insights)} insights")
                processed_count += 1
            else:
                # Even if we didn't find benefits, mark as processed
                # Create an empty entry to avoid reprocessing this content
                benefit_query = """
                INSERT INTO benefits (links, benefits_to_germany, insights)
                VALUES (%s, %s, %s)
                RETURNING id;
                """
                
                links_array = "{" + link + "}"
                cursor.execute(benefit_query, (links_array, None, None))
                benefit_id = cursor.fetchone()[0]
                
                # Create relationship in content_benefits table
                relation_query = """
                INSERT INTO content_benefits (content_id, benefit_id)
                VALUES (%s, %s);
                """
                
                cursor.execute(relation_query, (content_id, benefit_id))
                
                logger.info(f"Created empty benefit entry for content ID {content_id} (no benefits/insights found)")
                processed_count += 1
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Content analysis completed. Processed {processed_count} items")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return processed_count
        
    except Exception as e:
        logger.error(f"Error in content analysis: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        raise