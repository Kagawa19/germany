import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional, Tuple
from dotenv import load_dotenv
import json
import numpy as np
from openai import OpenAI

import re
from datetime import datetime
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables and set up OpenAI client
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=openai_api_key)

# Configure logging
logger = logging.getLogger("ContentDB")

def load_prompt(prompt_file):
    """
    Load a prompt template from the prompts folder
    
    Args:
        prompt_file: The name of the prompt file to load
        
    Returns:
        The content of the prompt file as a string
    """
    try:
        prompt_path = os.path.join("prompts", prompt_file)
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except Exception as e:
        logger.error(f"Error loading prompt from {prompt_file}: {str(e)}")
        return ""

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

def create_embedding(text: str) -> List[float]:
    """
    Generate an embedding vector for the provided text using OpenAI's API.
    
    Args:
        text: The text to create an embedding for
        
    Returns:
        List of floats representing the embedding vector
    """
    try:
        # Truncate text if needed to meet API limits
        max_text_length = 8000
        if len(text) > max_text_length:
            text = text[:max_text_length]
            
        logger.debug(f"Creating embedding for text (length: {len(text)})")
        
        # Call OpenAI's embedding API
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",  # Or the latest embedding model
            input=text
        )
        
        # Extract the embedding from the response
        embedding = response.data[0].embedding
        
        logger.debug(f"Successfully created embedding with dimensions: {len(embedding)}")
        return embedding
        
    except Exception as e:
        logger.error(f"Error creating embedding: {str(e)}")
        # Return empty vector in case of failure
        return []

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Calculate the cosine similarity between two vectors.
    
    Args:
        vec_a: First vector
        vec_b: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    if not vec_a or not vec_b:
        return 0.0
        
    try:
        vec_a = np.array(vec_a)
        vec_b = np.array(vec_b)
        
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
            
        return dot_product / (norm_a * norm_b)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0.0

def find_similar_content(embedding: List[float], limit: int = 5) -> List[Dict[str, Any]]:
    """
    Find content similar to the given embedding.
    
    Args:
        embedding: The embedding vector to compare against
        limit: Maximum number of similar items to return
        
    Returns:
        List of dictionaries containing similar content with similarity scores
    """
    logger.info(f"Finding similar content (limit: {limit})")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Query content from database
        query = """
        SELECT id, link, title, embedding
        FROM content_data
        WHERE embedding IS NOT NULL
        LIMIT 1000;  -- Fetch a reasonable number to compare in memory
        """
        
        cursor.execute(query)
        records = cursor.fetchall()
        
        # Calculate similarity scores in memory
        similarities = []
        for record in records:
            content_id, link, title, content_embedding = record
            
            # Skip if no embedding
            if not content_embedding:
                continue
                
            # Convert DB array to Python list if needed
            if isinstance(content_embedding, str):
                content_embedding = json.loads(content_embedding)
                
            # Calculate similarity
            similarity = cosine_similarity(embedding, content_embedding)
            
            similarities.append({
                "id": content_id,
                "link": link,
                "title": title,
                "similarity": similarity
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x["similarity"], reverse=True)
        
        # Take top N results
        top_results = similarities[:limit]
        
        logger.info(f"Found {len(top_results)} similar content items")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return top_results
            
    except Exception as e:
        logger.error(f"Error finding similar content: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        return []

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
        
        logger.debug(f"Sending content to OpenAI API for theme generation (length: {len(truncated_content)})")
        print(f"Sending content to OpenAI API for theme generation (length: {len(truncated_content)} chars)")
        
        # Load the themes prompt from file
        themes_prompt = load_prompt("themes.txt")
        
        # Call OpenAI API using the openai_client variable
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": themes_prompt},
                {"role": "user", "content": truncated_content}
            ],
            temperature=0.3
        )
        
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

def analyze_sentiment_with_openai(content: str) -> str:
    """
    Analyze sentiment of content using OpenAI API and sentiment.txt prompt.
    
    Args:
        content: The content to analyze
        
    Returns:
        Sentiment as string (Positive, Negative, or Neutral)
    """
    try:
        # Truncate content if it's too long
        max_content_length = 4000
        truncated_content = content[:max_content_length] if len(content) > max_content_length else content
        
        logger.debug(f"Sending content to OpenAI API for sentiment analysis (length: {len(truncated_content)})")
        print(f"Sending content to OpenAI API for sentiment analysis (length: {len(truncated_content)} chars)")
        
        # Load the sentiment prompt from file
        sentiment_prompt = load_prompt("sentiment.txt")
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sentiment_prompt},
                {"role": "user", "content": truncated_content}
            ],
            temperature=0.3
        )
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            sentiment = response.choices[0].message.content.strip()
            
            # Normalize sentiment to one of the three categories
            sentiment_lower = sentiment.lower()
            if "positive" in sentiment_lower:
                return "Positive"
            elif "negative" in sentiment_lower:
                return "Negative"
            else:
                return "Neutral"
        
        # Default to Neutral if we couldn't get a response
        return "Neutral"
        
    except Exception as e:
        logger.error(f"Error in OpenAI sentiment analysis: {str(e)}")
        print(f"ERROR in sentiment analysis: {str(e)}")
        return "Neutral"

def extract_benefits_with_openai(content: str) -> Optional[str]:
    """
    Extract benefits to Germany from content using OpenAI API and benefits.txt prompt.
    
    Args:
        content: The content to analyze
        
    Returns:
        Extracted benefits paragraph or None if not found
    """
    try:
        # Skip if content doesn't mention Germany
        content_lower = content.lower()
        if "germany" not in content_lower:
            return None
            
        # Truncate content if it's too long
        max_content_length = 4000
        truncated_content = content[:max_content_length] if len(content) > max_content_length else content
        
        logger.debug(f"Sending content to OpenAI API for benefits extraction (length: {len(truncated_content)})")
        print(f"Sending content to OpenAI API for benefits extraction (length: {len(truncated_content)} chars)")
        
        # Load the benefits prompt from file
        benefits_prompt = load_prompt("benefits.txt")
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": benefits_prompt},
                {"role": "user", "content": truncated_content}
            ],
            temperature=0.3
        )
        
        if hasattr(response, 'choices') and len(response.choices) > 0:
            benefits = response.choices[0].message.content.strip()
            
            # Return benefits if found, otherwise None
            if benefits and len(benefits) > 10 and not benefits.lower().startswith("no benefit"):
                return benefits
        
        return None
        
    except Exception as e:
        logger.error(f"Error in OpenAI benefits extraction: {str(e)}")
        print(f"ERROR in benefits extraction: {str(e)}")
        return None

def store_extract_data(extracted_data: List[Dict[str, Any]]) -> List[int]:
    """
    Store extracted data into the database using the new consolidated schema.
    Now with embedding generation for semantic search.
    
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
        # First, check if the embedding column exists in the content_data table
        # If not, add it
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        check_column_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='content_data' AND column_name='embedding';
        """
        
        cursor.execute(check_column_query)
        column_exists = cursor.fetchone() is not None
        
        # Add column if it doesn't exist
        if not column_exists:
            logger.info("Adding embedding column to content_data table")
            print("Adding embedding column to database schema...")
            
            add_column_query = """
            ALTER TABLE content_data
            ADD COLUMN embedding FLOAT[] DEFAULT NULL;
            """
            
            cursor.execute(add_column_query)
            conn.commit()
            logger.info("Successfully added embedding column")
            print("Added embedding column successfully")
        
        # Close connection and reconnect for main operation
        cursor.close()
        conn.close()
        
        # Now proceed with the main data insertion
        conn = get_db_connection()
        cursor = conn.cursor()
        
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
                
                # Generate themes using OpenAI with themes.txt prompt
                if openai_api_key:
                    try:
                        logger.info(f"Generating themes with OpenAI for: {title[:50]}...")
                        print(f"Generating themes for item {i+1}: {title[:30]}...")
                        item_themes = generate_themes_with_openai(content)
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
                
                # Use OpenAI for sentiment analysis with sentiment.txt prompt
                if openai_api_key:
                    try:
                        logger.info(f"Analyzing sentiment with OpenAI for: {title[:50]}...")
                        print(f"Analyzing sentiment for item {i+1}: {title[:30]}...")
                        sentiment = analyze_sentiment_with_openai(content)
                        logger.debug(f"Sentiment analysis result: {sentiment}")
                        print(f"Sentiment analysis: {sentiment}")
                    except Exception as sentiment_error:
                        error_msg = f"Error analyzing sentiment with OpenAI for URL {link}: {str(sentiment_error)}"
                        logger.warning(error_msg)
                        print(f"WARNING: {error_msg}")
                        sentiment = "Neutral"
                else:
                    logger.warning("OpenAI API key not found. Using Neutral as default sentiment.")
                    print("WARNING: OpenAI API key not found. Using Neutral as default sentiment.")
                    sentiment = "Neutral"
                
                # Extract potential benefits to Germany using OpenAI with benefits.txt prompt
                benefits_to_germany = None
                if openai_api_key and "germany" in content.lower():
                    try:
                        logger.info(f"Extracting benefits to Germany with OpenAI for: {title[:50]}...")
                        print(f"Extracting benefits for item {i+1}: {title[:30]}...")
                        benefits_to_germany = extract_benefits_with_openai(content)
                        if benefits_to_germany:
                            logger.info(f"Found potential benefits to Germany: {benefits_to_germany[:100]}...")
                            print(f"Benefits to Germany: Found potential content")
                        else:
                            print(f"Benefits to Germany: None identified")
                    except Exception as benefits_error:
                        error_msg = f"Error extracting benefits with OpenAI for URL {link}: {str(benefits_error)}"
                        logger.warning(error_msg)
                        print(f"WARNING: {error_msg}")
                
                # Generate embedding for semantic search
                embedding = None
                if openai_api_key:
                    try:
                        logger.info(f"Generating embedding for: {title[:50]}...")
                        print(f"Generating embedding for item {i+1}: {title[:30]}...")
                        
                        # Combine title and content for better embedding
                        embedding_text = f"{title}\n\n{content}"
                        embedding = create_embedding(embedding_text)
                        
                        logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                        print(f"Generated embedding vector with {len(embedding)} dimensions")
                    except Exception as embedding_error:
                        error_msg = f"Error generating embedding for URL {link}: {str(embedding_error)}"
                        logger.warning(error_msg)
                        print(f"WARNING: {error_msg}")
                        embedding = None
                
                # Insert into content_data table with the new structure
                # Now including embedding
                query = """
                INSERT INTO content_data 
                (link, title, date, summary, full_content, information, themes, organization, sentiment, benefits_to_germany, insights, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                
                logger.debug(f"Executing SQL insert for URL: {link}")
                cursor.execute(
                    query, 
                    (link, title, date_value, summary, content, summary, item_themes, organization, sentiment, benefits_to_germany, None, embedding)
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
               created_at, updated_at, embedding
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
                "updated_at": record[10].isoformat() if record[10] else None,
                "has_embedding": record[11] is not None
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
        SELECT id, link, title, date, summary, theme, organization, 
               (embedding IS NOT NULL) as has_embedding
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
                "organization": record[6],
                "has_embedding": record[7]
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
        
        # If content is being updated, regenerate the embedding
        regenerate_embedding = False
        embedding = None
        
        if "title" in update_data or "full_content" in update_data:
            regenerate_embedding = True
            
            # Get current data if needed
            current_title = None
            current_content = None
            
            if "title" not in update_data or "full_content" not in update_data:
                get_query = """
                SELECT title, full_content 
                FROM content_data 
                WHERE id = %s;
                """
                
                cursor.execute(get_query, (content_id,))
                current_data = cursor.fetchone()
                
                if current_data:
                    current_title = current_data[0]
                    current_content = current_data[1]
            
            # Combine new/existing data for embedding
            embedding_title = update_data.get("title", current_title)
            embedding_content = update_data.get("full_content", current_content)
            
            if embedding_title and embedding_content and openai_api_key:
                try:
                    logger.info(f"Regenerating embedding for content ID {content_id}")
                    print(f"Regenerating embedding for content ID {content_id}")
                    
                    embedding_text = f"{embedding_title}\n\n{embedding_content}"
                    embedding = create_embedding(embedding_text)
                    
                    # Add embedding to update data
                    update_data["embedding"] = embedding
                    
                    logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                    print(f"Generated embedding vector with {len(embedding)} dimensions")
                except Exception as e:
                    logger.error(f"Error generating embedding: {str(e)}")
                    print(f"WARNING: Error generating embedding: {str(e)}")
        
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
            if regenerate_embedding and embedding:
                logger.info(f"Successfully updated embedding for content ID: {content_id}")
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
    Now enhanced with semantic search using embeddings when available.
    
    Args:
        query_terms: String containing search terms
        limit: Maximum number of records to retrieve
        
    Returns:
        List of dictionaries containing matching content
    """
    logger.info(f"Searching for content with terms: {query_terms}")
    
    try:
        # First, try semantic search if OpenAI API key is available
        semantic_results = []
        
        if openai_api_key and query_terms.strip():
            try:
                logger.info(f"Generating query embedding for semantic search")
                print(f"Using semantic search for query: {query_terms}")
                
                # Generate embedding for the query
                query_embedding = create_embedding(query_terms)
                
                if query_embedding:
                    # Find similar content using embeddings
                    semantic_results = find_similar_content(query_embedding, limit=limit)
                    
                    if semantic_results:
                        logger.info(f"Found {len(semantic_results)} results using semantic search")
                        print(f"Found {len(semantic_results)} semantic matches")
                        return semantic_results
                    else:
                        logger.info("No semantic search results found, falling back to keyword search")
                        print("No semantic matches found, using keyword search as fallback")
            except Exception as e:
                logger.error(f"Error in semantic search: {str(e)}")
                print(f"Semantic search error: {str(e)}. Using keyword search instead.")
        
        # Fallback to traditional keyword search
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Keyword search query across multiple fields
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
                "organization": record[6],
                "search_method": "keyword"  # Indicate search method
            }
            content_list.append(content)
        
        logger.info(f"Found {len(content_list)} content records matching the search terms")
        print(f"Found {len(content_list)} keyword search results")
        
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
    Analyze content data to extract benefits and insights using OpenAI.
    This function uses the OpenAI API via benefits.txt prompt to extract benefits.
    
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
        print(f"Analyzing {len(content_items)} content items for benefits")
        
        for item in content_items:
            content_id, link, title, content = item
            
            # Skip if content is missing
            if not content:
                logger.warning(f"Skipping content ID {content_id}: No content available")
                print(f"Skipping content ID {content_id}: No content available")
                continue
                
            logger.info(f"Analyzing content ID {content_id}: {title}")
            print(f"Analyzing content ID {content_id}: {title[:50]}...")
            
            # Extract benefits using OpenAI API
            benefits_to_germany = None
            if openai_api_key and "germany" in content.lower():
                try:
                    logger.info(f"Extracting benefits to Germany with OpenAI for content ID {content_id}")
                    print(f"Extracting benefits for content ID {content_id}")
                    benefits_to_germany = extract_benefits_with_openai(content)
                except Exception as benefits_error:
                    logger.warning(f"Error extracting benefits with OpenAI: {str(benefits_error)}")
                    print(f"WARNING: Error extracting benefits with OpenAI: {str(benefits_error)}")
                    benefits_to_germany = None
            
            # Only proceed if we found something or to mark as processed
            benefits_text = benefits_to_germany if benefits_to_germany else None
            
            # Insert into benefits table
            benefit_query = """
            INSERT INTO benefits (links, benefits_to_germany, insights)
            VALUES (%s, %s, %s)
            RETURNING id;
            """
            
            links_array = "{" + link + "}"
            cursor.execute(benefit_query, (links_array, benefits_text, None))
            benefit_id = cursor.fetchone()[0]
            
            # Create relationship in content_benefits table
            relation_query = """
            INSERT INTO content_benefits (content_id, benefit_id)
            VALUES (%s, %s);
            """
            
            cursor.execute(relation_query, (content_id, benefit_id))
            
            if benefits_text:
                logger.info(f"Created benefit entry for content ID {content_id} with benefits")
                print(f"SUCCESS: Created benefit entry for content ID {content_id}")
            else:
                logger.info(f"Created empty benefit entry for content ID {content_id} (no benefits found)")
                print(f"INFO: Created empty benefit entry for content ID {content_id}")
            
            processed_count += 1
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Content analysis completed. Processed {processed_count} items")
        print(f"Content analysis completed. Processed {processed_count} items")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return processed_count
        
    except Exception as e:
        logger.error(f"Error in content analysis: {str(e)}")
        print(f"ERROR in content analysis: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        raise

def generate_embeddings_for_existing_content(limit=100):
    """
    Generate embeddings for existing content in the database that doesn't have embeddings yet.
    This is useful for upgrading existing databases to support semantic search.
    
    Args:
        limit: Maximum number of content items to process (default: 100)
        
    Returns:
        Number of items processed
    """
    logger.info(f"Generating embeddings for existing content (limit: {limit})")
    print(f"Generating embeddings for up to {limit} existing content items")
    
    if not openai_api_key:
        logger.error("OpenAI API key not found. Cannot generate embeddings.")
        print("ERROR: OpenAI API key not found. Cannot generate embeddings.")
        return 0
    
    try:
        # First, check if the embedding column exists in the content_data table
        # If not, add it
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if column exists
        check_column_query = """
        SELECT column_name 
        FROM information_schema.columns 
        WHERE table_name='content_data' AND column_name='embedding';
        """
        
        cursor.execute(check_column_query)
        column_exists = cursor.fetchone() is not None
        
        # Add column if it doesn't exist
        if not column_exists:
            logger.info("Adding embedding column to content_data table")
            print("Adding embedding column to database schema...")
            
            add_column_query = """
            ALTER TABLE content_data
            ADD COLUMN embedding FLOAT[] DEFAULT NULL;
            """
            
            cursor.execute(add_column_query)
            conn.commit()
            logger.info("Successfully added embedding column")
            print("Added embedding column successfully")
        
        # Get content that doesn't have embeddings yet
        query = """
        SELECT id, title, full_content
        FROM content_data
        WHERE embedding IS NULL
        LIMIT %s;
        """
        
        cursor.execute(query, (limit,))
        content_items = cursor.fetchall()
        
        processed_count = 0
        success_count = 0
        
        logger.info(f"Found {len(content_items)} content items without embeddings")
        print(f"Found {len(content_items)} content items without embeddings")
        
        for item in content_items:
            content_id, title, content = item
            
            # Skip if content is missing
            if not content or not title:
                logger.warning(f"Skipping content ID {content_id}: Missing title or content")
                print(f"Skipping content ID {content_id}: Missing title or content")
                processed_count += 1
                continue
                
            try:
                logger.info(f"Generating embedding for content ID {content_id}: {title[:50]}...")
                print(f"Generating embedding for content ID {content_id}: {title[:50]}...")
                
                # Combine title and content for better embedding
                embedding_text = f"{title}\n\n{content}"
                embedding = create_embedding(embedding_text)
                
                if embedding and len(embedding) > 0:
                    # Update the content with the embedding
                    update_query = """
                    UPDATE content_data
                    SET embedding = %s
                    WHERE id = %s;
                    """
                    
                    cursor.execute(update_query, (embedding, content_id))
                    
                    logger.info(f"Updated content ID {content_id} with embedding ({len(embedding)} dimensions)")
                    print(f"SUCCESS: Added embedding to content ID {content_id}")
                    success_count += 1
                else:
                    logger.warning(f"Failed to generate valid embedding for content ID {content_id}")
                    print(f"WARNING: Failed to generate valid embedding for content ID {content_id}")
            
            except Exception as e:
                logger.error(f"Error generating embedding for content ID {content_id}: {str(e)}")
                print(f"ERROR: Failed to generate embedding for content ID {content_id}: {str(e)}")
            
            processed_count += 1
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Embedding generation completed. Processed {processed_count} items, {success_count} successes")
        print(f"\nEMBEDDING GENERATION SUMMARY:")
        print(f"- Total processed: {processed_count}")
        print(f"- Successfully added: {success_count}")
        print(f"- Success rate: {(success_count/processed_count)*100:.1f}% if processed_count > 0 else 0}}%")
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return processed_count
        
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        print(f"CRITICAL ERROR in embedding generation: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
        raise

def semantic_search(query: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    Perform a semantic search using embeddings.
    
    Args:
        query: The search query text
        limit: Maximum number of results to return
        
    Returns:
        List of dictionaries containing matching content with similarity scores
    """
    logger.info(f"Performing semantic search for query: {query}")
    print(f"Performing semantic search for: {query}")
    
    if not openai_api_key:
        logger.error("OpenAI API key not found. Cannot perform semantic search.")
        print("ERROR: OpenAI API key not found. Cannot perform semantic search.")
        return []
    
    try:
        # Generate embedding for the query
        query_embedding = create_embedding(query)
        
        if not query_embedding:
            logger.error("Failed to generate embedding for query")
            print("ERROR: Failed to generate embedding for query")
            return []
        
        # Find similar content
        similar_content = find_similar_content(query_embedding, limit=limit)
        
        logger.info(f"Found {len(similar_content)} items matching the semantic query")
        print(f"Found {len(similar_content)} semantic matches")
        
        return similar_content
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        print(f"ERROR in semantic search: {str(e)}")
        return []