import psycopg2
import logging
from typing import List, Dict

# Using the logger from the main application
logger = logging.getLogger("WebExtractor")

def clean_text_for_database(text: str) -> str:
    """
    Clean text to remove NUL characters and ensure database compatibility.
    
    Args:
        text (str): Input text to clean
    
    Returns:
        str: Cleaned text safe for database insertion
    """
    # Remove NUL characters
    cleaned_text = text.replace('\x00', '')
    
    # Optional: Truncate extremely long text if needed
    max_length = 1_000_000  # Adjust based on your database column size
    if len(cleaned_text) > max_length:
        cleaned_text = cleaned_text[:max_length]
    
    return cleaned_text

def store_extract_data(extract_results: List[Dict]) -> List[int]:
    """
    Modified function to handle text cleaning before database insertion
    """
    logger.info(f"Storing {len(extract_results)} extraction results in database")
    
    conn = None
    stored_ids = []
    
    try:
        # Connect to the database
        conn = psycopg2.connect(
            host='postgres',
            port='5432',
            dbname='appdb',
            user='postgres',
            password='postgres'
        )
        
        cursor = conn.cursor()
        
        # Insert query for content_data table
        insert_query = """
        INSERT INTO content_data 
        (link, summary, full_content, information, theme, organization)
        VALUES (%s, %s, %s, %s, %s, %s)
        RETURNING id;
        """
        
        # Process each result with text cleaning
        for result in extract_results:
            # Clean text fields
            link = clean_text_for_database(result.get('link', ''))
            full_content = clean_text_for_database(result.get('content', ''))
            
            # Simple extraction for other fields
            summary = clean_text_for_database(
                result.get('snippet', '')[:255] if result.get('snippet') else result.get('title', '')[:255]
            )
            information = clean_text_for_database(
                full_content[:500] if len(full_content) > 500 else full_content
            )
            theme = "Environmental"  # Default theme
            organization = "Unknown"  # Default organization
            
            # Execute insert
            cursor.execute(insert_query, (link, summary, full_content, information, theme, organization))
            record_id = cursor.fetchone()[0]
            stored_ids.append(record_id)
            
            logger.info(f"Stored content with ID {record_id} for URL: {link}")
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Successfully stored {len(stored_ids)} results in database")
        
    except Exception as e:
        logger.error(f"Error storing data in database: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            cursor.close()
            conn.close()
    
    return stored_ids