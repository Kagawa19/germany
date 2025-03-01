import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import json
from datetime import datetime

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
    Format date string into ISO format (YYYY-MM-DD).
    Handles various date formats.
    
    Args:
        date_str: Date string in various formats
        
    Returns:
        ISO formatted date string or None if invalid
    """
    if not date_str:
        return None
        
    # Already in ISO format
    if isinstance(date_str, str) and re.match(r'^\d{4}-\d{2}-\d{2}', date_str):
        return date_str.split('T')[0]  # Remove time component if present
    
    # Try different date formats
    formats = [
        '%Y-%m-%d',  # 2023-01-15
        '%Y/%m/%d',  # 2023/01/15
        '%d-%m-%Y',  # 15-01-2023
        '%d/%m/%Y',  # 15/01/2023
        '%m-%d-%Y',  # 01-15-2023
        '%m/%d/%Y',  # 01/15/2023
        '%B %d, %Y',  # January 15, 2023
        '%d %B %Y',   # 15 January 2023
        '%b %d, %Y',  # Jan 15, 2023
        '%d %b %Y',   # 15 Jan 2023
        '%Y-%m-%dT%H:%M:%S',  # 2023-01-15T14:30:00
        '%Y-%m-%dT%H:%M:%SZ'  # 2023-01-15T14:30:00Z
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            return dt.strftime('%Y-%m-%d')
        except ValueError:
            continue
            
    # If all formats fail, try to extract a date with regex
    date_patterns = [
        r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
        r'(\d{2}/\d{2}/\d{4})',  # DD/MM/YYYY or MM/DD/YYYY
        r'(\d{2}\.\d{2}\.\d{4})'  # DD.MM.YYYY
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_str)
        if match:
            extracted_date = match.group(1)
            # Try to parse the extracted date
            for fmt in formats:
                try:
                    dt = datetime.strptime(extracted_date, fmt)
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    continue
    
    # If we reach here, we couldn't parse the date
    logger.warning(f"Could not parse date string: {date_str}")
    return None

def store_extract_data(extracted_data: List[Dict[str, Any]]) -> List[int]:
    """
    Store extracted data into the database.
    
    Args:
        extracted_data: List of dictionaries containing extracted web content
        
    Returns:
        List of database IDs for the stored records
    """
    if not extracted_data:
        logger.warning("No data to store")
        return []
    
    logger.info(f"Storing {len(extracted_data)} results in database")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Import re module for date validation
        import re
        
        # List to store inserted record IDs
        inserted_ids = []
        
        for item in extracted_data:
            try:
                # Extract item data
                title = item.get("title", "")
                link = item.get("link", "")
                date_str = item.get("date")
                content = item.get("content", "")
                snippet = item.get("snippet", "")
                summary = item.get("summary", snippet)
                theme = item.get("theme", "Environmental Sustainability")
                organization = item.get("organization")
                
                # Format and validate date
                date_value = None
                if date_str:
                    try:
                        date_value = format_date(date_str)
                    except Exception as date_error:
                        logger.warning(f"Error processing date '{date_str}' for URL {link}: {str(date_error)}")
                
                # Insert into content_data table
                query = """
                INSERT INTO content_data 
                (link, title, date, summary, full_content, information, theme, organization)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id;
                """
                
                cursor.execute(
                    query, 
                    (link, title, date_value, summary, content, summary, theme, organization)
                )
                
                # Get the ID of the inserted record
                record_id = cursor.fetchone()[0]
                inserted_ids.append(record_id)
                
                logger.info(f"Inserted record with ID {record_id} for URL: {link}")
            
            except Exception as item_error:
                logger.error(f"Error storing item with URL {item.get('link', 'unknown')}: {str(item_error)}")
                # Continue with next item
                continue
        
        # Commit the transaction
        conn.commit()
        logger.info(f"Successfully stored {len(inserted_ids)} records in database")
        
        # Close the cursor and connection
        cursor.close()
        conn.close()
        
        return inserted_ids
        
    except Exception as e:
        logger.error(f"Error storing data in database: {str(e)}")
        if 'conn' in locals() and conn:
            conn.rollback()
            conn.close()
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