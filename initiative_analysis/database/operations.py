import os
import logging
import psycopg2
from psycopg2.extras import execute_values
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import re
from urllib.parse import urlparse

from database.connection import get_db_connection
from utils.text_processing import format_date, clean_html_entities
from extraction.analysis import get_openai_client

logger = logging.getLogger("ContentDB")

def store_extract_data(extracted_data: List[Dict[str, Any]]) -> List[int]:
    """
    Store extracted data into the database without further processing.
    
    Args:
        extracted_data: List of dictionaries containing extracted web content
        
    Returns:
        List of database IDs for the stored records
    """
    if not extracted_data:
        logger.warning("No data to store")
        return []
    
    logger.info(f"Storing {len(extracted_data)} results in database")
    
    # List to store inserted record IDs
    inserted_ids = []
    success_count = 0
    error_count = 0
    
    # Process each item with individual transactions to avoid cascading failures
    for i, item in enumerate(extracted_data):
        conn = None
        cursor = None
        
        try:
            # Extract item data
            title = item.get("title", "")
            link = item.get("link", "")
            date_str = item.get("date")
            content = item.get("content", "")
            summary = item.get("summary", "")
            themes = item.get("themes", [])
            organization = item.get("organization", "")
            sentiment = item.get("sentiment", "Neutral")
            language = item.get("language", "English")
            initiative = item.get("initiative", "ABS Initiative")
            initiative_key = item.get("initiative_key", "abs_initiative")
            
            # Skip items with empty/invalid URLs
            if not link or len(link) < 5:
                logger.warning(f"Skipping item {i+1} with invalid URL: {link}")
                error_count += 1
                continue
            
            # Format date
            date_value = None
            if date_str:
                date_value = format_date(date_str)
            
            # Get database connection - separate connection for each item
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Construct query
            query = """
            INSERT INTO content_data 
            (link, title, date, summary, full_content, themes, organization, sentiment, 
             language, initiative, initiative_key, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
            RETURNING id;
            """
            
            cursor.execute(
                query, 
                (link, title, date_value, summary, content, themes, organization, sentiment,
                 language, initiative, initiative_key)
            )
            
            # Get the ID of the inserted record
            record_id = cursor.fetchone()[0]
            inserted_ids.append(record_id)
            success_count += 1
            
            # Commit this transaction
            conn.commit()
            
            logger.info(f"Inserted record with ID {record_id} for URL: {link}")
            
        except Exception as e:
            error_msg = f"Error storing item with URL {item.get('link', 'unknown')}: {str(e)}"
            logger.error(error_msg)
            if conn:
                conn.rollback()
            error_count += 1
        
        finally:
            # Close cursor and connection
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    # Summary after all items are processed
    logger.info(f"Successfully stored {success_count} records in database")
    logger.info(f"Failed to store {error_count} records")
    
    return inserted_ids

def fetch_data(limit=100, filters=None):
    """
    Fetch data from the database with optional filtering.
    
    Args:
        limit: Maximum number of records to retrieve
        filters: Dictionary of filter conditions
        
    Returns:
        List of dictionaries containing content data
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
            
            if filters.get('initiative'):
                query_parts.append("initiative = %s")
                params['initiative'] = filters['initiative']
            
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
            full_content, themes, organization, sentiment, 
            initiative, initiative_key, benefits_to_germany, 
            benefit_categories, benefit_examples,
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
            
            if filters.get('initiative'):
                param_values.append(filters['initiative'])
            
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
        
        # Parse JSON columns if present
        if 'benefit_categories' in df.columns:
            df['benefit_categories'] = df['benefit_categories'].apply(
                lambda x: json.loads(x) if x and isinstance(x, str) else None
            )
        
        if 'benefit_examples' in df.columns:
            df['benefit_examples'] = df['benefit_examples'].apply(
                lambda x: json.loads(x) if x and isinstance(x, str) else None
            )
        
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
        SELECT id, link, title, date, summary, themes, organization, sentiment, initiative, benefits_to_germany
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

# Remaining methods would be implemented similarly...
# I'll continue in the next message due to length constraints
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
        SELECT id, link, title, date, summary, full_content, themes, organization,
               sentiment, initiative, initiative_key, benefits_to_germany, 
               benefit_categories, benefit_examples, created_at, updated_at
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
            
            # Parse JSON fields if present
            if 'benefit_categories' in content and content['benefit_categories']:
                try:
                    if isinstance(content['benefit_categories'], str):
                        content['benefit_categories'] = json.loads(content['benefit_categories'])
                except:
                    pass
            
            if 'benefit_examples' in content and content['benefit_examples']:
                try:
                    if isinstance(content['benefit_examples'], str):
                        content['benefit_examples'] = json.loads(content['benefit_examples'])
                except:
                    pass
                
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
        
        query = """
        SELECT id, link, title, date, summary, themes, organization, sentiment, initiative, initiative_key, benefits_to_germany
        FROM content_data
        WHERE 
            to_tsvector('english', COALESCE(title, '')) @@ plainto_tsquery('english', %s) OR
            to_tsvector('english', COALESCE(summary, '')) @@ plainto_tsquery('english', %s) OR
            to_tsvector('english', COALESCE(full_content, '')) @@ plainto_tsquery('english', %s) OR
            to_tsvector('english', COALESCE(benefits_to_germany, '')) @@ plainto_tsquery('english', %s)
        ORDER BY 
            ts_rank(to_tsvector('english', COALESCE(title, '')), plainto_tsquery('english', %s)) +
            ts_rank(to_tsvector('english', COALESCE(summary, '')), plainto_tsquery('english', %s)) DESC
        LIMIT %s;
        """
        
        cursor.execute(query, (query_terms, query_terms, query_terms, query_terms, query_terms, query_terms, limit))
        
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

def create_schema():
    """
    Create the database schema if it doesn't exist.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Creating database schema")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create content_data table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_data (
            id SERIAL PRIMARY KEY,
            link VARCHAR(255) NOT NULL UNIQUE,
            title VARCHAR(255),
            date DATE,
            summary TEXT,
            full_content TEXT,
            themes TEXT[],
            organization VARCHAR(100),
            sentiment VARCHAR(50),
            initiative VARCHAR(100),
            initiative_key VARCHAR(50),
            benefits_to_germany TEXT,
            benefit_categories JSONB,
            benefit_examples JSONB,
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

def extract_benefits(content: str) -> Optional[str]:
    """
    Extract potential benefits using OpenAI.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Extracted benefits text or None
    """
    # Skip if no mention of relevant terms
    relevant_terms = ["germany", "german", "abs", "capacity", "bio-innovation", "africa"]
    if not any(term in content.lower() for term in relevant_terms):
        return None
    
    # Check content quality
    if not is_high_quality_content(content, "", ""):
        logger.info("Content didn't pass quality check for benefits extraction")
        return None
    
    client = get_openai_client()
    if client:
        try:
            # Prepare content - limit to first 3000 chars to save tokens
            excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
            
            # Create prompt for benefits extraction
            prompt = f"""
Extract specific benefits mentioned in this text related to the ABS Capacity Development Initiative or initiatives in developing countries. 
Focus on concrete, factual examples of benefits to Germany or other partner countries.

Text:
{excerpt}

Provide a concise summary of the most significant benefits, highlighting their potential impact and strategic importance.
"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            benefits = response.choices[0].message.content.strip()
            
            # Check if no benefits were found
            if "no specific benefits" in benefits.lower() or "no benefits" in benefits.lower():
                logger.info("OpenAI found no benefits")
                return None
                
            logger.info(f"Successfully extracted benefits with OpenAI ({len(benefits)} chars)")
            return benefits
        
        except Exception as e:
            logger.error(f"Error using OpenAI for benefits extraction: {str(e)}")
    
    return None

def extract_benefit_examples(content: str, initiative: str) -> List[Dict[str, Any]]:
    """
    Extract examples of benefits from the content.
    
    Args:
        content: Content text to analyze
        initiative: Initiative key
        
    Returns:
        List of extracted benefit examples
    """
    if not content or len(content) < 100:
        return []
            
    content_lower = content.lower()
    
    # Define benefit categories
    benefit_categories = {
        "environmental_benefits": [
            "biodiversity conservation", "ecosystem restoration", "sustainable use",
            "habitat protection", "ecological integrity", "conservation", "protected areas",
            "species protection", "environmental sustainability"
        ],
        "economic_benefits": [
            "poverty alleviation", "private sector", "technology transfer", 
            "sustainable development", "job creation", "employment", "income generation",
            "public-private partnerships", "market access", "trade", "investment",
            "economic growth", "livelihoods", "business opportunities", "value chains"
        ],
        "social_benefits": [
            "indigenous peoples", "local communities", "iplcs", "capacity building",
            "empowerment", "gender equality", "education", "training", "skills development",
            "participatory approach", "inclusion", "community development", "knowledge sharing"
        ],
        "strategic_benefits": [
            "global governance", "policy development", "legislation", "regulations",
            "institutional frameworks", "international cooperation", "partnerships",
            "stakeholder engagement", "compliance", "legal framework", "policy implementation"
        ]
    }
    
    # Find paragraphs that mention benefits
    paragraphs = content.split('\n\n')
    benefit_paragraphs = []
    
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        
        # Determine benefit category
        category = "general"
        max_score = 0
        
        for cat_key, cat_terms in benefit_categories.items():
            score = sum(paragraph_lower.count(term) for term in cat_terms)
            if score > max_score:
                max_score = score
                category = cat_key
                
        # Create benefit example
        if max_score > 0:
            benefit_example = {
                "text": paragraph.strip(),
                "category": category,
                "initiative": initiative,
                "word_count": len(paragraph.split())
            }
            
            benefit_paragraphs.append(benefit_example)
    
    return benefit_paragraphs

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
    
    # Skip if content doesn't mention relevant terms
    relevant_terms = ["germany", "german", "abs", "capacity", "bio-innovation", "africa"]
    if not any(term in content.lower() for term in relevant_terms):
        logger.info(f"Content doesn't mention relevant terms, skipping AI processing")
        return False
    
    # Check if the content is from a reliable domain
    reliable_domains = ["giz.de", "bmz.de", "kfw.de", "europa.eu", "un.org", "abs-initiative.info", "cbd.int"]
    is_reliable_source = any(domain in url.lower() for domain in reliable_domains)
    
    # Look for quality indicators in the content
    quality_keywords = [
        "cooperation", "sustainable", "development", "partnership", "initiative",
        "project", "bilateral", "agreement", "funding", "investment", 
        "climate", "conservation", "biodiversity", "renewable", "forest",
        "abs", "nagoya", "capacity", "benefit sharing"
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

def semantic_search(query_text, top_k=5):
    """
    Perform semantic search using embeddings.
    
    Args:
        query_text: Text to search for
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries containing search results
    """
    # Generate embedding for the query
    client = get_openai_client()
    if not client:
        logger.error("OpenAI client not available for semantic search")
        return []
    
    try:
        # Generate embedding for query
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=query_text
        )
        
        query_embedding = response.data[0].embedding
        
        # Get database connection
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if we're using pgvector or JSONB storage
        try:
            # Try pgvector approach first (using cosine similarity)
            query = """
            SELECT id, link, title, date, summary, themes, organization
            FROM content_data
            WHERE embedding IS NOT NULL
            ORDER BY embedding <=> %s
            LIMIT %s;
            """
            
            cursor.execute(query, (json.dumps(query_embedding), top_k))
        except Exception as e:
            logger.warning(f"pgvector query failed, falling back to JSONB: {str(e)}")
            # Fall back to manual calculation with JSONB
            query = """
            SELECT id, link, title, date, summary, themes, organization, embedding
            FROM content_data
            WHERE embedding IS NOT NULL
            LIMIT 100;
            """
            
            cursor.execute(query)
            
            # Get all results with embeddings
            results = cursor.fetchall()
            
            # Calculate similarity manually for each result
            results_with_scores = []
            for row in results:
                try:
                    # Parse embedding from JSONB
                    embedding = json.loads(row[7]) if row[7] else []
                    
                    if embedding:
                        # Calculate cosine similarity
                        similarity = cosine_similarity(query_embedding, embedding)
                        
                        # Create result with similarity score
                        result = {
                            "id": row[0],
                            "link": row[1], 
                            "title": row[2],
                            "date": row[3],
                            "summary": row[4],
                            "themes": row[5],
                            "organization": row[6],
                            "similarity": similarity
                        }
                        
                        results_with_scores.append(result)
                except Exception as calc_error:
                    logger.error(f"Error calculating similarity: {str(calc_error)}")
            
            # Sort by similarity (highest first) and limit to top_k
            results_with_scores.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            cursor.close()
            conn.close()
            
            return results_with_scores[:top_k]
        
        # Process results from pgvector approach
        column_names = [desc[0] for desc in cursor.description]
        
        results = []
        for row in cursor.fetchall():
            result = dict(zip(column_names, row))
            results.append(result)
        
        cursor.close()
        conn.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        return []

def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
    import math
    
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0
    
    try:
        # Calculate dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        
        # Calculate magnitudes
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        # Calculate cosine similarity
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
            
        return dot_product / (magnitude1 * magnitude2)
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {str(e)}")
        return 0