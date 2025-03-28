import os
import logging
import psycopg2
from psycopg2.extras import execute_values, Json
from typing import List, Dict, Any, Optional, Tuple
import json
from datetime import datetime
import re
from urllib.parse import urlparse

from database.connection import get_db_connection, get_sqlalchemy_engine
from utils.text_processing import format_date, clean_html_entities

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def store_extract_data(extracted_data: List[Dict[str, Any]]) -> List[int]:
    """
    Store extracted data into the database with enhanced error handling and logging.
    
    Args:
        extracted_data: List of dictionaries containing extracted web content
        
    Returns:
        List of database IDs for the stored records
    """
    if not extracted_data:
        logger.warning("No data to store")
        print("Warning: No data to store in database")
        return []
    
    logger.info(f"Storing {len(extracted_data)} results in database")
    print(f"Starting to store {len(extracted_data)} results in database")
    
    # List to store inserted record IDs
    inserted_ids = []
    success_count = 0
    error_count = 0
    
    # Process each item with individual transactions to avoid cascading failures
    for i, item in enumerate(extracted_data):
        conn = None
        cursor = None
        
        try:
            # Extract item data - with more robust extraction
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            date_str = item.get("date")
            content = item.get("content", "")
            summary = item.get("summary", "")
            
            # Enhanced sentiment handling
            sentiment_info = item.get("sentiment_info", {})
            sentiment = sentiment_info.get("overall_sentiment", "Neutral")
            sentiment_score = sentiment_info.get("sentiment_score", 0.0)
            sentiment_confidence = sentiment_info.get("sentiment_confidence", 0.5)
            
            # Validate sentiment values
            sentiment_score = max(-1.0, min(sentiment_score, 1.0))
            sentiment_confidence = max(0.0, min(sentiment_confidence, 1.0))
            
            # Embedding handling with explicit fallback
            embedding_vector = item.get("embedding", [])
            embedding_model = "text-embedding-ada-002" if embedding_vector else None
            
            # Get database connection
            conn = get_db_connection()
            cursor = conn.cursor()
            
            # Insert into content_sources with more robust handling
            source_query = """
            INSERT INTO content_sources 
            (url, domain_name, title, publication_date, source_type, language, full_content, content_summary)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            RETURNING id;
            """
            
            # Parse domain safely
            domain_name = urlparse(url).netloc if url else ""
            
            # Format date safely
            date_value = format_date(date_str) if date_str else None
            
            # Execute source insertion
            cursor.execute(
                source_query, 
                (
                    url, 
                    domain_name, 
                    title, 
                    date_value, 
                    item.get("source_type", "web"), 
                    item.get("language", "English"), 
                    content, 
                    summary
                )
            )
            
            # Get the ID of the inserted record
            source_id = cursor.fetchone()[0]
            inserted_ids.append(source_id)
            
            # Sentiment Analysis Insertion with Explicit Error Handling
            try:
                sentiment_query = """
                INSERT INTO sentiment_analysis
                (source_id, overall_sentiment, sentiment_score, sentiment_confidence)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (source_id) DO UPDATE 
                SET overall_sentiment = EXCLUDED.overall_sentiment,
                    sentiment_score = EXCLUDED.sentiment_score,
                    sentiment_confidence = EXCLUDED.sentiment_confidence;
                """
                
                cursor.execute(
                    sentiment_query,
                    (
                        source_id,
                        sentiment or 'Neutral',
                        sentiment_score,
                        sentiment_confidence
                    )
                )
            except Exception as sentiment_error:
                logger.warning(f"Failed to insert sentiment: {sentiment_error}")
                print(f"Warning: Sentiment insertion failed for {title}")
            
            # Embedding Insertion with Robust Fallback
            try:
                if embedding_vector:
                    # First attempt with PGVector
                    embedding_query = """
                    INSERT INTO content_embeddings
                    (source_id, embedding_vector, embedding_model)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_id) DO UPDATE
                    SET embedding_vector = EXCLUDED.embedding_vector,
                        embedding_model = EXCLUDED.embedding_model;
                    """
                    
                    cursor.execute(
                        embedding_query,
                        (
                            source_id,
                            embedding_vector,
                            embedding_model
                        )
                    )
                    print(f"Added vector embedding for {title}")
            except Exception as embedding_error:
                try:
                    # Fallback to JSON storage
                    json_embedding_query = """
                    INSERT INTO content_embeddings
                    (source_id, embedding_json, embedding_model)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_id) DO UPDATE
                    SET embedding_json = EXCLUDED.embedding_json,
                        embedding_model = EXCLUDED.embedding_model;
                    """
                    
                    cursor.execute(
                        json_embedding_query,
                        (
                            source_id,
                            json.dumps(embedding_vector),
                            embedding_model
                        )
                    )
                    print(f"Added JSON embedding for {title}")
                except Exception as json_error:
                    logger.warning(f"Failed to store embedding: {json_error}")
                    print(f"Could not store embedding for {title}")
            
            # Commit transaction
            conn.commit()
            success_count += 1
            logger.info(f"Successfully stored item {i+1}/{len(extracted_data)} with ID {source_id}")
            print(f"✓ Successfully stored item {i+1} with ID {source_id}")
        
        except Exception as e:
            error_msg = f"Error storing item {i+1}: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            
            if conn:
                conn.rollback()
            
            error_count += 1
        
        finally:
            # Ensure connections are closed
            if cursor:
                cursor.close()
            if conn:
                conn.close()
    
    # Summary logging
    total_processed = success_count + error_count
    success_rate = (success_count / total_processed * 100) if total_processed > 0 else 0
    
    logger.info(f"Data storage summary: {success_count}/{total_processed} successful ({success_rate:.1f}%)")
    print("="*50)
    print(f"DATA STORAGE SUMMARY")
    print(f"Total items processed: {total_processed}")
    print(f"Successfully stored: {success_count} records ({success_rate:.1f}%)")
    print(f"Failed to store: {error_count} records")
    print("="*50)
    
    return inserted_ids
def fetch_comprehensive_data(source_id: int) -> Dict[str, Any]:
    """
    Fetch comprehensive data for a single source ID, including all related tables.
    
    Args:
        source_id: The ID of the content source
        
    Returns:
        Dictionary with all data related to this source
    """
    logger.info(f"Fetching comprehensive data for source ID: {source_id}")
    
    result = {}
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get main content source data
        source_query = """
        SELECT 
            id, url, domain_name, title, publication_date, source_type, 
            language, full_content, content_summary, extracted_at, last_updated_at
        FROM content_sources
        WHERE id = %s;
        """
        
        cursor.execute(source_query, (source_id,))
        source_data = cursor.fetchone()
        
        if not source_data:
            logger.warning(f"No source found with ID: {source_id}")
            return {"error": "Source not found"}
        
        # Process main source data
        column_names = ["id", "url", "domain_name", "title", "publication_date", 
                       "source_type", "language", "full_content", "content_summary", 
                       "extracted_at", "last_updated_at"]
        
        result = dict(zip(column_names, source_data))
        
        # Format dates
        for date_field in ["publication_date", "extracted_at", "last_updated_at"]:
            if date_field in result and result[date_field]:
                result[date_field] = result[date_field].isoformat()
        
        # Get sentiment data
        sentiment_query = """
        SELECT overall_sentiment, sentiment_score, sentiment_confidence
        FROM sentiment_analysis
        WHERE source_id = %s;
        """
        
        cursor.execute(sentiment_query, (source_id,))
        sentiment_data = cursor.fetchone()
        
        if sentiment_data:
            result["sentiment"] = {
                "overall_sentiment": sentiment_data[0],
                "score": sentiment_data[1],
                "confidence": sentiment_data[2]
            }
        
        # Get themes
        themes_query = """
        SELECT theme
        FROM thematic_areas
        WHERE source_id = %s;
        """
        
        cursor.execute(themes_query, (source_id,))
        themes = [row[0] for row in cursor.fetchall()]
        
        if themes:
            result["themes"] = themes
        
        # Get ABS mentions
        mentions_query = """
        SELECT id, name_variant, mention_context, mention_type, relevance_score, mention_position
        FROM abs_mentions
        WHERE source_id = %s
        ORDER BY mention_position;
        """
        
        cursor.execute(mentions_query, (source_id,))
        mentions_data = cursor.fetchall()
        
        if mentions_data:
            mentions = []
            mention_columns = ["id", "name_variant", "mention_context", "mention_type", 
                              "relevance_score", "mention_position"]
            
            for row in mentions_data:
                mention = dict(zip(mention_columns, row))
                mentions.append(mention)
            
            result["abs_mentions"] = mentions
        
        # Get geographic focus
        geo_query = """
        SELECT id, country, region, scope
        FROM geographic_focus
        WHERE source_id = %s;
        """
        
        cursor.execute(geo_query, (source_id,))
        geo_data = cursor.fetchall()
        
        if geo_data:
            geo_focus = []
            geo_columns = ["id", "country", "region", "scope"]
            
            for row in geo_data:
                location = dict(zip(geo_columns, row))
                geo_focus.append(location)
            
            result["geographic_focus"] = geo_focus
        
        # Get project details
        project_query = """
        SELECT id, project_name, project_type, start_date, end_date, status, description
        FROM project_details
        WHERE source_id = %s;
        """
        
        cursor.execute(project_query, (source_id,))
        project_data = cursor.fetchall()
        
        if project_data:
            projects = []
            project_columns = ["id", "project_name", "project_type", "start_date", 
                              "end_date", "status", "description"]
            
            for row in project_data:
                project = dict(zip(project_columns, row))
                
                # Format dates
                for date_field in ["start_date", "end_date"]:
                    if date_field in project and project[date_field]:
                        project[date_field] = project[date_field].isoformat()
                
                projects.append(project)
            
            result["projects"] = projects
        
        # Get organizations
        org_query = """
        SELECT o.id, o.name, o.organization_type, o.website, om.relationship_type, om.description
        FROM organizations o
        JOIN organization_mentions om ON o.id = om.organization_id
        WHERE om.source_id = %s;
        """
        
        cursor.execute(org_query, (source_id,))
        org_data = cursor.fetchall()
        
        if org_data:
            organizations = []
            org_columns = ["id", "name", "organization_type", "website", "relationship_type", "description"]
            
            for row in org_data:
                org = dict(zip(org_columns, row))
                organizations.append(org)
            
            result["organizations"] = organizations
        
        # Get resources
        resource_query = """
        SELECT id, resource_type, resource_name, resource_url, description
        FROM resources
        WHERE source_id = %s;
        """
        
        cursor.execute(resource_query, (source_id,))
        resource_data = cursor.fetchall()
        
        if resource_data:
            resources = []
            resource_columns = ["id", "resource_type", "resource_name", "resource_url", "description"]
            
            for row in resource_data:
                resource = dict(zip(resource_columns, row))
                resources.append(resource)
            
            result["resources"] = resources
        
        # Get target audiences
        audience_query = """
        SELECT audience_type
        FROM target_audiences
        WHERE source_id = %s;
        """
        
        cursor.execute(audience_query, (source_id,))
        audiences = [row[0] for row in cursor.fetchall()]
        
        if audiences:
            result["target_audiences"] = audiences
        
        # Close cursor and connection
        cursor.close()
        conn.close()
        
        return result
        
    except Exception as e:
        logger.error(f"Error fetching comprehensive data: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        return {"error": str(e)}

def search_abs_mentions(search_term: str) -> List[Dict[str, Any]]:
    """
    Search for specific ABS Initiative name variants.
    
    Args:
        search_term: Term to search for in name variants
        
    Returns:
        List of dictionaries with mentions and their context
    """
    logger.info(f"Searching for ABS mentions with term: {search_term}")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT 
            cs.id AS source_id,
            cs.url,
            cs.title,
            cs.publication_date,
            am.name_variant,
            am.relevance_score,
            am.mention_context
        FROM content_sources cs
        JOIN abs_mentions am ON cs.id = am.source_id
        WHERE 
            am.name_variant ILIKE %s
        ORDER BY am.relevance_score DESC;
        """
        
        cursor.execute(query, (f"%{search_term}%",))
        
        results = []
        for row in cursor.fetchall():
            result = {
                "source_id": row[0],
                "url": row[1],
                "title": row[2],
                "date": row[3].isoformat() if row[3] else None,
                "name_variant": row[4],
                "relevance_score": row[5],
                "context": row[6]
            }
            results.append(result)
        
        cursor.close()
        conn.close()
        
        return results
        
    except Exception as e:
        logger.error(f"Error searching ABS mentions: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        return []

def get_abs_name_variants() -> List[str]:
    """
    Get all unique ABS Initiative name variants from the database.
    
    Returns:
        List of unique name variants
    """
    logger.info("Fetching all ABS name variants")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        query = """
        SELECT DISTINCT name_variant
        FROM abs_mentions
        WHERE name_variant IS NOT NULL AND name_variant != ''
        ORDER BY name_variant;
        """
        
        cursor.execute(query)
        variants = [row[0] for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return variants
        
    except Exception as e:
        logger.error(f"Error fetching ABS name variants: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        return []

def get_geographic_statistics() -> Dict[str, Any]:
    """
    Get statistics about geographic distribution of mentions.
    
    Returns:
        Dictionary with country and region statistics
    """
    logger.info("Fetching geographic statistics")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Get country statistics
        country_query = """
        SELECT country, COUNT(*) as count
        FROM geographic_focus
        WHERE country IS NOT NULL AND country != ''
        GROUP BY country
        ORDER BY count DESC;
        """
        
        cursor.execute(country_query)
        country_stats = [{"country": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # Get region statistics
        region_query = """
        SELECT region, COUNT(*) as count
        FROM geographic_focus
        WHERE region IS NOT NULL AND region != ''
        GROUP BY region
        ORDER BY count DESC;
        """
        
        cursor.execute(region_query)
        region_stats = [{"region": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        # Get scope statistics
        scope_query = """
        SELECT scope, COUNT(*) as count
        FROM geographic_focus
        WHERE scope IS NOT NULL AND scope != ''
        GROUP BY scope
        ORDER BY count DESC;
        """
        
        cursor.execute(scope_query)
        scope_stats = [{"scope": row[0], "count": row[1]} for row in cursor.fetchall()]
        
        cursor.close()
        conn.close()
        
        return {
            "countries": country_stats,
            "regions": region_stats,
            "scopes": scope_stats
        }
        
    except Exception as e:
        logger.error(f"Error fetching geographic statistics: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
        return {"countries": [], "regions": [], "scopes": []}

def create_schema() -> bool:
    """
    Create the complete database schema for ABS Initiative data.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Creating database schema")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create content_sources table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS content_sources (
            id SERIAL PRIMARY KEY,
            url VARCHAR(255) NOT NULL UNIQUE,
            domain_name VARCHAR(100),
            title VARCHAR(255),
            publication_date DATE,
            source_type VARCHAR(50),
            language VARCHAR(50),
            full_content TEXT,
            content_summary TEXT,
            extracted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Create abs_mentions table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS abs_mentions (
            id SERIAL PRIMARY KEY,
            source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
            name_variant VARCHAR(255),
            mention_context TEXT,
            mention_type VARCHAR(100),
            relevance_score FLOAT,
            mention_position INTEGER,
            UNIQUE(source_id, mention_position)
        );
        """)
        
        # Create geographic_focus table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS geographic_focus (
            id SERIAL PRIMARY KEY,
            source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
            mention_id INTEGER REFERENCES abs_mentions(id) ON DELETE CASCADE,
            country VARCHAR(100),
            region VARCHAR(100),
            scope VARCHAR(50)
        );
        """)
        
        # Create thematic_areas table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS thematic_areas (
            source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
            theme VARCHAR(100),
            PRIMARY KEY (source_id, theme)
        );
        """)
        
        # Create project_details table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS project_details (
            id SERIAL PRIMARY KEY,
            source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
            project_name VARCHAR(255),
            project_type VARCHAR(100),
            start_date DATE,
            end_date DATE,
            status VARCHAR(50),
            description TEXT
        );
        """)
        
        # Create organizations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS organizations (
            id SERIAL PRIMARY KEY,
            name VARCHAR(255) UNIQUE,
            organization_type VARCHAR(100),
            website VARCHAR(255)
        );
        """)
        
        # Create organization_mentions junction table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS organization_mentions (
            source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
            organization_id INTEGER REFERENCES organizations(id) ON DELETE CASCADE,
            relationship_type VARCHAR(100),
            description TEXT,
            PRIMARY KEY (source_id, organization_id)
        );
        """)
        
        # Create resources table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS resources (
            id SERIAL PRIMARY KEY,
            source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
            resource_type VARCHAR(100),
            resource_name VARCHAR(255),
            resource_url VARCHAR(255),
            description TEXT
        );
        """)
        
        # Create target_audiences table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS target_audiences (
            source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE,
            audience_type VARCHAR(100),
            PRIMARY KEY (source_id, audience_type)
        );
        """)
        
        # Create sentiment_analysis table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE PRIMARY KEY,
            overall_sentiment VARCHAR(50),
            sentiment_score FLOAT,
            sentiment_confidence FLOAT
        );
        """)
        
        # Try to create content_embeddings table with pgvector if available
        try:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_embeddings (
                source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE PRIMARY KEY,
                embedding_vector VECTOR(1536),
                embedding_model VARCHAR(100)
            );
            """)
            logger.info("Created content_embeddings table with pgvector support")
        except Exception as e:
            logger.warning(f"pgvector not available, creating fallback embeddings table: {str(e)}")
            # Create fallback embedding table with JSONB
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS content_embeddings (
                source_id INTEGER REFERENCES content_sources(id) ON DELETE CASCADE PRIMARY KEY,
                embedding_json JSONB,
                embedding_model VARCHAR(100)
            );
            """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_sources_domain ON content_sources(domain_name);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_sources_date ON content_sources(publication_date);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_sources_language ON content_sources(language);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_sources_type ON content_sources(source_type);")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_abs_mentions_variant ON abs_mentions(name_variant);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_abs_mentions_type ON abs_mentions(mention_type);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_abs_mentions_relevance ON abs_mentions(relevance_score);")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_geographic_focus_country ON geographic_focus(country);")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_geographic_focus_region ON geographic_focus(region);")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_organization_mentions_relation ON organization_mentions(relationship_type);")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_resources_type ON resources(resource_type);")
        
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_sentiment_overall ON sentiment_analysis(overall_sentiment);")
        
        # Create trigger for updated_at
        cursor.execute("""
        CREATE OR REPLACE FUNCTION update_modified_column() RETURNS TRIGGER AS $
        BEGIN
            NEW.last_updated_at = now();
            RETURN NEW;
        END;
        $ LANGUAGE 'plpgsql';
        """)
        
        cursor.execute("""
        DROP TRIGGER IF EXISTS update_content_timestamp ON content_sources;
        CREATE TRIGGER update_content_timestamp
        BEFORE UPDATE ON content_sources
        FOR EACH ROW
        EXECUTE FUNCTION update_modified_column();
        """)
        
        # Create view for comprehensive content analysis
        cursor.execute("""
        CREATE OR REPLACE VIEW v_abs_content_analysis AS
        SELECT 
            cs.id AS source_id,
            cs.url,
            cs.title,
            cs.publication_date,
            cs.language,
            cs.source_type,
            ARRAY_AGG(DISTINCT am.name_variant) AS name_variants,
            ARRAY_AGG(DISTINCT ta.theme) AS themes,
            ARRAY_AGG(DISTINCT gf.country) AS countries,
            ARRAY_AGG(DISTINCT gf.region) AS regions,
            ARRAY_AGG(DISTINCT o.name) AS organizations,
            ARRAY_AGG(DISTINCT r.resource_name) AS resources,
            ARRAY_AGG(DISTINCT aud.audience_type) AS target_audiences,
            sa.overall_sentiment,
            MAX(am.relevance_score) AS max_relevance_score,
            cs.content_summary
        FROM 
            content_sources cs
        LEFT JOIN abs_mentions am ON cs.id = am.source_id
        LEFT JOIN thematic_areas ta ON cs.id = ta.source_id
        LEFT JOIN geographic_focus gf ON cs.id = gf.source_id
        LEFT JOIN organization_mentions om ON cs.id = om.source_id
        LEFT JOIN organizations o ON om.organization_id = o.id
        LEFT JOIN resources r ON cs.id = r.source_id
        LEFT JOIN target_audiences aud ON cs.id = aud.source_id
        LEFT JOIN sentiment_analysis sa ON cs.id = sa.source_id
        GROUP BY 
            cs.id, cs.url, cs.title, cs.publication_date, cs.language, 
            cs.source_type, sa.overall_sentiment, cs.content_summary;
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

def fetch_data(limit=100, filters=None):
    """
    Fetch data from the database with optional filtering.
    
    Args:
        limit: Maximum number of records to retrieve
        filters: Dictionary of filter conditions
        
    Returns:
        DataFrame containing content data
    """
    logger.info(f"Fetching data from database (limit: {limit}, filters: {filters})")
    
    try:
        import pandas as pd
        engine = get_sqlalchemy_engine()
        
        # Use the view for comprehensive data
        query_parts = ["SELECT * FROM v_abs_content_analysis"]
        params = {}
        
        # Add filter conditions if provided
        where_clauses = []
        if filters:
            if filters.get('theme'):
                where_clauses.append("themes @> ARRAY[%(theme)s]")
                params['theme'] = filters['theme']
            
            if filters.get('organization'):
                where_clauses.append("organizations @> ARRAY[%(organization)s]")
                params['organization'] = filters['organization']
            
            if filters.get('sentiment'):
                where_clauses.append("overall_sentiment = %(sentiment)s")
                params['sentiment'] = filters['sentiment']
            
            if filters.get('country'):
                where_clauses.append("countries @> ARRAY[%(country)s]")
                params['country'] = filters['country']
            
            if filters.get('region'):
                where_clauses.append("regions @> ARRAY[%(region)s]")
                params['region'] = filters['region']
            
            if filters.get('start_date') and filters.get('end_date'):
                where_clauses.append("publication_date BETWEEN %(start_date)s AND %(end_date)s")
                params['start_date'] = filters['start_date']
                params['end_date'] = filters['end_date']
        
        # Construct WHERE clause
        if where_clauses:
            query_parts.append("WHERE " + " AND ".join(where_clauses))
        
        # Add order and limit
        query_parts.append("ORDER BY max_relevance_score DESC, publication_date DESC")
        query_parts.append(f"LIMIT {limit}")
        
        # Combine query parts
        query = " ".join(query_parts)
        
        # Execute query
        df = pd.read_sql(query, engine, params=params)
        
        logger.info(f"Fetched {len(df)} rows from database")
        return df
    
    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        # Return empty DataFrame in case of error
        import pandas as pd
        return pd.DataFrame()

def semantic_search(query_text, top_k=5):
    """
    Perform semantic search using embeddings.
    
    Args:
        query_text: Text to search for
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries containing search results
    """
    # Get OpenAI client for embedding generation
    from extraction.analysis import get_openai_client
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
            SELECT 
                cs.id,
                cs.url, 
                cs.title,
                cs.publication_date,
                cs.content_summary,
                sa.overall_sentiment,
                1 - (embedding_vector <=> %s) AS similarity
            FROM content_sources cs
            JOIN content_embeddings ce ON cs.id = ce.source_id
            LEFT JOIN sentiment_analysis sa ON cs.id = sa.source_id
            ORDER BY embedding_vector <=> %s
            LIMIT %s;
            """
            
            cursor.execute(query, (query_embedding, query_embedding, top_k))
            
            # Process results
            results = []
            for row in cursor.fetchall():
                result = {
                    "id": row[0],
                    "url": row[1],
                    "title": row[2],
                    "date": row[3].isoformat() if row[3] else None,
                    "summary": row[4],
                    "sentiment": row[5],
                    "similarity": row[6]
                }
                results.append(result)
            
            cursor.close()
            conn.close()
            
            return results
            
        except Exception as e:
            logger.warning(f"pgvector query failed, falling back to JSONB: {str(e)}")
            # Fall back to manual calculation with JSONB
            query = """
            SELECT 
                cs.id,
                cs.url, 
                cs.title,
                cs.publication_date,
                cs.content_summary,
                sa.overall_sentiment,
                ce.embedding_json
            FROM content_sources cs
            JOIN content_embeddings ce ON cs.id = ce.source_id
            LEFT JOIN sentiment_analysis sa ON cs.id = sa.source_id
            LIMIT 100;
            """
            
            cursor.execute(query)
            
            # Get all results with embeddings
            results = []
            for row in cursor.fetchall():
                try:
                    # Parse embedding from JSONB
                    embedding = json.loads(row[6]) if row[6] else []
                    
                    if embedding:
                        # Calculate cosine similarity
                        similarity = cosine_similarity(query_embedding, embedding)
                        
                        # Create result with similarity score
                        result = {
                            "id": row[0],
                            "url": row[1],
                            "title": row[2],
                            "date": row[3].isoformat() if row[3] else None,
                            "summary": row[4],
                            "sentiment": row[5],
                            "similarity": similarity
                        }
                        
                        results.append(result)
                except Exception as calc_error:
                    logger.error(f"Error calculating similarity: {str(calc_error)}")
            
            # Sort by similarity (highest first) and limit to top_k
            results.sort(key=lambda x: x.get("similarity", 0), reverse=True)
            
            cursor.close()
            conn.close()
            
            return results[:top_k]
            
    except Exception as e:
        logger.error(f"Error in semantic search: {str(e)}")
        if 'conn' in locals() and conn:
            conn.close()
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