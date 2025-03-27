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
    try:
        from openai import OpenAI
        import os
        from dotenv import load_dotenv
        
        # Load environment variables
        load_dotenv()
        
        # Get API key
        openai_api_key = os.getenv("OPENAI_API_KEY")
        if not openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment variables")
            return None
        
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
    Extract potential benefits using OpenAI with prompt from benefits.txt file.
    Only processes high-quality content.
    
    Args:
        content: Text content to analyze
        
    Returns:
        Extracted benefits text or None
    """
    # Skip if no mention of relevant terms
    relevant_terms = ["germany", "german", "abs", "capacity", "bio-innovation", "africa"]
    if not any(term in content.lower() for term in relevant_terms):
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
            # Extract sections most relevant to benefits
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
            
            # Find sentences that mention both initiatives and potential benefits
            benefit_sentences = []
            for sentence in sentences:
                sentence_lower = sentence.lower()
                if any(term in sentence_lower for term in ["abs", "capacity development", "bio-innovation", "africa"]):
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
                # Fallback prompt
                system_prompt = "Extract specific benefits mentioned in the text related to the ABS Capacity Development Initiative or Bio-innovation Africa initiatives. Focus on concrete, factual examples rather than general claims. If no specific benefits are mentioned, indicate this clearly."
            
            logger.info("Extracting benefits using OpenAI")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Extract any specific benefits from this text, focusing on concrete advantages, opportunities, or gains related to the ABS Capacity Development Initiative or Bio-innovation Africa: {content_to_analyze}"}
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

def identify_initiative(content: str) -> Tuple[str, float]:
    """
    Identify which initiative is mentioned in the content and calculate confidence.
    
    Args:
        content: Content text to analyze
        
    Returns:
        Tuple of (initiative_key, confidence_score)
    """
    content_lower = content.lower()
    
    # Define initiatives with their names and keywords
    initiatives = {
        "abs_cdi": {
            "names": [
                "ABS Capacity Development Initiative",
                "ABS CDI",
                "ABS Capacity Development Initiative for Africa",
                "Capacity Development Initiative",
                "Initiative pour le renforcement des capacités en matière d'APA",
                "Accès et Partage des Avantages",
                "Initiative sur le développement des capacités pour l'APA",
                "Initiative de renforcement des capacités sur l'APA",
                "Initiative APA"
            ],
            "keywords": [
                "nagoya protocol", "access and benefit sharing", "genetic resources", 
                "traditional knowledge", "convention on biological diversity", "cbd",
                "fair and equitable sharing", "bioprospecting", "ABS", "APA",
                "capacity development", "benefit sharing"
            ]
        },
        "bio_innovation_africa": {
            "names": [
                "Bio-innovation Africa",
                "BioInnovation Africa"
            ],
            "keywords": [
                "biotrade", "biodiversity", "sustainable use", "value chains",
                "bio-economy", "biological resources", "green business", 
                "natural ingredients", "africa innovation", "sustainable sourcing"
            ]
        }
    }
    
    # Track mentions of each initiative
    initiative_scores = {}
    
    for initiative_key, initiative_data in initiatives.items():
        # Count name mentions (more important)
        name_count = 0
        for name in initiative_data["names"]:
            name_count += content_lower.count(name.lower())
        
        # Count keyword mentions (less important)
        keyword_count = 0
        for keyword in initiative_data["keywords"]:
            keyword_count += content_lower.count(keyword.lower())
        
        # Calculate score - names are weighted higher than keywords
        score = (name_count * 3) + keyword_count
        
        # Normalize score based on content length
        content_length = max(1, len(content_lower))
        normalized_score = min(1.0, (score * 500) / content_length)
        
        initiative_scores[initiative_key] = normalized_score
    
    # Find initiative with highest score
    if not initiative_scores:
        return "unknown", 0.0
            
    best_initiative = max(initiative_scores.items(), key=lambda x: x[1])
    
    # Only return initiative if score is above threshold
    if best_initiative[1] >= 0.1:
        return best_initiative
    else:
        return "unknown", best_initiative[1]

def identify_themes(self, content: str) -> List[str]:
        """
        Identify diverse themes in content using OpenAI with a data-driven approach.
        Does not include "ABS Initiative" as a default theme.
        
        Args:
            content: Content text to analyze
            
        Returns:
            List of identified themes
        """
        # Skip if content is too short
        if not content or len(content) < 100:
            return ["Content Analysis", "Documentation"]
        
        # Try to use OpenAI for theme extraction
        try:
            from openai import OpenAI
            import os
            from dotenv import load_dotenv
            
            # Load environment variables and get API key
            load_dotenv()
            api_key = os.getenv("OPENAI_API_KEY")
            
            # If API key is available, use OpenAI
            if api_key:
                client = OpenAI(api_key=api_key)
                
                # Prepare content - limit to first 3000 chars to save tokens
                excerpt = content[:3000] + ("..." if len(content) > 3000 else "")
                
                # Create a prompt for theme extraction without suggesting ABS Initiative
                prompt = f"""
    Analyze this text and identify the main substantive themes it discusses. Focus on the actual subject matter.

    Text excerpt:
    {excerpt}

    Extract exactly 5 specific, substantive themes from this content. Do NOT use generic themes like "ABS Initiative" or "Capacity Development" unless they're discussed in detail as topics themselves. Focus on the actual subjects being discussed such as "Biodiversity Conservation", "Traditional Knowledge", "Genetic Resources", etc.

    Return ONLY a comma-separated list of identified themes without explanations or additional text.
    """

                # Make API call
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,  # Slightly higher temperature for more diversity
                    max_tokens=100
                )
                
                # Extract themes from response
                ai_themes_text = response.choices[0].message.content.strip()
                
                # Convert to list and clean up each theme
                ai_themes = [theme.strip() for theme in ai_themes_text.split(',')]
                
                # Remove any empty themes
                ai_themes = [theme for theme in ai_themes if theme]
                
                # Ensure we have at least some themes
                if ai_themes and len(ai_themes) >= 2:
                    return ai_themes
        
        except Exception as e:
            # Log the error but continue to fallback method
            logger.warning(f"Error using OpenAI for theme extraction: {str(e)}. Falling back to simple content analysis.")
        
        # Fallback approach without using "ABS Initiative" as default
        import re
        from collections import Counter
        
        # Define some substantive topics related to biodiversity and conservation
        potential_topics = [
            "Biodiversity", "Conservation", "Sustainable Development", "Genetic Resources",
            "Traditional Knowledge", "Indigenous Rights", "Policy Development", 
            "Legal Framework", "Compliance", "Implementation", "Benefit Sharing",
            "Sustainable Use", "Ecosystem Services", "Stakeholder Engagement",
            "Technology Transfer", "Capacity Building", "International Cooperation",
            "Research", "Innovation", "Monitoring", "Evaluation", "Governance"
        ]
        
        # Check which topics are present in the content
        found_topics = []
        content_lower = content.lower()
        
        for topic in potential_topics:
            if topic.lower() in content_lower:
                found_topics.append(topic)
                # Stop once we have 5 topics
                if len(found_topics) >= 5:
                    break
        
        # If we found specific topics, return them
        if found_topics:
            return found_topics
        
        # Otherwise use a more general approach - extract key terms
        # Extract all words and simple phrases
        text = re.sub(r'[^\w\s]', ' ', content_lower)  # Remove punctuation
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text)  # Find words of 4+ characters
        
        # Count word frequencies
        word_counts = Counter(words)
        
        # Remove common stopwords
        stopwords = {"this", "that", "these", "those", "with", "from", "their", "would", "could", "should", 
                    "about", "which", "there", "where", "when", "what", "have", "will", "they", 
                    "them", "then", "than", "were", "been", "being", "other", "initiative", "development",
                    "capacity", "through", "between", "information", "because", "system", "process"}
        
        # Filter out stop words
        potential_themes = {word: count for word, count in word_counts.items() 
                        if word not in stopwords and count > 1}
        
        # Extract 5 most common potential theme words
        top_words = [word.capitalize() for word, _ in sorted(potential_themes.items(), key=lambda x: x[1], reverse=True)[:5]]
        
        # If we couldn't find good topic words, return generic research categories
        if not top_words:
            return ["Policy Analysis", "Research Findings", "Environmental Studies", "International Relations", "Resource Management"]
        
        return top_words
        

def extract_benefit_examples(content: str, initiative: str) -> List[Dict[str, Any]]:
    """
    Extract examples of benefits from the content using a pattern-based approach.
    
    Args:
        content: Content text to analyze
        initiative: Initiative key (abs_cdi or bio_innovation_africa)
        
    Returns:
        List of extracted benefit examples
    """
    if not content or len(content) < 100:
        return []
            
    content_lower = content.lower()
    
    # Define initiatives with names and keywords
    initiatives = {
        "abs_cdi": {
            "names": [
                "abs capacity development initiative",
                "abs cdi",
                "capacity development initiative for africa",
                "initiative pour le renforcement des capacités en matière d'apa"
            ]
        },
        "bio_innovation_africa": {
            "names": [
                "bio-innovation africa",
                "bioinnovation africa"
            ]
        },
        "unknown": {
            "names": ["abs", "capacity development", "benefit sharing", "nagoya protocol"]
        }
    }
    
    # Get initiative names
    initiative_names = []
    if initiative in initiatives:
        initiative_names = initiatives[initiative]["names"]
    else:
        initiative_names = initiatives["unknown"]["names"]
    
    # Find paragraphs that mention benefits
    paragraphs = content.split('\n\n')
    benefit_paragraphs = []
    
    # Benefit indicator terms
    benefit_terms = [
        "benefit", "advantage", "impact", "result", "outcome", "achievement", 
        "success", "improvement", "contribution", "led to", "resulted in",
        "development of", "establishment of", "creation of", "implementation of"
    ]
    
    # Country and region terms to look for
    countries = [
        "africa", "ghana", "kenya", "south africa", "ethiopia", "cameroon", 
        "benin", "madagascar", "namibia", "senegal", "uganda", "tanzania",
        "nigeria", "morocco", "developing country", "developing countries"
    ]
    
    # Find paragraphs that mention benefits
    for paragraph in paragraphs:
        paragraph_lower = paragraph.lower()
        
        # Check if the paragraph mentions the initiative
        initiative_mentioned = any(name in paragraph_lower for name in initiative_names)
        
        # Check if benefit terms are mentioned
        benefit_mentioned = any(term in paragraph_lower for term in benefit_terms)
        
        # Check if countries or regions are mentioned
        country_mentioned = any(country in paragraph_lower for country in countries)
        
        # If the paragraph meets criteria, include it
        if initiative_mentioned and (benefit_mentioned or country_mentioned):
            # Only include paragraphs of reasonable length to avoid fragments
            if len(paragraph.split()) >= 10:
                benefit_paragraphs.append(paragraph)
    
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
    
    # Extract structured benefit examples
    benefit_examples = []
    
    for paragraph in benefit_paragraphs:
        paragraph_lower = paragraph.lower()
        
        # Determine benefit category
        category = "general"
        max_score = 0
        
        for cat_key, cat_terms in benefit_categories.items():
            score = sum(paragraph_lower.count(term) for term in cat_terms)
            if score > max_score:
                max_score = score
                category = cat_key
                
        # Identify countries mentioned
        mentioned_countries = []
        for country in countries:
            if country in paragraph_lower:
                mentioned_countries.append(country.title())
                    
        # Create benefit example
        benefit_example = {
            "text": paragraph.strip(),
            "category": category,
            "countries": mentioned_countries,
            "initiative": initiative,
            "word_count": len(paragraph.split())
        }
            
        benefit_examples.append(benefit_example)
    
    return benefit_examples

def store_extract_data(extracted_data: List[Dict[str, Any]]) -> List[int]:
    """
    Store extracted data into the database without further processing.
    Just stores the data exactly as provided.
    
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
            
            # Check if embedding column exists
            has_embedding_column = False
            try:
                check_query = """
                SELECT EXISTS (
                    SELECT FROM information_schema.columns 
                    WHERE table_name = 'content_data' AND column_name = 'embedding'
                );
                """
                cursor.execute(check_query)
                has_embedding_column = cursor.fetchone()[0]
                
                # Add embedding column if it doesn't exist - but use JSONB
                if not has_embedding_column:
                    try:
                        # Try with JSONB type first which should work on all PostgreSQL versions
                        alter_query = """
                        ALTER TABLE content_data ADD COLUMN embedding JSONB;
                        """
                        cursor.execute(alter_query)
                        conn.commit()  # Commit this change right away
                        logger.info("Added embedding JSONB column to content_data table")
                        has_embedding_column = True
                    except Exception as e:
                        logger.warning(f"Failed to add embedding column: {str(e)}")
                        conn.rollback()
                        # Continue without the embedding column
            except Exception as e:
                logger.warning(f"Error checking for embedding column: {str(e)}")
                # Continue without checking column
            
            # Convert embedding to JSON string if present
            embedding_json = None
            embedding = item.get("embedding")
            if embedding and has_embedding_column:
                try:
                    embedding_json = json.dumps(embedding)
                except Exception as e:
                    logger.warning(f"Failed to convert embedding to JSON: {str(e)}")
            
            # Construct query based on column availability
            if has_embedding_column:
                query = """
                INSERT INTO content_data 
                (link, title, date, summary, full_content, themes, organization, sentiment, 
                 language, initiative, initiative_key, embedding, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                RETURNING id;
                """
                
                cursor.execute(
                    query, 
                    (link, title, date_value, summary, content, themes, organization, sentiment,
                     language, initiative, initiative_key, embedding_json)
                )
            else:
                # Fallback without embedding column
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


def cosine_similarity(vec1, vec2):
    """
    Calculate cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity score (0-1)
    """
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
        
        # Convert embedding to string format for database
        query_embedding_str = json.dumps(query_embedding)
        
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
            
            cursor.execute(query, (query_embedding_str, top_k))
        except Exception as e:
            logger.warning(f"pgvector query failed, falling back to JSONB: {str(e)}")
            # Fall back to manual calculation with JSONB
            # This is much less efficient but works without pgvector
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
def validate_summary(self, summary, original_content):
    """
    Validate that a summary contains only information present in the original content.
    
    Args:
        summary: Generated summary to validate
        original_content: Original content to check against
        
    Returns:
        Dictionary with validation results
    """
    client = get_openai_client()
    if not client:
        # If we can't validate, assume it's valid but log a warning
        logger.warning("Cannot validate summary: OpenAI client not available")
        return {"valid": True, "issues": []}
    
    try:
        # Create a validation prompt
        validation_prompt = f"""
Your task is to verify if the summary below contains ONLY information that is explicitly stated in the original content.

Summary to validate:
{summary}

Original content:
{original_content}

Check for these issues:
1. Hallucinations - information in the summary not present in the original
2. Misrepresentations - information that distorts what's in the original
3. Omissions of critical context that change meaning

Return a JSON object with the following structure:
{{
  "valid": true/false,
  "issues": ["specific issue 1", "specific issue 2", ...],
  "explanation": "Brief explanation of validation result"
}}
"""

        # Make API call for validation
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": validation_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.2,
            max_tokens=300
        )
        
        # Parse response
        validation_result = json.loads(response.choices[0].message.content)
        logger.info(f"Summary validation result: {validation_result['valid']}")
        
        if not validation_result['valid']:
            issues = validation_result.get('issues', [])
            logger.warning(f"Summary validation issues: {issues}")
        
        return validation_result
        
    except Exception as e:
        logger.error(f"Error validating summary: {str(e)}")
        # If validation fails, assume the summary is valid but log the error
        return {"valid": True, "issues": []}
def extract_organization_from_url(url: str) -> str:
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
        
        # Special case handling for known domains
        domain_map = {
            "abs-initiative.info": "ABS Initiative",
            "cbd.int": "CBD",
            "absch.cbd.int": "ABS Clearing-House",
            "giz.de": "GIZ",
            "bmz.de": "BMZ",
            "kfw.de": "KfW",
            "unctad.org": "UNCTAD",
            "uebt.org": "UEBT",
            "ethicalbiotrade.org": "UEBT"
        }
        
        # Check if domain matches any in the map
        for key, value in domain_map.items():
            if key in domain:
                return value
        
        # Extract main domain (before first dot)
        main_domain = domain.split('.')[0].upper()
        
        return main_domain
    except:
        return "Unknown"

def fetch_data(limit=100, filters=None):
    """
    Fetch data from the database with optional filtering.
    Enhanced to support initiative filtering.
    
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
            
            if filters.get('initiative'):
                query_parts.append("initiative = %s")
                params['initiative'] = filters['initiative']
            
            if filters.get('start_date') and filters.get('end_date'):
                query_parts.append("date BETWEEN %s AND %s")
                params['start_date'] = filters['start_date']
                params['end_date'] = filters['end_date']
        
        # Construct WHERE clause
        where_clause = "WHERE " + " AND ".join(query_parts) if query_parts else ""
        
        # Check if initiative columns exist
        try:
            cursor = conn.cursor()
            check_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'content_data' AND column_name = 'initiative'
            );
            """
            cursor.execute(check_query)
            has_initiative_columns = cursor.fetchone()[0]
            cursor.close()
        except:
            # If check fails, assume no initiative columns
            has_initiative_columns = False
        
        # Full query with appropriate columns
        if has_initiative_columns:
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
        else:
            # Fall back to original schema
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
        
        # Check if initiative columns exist
        try:
            check_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'content_data' AND column_name = 'initiative'
            );
            """
            cursor.execute(check_query)
            has_initiative_columns = cursor.fetchone()[0]
        except:
            # If check fails, assume no initiative columns
            has_initiative_columns = False
        
        # Prepare query based on available columns
        if has_initiative_columns:
            query = """
            SELECT id, link, title, date, summary, themes, organization, sentiment, initiative, benefits_to_germany
            FROM content_data
            ORDER BY id DESC
            LIMIT %s OFFSET %s;
            """
        else:
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
        
        # Check if initiative columns exist
        try:
            check_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'content_data' AND column_name = 'initiative'
            );
            """
            cursor.execute(check_query)
            has_initiative_columns = cursor.fetchone()[0]
        except:
            # If check fails, assume no initiative columns
            has_initiative_columns = False
        
        # Prepare query based on available columns
        if has_initiative_columns:
            query = """
            SELECT id, link, title, date, summary, full_content, themes, organization,
                   sentiment, initiative, initiative_key, benefits_to_germany, 
                   benefit_categories, benefit_examples, created_at, updated_at
            FROM content_data
            WHERE id = %s;
            """
        else:
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
    Enhanced to include initiative data.
    
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
        
        # Check if initiative columns exist
        try:
            check_query = """
            SELECT EXISTS (
                SELECT FROM information_schema.columns 
                WHERE table_name = 'content_data' AND column_name = 'initiative'
            );
            """
            cursor.execute(check_query)
            has_initiative_columns = cursor.fetchone()[0]
        except:
            # If check fails, assume no initiative columns
            has_initiative_columns = False
        
        # Create a tsquery from the search terms
        if has_initiative_columns:
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
        else:
            query = """
            SELECT id, link, title, date, summary, themes, organization, sentiment, benefits_to_germany
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

def create_schema():
    """
    Create the database schema if it doesn't exist.
    Updated to include new initiative-specific fields.
    
    Returns:
        True if successful, False otherwise
    """
    logger.info("Creating database schema")
    
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Create content_data table if it doesn't exist (with original schema)
        # Add language column to schema creation
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
            language VARCHAR(50) DEFAULT 'English',
            initiative VARCHAR(100),
            initiative_key VARCHAR(50),
            benefit_categories JSONB,
            benefit_examples JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Add index for language column
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_language ON content_data (language)")
        
        # Add new columns for initiatives if they don't exist
        cursor.execute("""
        DO $$
        BEGIN
            -- Check if initiative column exists
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_name='content_data' AND column_name='initiative') THEN
                ALTER TABLE content_data ADD COLUMN initiative VARCHAR(100);
            END IF;

            -- Check if initiative_key column exists
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_name='content_data' AND column_name='initiative_key') THEN
                ALTER TABLE content_data ADD COLUMN initiative_key VARCHAR(50);
            END IF;

            -- Check if benefit_categories column exists
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_name='content_data' AND column_name='benefit_categories') THEN
                ALTER TABLE content_data ADD COLUMN benefit_categories JSONB;
            END IF;

            -- Check if benefit_examples column exists
            IF NOT EXISTS (SELECT 1 FROM information_schema.columns 
                          WHERE table_name='content_data' AND column_name='benefit_examples') THEN
                ALTER TABLE content_data ADD COLUMN benefit_examples JSONB;
            END IF;
        END
        $$;
        """)
        
        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_link ON content_data (link)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_themes ON content_data USING GIN (themes)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_organization ON content_data (organization)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_date ON content_data (date)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_sentiment ON content_data (sentiment)")
        
        # Create indexes for new columns
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_initiative ON content_data (initiative)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_content_data_initiative_key ON content_data (initiative_key)")
        
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