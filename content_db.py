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
            # Extract sections most relevant to initiatives
            relevant_paragraphs = []
            paragraphs = content.split('\n\n')
            for para in paragraphs:
                para_lower = para.lower()
                if any(term in para_lower for term in ["abs", "capacity development", "bio-innovation", "africa", "nagoya", "benefit sharing"]):
                    relevant_paragraphs.append(para)
            
            # Use either initiative-focused paragraphs or first part of content
            if relevant_paragraphs and len(' '.join(relevant_paragraphs)) >= 300:
                content_to_summarize = ' '.join(relevant_paragraphs[:3])  # Top 3 most relevant paragraphs
                logger.info(f"Using {len(relevant_paragraphs)} initiative-specific paragraphs for summary")
            else:
                # Use only first 3000 chars to save on token costs
                content_to_summarize = content[:3000] + ("..." if len(content) > 3000 else "")
            
            logger.info("Generating summary using OpenAI")
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": f"Summarize this content, focusing particularly on ABS Capacity Development Initiative, Bio-innovation Africa, or other initiatives mentioned: {content_to_summarize}"}
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
    Store extracted data into the database using a batch transaction.
    Enhanced to handle initiative-specific data and language information.
    
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
                language = item.get("language", "English")  # Default to English if not specified
                
                # Skip items with empty/invalid URLs
                if not link or len(link) < 5:
                    logger.warning(f"Skipping item {i+1} with invalid URL: {link}")
                    error_count += 1
                    continue
                    
                # Use existing summary or generate one
                summary = item.get("summary", snippet)
                if not summary and content:
                    # Simple summary extraction (first 500 chars)
                    summary = content[:500] + "..." if len(content) > 500 else content
                
                # Get themes from item or identify them
                themes = item.get("themes", [])
                if not themes and content:
                    themes = identify_themes(content)
                
                # Get organization
                organization = item.get("organization", extract_organization_from_url(link))
                
                # Get sentiment (either from item or analyze it)
                sentiment = item.get("sentiment", "Neutral")
                if not sentiment and content:
                    sentiment = analyze_sentiment(content)
                
                # Get initiative information or identify it
                initiative = item.get("initiative", "Unknown Initiative")
                initiative_key = item.get("initiative_key", "unknown")
                
                if initiative == "Unknown Initiative" and content:
                    identified_initiative, score = identify_initiative(content)
                    if identified_initiative != "unknown" and score >= 0.1:
                        initiative_key = identified_initiative
                        initiative_display_names = {
                            "abs_cdi": "ABS Capacity Development Initiative",
                            "bio_innovation_africa": "Bio-innovation Africa"
                        }
                        initiative = initiative_display_names.get(identified_initiative, "Unknown Initiative")
                
                # Extract benefits if not already provided
                benefits_to_germany = item.get("benefits_to_germany")
                if not benefits_to_germany and content:
                    benefits_to_germany = extract_benefits(content)
                
                # Extract benefit examples if not already provided
                benefit_examples = item.get("benefit_examples", [])
                if not benefit_examples and content and initiative_key != "unknown":
                    benefit_examples = extract_benefit_examples(content, initiative_key)
                
                # Format benefit categories as JSON if present
                benefit_categories_json = None
                if item.get("benefit_categories"):
                    if isinstance(item["benefit_categories"], dict):
                        benefit_categories_json = json.dumps(item["benefit_categories"])
                    elif isinstance(item["benefit_categories"], str):
                        # Already a JSON string
                        benefit_categories_json = item["benefit_categories"]
                
                # Format benefit examples as JSON if present
                benefit_examples_json = None
                if benefit_examples:
                    benefit_examples_json = json.dumps(benefit_examples)
                
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
                    "language": language,  # Include language field
                    "initiative": initiative,
                    "initiative_key": initiative_key,
                    "benefits_to_germany": benefits_to_germany,
                    "benefit_categories": benefit_categories_json,
                    "benefit_examples": benefit_examples_json
                })
                
            except Exception as prep_error:
                error_msg = f"Error preparing item {i+1}: {str(prep_error)}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                error_count += 1
        
        # Now insert all valid items in a single transaction
        for i, item in enumerate(valid_items):
            try:
                # Check if all required fields needed for the original insert are present
                required_fields = {
                    "link": item["link"],
                    "title": item["title"],
                    "date_value": item["date_value"],
                    "summary": item["summary"],
                    "content": item["content"],
                    "themes": item["themes"],
                    "organization": item["organization"],
                    "sentiment": item["sentiment"],
                    "language": item["language"],
                    "benefits_to_germany": item["benefits_to_germany"]
                }
                
                # Check if table has the language column
                try:
                    # Check if table has language column
                    check_query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'content_data' AND column_name = 'language'
                    );
                    """
                    cursor.execute(check_query)
                    has_language_column = cursor.fetchone()[0]
                    
                    # If language column doesn't exist, add it
                    if not has_language_column:
                        alter_query = """
                        ALTER TABLE content_data ADD COLUMN language VARCHAR(50) DEFAULT 'English';
                        CREATE INDEX IF NOT EXISTS idx_content_data_language ON content_data (language);
                        """
                        cursor.execute(alter_query)
                        logger.info("Added language column to content_data table")
                        print("Added language column to database schema")
                except Exception as column_error:
                    logger.warning(f"Error checking language column: {str(column_error)}")
                
                # Construct the SQL query dynamically based on existing columns
                try:
                    # Check if table has initiative columns
                    check_query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'content_data' AND column_name = 'initiative'
                    );
                    """
                    cursor.execute(check_query)
                    has_initiative_columns = cursor.fetchone()[0]
                    
                    # Check for language column (should be added by now)
                    check_query = """
                    SELECT EXISTS (
                        SELECT FROM information_schema.columns 
                        WHERE table_name = 'content_data' AND column_name = 'language'
                    );
                    """
                    cursor.execute(check_query)
                    has_language_column = cursor.fetchone()[0]
                    
                    if has_initiative_columns and has_language_column:
                        # Use the enhanced schema with initiative and language columns
                        query = """
                        INSERT INTO content_data 
                        (link, title, date, summary, full_content, information, themes, organization, sentiment, 
                         language, initiative, initiative_key, benefits_to_germany, benefit_categories, benefit_examples,
                         insights, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id;
                        """
                        
                        cursor.execute(
                            query, 
                            (item["link"], item["title"], item["date_value"], item["summary"], 
                             item["content"], item["summary"], item["themes"], item["organization"], item["sentiment"],
                             item["language"], item["initiative"], item["initiative_key"], item["benefits_to_germany"],
                             item["benefit_categories"], item["benefit_examples"], None)
                        )
                    elif has_initiative_columns:
                        # Use schema with initiative but without language
                        query = """
                        INSERT INTO content_data 
                        (link, title, date, summary, full_content, information, themes, organization, sentiment, 
                         initiative, initiative_key, benefits_to_germany, benefit_categories, benefit_examples,
                         insights, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id;
                        """
                        
                        cursor.execute(
                            query, 
                            (item["link"], item["title"], item["date_value"], item["summary"], 
                             item["content"], item["summary"], item["themes"], item["organization"], item["sentiment"],
                             item["initiative"], item["initiative_key"], item["benefits_to_germany"],
                             item["benefit_categories"], item["benefit_examples"], None)
                        )
                    elif has_language_column:
                        # Use schema with language but without initiative
                        query = """
                        INSERT INTO content_data 
                        (link, title, date, summary, full_content, information, themes, organization, sentiment, 
                         language, benefits_to_germany, insights, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id;
                        """
                        
                        cursor.execute(
                            query, 
                            (item["link"], item["title"], item["date_value"], item["summary"], 
                             item["content"], item["summary"], item["themes"], item["organization"], item["sentiment"],
                             item["language"], item["benefits_to_germany"], None)
                        )
                    else:
                        # Use the original schema without initiative or language
                        query = """
                        INSERT INTO content_data 
                        (link, title, date, summary, full_content, information, themes, organization, sentiment, 
                         benefits_to_germany, insights, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id;
                        """
                        
                        cursor.execute(
                            query, 
                            (item["link"], item["title"], item["date_value"], item["summary"], 
                             item["content"], item["summary"], item["themes"], item["organization"], item["sentiment"],
                             item["benefits_to_germany"], None)
                        )
                except Exception as column_error:
                    logger.warning(f"Error checking columns, falling back to original schema: {str(column_error)}")
                    # Fall back to original schema if column check fails
                    query = """
                    INSERT INTO content_data 
                    (link, title, date, summary, full_content, information, themes, organization, sentiment, 
                     benefits_to_germany, insights, created_at, updated_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                    RETURNING id;
                    """
                    
                    cursor.execute(
                        query, 
                        (item["link"], item["title"], item["date_value"], item["summary"], 
                         item["content"], item["summary"], item["themes"], item["organization"], item["sentiment"],
                         item["benefits_to_germany"], None)
                    )
                
                # Get the ID of the inserted record
                record_id = cursor.fetchone()[0]
                inserted_ids.append(record_id)
                success_count += 1
                
                logger.info(f"Inserted record with ID {record_id} for URL: {item['link']} (Language: {item['language']})")
                print(f"SUCCESS: Inserted record ID {record_id} | {item['title'][:50]} | {item['language']}")
                
            except Exception as item_error:
                error_msg = f"Error storing item with URL {item.get('link', 'unknown')}: {str(item_error)}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                
                # Individual insert failures don't abort the whole transaction
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
                    
                    # Check if table has all required columns
                    has_language_column = False
                    has_initiative_columns = False
                    
                    try:
                        # Check if table has language column
                        check_query = """
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_name = 'content_data' AND column_name = 'language'
                        );
                        """
                        item_cursor.execute(check_query)
                        has_language_column = item_cursor.fetchone()[0]
                        
                        # Check if table has initiative columns
                        check_query = """
                        SELECT EXISTS (
                            SELECT FROM information_schema.columns 
                            WHERE table_name = 'content_data' AND column_name = 'initiative'
                        );
                        """
                        item_cursor.execute(check_query)
                        has_initiative_columns = item_cursor.fetchone()[0]
                    except:
                        pass
                    
                    # Use appropriate query based on available columns
                    if has_initiative_columns and has_language_column:
                        # Use the enhanced schema with initiative and language columns
                        query = """
                        INSERT INTO content_data 
                        (link, title, date, summary, full_content, information, themes, organization, sentiment, 
                         language, initiative, initiative_key, benefits_to_germany, benefit_categories, benefit_examples,
                         insights, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id;
                        """
                        
                        item_cursor.execute(
                            query, 
                            (item["link"], item["title"], item["date_value"], item["summary"], 
                             item["content"], item["summary"], item["themes"], item["organization"], item["sentiment"],
                             item["language"], item["initiative"], item["initiative_key"], item["benefits_to_germany"],
                             item["benefit_categories"], item["benefit_examples"], None)
                        )
                    else:
                        # Use the original schema without special columns
                        query = """
                        INSERT INTO content_data 
                        (link, title, date, summary, full_content, information, themes, organization, sentiment, 
                         benefits_to_germany, insights, created_at, updated_at)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NOW(), NOW())
                        RETURNING id;
                        """
                        
                        item_cursor.execute(
                            query, 
                            (item["link"], item["title"], item["date_value"], item["summary"], 
                             item["content"], item["summary"], item["themes"], item["organization"], item["sentiment"],
                             item["benefits_to_germany"], None)
                        )
                    
                    record_id = item_cursor.fetchone()[0]
                    item_conn.commit()
                    
                    inserted_ids.append(record_id)
                    success_count += 1
                    
                    logger.info(f"Individual insert succeeded for URL: {item['link']} (Language: {item['language']})")
                    print(f"RECOVERY SUCCESS: Inserted record ID {record_id} | {item['title'][:50]} | {item['language']}")
                    
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