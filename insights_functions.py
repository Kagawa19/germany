import os
import logging
from typing import Optional, List
from openai import OpenAI
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("insights_functions.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize OpenAI client
try:
    openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    logger.info("OpenAI client initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize OpenAI client: {str(e)}")
    openai_client = None

def load_prompt(filename: str) -> str:
    """
    Load a prompt from the prompts directory.
    
    Args:
        filename (str): Name of the prompt file
    
    Returns:
        str: Prompt content or empty string if file not found
    """
    try:
        prompt_path = os.path.join('prompts', filename)
        with open(prompt_path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except FileNotFoundError:
        logger.warning(f"Prompt file not found: {filename}")
        return ""
    except Exception as e:
        logger.error(f"Error reading prompt file {filename}: {str(e)}")
        return ""

def generate_insights_with_openai(content: str) -> Optional[str]:
    """
    Generate strategic insights using OpenAI.
    
    Args:
        content (str): The text content to analyze
    
    Returns:
        Optional[str]: Generated insights or None if generation fails
    """
    # Validate inputs
    if not content or not isinstance(content, str):
        logger.warning("Invalid content provided for insights generation")
        return None

    # Check OpenAI client
    if not openai_client:
        logger.error("OpenAI client not initialized")
        return None
    
    try:
        # Load insights prompt
        insights_prompt = load_prompt('insights.txt')
        if not insights_prompt:
            insights_prompt = (
                "Extract key strategic insights from the following content. "
                "Focus on unique approaches, innovative solutions, "
                "and critical observations about international cooperation."
            )
        
        # Limit content length to prevent token overflow
        truncated_content = content[:15000]
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": insights_prompt},
                {"role": "user", "content": truncated_content}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        # Extract and clean insights
        insights_text = response.choices[0].message.content.strip()
        
        logger.info(f"Successfully generated insights (length: {len(insights_text)} chars)")
        return insights_text
    
    except Exception as e:
        logger.error(f"Error generating insights with OpenAI: {str(e)}")
        return None

def generate_benefits_to_germany_with_openai(content: str) -> Optional[str]:
    """
    Generate benefits to Germany using OpenAI.
    
    Args:
        content (str): The text content to analyze
    
    Returns:
        Optional[str]: Generated benefits or None if generation fails
    """
    # Validate inputs
    if not content or not isinstance(content, str):
        logger.warning("Invalid content provided for benefits generation")
        return None

    # Check OpenAI client
    if not openai_client:
        logger.error("OpenAI client not initialized")
        return None
    
    try:
        # Load benefits prompt
        benefits_prompt = load_prompt('benefits.txt')
        if not benefits_prompt:
            benefits_prompt = (
                "Extract specific benefits to Germany from the given content. "
                "Focus on economic, technological, environmental, and strategic advantages. "
                "Provide a clear, concise summary of how the content relates to Germany's interests."
            )
        
        # Limit content length to prevent token overflow
        truncated_content = content[:15000]
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": benefits_prompt},
                {"role": "user", "content": truncated_content}
            ],
            max_tokens=500,
            temperature=0.3
        )
        
        # Extract and clean benefits
        benefits_text = response.choices[0].message.content.strip()
        
        logger.info(f"Successfully generated benefits (length: {len(benefits_text)} chars)")
        return benefits_text
    
    except Exception as e:
        logger.error(f"Error generating benefits with OpenAI: {str(e)}")
        return None

def analyze_sentiment_with_openai(content: str) -> str:
    """
    Analyze sentiment using OpenAI.
    
    Args:
        content (str): The text content to analyze
    
    Returns:
        str: Sentiment category (Positive/Neutral/Negative)
    """
    # Validate inputs
    if not content or not isinstance(content, str):
        logger.warning("Invalid content provided for sentiment analysis")
        return "Neutral"

    # Check OpenAI client
    if not openai_client:
        logger.error("OpenAI client not initialized")
        return "Neutral"
    
    try:
        # Load sentiment prompt
        sentiment_prompt = load_prompt('sentiment.txt')
        if not sentiment_prompt:
            sentiment_prompt = (
                "Analyze the sentiment of the following text in the context of "
                "international environmental cooperation. "
                "Categorize the overall sentiment as Positive, Neutral, or Negative. "
                "Consider the tone, language, and underlying message."
            )
        
        # Limit content length to prevent token overflow
        truncated_content = content[:15000]
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": sentiment_prompt},
                {"role": "user", "content": truncated_content}
            ],
            max_tokens=50,
            temperature=0.1
        )
        
        # Extract and normalize sentiment
        sentiment_text = response.choices[0].message.content.strip().lower()
        
        # Classify sentiment
        if "positive" in sentiment_text:
            return "Positive"
        elif "negative" in sentiment_text:
            return "Negative"
        else:
            return "Neutral"
    
    except Exception as e:
        logger.error(f"Error analyzing sentiment with OpenAI: {str(e)}")
        return "Neutral"

def classify_themes_with_openai(content: str) -> List[str]:
    """
    Classify themes using OpenAI.
    
    Args:
        content (str): The text content to analyze
    
    Returns:
        List[str]: List of identified themes
    """
    # Validate inputs
    if not content or not isinstance(content, str):
        logger.warning("Invalid content provided for theme classification")
        return []

    # Check OpenAI client
    if not openai_client:
        logger.error("OpenAI client not initialized")
        return []
    
    try:
        # Load themes prompt
        themes_prompt = load_prompt('themes.txt')
        if not themes_prompt:
            themes_prompt = (
                "Extract the most relevant themes from the content. "
                "Choose from these predefined themes: "
                "Promotion of indigenous peoples, Promotion of protected areas, "
                "Restoration, Marine protection, Ecosystem services, "
                "Sustainable agriculture, Sustainable forestry, "
                "Sustainable fisheries, Sustainable aquaculture, "
                "Job creation and value creation in Germany, "
                "Security of supply for Germany. "
                "Return themes as a comma-separated list."
            )
        
        # Limit content length to prevent token overflow
        truncated_content = content[:15000]
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": themes_prompt},
                {"role": "user", "content": truncated_content}
            ],
            max_tokens=100,
            temperature=0.3
        )
        
        # Extract and clean themes
        themes_str = response.choices[0].message.content.strip()
        themes_list = [theme.strip() for theme in themes_str.split(",") if theme.strip()]
        
        logger.info(f"Successfully classified {len(themes_list)} themes")
        return themes_list
    
    except Exception as e:
        logger.error(f"Error classifying themes with OpenAI: {str(e)}")
        return []

# Optional: Add a main block for testing
if __name__ == "__main__":
    # Example usage and testing
    test_content = """
    Germany's international cooperation in environmental sustainability 
    has shown remarkable progress in recent years. Through innovative 
    partnerships with developing countries, Germany has been instrumental 
    in promoting sustainable agriculture and protecting marine ecosystems.
    """
    
    print("Testing Insights Generation:")
    insights = generate_insights_with_openai(test_content)
    print(insights)
    
    print("\nTesting Benefits Generation:")
    benefits = generate_benefits_to_germany_with_openai(test_content)
    print(benefits)
    
    print("\nTesting Sentiment Analysis:")
    sentiment = analyze_sentiment_with_openai(test_content)
    print(sentiment)
    
    print("\nTesting Theme Classification:")
    themes = classify_themes_with_openai(test_content)
    print(themes)