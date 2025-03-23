import pandas as pd
import os
import re
import time
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import logging
import html

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("csv_cleaner.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("CSV_Cleaner")

def clean_html_entities(text):
    """Clean HTML entities and common encoding issues in text"""
    if not isinstance(text, str):
        return text
        
    # First decode HTML entities
    text = html.unescape(text)
    
    # Replace common UTF-8 encoding issues
    replacements = {
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'Â': ' ',
        'â€¦': '...',
        'â€"': '—',
        'â€"': '-',
        'â€˜': "'",
        'Ã©': 'é',
        'Ã¨': 'è',
        'Ã¢': 'â',
        'Ã»': 'û',
        'Ã´': 'ô',
        'Ã®': 'î',
        'Ã¯': 'ï',
        'Ã': 'à',
        'Ã§': 'ç',
        'Ãª': 'ê',
        'Ã¹': 'ù',
        'Ã³': 'ó',
        'Ã±': 'ñ',
        'ï»¿': '',  # BOM
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text

def clean_csv_file(input_file, output_file):
    """Clean encoding issues in the CSV file"""
    logger.info(f"Cleaning CSV file: {input_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file, encoding='utf-8', on_bad_lines='skip')
        logger.info(f"Read CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # List of columns to clean
        text_columns = df.columns[df.dtypes == 'object'].tolist()
        
        # Clean each column
        for column in text_columns:
            logger.info(f"Cleaning column: {column}")
            df[column] = df[column].apply(clean_html_entities)
        
        # Save the cleaned CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        logger.info(f"Saved cleaned CSV to {output_file}")
        
        return df
    
    except Exception as e:
        logger.error(f"Error cleaning CSV: {str(e)}")
        raise

def get_openai_client():
    """Get or initialize OpenAI client."""
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    if not openai_api_key:
        logger.error("OPENAI_API_KEY not found in environment variables")
        return None
    
    try:
        return OpenAI(api_key=openai_api_key)
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        return None

def enhance_summary(summary, title, context_text=''):
    """Use OpenAI to enhance and clean a summary based on context"""
    client = get_openai_client()
    if not client:
        logger.error("OpenAI client not available")
        return summary
    
    try:
        prompt = f"""
Please improve and clean up this summary from a website. Fix any encoding issues, and make it clear and coherent
without adding any fictional information. Keep the meaning, just fix structural issues and make it easy to read.

Title: {title}

Original Summary: 
{summary}

Additional Context (if available):
{context_text}

Only output the cleaned summary text, with no preface, commentary, or explanation.
Keep the summary factual and based only on the information provided.
"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        enhanced_summary = response.choices[0].message.content.strip()
        logger.info(f"Enhanced summary: {len(summary)} chars → {len(enhanced_summary)} chars")
        return enhanced_summary
    
    except Exception as e:
        logger.error(f"Error enhancing summary: {str(e)}")
        return summary

def process_csv_summaries(input_file, output_file, batch_size=10):
    """Process and enhance summaries in the CSV file"""
    logger.info(f"Processing summaries in {input_file}")
    
    # First clean the CSV
    cleaned_file = 'cleaned_temp.csv'
    df = clean_csv_file(input_file, cleaned_file)
    
    # Create a copy for the enhanced summaries
    df_enhanced = df.copy()
    
    # Set up a progress bar
    total_rows = len(df)
    print(f"Enhancing summaries for {total_rows} rows...")
    
    # Check if we have OpenAI API key for summary enhancement
    client = get_openai_client()
    if not client:
        logger.warning("OpenAI client not available. Skipping summary enhancement.")
        print("WARNING: OpenAI API key not found. Skipping summary enhancement.")
        # Save the basic cleaned file as the output
        df.to_csv(output_file, index=False, encoding='utf-8')
        return df
    
    # Process in batches to avoid API rate limits
    for i in tqdm(range(0, total_rows, batch_size), desc="Processing batches"):
        batch = df.iloc[i:min(i+batch_size, total_rows)]
        
        for idx, row in batch.iterrows():
            # Skip if summary is too short or missing
            if 'summary' not in row or not isinstance(row['summary'], str) or len(row['summary']) < 10:
                continue
                
            # Get context from other fields
            context_parts = []
            if 'title' in row and isinstance(row['title'], str):
                context_parts.append(f"Title: {row['title']}")
            if 'organization' in row and isinstance(row['organization'], str):
                context_parts.append(f"Organization: {row['organization']}")
            if 'initiative' in row and isinstance(row['initiative'], str):
                context_parts.append(f"Initiative: {row['initiative']}")
            if 'themes' in row and isinstance(row['themes'], str):
                context_parts.append(f"Themes: {row['themes']}")
                
            context = "\n".join(context_parts)
            
            # Enhance the summary
            original_summary = row['summary']
            title = row.get('title', '')
            
            enhanced_summary = enhance_summary(original_summary, title, context)
            
            # Update the enhanced dataframe
            df_enhanced.at[idx, 'summary'] = enhanced_summary
            
        # Save after each batch as a checkpoint
        df_enhanced.to_csv('enhanced_checkpoint.csv', index=False, encoding='utf-8')
        
        # Sleep to avoid rate limits
        time.sleep(1)
    
    # Save the final enhanced CSV
    df_enhanced.to_csv(output_file, index=False, encoding='utf-8')
    logger.info(f"Saved enhanced CSV to {output_file}")
    print(f"Saved enhanced CSV to {output_file}")
    
    # Clean up temp files
    if os.path.exists(cleaned_file):
        os.remove(cleaned_file)
        print(f"Removed temporary file: {cleaned_file}")
    if os.path.exists('enhanced_checkpoint.csv'):
        os.remove('enhanced_checkpoint.csv')
        print(f"Removed temporary file: enhanced_checkpoint.csv")
    
    return df_enhanced

if __name__ == "__main__":
    print("CSV Cleaner and Summary Enhancer")
    print("================================")
    
    # Check environment
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not found in environment variables or .env file")
        print("Summary enhancement will be skipped.")
    
    # Get input file
    input_file = input("Enter input CSV file name (default: results.csv): ").strip() or "results.csv"
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File '{input_file}' not found.")
        exit(1)
    
    # Get output file
    output_file = input("Enter output CSV file name (default: results_enhanced.csv): ").strip() or "results_enhanced.csv"
    
    # Process the CSV
    print(f"Processing {input_file}...")
    df_result = process_csv_summaries(input_file, output_file)
    
    # Display stats
    print("\nProcessing complete!")
    print(f"Input file: {input_file} ({os.path.getsize(input_file) / 1024:.1f} KB)")
    print(f"Output file: {output_file} ({os.path.getsize(output_file) / 1024:.1f} KB)")
    print(f"Records processed: {len(df_result)}")
    
    print("\nColumn statistics:")
    for col in df_result.columns:
        non_null = df_result[col].count()
        print(f"- {col}: {non_null} non-null values ({non_null/len(df_result)*100:.1f}%)")
    
    print("\nSample of enhanced summaries:")
    if 'summary' in df_result.columns:
        for _, row in df_result.head(3).iterrows():
            print(f"\nTitle: {row.get('title', 'N/A')}")
            print(f"Summary: {row.get('summary', 'N/A')[:200]}...")
    
    print("\nDone!")