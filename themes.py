#!/usr/bin/env python3
"""
Theme Generator - Analyzes CSV data and assigns relevant themes using OpenAI
"""

import pandas as pd
import os
import time
import random
from openai import OpenAI
from dotenv import load_dotenv
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("theme_generator.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("ThemeGenerator")

def get_openai_client():
    """Get OpenAI client from environment variables"""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in environment variables")
    return OpenAI(api_key=api_key)

def generate_global_themes(df, num_themes=30):
    """
    Generate a set of global themes by analyzing the entire dataset
    
    Args:
        df: DataFrame containing the data
        num_themes: Number of themes to generate
        
    Returns:
        List of theme strings
    """
    logger.info(f"Generating {num_themes} global themes from data with {len(df)} rows")
    
    client = get_openai_client()
    
    # Prepare sample of data to analyze
    # Get a sample of rows to keep the prompt manageable
    sample_size = min(25, len(df))
    sample_rows = df.sample(sample_size)
    
    # Prepare a description of the data
    columns_desc = ", ".join(df.columns)
    
    # Create a text representation of sample data
    sample_data = []
    for _, row in sample_rows.iterrows():
        row_text = []
        for col in df.columns:
            if col != 'themes':  # Skip the themes column
                value = row[col]
                # Truncate long text fields
                if isinstance(value, str) and len(value) > 200:
                    value = value[:200] + "..."
                row_text.append(f"{col}: {value}")
        sample_data.append(" | ".join(row_text))
    
    sample_text = "\n".join(sample_data)
    
    # Create prompt for OpenAI
    prompt = f"""
I have a dataset about the ABS Initiative (Access and Benefit Sharing Capacity Development Initiative) with columns: {columns_desc}.

Below is a sample of {sample_size} rows from the data:

{sample_text}

Based on this data, generate {num_themes} distinct, specific themes that would be relevant for categorizing this kind of content.
Generate themes that capture the diverse topics and subjects discussed in relation to ABS Initiative, biodiversity, genetic resources, etc.

The themes should be:
1. Specific and substantive (avoid generic themes like "General Information")
2. Diverse and covering different aspects of the content
3. Relevant to the domain of biodiversity, genetic resources, etc.
4. Short (1-4 words each)

Provide exactly {num_themes} themes as a comma-separated list without numbering or explanations.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,  # Higher temperature for more diversity
            max_tokens=500
        )
        
        themes_text = response.choices[0].message.content.strip()
        
        # Parse the themes
        themes = [theme.strip() for theme in themes_text.split(',')]
        
        # Ensure we have the right number of themes
        if len(themes) < num_themes:
            logger.warning(f"OpenAI returned fewer themes than requested: {len(themes)}/{num_themes}")
            # Duplicate some themes to reach the desired count
            while len(themes) < num_themes:
                themes.append(random.choice(themes) + " (Additional)")
        elif len(themes) > num_themes:
            logger.warning(f"OpenAI returned more themes than requested: {len(themes)}/{num_themes}")
            themes = themes[:num_themes]
            
        logger.info(f"Successfully generated {len(themes)} global themes")
        return themes
        
    except Exception as e:
        logger.error(f"Error generating global themes: {str(e)}")
        # Return basic fallback themes
        return [f"Theme {i+1}" for i in range(num_themes)]

def assign_themes_to_row(row, global_themes, client):
    """
    Assign the most relevant themes from the global theme list to a specific row
    
    Args:
        row: DataFrame row
        global_themes: List of all available themes
        client: OpenAI client
        
    Returns:
        List of assigned themes
    """
    # Combine row data into a text representation
    row_text = []
    for col, value in row.items():
        if col != 'themes' and isinstance(value, str):  # Skip the themes column and non-text
            # Truncate long text fields
            if len(value) > 500:
                value = value[:500] + "..."
            row_text.append(f"{col}: {value}")
    
    content = "\n".join(row_text)
    
    # Create a compact list of all available themes
    themes_list = ", ".join(global_themes)
    
    # Create prompt for OpenAI
    prompt = f"""
Choose the 3-5 most relevant themes for this content from the following list of themes: {themes_list}

Content:
{content}

Return ONLY the selected themes as a comma-separated list without any explanations or additional text.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=100
        )
        
        themes_text = response.choices[0].message.content.strip()
        
        # Parse the themes
        assigned_themes = [theme.strip() for theme in themes_text.split(',')]
        
        # Validate that the assigned themes are from the global list
        valid_themes = [theme for theme in assigned_themes if theme in global_themes]
        
        # If no valid themes were assigned, pick some random ones
        if not valid_themes:
            valid_themes = random.sample(global_themes, min(3, len(global_themes)))
            
        return valid_themes
        
    except Exception as e:
        logger.error(f"Error assigning themes to row: {str(e)}")
        # Return random themes as fallback
        return random.sample(global_themes, min(3, len(global_themes)))

def process_csv(input_file='themes.csv', output_file='new_themes.csv'):
    """
    Process the CSV file, generate global themes, and assign them to each row
    
    Args:
        input_file: Input CSV filename
        output_file: Output CSV filename
    """
    logger.info(f"Starting to process {input_file}")
    
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)
        logger.info(f"Read CSV with {len(df)} rows and {len(df.columns)} columns")
        
        # Generate global themes
        global_themes = generate_global_themes(df)
        logger.info(f"Generated {len(global_themes)} global themes")
        
        # Clear existing themes column
        if 'themes' in df.columns:
            logger.info("Clearing existing themes column")
            df['themes'] = None
        else:
            logger.info("Creating new themes column")
            df['themes'] = None
        
        # Get OpenAI client
        client = get_openai_client()
        
        # Assign themes to each row
        logger.info("Assigning themes to each row")
        
        # Create a progress bar
        with tqdm(total=len(df)) as pbar:
            for idx, row in df.iterrows():
                # Assign themes to this row
                assigned_themes = assign_themes_to_row(row, global_themes, client)
                
                # Update the DataFrame
                df.at[idx, 'themes'] = ', '.join(assigned_themes)
                
                # Update progress bar
                pbar.update(1)
                
                # Add a small delay to avoid rate limits
                time.sleep(0.1)
        
        # Save the updated DataFrame
        logger.info(f"Saving updated data to {output_file}")
        df.to_csv(output_file, index=False)
        
        logger.info("Processing completed successfully")
        
        # Print summary
        print(f"\nProcessing completed successfully!")
        print(f"Generated {len(global_themes)} global themes:")
        for i, theme in enumerate(global_themes, 1):
            print(f"  {i}. {theme}")
        print(f"\nAssigned themes to {len(df)} rows")
        print(f"Results saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error processing CSV: {str(e)}")
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    print("Theme Generator - Assigns relevant themes to CSV data\n")
    
    # Check for OpenAI key
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in .env file")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        exit(1)
    
    # Get input and output filenames (optional)
    input_file = input("Enter input CSV filename (default: themes.csv): ").strip() or "themes.csv"
    output_file = input("Enter output CSV filename (default: new_themes.csv): ").strip() or "new_themes.csv"
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found")
        exit(1)
    
    # Process the CSV
    print(f"Processing {input_file}...")
    process_csv(input_file, output_file)