import os
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re

def preprocess_text(text):
    """
    Preprocess text for similarity analysis using simple regex
    """
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_initiative_descriptions():
    """
    Provide comprehensive descriptions of ABS initiatives
    """
    return [
        "Access and Benefit Sharing Initiative focusing on biodiversity conservation and genetic resources management in developing countries",
        "Capacity development program for sustainable use of biological resources supporting indigenous communities",
        "Strategic framework for implementing Nagoya Protocol on genetic resources and traditional knowledge",
        "Multilateral initiative supporting equitable sharing of benefits derived from genetic resources",
        "Comprehensive program for biodiversity protection and sustainable development in African regions"
    ]

def calculate_similarity(documents, reference_texts, threshold=0.3):
    """
    Calculate cosine similarity between documents and reference texts
    
    Returns:
    - List of similarities
    - List of similarity scores
    """
    # Preprocess all texts
    processed_documents = [preprocess_text(doc) for doc in documents]
    processed_references = [preprocess_text(ref) for ref in reference_texts]
    
    # Combine all texts for vectorization
    all_texts = processed_documents + processed_references
    
    # TF-IDF Vectorization
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)
    
    # Split back into documents and references
    doc_vectors = tfidf_matrix[:len(processed_documents)]
    ref_vectors = tfidf_matrix[len(processed_documents):]
    
    # Calculate similarity
    similarities = []
    similarity_scores = []
    for doc_vector in doc_vectors:
        # Calculate max similarity to any reference text
        doc_similarities = cosine_similarity(doc_vector, ref_vectors)[0]
        max_similarity = np.max(doc_similarities)
        
        similarities.append(max_similarity >= threshold)
        similarity_scores.append(max_similarity)
    
    return similarities, similarity_scores

def filter_csv(input_file, output_file, threshold=0.3):
    """
    Filter CSV using cosine similarity
    Saves both filtered and removed rows
    """
    # Read CSV
    df = pd.read_csv(input_file)
    
    # Combine all text columns
    def combine_text_columns(row):
        return ' '.join(str(val) for val in row.values if pd.notna(val))
    
    # Combine text for each row
    documents = df.apply(combine_text_columns, axis=1)
    
    # Get initiative descriptions
    initiative_descriptions = get_initiative_descriptions()
    
    # Calculate relevance
    relevance_mask, similarity_scores = calculate_similarity(
        documents, 
        initiative_descriptions, 
        threshold=threshold
    )
    
    # Add similarity scores to DataFrame
    df['similarity_score'] = similarity_scores
    df['is_relevant'] = relevance_mask
    
    # Filter DataFrames
    df_filtered = df[df['is_relevant']]
    df_removed = df[~df['is_relevant']]
    
    # Remove temporary columns for saving
    df_filtered = df_filtered.drop(columns=['is_relevant', 'similarity_score'])
    df_removed = df_removed.drop(columns=['is_relevant'])
    
    # Save results
    df_filtered.to_csv(output_file, index=False)
    df_removed.to_csv('removed_content.csv', index=False)
    
    # Print stats
    print(f"Total original rows: {len(df)}")
    print(f"Filtered rows: {len(df_filtered)}")
    print(f"Rows removed: {len(df_removed)}")
    
    # Optional: Save removed rows with their similarity scores
    df_removed_with_scores = df[~df['is_relevant']].copy()
    df_removed_with_scores['similarity_score'] = df_removed_with_scores['similarity_score']
    df_removed_with_scores.drop(columns=['is_relevant'], inplace=True)
    df_removed_with_scores.to_csv('removed_content_with_scores.csv', index=False)
    
    return df_filtered, df_removed

def main():
    input_file = 're.csv'
    output_file = 'filtered_relevant_content.csv'
    
    # Run filtering with adjustable threshold
    filter_csv(input_file, output_file, threshold=0.3)

if __name__ == "__main__":
    main()