import os
import pandas as pd
from dotenv import load_dotenv
import openai

def translate_text(text, target_language="German"):
    if pd.isna(text) or text.strip() == "":
        return text  # Return empty values as they are
    
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful translator."},
            {"role": "user", "content": f"Translate the following text to {target_language}: {text}"}
        ]
    )
    return response["choices"][0]["message"]["content"].strip()

def main():
    load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai.api_key:
        print("Error: OPENAI_API_KEY not found in .env file.")
        return
    
    file_path = "german.csv"
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    df = pd.read_csv(file_path)
    if "summary" not in df.columns or "themes" not in df.columns:
        print("Error: CSV file must contain 'summary' and 'themes' columns.")
        return
    
    df["summary"] = df["summary"].apply(lambda x: translate_text(x))
    df["themes"] = df["themes"].apply(lambda x: translate_text(x))
    
    output_path = "german_translated.csv"
    df.to_csv(output_path, index=False)
    print(f"Translation completed. Saved as {output_path}")

if __name__ == "__main__":
    main()
