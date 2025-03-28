import os
import pandas as pd
import openai
from dotenv import load_dotenv

# Load API Key from .env
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def translate_text(text, target_language):
    if pd.isna(text) or not text.strip():
        return text  # Return empty or NaN as is

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": f"Translate the following text into {target_language}."},
            {"role": "user", "content": text}
        ]
    )
    return response["choices"][0]["message"]["content"]

def process_csv(file_path, target_language, new_file_name):
    df = pd.read_csv(file_path)

    if "summary" in df.columns:
        df["summary"] = df["summary"].apply(lambda x: translate_text(x, target_language))
        df.to_csv(new_file_name, index=False)
        print(f"Translated file saved as {new_file_name}")
    else:
        print(f"No 'summary' column found in {file_path}")

# Process German CSV
process_csv("german.csv", "German", "german_translated.csv")

# Process French CSV
process_csv("french.csv", "French", "french_translated.csv")
