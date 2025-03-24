import os
import pandas as pd

def filter_summary():
    file_path = "summary.csv"
    output_path = "new_summary.csv"
    
    keywords = [
        "ABS Capacity Development Initiative", "ABS CDI", "ABS Capacity Development Initiative for Africa", 
        "ABS Inititiave", "Initiative pour le renforcement des capacités en matière d’APA", 
        "Initiative Accès et Partage des Avantages", "Initiative sur le développement des capacités pour l’APA", 
        "Initiative de renforcement des capacités sur l’APA", "Initiative APA", 
        "Initiative de développement des capacités en matière d'accès et de partage des avantages", 
        "Initiative für Zugang und Vorteilsausgleich", "ABS-Kapazitätenentwicklungsinitiative für Afrika", 
        "ABS-Initiative"
    ]
    
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return
    
    df = pd.read_csv(file_path)
    if "summary" not in df.columns:
        print("Error: CSV file must contain a 'summary' column.")
        return
    
    df_filtered = df[df["summary"].astype(str).apply(lambda x: any(keyword in x for keyword in keywords))]
    
    df_filtered.to_csv(output_path, index=False)
    print(f"Filtered data saved as {output_path}")

if __name__ == "__main__":
    filter_summary()
