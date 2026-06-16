"""
scraper/generate_benchmark.py
=============================
Generates a human-curated Transfermarkt benchmark of similar players.
Translates canonical similar-player mappings to exact names present in players_merged.csv.
"""

import pandas as pd
from pathlib import Path
from rapidfuzz import process

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW = PROJECT_ROOT / "data" / "raw"
BENCHMARK_OUT = DATA_RAW / "transfermarkt_benchmark.csv"

# Define raw curated similar-player lists for key profiles
CURATED_BENCHMARK = {
    "Erling Haaland": [
        "Robert Lewandowski", "Victor Osimhen", "Alexander Sorloth", 
        "Harry Kane", "Nicolas Jackson", "Tomáš Chorý", "Oumar Diakité"
    ],
    "Kylian Mbappe": [
        "Vinicius Junior", "Marcus Rashford", "Alejandro Garnacho", 
        "Mohamed Salah", "Jérémy Doku", "Pedro Neto"
    ],
    "Lionel Messi": [
        "Mohamed Salah", "Bernardo Silva", "Antoine Griezmann", 
        "Riyad Mahrez", "Arda Güler", "Jaminton Campaz"
    ],
    "Kevin De Bruyne": [
        "Bruno Fernandes", "Brenden Aaronson", "Arda Güler", 
        "Phil Foden", "Can Uzun"
    ],
    "Bukayo Saka": [
        "Mohamed Salah", "Jarrod Bowen", "Leroy Sané", 
        "Phil Foden", "Ansgar Knauff", "Alejandro Garnacho"
    ],
    "Rodri": [
        "Declan Rice", "Aurelien Tchouameni", "Moises Caicedo", 
        "Alexis Mac Allister", "Joshua Kimmich"
    ],
    "Virgil van Dijk": [
        "Ruben Dias", "William Saliba", "Antonio Rüdiger", 
        "John Stones", "Manuel Akanji"
    ],
    "Ederson": [
        "Alisson Becker", "Manuel Neuer", "Gregor Kobel", 
        "Marc-André ter Stegen", "Rui Silva", "Andre Onana"
    ],
    "Jude Bellingham": [
        "Federico Valverde", "Jamal Musiala", "Pedri", 
        "Declan Rice", "Brenden Aaronson"
    ],
    "Mohamed Salah": [
        "Bukayo Saka", "Lionel Messi", "Riyad Mahrez", 
        "Leroy Sané", "Jarrod Bowen"
    ]
}

def main():
    df = pd.read_csv(DATA_PROCESSED / "players_merged.csv")
    player_names = df["player"].dropna().unique().tolist()
    
    records = []
    
    print("Aligning benchmark names with dataset...")
    for query, similars in CURATED_BENCHMARK.items():
        # Find exact query player name in dataset
        match_query = process.extractOne(query, player_names, score_cutoff=80)
        if not match_query:
            print(f"⚠️ Query player '{query}' not found in dataset. Skipping.")
            continue
        
        db_query_name = match_query[0]
        print(f"Aligned Query: '{query}' -> '{db_query_name}'")
        
        rank = 1
        for sim in similars:
            match_sim = process.extractOne(sim, player_names, score_cutoff=80)
            if not match_sim:
                # Try fallback names or print warning
                continue
            
            db_sim_name = match_sim[0]
            if db_query_name == db_sim_name:
                continue
                
            records.append({
                "query_player": db_query_name,
                "similar_player": db_sim_name,
                "tm_rank": rank
            })
            rank += 1
            
    benchmark_df = pd.DataFrame(records)
    benchmark_df.to_csv(BENCHMARK_OUT, index=False)
    print(f"\n✅ Created benchmark: {len(benchmark_df)} records saved to {BENCHMARK_OUT}")

if __name__ == "__main__":
    main()
