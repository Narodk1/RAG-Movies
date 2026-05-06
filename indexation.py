import pandas as pd
import json
from os import name 
DATA_PATH = "data/tmdb_5000_movies.csv"


def load_and_clean_data(path: str) -> pd.DataFrame:
    print("Chargement des données...")
    df = pd.read_csv(path)

    df = df.sort_values('vote_count', ascending=False).reset_index(drop=True)
    df = df.head(2000).copy()

    cols = ['title', 'overview', 'genres', 'release_date',
            'vote_average', 'original_language']
    df = df[cols].copy()
    df = df.dropna(subset=['overview'])

    df['genres_cleaned'] = df['genres'].apply(_extract_genres)
    return df


def _extract_genres(genre_str: str) -> str:
    try:
        genres = json.loads(genre_str)
        return ", ".join(g['name'] for g in genres)
    except Exception:
        return ""


if name == "main":
    df = load_and_clean_data(DATA_PATH)
    print(df.head())
