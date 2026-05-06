import pandas as pd
import json
from os import name 
import pandas as pd
import json
DATA_PATH = "data/tmdb_5000_movies.csv"


def load_and_clean_data(path: str) -> pd.DataFrame:
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


def create_textdescriptions(df: pd.DataFrame) -> list[str]:
    descriptions = []
    for _, row in df.iterrows():
        text = (
            f"Title: {row['title']}. "
            f"Genres: {row['genres_cleaned']}. "
            f"Date: {row['release_date']}. "
            f"Rating: {row['vote_average']}/10. "
            f"Overview: {row['overview']}"
        )
        descriptions.append(text)
    return descriptions


if name == "main":
    df = load_and_clean_data(DATA_PATH)
    descriptions = create_textdescriptions(df)

    print(descriptions[0])