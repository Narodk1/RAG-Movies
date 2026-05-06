import os
# Fix pour les crashs sur Mac (conflits de bibliothèques mathématiques)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import pandas as pd
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

# ─── Configuration ────────────────────────────────────────────────────────────
DATA_PATH      = "data/tmdb_5000_movies.csv"
INDEX_PATH     = "data/movies.index"
METADATA_PATH  = "data/movies_metadata.pkl"
MODEL_NAME     = "all-mpnet-base-v2"

CHUNK_SIZE     = 600   # caractères max par chunk
CHUNK_OVERLAP  = 80    # chevauchement entre chunks consécutifs
# ──────────────────────────────────────────────────────────────────────────────


# ─── Étape 1 : Chargement et nettoyage des données ────────────────────────────

def load_and_clean_data(path: str) -> pd.DataFrame:
    """Charge le CSV, mélange les lignes et conserve les 5 000 premiers films."""
    print("Chargement et mélange des données (cible : 5 000 films)...")
    df = pd.read_csv(path)

    # Mélange pour assurer la diversité (genres, époques, langues)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.head(5000).copy()

    # Colonnes utiles uniquement
    cols = ['title', 'overview', 'genres', 'release_date',
            'vote_average', 'original_language']
    df = df[cols].copy()
    df = df.dropna(subset=['overview'])

    df['genres_cleaned'] = df['genres'].apply(_extract_genres)
    return df


def _extract_genres(genre_str: str) -> str:
    """Convertit le JSON imbriqué de la colonne genres en chaîne lisible."""
    try:
        genres = json.loads(genre_str)
        return ", ".join(g['name'] for g in genres)
    except Exception:
        return ""


# ─── Étape 2 : Création des descriptions textuelles ───────────────────────────

def create_text_descriptions(df: pd.DataFrame) -> list[str]:
    """
    Convertit chaque ligne CSV en un texte cohérent destiné à l'embedding.
    Format : Title · Genres · Date · Rating · Overview
    """
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


# ─── Étape 3 : Chunking ───────────────────────────────────────────────────────

def chunker(texte: str, taille_max: int = CHUNK_SIZE,
            overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Découpe un texte en chunks avec chevauchement.

    Pour les descriptions de films, un seul chunk suffit en général
    (< 600 caractères). La fonction est néanmoins générique : elle
    s'applique à tout contenu plus long.

    Args:
        texte     : texte à découper
        taille_max: nombre maximum de caractères par chunk
        overlap   : chevauchement entre deux chunks consécutifs

    Returns:
        Liste de chunks (au moins un élément, même si texte < taille_max)
    """
    if not texte:
        return []

    chunks = []
    debut = 0
    while debut < len(texte):
        fin = debut + taille_max
        chunk = texte[debut:fin]
        chunks.append(chunk.strip())
        if fin >= len(texte):
            break
        debut = fin - overlap   # recul de `overlap` pour assurer la continuité

    return chunks


def chunker_descriptions(descriptions: list[str]) -> tuple[list[str], list[int]]:
    """
    Applique le chunker à toutes les descriptions.

    Returns:
        chunks      : liste de tous les chunks
        chunk_to_doc: mapping chunk_index → index du film d'origine
    """
    all_chunks   = []
    chunk_to_doc = []   # pour retrouver quel film correspond à chaque chunk

    for doc_idx, desc in enumerate(descriptions):
        c = chunker(desc)
        all_chunks.extend(c)
        chunk_to_doc.extend([doc_idx] * len(c))

    return all_chunks, chunk_to_doc


# ─── Étape 4 : Index FAISS ────────────────────────────────────────────────────

def creer_index_faiss(vecteurs: np.ndarray) -> faiss.Index:
    """Crée un index FAISS L2 et y insère les vecteurs."""
    print("Initialisation de l'index FAISS (IndexFlatL2)...")
    dimension = vecteurs.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(vecteurs)
    print(f"  → {index.ntotal} vecteurs indexés (dimension={dimension})")
    return index


def sauvegarder_index(index: faiss.Index, df: pd.DataFrame,
                      descriptions: list[str], all_chunks: list[str],
                      chunk_to_doc: list[int],
                      chemin_index: str, chemin_meta: str) -> None:
    """Sauvegarde l'index FAISS et les métadonnées associées sur disque."""
    print("Sauvegarde de l'index et des métadonnées...")
    faiss.write_index(index, chemin_index)

    metadata = {
        # Données au niveau du film
        "titles":       df['title'].tolist(),
        "descriptions": descriptions,
        "ratings":      df['vote_average'].tolist(),
        "languages":    df['original_language'].tolist(),
        # Données au niveau du chunk (pour la recherche)
        "chunks":       all_chunks,
        "chunk_to_doc": chunk_to_doc,
    }
    with open(chemin_meta, "wb") as f:
        pickle.dump(metadata, f)

    print(f"  → Index sauvegardé : {chemin_index}")
    print(f"  → Métadonnées sauvegardées : {chemin_meta}")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        # 1. Préparation des données
        df           = load_and_clean_data(DATA_PATH)
        descriptions = create_text_descriptions(df)
        print(f"  → {len(descriptions)} films chargés après nettoyage")

        # 2. Chunking
        print("Découpage des descriptions en chunks...")
        all_chunks, chunk_to_doc = chunker_descriptions(descriptions)
        print(f"  → {len(all_chunks)} chunks créés "
              f"(taille_max={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")

        # 3. Embeddings
        print(f"\nChargement du modèle '{MODEL_NAME}' (CPU)...")
        model = SentenceTransformer(MODEL_NAME, device="cpu")

        print(f"Génération des embeddings pour {len(all_chunks)} chunks...")
        embeddings = model.encode(
            all_chunks,
            show_progress_bar=True,
            batch_size=32
        )
        embeddings = np.array(embeddings, dtype='float32')

        # 4. Index FAISS + sauvegarde
        index = creer_index_faiss(embeddings)
        sauvegarder_index(
            index, df, descriptions,
            all_chunks, chunk_to_doc,
            INDEX_PATH, METADATA_PATH
        )

        print("\n✅ SUCCÈS : Indexation terminée !")
        print(f"   {len(df)} films · {len(all_chunks)} chunks · "
              f"dimension={embeddings.shape[1]}")

    except FileNotFoundError:
        print(f"\n❌ Fichier introuvable : {DATA_PATH}")
        print("   Téléchargez le dataset sur kaggle.com/datasets/tmdb/tmdb-movie-metadata")
        print("   et placez tmdb_5000_movies.csv dans le dossier data/")
    except Exception as e:
        print(f"\n❌ Erreur : {e}")
        raise
