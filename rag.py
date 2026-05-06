import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from generator import generer_reponse_avec_historique

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

load_dotenv()
INDEX_PATH      = "data/movies.index"
METADATA_PATH   = "data/movies_metadata.pkl"
MODEL_NAME      = "all-mpnet-base-v2"
SCORE_THRESHOLD = 1.2

def charger_index(chemin_index: str, chemin_meta: str):
    """Load FAISS index and metadata from disk."""
    print("Chargement de l'index FAISS et des métadonnées...")
    index = faiss.read_index(chemin_index)
    with open(chemin_meta, "rb") as f:
        metadata = pickle.load(f)
    print(f"  → {index.ntotal} vecteurs chargés")
    return index, metadata

def rechercher(question: str, modele_emb, index, metadata: dict,
               langue_pref: str, k: int = 15) -> tuple[list[dict], float]:
    """
    Search for the k most relevant films for a given question.
    Pipeline:
      1. Embed the question
      2. Broad FAISS search (top-300) to allow filtering
      3. Filter by language
      4. Fallback without filter if no results found
    """
    question_embedding = modele_emb.encode([question]).astype('float32')
    scores, indices    = index.search(question_embedding, 300)
    best_score         = float(scores[0][0])

    chunk_to_doc  = metadata.get("chunk_to_doc", list(range(len(metadata["titles"]))))
    resultats_idx = []

    for i, idx in enumerate(indices[0]):
        doc_idx = chunk_to_doc[idx] if idx < len(chunk_to_doc) else idx
        lang    = metadata['languages'][doc_idx]

        if langue_pref == "fr":
            if lang == "fr":
                resultats_idx.append((idx, doc_idx))
        else:
            if lang != "fr":
                resultats_idx.append((idx, doc_idx))

        if len(resultats_idx) >= k:
            break

    if not resultats_idx:
        for idx in indices[0][:k]:
            doc_idx = chunk_to_doc[idx] if idx < len(chunk_to_doc) else idx
            resultats_idx.append((idx, doc_idx))

    seen_docs     = set()
    final_results = []
    chunks_key    = "chunks" in metadata

    for chunk_idx, doc_idx in resultats_idx:
        if doc_idx in seen_docs:
            continue
        seen_docs.add(doc_idx)
        final_results.append({
            "title":       metadata['titles'][doc_idx],
            "description": metadata['descriptions'][doc_idx],
            "rating":      metadata['ratings'][doc_idx],
            "lang":        metadata['languages'][doc_idx],
            "chunk":       metadata['chunks'][chunk_idx] if chunks_key else metadata['descriptions'][doc_idx],
        })
        if len(final_results) >= k:
            break

    return final_results[:5], best_score