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

def main():
    try:
        index, metadata = charger_index(INDEX_PATH, METADATA_PATH)
    except FileNotFoundError:
        print("❌ Index non trouvé. Lancez d'abord : python3 indexation.py")
        return

    print(f"Chargement du modèle d'embedding '{MODEL_NAME}'...")
    model_emb = SentenceTransformer(MODEL_NAME, device="cpu")

    print("\n🎬  Cine_RAG — Assistant de recommandation de films")
    print("     Tapez 'q' pour quitter · 'reset' pour vider l'historique\n")

    historique: list[dict] = []

    while True:
        print("-" * 50)

        choix_langue = input(
            "🌐 Langue : (1) Français  (2) International / VO  [Défaut: 2] : "
        ).strip()
        lang_pref = "fr" if choix_langue == "1" else "en"

        query = input("🤔 Que voulez-vous regarder ? (q=quitter, reset=historique) : ").strip()

        if query.lower() in ("q", "quit", "exit"):
            print("À bientôt !")
            break
        if query.lower() == "reset":
            historique = []
            print("🔄 Historique vidé.\n")
            continue
        if not query:
            continue

        label_langue = "Français" if lang_pref == "fr" else "International"
        print(f"🔍 Recherche vectorielle ({label_langue})...")
        films, best_score = rechercher(query, model_emb, index, metadata, lang_pref)

        if best_score > SCORE_THRESHOLD:
            print(
                f"⚠️  Score de similarité faible ({best_score:.3f} > seuil {SCORE_THRESHOLD})."
                " Les résultats peuvent être peu pertinents."
            )
        else:
            print(f"   Score de similarité : {best_score:.3f} ✓")

        historique.append({"role": "user", "content": query})

        print("🤖 Consultation de l'expert Groq (avec historique)...")
        reponse = generer_reponse_avec_historique(historique, films)

        historique.append({"role": "assistant", "content": reponse})

        print("\n" + "=" * 60)
        print(reponse)
        print("=" * 60 + "\n")

if __name__ == "__main__":
    main()