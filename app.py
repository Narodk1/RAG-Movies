"""
app.py — Backend Flask pour Cine_RAG
Lance avec : python3 app.py
"""

import os
import pickle
import faiss
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

# Fix Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────
INDEX_PATH      = "data/movies.index"
METADATA_PATH   = "data/movies_metadata.pkl"
MODEL_NAME      = "all-mpnet-base-v2"
SCORE_THRESHOLD = 1.2
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

app = Flask(__name__)
CORS(app)

@app.route("/")
def index_page():
    return render_template("index.html")

# ─── Chargement au démarrage ──────────────────────────────────────────────────
print("Chargement de l'index FAISS...")
index    = faiss.read_index(INDEX_PATH)
with open(METADATA_PATH, "rb") as f:
    metadata = pickle.load(f)

print(f"Chargement du modèle '{MODEL_NAME}'...")
model_emb = SentenceTransformer(MODEL_NAME, device="cpu")

client = Groq(api_key=GROQ_API_KEY)
print("✅ Backend prêt sur http://localhost:5000")

# ─── Helpers ──────────────────────────────────────────────────────────────────

def rechercher(question, langue_pref, k=5):
    question_embedding = model_emb.encode([question]).astype("float32")
    scores, indices    = index.search(question_embedding, 50)
    best_score         = float(scores[0][0])

    chunk_to_doc  = metadata.get("chunk_to_doc", list(range(len(metadata["titles"]))))
    resultats_idx = []

    for idx in indices[0]:
        doc_idx = chunk_to_doc[idx] if idx < len(chunk_to_doc) else idx
        lang    = metadata["languages"][doc_idx]
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

    seen, results = set(), []
    chunks_key = "chunks" in metadata
    for chunk_idx, doc_idx in resultats_idx:
        if doc_idx in seen:
            continue
        seen.add(doc_idx)
        results.append({
            "title":       metadata["titles"][doc_idx],
            "description": metadata["descriptions"][doc_idx],
            "rating":      metadata["ratings"][doc_idx],
            "lang":        metadata["languages"][doc_idx],
            "chunk":       metadata["chunks"][chunk_idx] if chunks_key else metadata["descriptions"][doc_idx],
        })
        if len(results) >= k:
            break

    return results, best_score


def construire_prompt_systeme():
    return """Tu es un expert cinéphile chargé de recommander des films.

CONTRAINTES STRICTES :
1. Utilise UNIQUEMENT les films fournis dans le contexte ci-dessous.
2. Pour chaque recommandation, cite obligatoirement le TITRE et la NOTE (/10).
3. Justifie chaque choix de manière cinéphile en liant la demande au synopsis.
4. Si aucun film du contexte ne correspond à la demande, dis-le clairement sans inventer.
5. Termine toujours par : "Source : Base de données TMDB"."""


def generer_reponse(question, films, historique):
    contexte = ""
    for i, f in enumerate(films, 1):
        contexte += (
            f"--- FILM {i} ---\n"
            f"Titre : {f['title']}\n"
            f"Note  : {f['rating']}/10\n"
            f"Description : {f['description']}\n\n"
        )

    messages = [{"role": "system", "content": construire_prompt_systeme()}]

    # Historique complet sauf le dernier message
    for msg in historique[:-1]:
        messages.append(msg)

    # Dernier message enrichi avec le contexte
    messages.append({
        "role":    "user",
        "content": f"Question : {question}\n\nContexte films :\n{contexte}",
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "films": len(metadata["titles"])})


@app.route("/search", methods=["POST"])
def search():
    data      = request.json
    question  = data.get("question", "").strip()
    langue    = data.get("langue", "en")
    historique = data.get("historique", [])

    if not question:
        return jsonify({"error": "Question vide"}), 400

    films, best_score = rechercher(question, langue)
    reponse = generer_reponse(question, films, historique)

    return jsonify({
        "reponse":    reponse,
        "films":      films,
        "score":      best_score,
        "low_confidence": best_score > SCORE_THRESHOLD,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5001)
