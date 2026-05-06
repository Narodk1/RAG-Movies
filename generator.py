import os
from groq import Groq
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()
GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
CONTEXT_PATH   = "context.txt"

if not GROQ_API_KEY:
    raise EnvironmentError("❌ GROQ_API_KEY not found. Check your .env file.")

client = Groq(api_key=GROQ_API_KEY)


# ─── System Prompt ────────────────────────────────────────────────────────────

def charger_prompt_systeme(chemin: str = CONTEXT_PATH) -> str:
    """
    Loads the system prompt from context.txt.
    Keeping the prompt in a separate file makes it easy to
    update behavior without touching the code.
    """
    with open(chemin, "r", encoding="utf-8") as f:
        return f.read().strip()


# ─── Context Builder ──────────────────────────────────────────────────────────

def construire_contexte(films_pertinents: list[dict]) -> str:
    """
    Formats the FAISS results into a structured text block
    to inject into the LLM user prompt.

    Args:
        films_pertinents: list of dicts returned by rechercher()

    Returns:
        Formatted context string
    """
    contexte = ""
    for i, f in enumerate(films_pertinents, start=1):
        contexte += (
            f"--- FILM {i} ---\n"
            f"Titre : {f['title']}\n"
            f"Note  : {f['rating']}/10\n"
            f"Extrait pertinent : {f['chunk']}\n"
            f"Description complète : {f['description']}\n\n"
        )
    return contexte


# ─── Simple Generation ────────────────────────────────────────────────────────

def generer_reponse(question: str, films_pertinents: list[dict]) -> str:
    """
    Calls the Groq API with FAISS context to generate a recommendation.
    No conversation history — single turn only.

    Args:
        question        : the user's query
        films_pertinents: results returned by rechercher()

    Returns:
        LLM response string
    """
    contexte    = construire_contexte(films_pertinents)
    user_prompt = f"Question : {question}\n\nContexte :\n{contexte}"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": charger_prompt_systeme()},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ─── Generation with History (Bonus A) ───────────────────────────────────────

def generer_reponse_avec_historique(historique: list[dict],
                                    films_pertinents: list[dict]) -> str:
    """
    Bonus A — Generates a response using the full conversation history.
    The FAISS context is injected into the last user message so the
    LLM can reason across multiple turns.

    Args:
        historique      : full conversation history
                          [{"role": "user"|"assistant", "content": "..."}]
                          Last entry must be the current user question.
        films_pertinents: FAISS results for the current question

    Returns:
        LLM response string
    """
    contexte = construire_contexte(films_pertinents)

    # Inject FAISS context into the last user message only
    messages_enrichis = historique[:-1] + [{
        "role":    "user",
        "content": historique[-1]["content"] + f"\n\nContexte films disponibles :\n{contexte}",
    }]

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": charger_prompt_systeme()},
            *messages_enrichis,
        ],
        temperature=0.7,
        max_tokens=1024,
    )
    return response.choices[0].message.content
