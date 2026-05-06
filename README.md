# RAG — Assistant de Recommandation de Films

Système **RAG (Retrieval-Augmented Generation)** pour recommander des films à partir du dataset TMDB 5000, en utilisant **FAISS** pour la recherche vectorielle et **Groq (Llama 3.3 70B)** pour la génération de réponses.

---

## 🧠 Questions de réflexion préliminaires

### Q1. Conversion du CSV en texte cohérent
Les données tabulaires sont converties via un **template structuré** qui concatène les colonnes les plus sémantiquement riches :
- `title` : identifie le sujet du document
- `overview` : source principale de sémantique (synopsis)
- `genres` : catégorisation thématique
- `release_date` : contexte temporel
- `vote_average` : indicateur de qualité pour le LLM

**Format final :** `"Title: [Titre]. Genres: [Genres]. Date: [Date]. Rating: [Note]/10. Overview: [Synopsis]"`

### Q2. Extraction de la colonne `genres`
La colonne contient du JSON imbriqué (ex : `[{"id": 18, "name": "Drama"}]`). On utilise `json.loads()` dans une fonction de nettoyage dédiée pour extraire uniquement le champ `name` de chaque dictionnaire et les joindre par des virgules (ex : `"Action, Adventure"`).

### Q3. Stratégie d'indexation
Pour éviter de relancer l'indexation à chaque test, on utilise une **stratégie de persistance** :
- L'index vectoriel est sauvegardé via `faiss.write_index()`
- Les métadonnées (titres, textes, notes, langues, chunks, mapping chunk→film) sont sérialisées dans un fichier `.pkl` via `pickle`

Le script `rag.py` charge ces fichiers en quelques millisecondes au démarrage.

### Q4. Guidage du LLM (Prompt Engineering)
Le LLM est guidé par un **System Prompt** strict qui :
- Lui donne le rôle d'expert cinéphile
- L'oblige à n'utiliser que le contexte fourni (pas d'hallucination)
- Impose de citer le titre et la note pour chaque recommandation
- Lui demande d'expliquer ses choix en liant la demande au synopsis
- Le force à avouer honnêtement si aucun film ne correspond

### Q5. Gestion des films hors base (ex : 2024)
Le système gère ce cas à deux niveaux :
1. **Instruction système explicite** : "Si aucun film du contexte ne correspond, dis-le clairement sans inventer"
2. **Score de confiance (Bonus B)** : si le meilleur score L2 dépasse le seuil `SCORE_THRESHOLD`, un avertissement est affiché *avant* la réponse pour signaler que les résultats sont probablement hors sujet

---

## 🚀 Installation et Lancement

### 1. Prérequis
```bash
python3 -m venv venv
source venv/bin/activate        # Linux / Mac
# venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Configuration
Créez un fichier `.env` à la racine :
```text
GROQ_API_KEY=gsk_votre_clé_ici
```
> ⚠️ Ne commitez jamais ce fichier. Il est dans `.gitignore`.

### 3. Données
Téléchargez `tmdb_5000_movies.csv` sur [kaggle.com/datasets/tmdb/tmdb-movie-metadata](https://kaggle.com/datasets/tmdb/tmdb-movie-metadata) et placez-le dans `data/`.

### 4. Indexation (une seule fois)
```bash
python3 indexation.py
```

### 5. Lancement de l'assistant
```bash
python3 rag.py
```

---

## 🛠 Choix Techniques

| Composant | Choix | Justification |
|---|---|---|
| Modèle d'embedding | `all-mpnet-base-v2` | Le dataset est en anglais ; ce modèle offre les meilleures performances sur des textes anglais courts à moyens |
| Base vectorielle | `FAISS IndexFlatL2` | Recherche exacte, simple à implémenter, suffisant pour 5 000 films |
| LLM | `llama-3.3-70b-versatile` via Groq | Équilibre qualité / vitesse ; gratuit via Groq |
| Chunking | `taille_max=600, overlap=80` | Les descriptions de films font 300–500 caractères en moyenne ; la taille est calibrée pour ne pas couper un synopsis en plein milieu. L'overlap assure la continuité contextuelle. |
| Diversité | Shuffle aléatoire avant sélection | Garantit un échantillon varié (genres, décennies, langues) |

---

## ✨ Bonus implémentés

### Bonus A — Historique de conversation
La boucle `main()` maintient un historique complet des échanges. Chaque nouvelle question est envoyée à Groq **avec** l'historique de la session, ce qui permet au LLM de comprendre les reformulations et les demandes de comparaison. Tapez `reset` pour vider l'historique.

### Bonus B — Score de confiance
Après chaque recherche FAISS, le meilleur score L2 est affiché. Si ce score dépasse le seuil `SCORE_THRESHOLD = 1.2`, un avertissement est affiché avant la réponse pour signaler que la base ne contient probablement pas de film correspondant à la demande.

---

## 📋 Compte-rendu — Difficultés et décisions de conception

### Difficultés rencontrées

**1. Crash Mac avec les bibliothèques mathématiques**
L'utilisation de `SentenceTransformer` sur Mac provoquait des crashs liés aux conflits entre OpenMP (`libiomp5`) et le runtime MKL embarqué dans FAISS. Solution : forcer `KMP_DUPLICATE_LIB_OK=TRUE` et limiter les threads avec `OMP_NUM_THREADS=1`.

**2. Filtre linguistique et recherche élargie**
Le filtre par langue (français vs. international) nécessite de chercher bien au-delà des `k` résultats voulus. Avec `k=5`, il faut parfois parcourir les 50 premiers résultats pour trouver 5 films français — surtout que le dataset TMDB contient très peu de films `lang="fr"`. Le fallback sans filtre évite les réponses vides.

**3. Mapping chunk → film**
Après l'introduction du chunking, chaque vecteur FAISS correspond à un *chunk*, pas directement à un film. Il a fallu ajouter un tableau `chunk_to_doc` pour retrouver les métadonnées du film (titre, note, langue) à partir de l'index d'un chunk. La dé-duplication dans `rechercher()` évite de recommander le même film deux fois si plusieurs de ses chunks ressortent.

**4. Taille du contexte envoyé à Groq**
Envoyer trop de chunks gonflait le prompt et ralentissait la génération. On limite à `k=5` films et on envoie à la fois le chunk le plus pertinent *et* la description complète, pour donner au LLM le contexte local et global.

### Décisions de conception

- **Persistance séparée** (`.index` + `.pkl`) plutôt qu'un seul fichier, pour pouvoir recharger l'index FAISS nativement sans passer par pickle (plus rapide et plus robuste).
- **`IndexFlatL2` plutôt que `IndexFlatIP`** : la similarité cosinus aurait nécessité une normalisation préalable des vecteurs. L2 donne des résultats équivalents sur des vecteurs de même norme et évite cette étape.
- **Historique côté client** : l'historique est géré dans `rag.py`, pas côté Groq (qui est stateless). Cela donne un contrôle total sur ce qui est transmis et permet le `reset`.
