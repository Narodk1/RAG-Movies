import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from generator import generer_reponse_avec_historique

# Fix Mac
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

load_dotenv()
INDEX_PATH      = "data/movies.index"
METADATA_PATH   = "data/movies_metadata.pkl"
MODEL_NAME      = "all-mpnet-base-v2"
SCORE_THRESHOLD = 1.2
