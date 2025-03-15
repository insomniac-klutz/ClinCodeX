from langchain.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from joblib import Memory
import pandas as pd

def load_sentence_transformer(model_name):
    return SentenceTransformerEmbeddings(model_name=model_name)

def set_mem_cache_for_gpt(cache_dir):
    return Memory(cache_dir, verbose=0)

def df_dataloader(path):
    return pd.read_csv(path)