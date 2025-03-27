import random, torch, faiss, numpy as np, pandas as pd
from index_creation import pdf_load
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def load_faiss_index():
    '''
    load faiss index
    '''
    index = faiss.read_index("rag_index/faiss_db.idx")
    print("FAISS index loaded successfully!")
    return index 


def get_embedding(query):
    '''
    Take in query from user and convert to embeddings for FAISS search
    '''
    tokens = tokenizer(query, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        output = model(**tokens)

    cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()

    return cls_embedding  

def search_similar_projections(query, k=5):
    '''
    Search FAISS index for top-k similar projections (default is 5)
    distances: euclidian distance of k closest vectors from query (smaller distance = closer)
    index: approx nearest neighbours containing indicies of query vectors
    '''
    index = load_faiss_index()  # Load the FAISS index
    query_embedding = get_embedding(query)  # Get the query embedding
    # faiss.normalize_L2(np.array(query_embedding)) #normalize to compare with angles

    distances, indices = index.search(query_embedding, k) #looking for the k closest vectors
    
    return indices, distances

### TEST IF WORKING ###

query = "Crazy boy."
indices, distances = search_similar_projections(query, k=5)

print("Similar Projections Indices:", indices)
print("Distances from Query:", distances)
