'''
BackEnd of The LLM
Various function from the frontend are triggered
'''
import random, torch, faiss, numpy as np, pandas as pd
from index_creation import pdf_load
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def search_index(financial_embedding, k=5):
    '''
    Search FAISS index for top-k similar projections (default is 5).
    distances: Euclidean distance of k closest vectors from the query (smaller distance = closer).
    index: Approx nearest neighbors containing indices of query vectors.
    '''
    # Read the FAISS index
    index = faiss.read_index("rag_index/faiss_db.idx")

    # Normalize embeddings to for consine, angled similarity (optional, can be removed if not needed)
    faiss.normalize_L2(financial_embedding)  

    # Perform the search in FAISS index
    distances, indices = index.search(np.array(financial_embedding).reshape(1, -1), k) 

    return indices, distances  # Return the indices of the top k similar projections


def search_data(indices):
    '''
    look at where indicies lie in text
    '''
    text = pdf_load.loadPDF()

    similar_texts = []

    for idx in indices[0]:
        projection = text[idx]  # Retrieve the financial projection corresponding to the index
        similar_texts.append(projection)  # Add it to the list of similar texts
    
    return similar_texts

#### STEPS FOR FINBERT ANALYSIS
def analyze_sentiment(text):
    '''
    Take in the text, pass it through FinBERT, and return the sentiment.
    '''
    text = f"text" #make the text a string

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    sentiments = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiments[prediction]
    
def sentiment_helper(financial_embedding):
    indices, distances = search_index(financial_embedding, k=5) #search projections based on projections
    similar_projections = search_data(indices)

    sentiment = analyze_sentiment(similar_projections)
    return sentiment