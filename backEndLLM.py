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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()

    sentiments = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiments[prediction]
    
def generate_advice_from_similar_projections(financial_embedding):
    indices, distances = search_index(financial_embedding, k=5) #search projections based on projections
    similar_projections = search_data(indices)

    advice = []
    for projection in similar_projections:
        print(projection)
        
        projection = f"{projection}" #string conversion so model can analyze

        sentiment = analyze_sentiment(projection)
        
        # Generate advice based on sentiment
        if sentiment == "positive":
            sentiment_advice = "This projection has positive sentiment. You might consider expanding investments."
        elif sentiment == "neutral":
            sentiment_advice = "The sentiment is neutral. Keep monitoring the market and adjust accordingly."
        else:
            sentiment_advice = "This projection has negative sentiment. Consider re-evaluating your strategy."
        
        advice.append(f"Projection Sentiment: {sentiment}, Advice: {sentiment_advice}")
    
    return advice