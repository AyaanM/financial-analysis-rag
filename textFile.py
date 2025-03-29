import random
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import main
import pandas as pd

# Load FinBERT model for sentiment analysis
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Placeholder for financial data
financial_data = [
    {
        'cash_flow': -5000,  # Negative cash flow
        'revenue': 25000,    # Revenue
        'expenses': 30000,   # Expenses
        'profit_margin': 0.1,  # 10% profit margin
        'revenue_growth': -0.02,  # Negative growth
    },
    {
        'cash_flow': 10000,  # Positive cash flow
        'revenue': 40000,    # Revenue
        'expenses': 25000,   # Expenses
        'profit_margin': 0.2,  # 20% profit margin
        'revenue_growth': 0.05,  # Positive growth
    },
    # Add more projections here as needed
]

# Load FAISS index
def load_faiss_index():
    '''
    Load the FAISS index from a specified file
    '''
    index = faiss.read_index("rag_index/faiss_db.idx")
    print("FAISS index loaded successfully!")
    return index

# Search for top-k similar projections from the FAISS index
def search_similar_projections(query_embedding, k=5):
    '''
    Search the FAISS index for the top-k similar projections.
    Returns the indices and distances of the closest vectors.
    '''
    index = load_faiss_index()

    # Perform the search to find the most similar vectors to the query
    distances, indices = index.search(np.array(query_embedding).reshape(1, -1), k)

    # Return the indices and distances of the top-k closest vectors
    return indices, distances

# Retrieve the similar projections based on the FAISS search
def get_similar_texts(indices):
    '''
    Retrieve the text associated with the indices of the most similar projections.
    '''
    similar_texts = []

    # Iterate over indices to retrieve the associated projections
    for idx in indices[0]:
        if idx < len(financial_data):  # Ensure the index is within bounds
            projection = financial_data[idx]
            similar_texts.append(projection)
        else:
            print(f"Warning: Index {idx} out of range for financial_data list.")
    
    return similar_texts

# Generate embedding for a financial projection using FinBERT
def get_financial_embedding(financial_projection):
    '''
    Convert a financial projection into an embedding using FinBERT.
    '''
    # Convert the financial projection into a textual string for embedding
    projection_text = (
        f"Cash flow: {financial_projection['cash_flow']}, "
        f"Revenue: {financial_projection['revenue']}, "
        f"Expenses: {financial_projection['expenses']}, "
        f"Profit margin: {financial_projection['profit_margin']}, "
        f"Revenue growth: {financial_projection['revenue_growth']}"
    )
    
    # Tokenize and get embeddings
    inputs = tokenizer(projection_text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract the embedding (using the last hidden state mean)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return embedding

# Sentiment analysis using FinBERT
def analyze_sentiment(text):
    '''
    Perform sentiment analysis on the text and return the sentiment label.
    '''
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    # Predict sentiment
    prediction = torch.argmax(outputs.logits, dim=1).item()

    # Sentiment labels
    sentiments = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiments[prediction]

# Generate advice based on similar projections
def generate_advice_from_similar_projections(query_embedding):
    '''
    Generate advice by analyzing the sentiment of projections most similar to the input query.
    '''
    # Step 1: Get the top-k most similar projections from FAISS
    indices, distances = search_similar_projections(query_embedding, k=5)

    # Step 2: Retrieve the financial data corresponding to the similar projections
    similar_projections = get_similar_texts(indices)

    # Step 3: Perform sentiment analysis and generate advice based on the sentiment
    for projection in similar_projections:
        # Here, assuming the projection is a dictionary containing financial data (not text)
        projection_text = (
            f"Cash flow: {projection['cash_flow']}, "
            f"Revenue: {projection['revenue']}, "
            f"Expenses: {projection['expenses']}, "
            f"Profit margin: {projection['profit_margin']}, "
            f"Revenue growth: {projection['revenue_growth']}"
        )
        
        sentiment = analyze_sentiment(projection_text)
        
        # Generate advice based on sentiment
        if sentiment == "positive":
            sentiment_advice = "This projection has positive sentiment. You might consider expanding investments."
        elif sentiment == "neutral":
            sentiment_advice = "The sentiment is neutral. Keep monitoring the market and adjust accordingly."
        else:
            sentiment_advice = "This projection has negative sentiment. Consider re-evaluating your strategy."

        # Print sentiment and advice
        print(f"Projection Sentiment: {sentiment}, Advice: {sentiment_advice}")
    
    # Quantitative analysis can also be done here for financial advice (e.g., cost-cutting, increasing revenue)
    for projection in similar_projections:
        # Example: Suggesting reducing expenses if the cash flow is negative
        if projection['cash_flow'] < 0:
            quantitative_advice = "Consider reducing expenses to improve cash flow."
        elif projection['revenue_growth'] < 0:
            quantitative_advice = "Focus on increasing revenue and reducing unnecessary expenses."
        else:
            quantitative_advice = "Your projections look stable. Continue with the current strategy."

        print(f"Quantitative Advice: {quantitative_advice}")

#### Example Usage ####

# Example new financial data for comparison
new_financial_data = {
    'cash_flow': -5000,  # Negative cash flow
    'revenue': 25000,    # Revenue
    'expenses': 30000,   # Expenses
    'profit_margin': 0.1,  # 10% profit margin
    'revenue_growth': -0.02,  # Negative growth
}

# Get the embedding for the new financial data
new_embedding = get_financial_embedding(new_financial_data)

# Generate advice from the most similar projections in the FAISS index
generate_advice_from_similar_projections(new_embedding)
