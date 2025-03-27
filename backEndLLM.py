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

def load_faiss_index():
    '''
    load faiss index
    '''
    index = faiss.read_index("rag_index/faiss_db.idx")
    print("FAISS index loaded successfully!")
    return index 

def search_index(financial_embedding, k=5):
    '''
    Search FAISS index for top-k similar projections (default is 5)
    distances: euclidian distance of k closest vectors from query (smaller distance = closer)
    index: approx nearest neighbours containing indicies of query vectors
    '''
    index = load_faiss_index()

    #search for most similar vectors to query
    distances, indices = index.search(np.array(financial_embedding).reshape(1, -1), k) #search in faiss index

    return indices, distances #return for top k vectors

def get_financial_embedding(financial_projection):
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
    
    # Extract the embedding (usually from the last hidden state or pooler output)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    
    return embedding

def search_data(indices):
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
    indices, distances = search_index(financial_embedding, k=5) #search projections

    similar_projections = search_data(indices)

    advice = []
    for projection in search_similar_projections:
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
    
    # Step 3: Perform sentiment analysis on each retrieved projection
    # for projection in similar_projections:
    #     # Here, you assume the projection is a string (could be a report or text)
    #     sentiment = analyze_sentiment(projection['text'])  # Assume 'text' is the key for financial text
        
    #     # Generate advice based on sentiment
    #     if sentiment == "positive":
    #         sentiment_advice = "This projection has positive sentiment. You might consider expanding investments."
    #     elif sentiment == "neutral":
    #         sentiment_advice = "The sentiment is neutral. Keep monitoring the market and adjust accordingly."
    #     else:
    #         sentiment_advice = "This projection has negative sentiment. Consider re-evaluating your strategy."
        
    #     print(f"Projection Sentiment: {sentiment}, Advice: {sentiment_advice}")
    
    # You can add your quantitative analysis here to generate more specific advice on cost-cutting, revenue increase, etc.



### TEST IF WORKING ###
# query = "How can I improve cash flow over the next 5 years"
# indicies, distances = search_similar_projections(query, k=5)
# advice = generate_sentiment_advice(get_similar_texts(indicies))

# # Print the similar texts
# for idx, advice_item in enumerate(advice):
#     print(f"Advice {idx + 1}: {advice_item}")

# #Example projections
# projections = [
#     {
#     'cash_flow': -5000,  # Negative cash flow
#     'revenue': 25000,    # Revenue
#     'expenses': 30000,   # Expenses
#     'profit_margin': 0.1,  # 10% profit margin
#     'revenue_growth': -0.02,  # Negative growth
#     },
#     {
#     'cash_flow': 10000,  # Positive cash flow
#     'revenue': 40000,    # Revenue
#     'expenses': 25000,   # Expenses
#     'profit_margin': 0.2,  # 20% profit margin
#     'revenue_growth': 0.05,  # Positive growth
#     },
#     # Add more projections here as needed
# ]
# financial_data = projections

# query_embedding = frontEndLLM.get_embedding("How can I improve cash flow over the next 5 years")

# # Generate advice based on similar projections found by FAISS and sentiment analysis
# generate_advice_from_similar_projections(query_embedding)




# def load_projection_data(idx):
#     '''
#     Load projection data (you need to define this based on your data storage)
#     '''
#     # Placeholder for actual projection loading logic (e.g., from a database or CSV)
#     projection_data = {
#         'cash_flow': random.uniform(-10000, 10000),  # Example data
#         'revenue_growth': random.uniform(5, 20),     # Example data
#         'expenses': random.uniform(5000, 20000)      # Example data
#     }
#     return projection_data

# def retrieve_similar_projections(indices):
#     '''
#     Retrieve similar projections from the dataset using FAISS indices
#     Data must be a list
#     '''
#     # Assuming the projections are stored in some data structure (list, database, etc.)
#     projections = []
#     for i in indices[0]:
#         projection = load_projection_data(i)  # Load data from your projections dataset (modify as needed)
#         projections.append(projection)
#     return projections

# def generate_advice_based_on_finbert(projections):
#     '''
#     Generate actionable financial advice using FinBERT
#     This function analyzes each projection and generates insights
#     '''
#     advice = "Here are some strategies based on historical projections:\n"
    
#     for projection in projections:
#         # Use FinBERT to analyze the sentiment or financial characteristics of the projection
#         projection_text = f"Cash flow: {projection['cash_flow']}, Revenue growth: {projection['revenue_growth']}, Expenses: {projection['expenses']}"
        
#         sentiment = analyze_projection_sentiment(projection_text)
        
#         # Based on the sentiment and analysis, generate advice
#         if sentiment == "negative":
#             advice += "Consider cutting down on operational costs and focusing on high-margin products.\n"
#         elif sentiment == "positive":
#             advice += "Consider expanding your marketing efforts and exploring new revenue channels.\n"
#         else:
#             advice += "The projections seem neutral, focus on maintaining steady cash flow and cost control.\n"
    
#     return advice

# def analyze_projection_sentiment(projection_text):
#     '''
#     Use FinBERT to analyze sentiment or classify financial projections
#     '''
#     tokens = tokenizer(projection_text, padding=True, truncation=True, return_tensors="pt", max_length=512)

#     with torch.no_grad():
#         output = model(**tokens)

#     # FinBERT sentiment classification
#     prediction = torch.argmax(output.logits, dim=1).item()  # Get predicted class (0: negative, 1: neutral, 2: positive)

#     # Map prediction to sentiment
#     sentiment_map = {0: "negative", 1: "neutral", 2: "positive"}
#     return sentiment_map[prediction]

# def get_financial_advice(projection_query):
#     '''
#     Search FAISS for similar projections and generate financial advice using FinBERT
#     '''
#     # Step 1: Get similar projections
#     indices, distances = search_similar_projections(projection_query, k=5)

#     # Step 2: Retrieve similar projections based on indices
#     similar_projections = retrieve_similar_projections(indices)

#     # Step 3: Generate advice based on these projections
#     advice = generate_advice_based_on_finbert(similar_projections)
    
#     return advice 