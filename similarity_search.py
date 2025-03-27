import random, torch, faiss, numpy as np, pandas as pd
from index_creation import pdf_load
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

#### GET ALL NEEDED EXTRNAL DOCUMENTS ####
def load_faiss_index():
    '''
    load faiss index
    '''
    index = faiss.read_index("rag_index/faiss_db.idx")
    print("FAISS index loaded successfully!")
    return index 

def get_similar_texts(indices):
    '''
    Reload all pds, based on indicies of the closest vectors according to faiss idx, generate similar texts
    '''
    documents = pdf_load.loadPDF()  # Function to load your dataset
    
    similar_texts = [documents[idx] for idx in indices[0]]  # Indices are in the first array
    
    return similar_texts

#### EMBED QUERIES
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

#### STEPS FOR FINBERT ANALYSIS
def analyze_sentiment(text):
    '''
    Take in the text, pass it through FinBERT, and return the sentiment.
    '''
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    with torch.no_grad():
        outputs = model(**inputs)

    prediction = torch.argmax(outputs.logits, dim=1).item()  # 0: negative, 1: neutral, 2: positive

    # Return a sentiment description
    sentiments = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiments[prediction]
    
def generate_advice_based_on_sentiment(similar_texts):
    '''
    Analyze sentiment of similar texts and generate advice based on it.
    '''
    # Ensure that each item in similar_texts is a string
    for doc in similar_texts:
        if isinstance(doc, str):  # If already a string, process it
            text = doc
        elif hasattr(doc, 'page_content'):  # If it's a Document object, extract text
            text = doc.page_content
        else:
            raise ValueError(f"Unexpected type in similar_texts: {type(doc)}")
        
        sentiment = analyze_sentiment(text)  # Analyze sentiment of the extracted text
        
        print(f"Sentiment: {sentiment}")
        # Based on sentiment, generate advice (you can expand this logic)
        if sentiment == "positive":
            advice = "Consider expanding your investment."
        elif sentiment == "neutral":
            advice = "Keep monitoring market conditions."
        else:
            advice = "Reevaluate your strategy and reduce risks."
        
        print(f"Advice: {advice}")



### TEST IF WORKING ###
query = "How can I improve cash flow over the next 5 years"
indicies, distances = search_similar_projections(query, k=5)
advice = generate_advice_based_on_sentiment(get_similar_texts(indicies))

# Print the similar texts
for idx, advice_item in enumerate(advice):
    print(f"Advice {idx + 1}: {advice_item}")



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