import random, torch, faiss, numpy as np, pandas as pd
from index_creation import pdf_load
import faiss
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import backEndLLM

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

#### EMBED QUERIES
def get_embedding(finiancial_projections):
    '''
    Take in query from user and convert to embeddings for FAISS search
    '''
    projection_dict = (
        f"Cash flow: {finiancial_projections['cash_flow']}, "
        f"Revenue: {finiancial_projections['revenue']}, "
        f"Expenses: {finiancial_projections['expenses']}, "
        f"Profit margin: {finiancial_projections['profit_margin']}, "
        f"Revenue growth: {finiancial_projections['revenue_growth']}"
    )
    tokens = tokenizer(projection_dict, padding=True, truncation=True, return_tensors="pt", max_length=512)

    with torch.no_grad():
        output = model(**tokens)

    cls_embedding = output.last_hidden_state[:, 0, :].cpu().numpy()
    return cls_embedding 

#TESTS
financial_data = {
    'cash_flow': -5000,  # Negative cash flow
    'revenue': 25000,    # Revenue
    'expenses': 30000,   # Expenses
    'profit_margin': 0.1,  # 10% profit margin
    'revenue_growth': -0.02,  # Negative growth
}

query_embeddings = get_embedding(financial_data)
backEndLLM.generate_advice_from_similar_projections(query_embeddings)

