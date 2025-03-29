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
def get_embedding(financial_projections):
    projection_dict = (
        f"Revenue: {financial_projections['revenue']}, "
        f"Expenses: {financial_projections['expenses']}, "
        f"Profits: {financial_projections['profits']}, "
        f"Cashflow: {financial_projections['cashflow']}, "
        f"Capital Needed: {financial_projections['captial_needed']}"
    )
    tokens = tokenizer(projection_dict, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**tokens)

    
    cls_embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()
    return cls_embedding

#TESTS
financial_data = {
    'revenue': 25000,
    'expenses': 1000, 
    'profits': 30000,  
    'cashflow': 0.1, 
    'capital_needed': -0.02
}

query_embeddings = get_embedding(financial_data)
advice = backEndLLM.generate_advice_from_similar_projections(query_embeddings)

print(advice)

