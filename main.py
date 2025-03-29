import pandas as pd
import torch
import faiss
import numpy as np
from transformers import AutoTokenizer, AutoModel
import backEndLLM

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

# Load projections from CSV
def load_projections(csv_path):
    '''
    load financial projections from csv into dataframe
    '''
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

def get_embedding(projection):
    '''
    get embeddings of all financial projectios
    '''
    projection_dict = (
        f"Month: {projection['month']}," 
        f"Revenue: {projection['revenue']},"
        f"Expenses: {projection['expenses']}," 
        f"Profits: {projection['profits']},"
        f"Cashflow: {projection['cashflow']},"
        f"Capital Needed: {projection['capital_need']},"
    )
    #print(projection_dict)

    tokens = tokenizer(projection_dict, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model(**tokens)

    cls_embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()
    #print(cls_embedding)
    return cls_embedding

def main(csv_path):
    '''
    load projections, and for each
    '''
    projections = load_projections(csv_path)
    
    for projection in projections:
        embedding = get_embedding(projection)
        
        # Generate actionable advice
        advice = backEndLLM.generate_advice_from_similar_projections(embedding)
        print(f"Advice for Month {projection['month']}:\n{advice}\n")

if __name__ == "__main__":
    main("financial_projections.csv")

