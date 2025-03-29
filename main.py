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

def generate_advice(sentiment, financial_data):
    advice = []
    
    # Example financial data
    revenue = financial_data['revenue']
    expenses = financial_data['expenses']
    profits = financial_data['profits']
    profit_margin = profits / revenue
    cash_flow = financial_data['cashflow']
    capital_needed = financial_data['capital_need']
    
    if sentiment == "positive":
        print("hi")
        # if profit_margin > 0.2:
        #     advice.append("The company is in a strong position with high profit margins. Consider reinvesting in marketing or R&D for further growth.")
        # if revenue > 100000:
        #     advice.append("With strong revenue, you should consider exploring new business opportunities or geographic expansion.")
    elif sentiment == "neutral":
        print("days")
        # if expenses are increasing but revenue is steady, suggest focusing on **cost optimization**:
        #     advice.append("Revenue is steady, but costs are creeping up. Review your operational efficiency and renegotiate supplier contracts.")
        # if cash_flow is low:
        #     advice.append("Cash flow is low; focus on improving receivables or deferring capital expenditures.")
    elif sentiment == "negative":
        print("low")
        # if profit_margin < 0.1:
        #     advice.append("Profit margins are below expectations. Consider revisiting your pricing strategy and cutting non-essential expenses.")
        # if cash_flow is negative:
        #     advice.append("Negative cash flow is a concern. You should focus on reducing operational costs and improving liquidity.")
    
    return advice

def main(csv_path):
    '''
    load projections, and for each
    '''
    projections = load_projections(csv_path)
    
    for projection in projections:
        embedding = get_embedding(projection)
        sentiment = backEndLLM.sentiment_helper(embedding)

        advice = generate_advice(sentiment, projection)
        print(f"Advice for Month {projection['month']}:\n{advice}\n")

if __name__ == "__main__":
    main("financial_projections.csv")

