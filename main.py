import pandas as pd
import backEndLLM
from transformers import AutoTokenizer, AutoModel

MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def load_projections(csv_path):
    '''
    Load financial projections from CSV into a dataframe.
    '''
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

def main(csv_path):
    '''
    Load projections, feed into backend, generate advice for each projection.
    '''
    projections = load_projections(csv_path)
    
    for projection in projections:

        advice = backEndLLM.backEnd_Helper(projection)
        print(f"Advice for Month {projection['month']}:\n{advice}\n")

if __name__ == "__main__":
    main("financial_projections.csv")

