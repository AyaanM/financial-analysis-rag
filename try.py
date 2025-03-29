'''
BackEnd of The LLM
Various functions from the frontend are triggered
'''
import random, torch, faiss, numpy as np, pandas as pd
from index_creation import pdf_load
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoModelForCausalLM, AutoModelForSeq2SeqLM

# Load FinBERT for embeddings & sentiment
MODEL_NAME_FINBERT = "ProsusAI/finbert" #change if wanting to use any other sentiment model
tokenizer_finbert = AutoTokenizer.from_pretrained(MODEL_NAME_FINBERT)
model_finbert = AutoModel.from_pretrained(MODEL_NAME_FINBERT)
model_sentiment = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME_FINBERT)

# Load FLAN-T5 for advice generation
MODEL_NAME_T5 = "google/flan-t5-large" #change if wanting to use any other text gen
tokenizer_t5 = AutoTokenizer.from_pretrained(MODEL_NAME_T5)
model_t5 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME_T5)

### ========== FAISS INDEX SEARCH ========== ###
def search_index(financial_embedding, k=5):
    '''
    Search FAISS index for top-k similar projections (default is 5).
    '''
    index = faiss.read_index("rag_index/faiss_db.idx")
    faiss.normalize_L2(financial_embedding)  # Normalize if cosine similarity is used
    print(f"Index loaded. Number of vectors: {index.ntotal}")
    distances, indices = index.search(np.array(financial_embedding).reshape(1, -1), k)
    print(f"Retrieved indices: {indices}, Distances: {distances}")
    return indices, distances

def search_data(indices):
    '''
    Look at where indices lie in text and return matching documents.
    '''
    text = pdf_load.loadPDF()
    similar_texts = []

    for idx in indices[0]:
        projection = text[idx]
        similar_texts.append(projection)

    random.shuffle(similar_texts)

    combined_context = " ".join([doc.page_content for doc in similar_texts])
    return combined_context


### ========== FINBERT EMBEDDING + SENTIMENT ANALYSIS ========== ###
def get_embedding(projection):
    '''
    Get embeddings of financial projections using FinBERT.
    '''
    projection_dict = (
        f"Month: {projection['month']}, "
        f"Revenue: {projection['revenue']}, "
        f"Expenses: {projection['expenses']}, "
        f"Profits: {projection['profits']}, "
        f"Cashflow: {projection['cashflow']}, "
        f"Capital Needed: {projection['capital_need']}"
    )
    tokens = tokenizer_finbert(projection_dict, padding=True, truncation=True, return_tensors="pt", max_length=512)
    with torch.no_grad():
        output = model_finbert(**tokens)
    cls_embedding = output.last_hidden_state.mean(dim=1).cpu().numpy()
    return cls_embedding

def analyze_sentiment(text):
    '''
    Analyze sentiment with FinBERT.
    '''
    inputs = tokenizer_finbert(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model_sentiment(**inputs)
    prediction = torch.argmax(outputs.logits, dim=1).item()
    sentiments = {0: "negative", 1: "neutral", 2: "positive"}
    return sentiments[prediction]

def sentiment_helper(financial_embedding):
    '''
    Search FAISS and run sentiment analysis on retrieved documents.
    '''
    indices, distances = search_index(financial_embedding, k=5)
    similar_projections = search_data(indices)
    sentiment = analyze_sentiment(similar_projections)
    return sentiment, similar_projections


### ========== GENERATE DYNAMIC FINANCIAL ADVICE ========== ###
def generate_advice(sentiment, financial_data, context):
    '''
    Generate actionable advice using Flan-T5 with contextual information.
    '''
    
    # Build prompt with context and sentiment - Updated and Fixed
    prompt = f"""
You are a highly experienced financial analyst with expertise in optimizing financial performance.
The company's financial projection data and sentiment are provided below:

Financial Projection:
- Revenue: ${financial_data['revenue']}
- Expenses: ${financial_data['expenses']}
- Profits: ${financial_data['profits']}
- Cashflow: {financial_data['cashflow']}
- Capital Needed: ${financial_data['capital_need']}

Sentiment Analysis: {sentiment}

### Contextual Insights from Similar Companies:
{context}

### Task:
Based on this data, generate 3-4 actionable, data-driven, and quantifiable recommendations that aim to:
1. Increase revenue or reduce expenses by at least 10%.
2. Optimize cash flow to maintain a positive balance.
3. Improve capital efficiency and ROI.

### Output Format:
1. Recommendation #1: [Specific action with numeric target]\n
2. Recommendation #2: [Specific action with numeric target]\n
3. Recommendation #3: [Specific action with numeric target]\n
    
Your recommendations should be based on industry best practices, financial trends, and the provided data.
"""
    print(f"This is the Prompt that was fed into {MODEL_NAME_T5}")
    print(prompt)

    # Tokenize and generate advice
    inputs = tokenizer_t5(prompt, return_tensors="pt", max_length=512, truncation=True)
    output = model_t5.generate(**inputs, max_length=200, num_return_sequences=1, early_stopping=True)
    advice = tokenizer_t5.decode(output[0], skip_special_tokens=True)
    
    return advice


### ========== MAIN FUNCTIONS ========== ###
def load_projections(csv_path):
    '''
    Load financial projections from CSV into a dataframe.
    '''
    df = pd.read_csv(csv_path)
    return df.to_dict(orient="records")

def main(csv_path):
    '''
    Load projections and analyze + generate advice for each projection.
    '''
    projections = load_projections(csv_path)
    
    for projection in projections:
        # Get embedding and search FAISS
        embedding = get_embedding(projection)
        sentiment, context = sentiment_helper(embedding)

        # Generate detailed financial advice
        advice = generate_advice(sentiment, projection, context)
        print(f"Advice for Month {projection['month']}:\n{advice}\n")

if __name__ == "__main__":
    main("financial_projections.csv")
