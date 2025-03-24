from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from src import pdf_load

# Load FinBERT
MODEL_NAME = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def getFinBertEmbeddings(texts):
    """Generate embeddings from FinBERT"""
    embeddings_list = []
    batch_size = 32  # Adjust based on available resources

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512)

        with torch.no_grad():
            outputs = model(**tokens)  # Forward pass

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # Extract CLS token
        embeddings_list.extend(cls_embeddings)

        if i % 100 == 0:
            print(f"Processed {i+1}/{len(texts)} texts")

    return np.array(embeddings_list)

def saveEmbeddingsFaiss():
    texts = [doc.page_content for doc in pdf_load.loadPDF()]  # Extract text from documents
    embeddings = getFinBertEmbeddings(texts)

    # Create FAISS index
    d = embeddings.shape[1]  # Dimension of embeddings
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)  # Add embeddings to FAISS

    # Save FAISS index
    faiss.write_index(index, "faiss_db.idx")
    print("Embeddings saved to FAISS index!")

if __name__ == "__main__":
    saveEmbeddingsFaiss()