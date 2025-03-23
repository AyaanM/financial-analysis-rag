from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from flask import Flask, request, jsonify #only for web server llm type stuff, not needed right now
import torch #operations to handle model no gradient
import faiss
import numpy as np
from src import pdf_load

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def getEmbeddings():
    texts = pdf_load.loadPDF()

    embeddings_list = [] 
    
    for text in texts: #loop thru each pdf, create tokens individually
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512) #convert to tokens

        with torch.no_grad(): #disable gradient calculation for less computation usage
            outputs = model(**tokens) #feeds tokens into model and create vectors

        embeddings_list.append(outputs.last_hidden_state[:, 0, :].squeeze().tolist()) #[:, 0, :] for CLS (classification token prepend)
    
    return embeddings_list

if __name__ == "__main__":
    getEmbeddings.run(host="0.0.0.0", port=8080)