from transformers import AutoTokenizer, AutoModelForSequenceClassification
#from flask import Flask, request, jsonify #only for web server llm type stuff, not needed right now
import torch #operations to handle model no gradient
import faiss
from langchain.vectorstores import FAISS
import pandas as pd #to visualize vector db better
import numpy as np
from src import pdf_load

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def getEmbeddings(): 
    '''
    Embedding basically captures sentence similarity meaning, no matching by word
    Comverts text into tokens thru tokenizer
    Sends those tokens in the embedding model to create vectors
    '''
    texts = pdf_load.loadPDF()

    embeddings_list = [] 
    
    for text in texts: #loop thru each pdf, create tokens individually
        tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512) #convert to tokens

        with torch.no_grad(): #disable gradient calculation for less computation usage
            outputs = model(**tokens) #feeds tokens into model and create vectors

        embeddings_list.append(outputs.last_hidden_state[:, 0, :].squeeze().tolist()) #[:, 0, :] for CLS (classification token prepend)
    
    return embeddings_list

def saveEmbeddingsFaiss(embeddings_list):
    '''
    save embeddings to fasiss vector database for similarity search, use elucidean L2 distance
    '''
    embeddings_array = np.array(embeddings_list).astype('float32')

    index = faiss.IndexFlatL2(embeddings_array.shape[1]) #eucildean distance search (measures straight line between points for similarity search)
    index.add(embeddings_array)

if __name__ == "__main__":
    embeddings = getEmbeddings() 
    faiss_db = saveEmbeddingsFaiss(embeddings)

    faiss.write_index(faiss_db, 'faiss_db.idx') #save fiass db as index for future reference

    #test if it actually works
    print("faiss_db.idx")
    print("Embeddings saved to FAISS index!")