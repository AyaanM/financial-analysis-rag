from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy
#from flask import Flask, request, jsonify #only for web server llm type stuff, not needed right now
import torch #operations to handle model
from src import rag_create

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

def getEmbeddings():
    text = rag_create.loadPDF()
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512) #convert to tokens

    with torch.no_grad(): #disable gradient calculation for less computation usage
        outputs = model(**tokens) #feeds tokens into model

    return outputs.last_hidden_state[:, 0, :].squeeze().tolist() #
