from transformers import AutoTokenizer, AutoModel
import torch
import faiss
import numpy as np
import pandas as pd
import pdf_load

MODEL_NAME = "ProsusAI/finbert" #replace with if any other model used later
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def getEmbeddings(texts):
    '''
    takes in text, tokenizes them, generates FinBert embeddings
    '''
    embeddings_list = []
    batch_size = 8 #process texts in chunks of 8 for memory

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]

        tokens = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=512) #generate tokens

        with torch.no_grad():
            outputs = model(**tokens) #pass tokens in model to get embeddings

        cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy() #extract CLS, first token of each sentence, which represents entire sentence
        embeddings_list.extend(cls_embeddings)

        if i % 100 == 0: #print statement to verify
            print(f"Processed {i+1}/{len(texts)} texts")

    return np.array(embeddings_list)

def saveEmbeddingsFaiss():
    '''
    process pdf documents, generate embeddings using external function, and feed those generated vector embeddings into 
    faiss vector database
    '''
    texts = [doc.page_content for doc in pdf_load.loadPDF()] #get text
    embeddings = getEmbeddings(texts)

    # Create FAISS index
    dimention = embeddings.shape[1] #dimension = length of vector

    index = faiss.IndexFlatL2(dimention) #use L2 Eucdlian distance to get all points between vectors
    faiss.normalize_L2(embeddings) #search is based on angle between vectors (not magnitudes) - might be useful, if not, can remove no harm
    index.add(embeddings)  #add embeddings to index

    # Save FAISS index
    faiss.write_index(index, "rag_index/faiss_db.idx")
    print("Embeddings saved to FAISS index!")

def visualizeIndex():
    '''
    visualize the faiss index database to get a better idea of whats happening
    save database to dataframe, export to retain knowledge
    '''
    index = faiss.read_index("faiss_db.idx")
    texts = [doc.page_content for doc in pdf_load.loadPDF()] #reload the texts for comparison

    stored_embeddings = np.array([index.reconstruct(i) for i in range(index.ntotal)]) #store embeddings in array

    df = pd.DataFrame(stored_embeddings)

    df.insert(0, "Text", texts) #put original texts in data frame

    df.to_csv('rag_index/visualizedIndex.csv')

    print(df.head()) #print first few rows

if __name__ == "__main__":
    saveEmbeddingsFaiss()
    visualizeIndex()