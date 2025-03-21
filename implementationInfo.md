# Why RAG
Retrieval augemented generation can take info, pass it to LLM and generate outputs based on info.
- Retrieval - retrieves passages of text related to our query
- Augementation - take relevant information and augment prompt to LLM with that relevant information
- Generation - Pass information into LLM for generative outputs

Using RAG in approach also:
- Prevents hallucinations: give factual information
- Creates specific responses based on specific documents

This RAG will not be run locally, as thats very computationally expensive, the time/computation required will increase with the number of documents.

### Risk with this system
- If projections are unique/innovative (ex. industry disturbing), this might lead to inaccurate predictions
- Overfitting: If provided dataset is too narrow/biased, data might be overfitted and skewed

# Techstack

## Python Libraries

### Langchain: interface for LLM interaction
- CSV: load and open CSV
- FAISS (Facebook AI Similarity Search): Similarity searching and clustering of vectors
    need to retrieve data with vector embeddings, allows to search for meaning and similarities
    data is unstructured, need to find similar vectors in vector space

### Embedding Model
- Use of ~text-embedding-ada-002~ initially
  - Cost: ~0.0001 cents per token
  - General embedding model, runs on cloud, fast run, low cost
- For scalabiilty can switch to ~FinBert~ (Open Source Hugging Face)[https://huggingface.co/ProsusAI/finbert]
  - No cost if embeddings are done locally -> no per token costs, but requires alot of GPU resources
  - Fine tuned on financial analysis text




