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

# Steps (for vector database creation)
1) Loads in pdf texts from `data`
2) Texts are processed in batches for faster process, variable `batch_size` in `index_creation/index_create.py`
3) Texts are passed through a tokenizer to create tokens
4) Tokens are passed through the sentiment model to create embeddings
- NOTE: these are the same steps to create any embeddings throughout the program

# Steps (Post vector database creation)
## 1) Takes financial projections as input
- Assume input is a csv file and convert into dataframe projections
- Convert projections per month into embeddings through sentiment model
## 2) Searches embeddings in finacial database to generate indicies and distances
## 3) Conducts sentiment analysis through indicies (and why?)
- Outputs `Positive`, `Negative`, or `Neutral` based on input (financial projections)
### why?
- Helps with actionable financial advice
- Indentifies financial strength/stress
- Forecasts risks/recommendations
## LLM text advice generation
- Combines input, sentiment, and untokenized data into single prompt
- Feeds the prompt into LLM to generate advice

# Python Libraries Used
- Downloadable using `requirements.txt`
## Langchain: LLM Interaction/faiss-cpu
- CSV: load and open CSV
- FAISS (Facebook AI Similarity Search): Similarity searching and clustering of vectors
    need to retrieve data with vector embeddings, allows to search for meaning and similarities
    data is unstructured, need to find similar vectors in vector space
## Torch: for gradient descent and performing math funcs
## Pypdf: for pdf processing
## Numpy/Pandas: array and dataframes

# Embedding Model(s)
- Huggingface is a website for building and deploying Machine Learning models
- The models currently used in this program are accessed through Huggingface
## OpenAI model (not used)
- Can be imported using `text-embedding-ada-002`
- Cost: ~0.0001 cents per token
- Can be used for LLM
- Reccomended more than what we're currenly using - Flan-t5
- Not very costly, only need OpenAI API key
## Finbert (Used for Sentiment Analysis and Vector Database Creation)
- [Finbert HuggingFace Link](https://huggingface.co/ProsusAI/finbert)
- Can be imported using `ProsusAI/finbert`
- Needs libraries from `tranformers` such as `AutoTokenizer`, `AutoModelForSequenceClassification` and `AutoModel`
- Open Source and free
## Flan-t5 (used as LLM)
- [Flan-t5 HuggingFace Link](https://huggingface.co/google/flan-t5-large)
- Can be imported using `google/flan-t5-large`
- Needs libraries from `tranformers` such as `AutoTokenizer` and `AutoModelForSeq2SeqLM`
- Open Source and free, fine tuned on 1000 tasks
- Not finance specific but gets the job done
## Llama2-7b-Finance (not used, reccomended for future)
- [lama2-7b-Finance HuggingFace Link](https://huggingface.co/cxllin/Llama2-7b-Finance)
- Can be imported using `cxllin/Llama2-7b-Finance`
- Needs libraries from `tranformers` such as `AutoTokenizer` and `AutoModelForCausalLM`
- Open Source and free, generates actionable powerful text for financial domain
- Requires lots of GPU resources and takes along time to generate text




