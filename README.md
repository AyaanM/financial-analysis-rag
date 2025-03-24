# Financial Analysis Rag

#### Author: Ayaan Merchant

# The Project
GoProjections, throught their website takes in 60-months worth of business projections from users. These projections are then meant to be fed in, through a CSV format to the "AI Business Analyzer" platform. 

Working with the data, the plaform then generates insights to identify financial anaomlies, outliers, and more. This insight is used to provide financial advise to the user, and ask any questions.

# The Platform
The GoProjections website provides this plaform with **pre-cleaned** data, and 60-month worth of projections (context) for each of the following:
- Revenue
- Expenses
- Profits
- Cashflow

The platform must generate best case, nomral case, and worst case scenarios using RAG, with stored documents. This RAG system retrives documents, and compares it with the context to generate insights.

Risk analysis may be performed. Beyond this, an LLM will be able to answer user queries by looking at the insights generated.

# How to work this model
- The FIASS vector database has already been generated - its stored in a file called faiss_db.idx
- This data can also be visualized through a pandas dataframe called visualizedIndex.csv

### If you wish to retrain the model, follow the steps below
    - Open this project up in the Google Codespace from Main
      - While thats what was done to intially train the model, an alternative could be to do it more locally on a PC with a powerful GPU
    - Run the following command ~pip install -r requirements.txt~
      - This will install all the necessary libraries to 
