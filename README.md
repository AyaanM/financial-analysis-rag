#Financial Analysis Rag

####Author: Ayaan Merchant

##The Project
GoProjections, throught their website takes in 60-months worth of business projections from users. These projections are then meant to be fed in, through a CSV format to the "AI Business Analyzer" platform. 

Working with the data, the plaform then generates insights to identify financial anaomlies, outliers, and more. This insight is used to provide financial advise to the user, and ask any questions.

##The Platform
The GoProjections website provides this plaform with **pre-cleaned** data, and 60-month worth of projections (context) for each of the following:
- Revenue
- Expenses
- Profits
- Cashflow

The platform must generate best case, nomral case, and worst case scenarios using RAG, with stored documents. This RAG system retrives documents, and compares it with the context to generate insights.

Risk analysis may be performed. Beyond this, an LLM will be able to answer user queries by looking at the insights generated.

###RAG
Retrieval augemented generation can take info, pass it to LLM and generate outputs based on info.
- Retrieval - retrieves passages of text related to our query
- Augementation - take relevant information and augment prompt to LLM with that relevant information
- Generation - Pass information into LLM for generative outputs

Using RAG in approach also:
- Prevents hallucinations: give factual information
- Creates specific responses based on specific documents

This RAG will not be run locally, as thats very computationally expensive, the time/computation required will increase with the number of documents.

###Risk with this system
- If projections are unique/innovative (ex. industry disturbing), this might lead to inaccurate predictions
- Overfitting: If provided dataset is too narrow/biased, data might be overfitted and skewed

