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
- The model is too large to push through Github or any other remote server, perhaps AWS/Azure might work
- To run this model you will have to train it on your computer in order to get the vector database

## Step 1: (Re)training the Model (time taken by all the steps depend heavily on GPU specs)
  - Any data must be stored under the ~/data~ folder
  - Currently the model can only be trained on pdfs, to change and process files of other format, edit ~src/pdf_load.py~
  ### Option 1: Train the model locally
    - Open up this Github repository in any IDE
    - Run ~pip install -r requirements.txt~ on your terminal
    - Run ~python3 src/app.py~ on your terminal
  ### Option 2: Train the model using Github codespaces (cloud)
    - On the github repository for this project open up Github Codespaces for main
    - Run ~pip install -r requirements.txt~ on the Codespace terminal
    - Run ~python3 src/app.py~ on the Codespace terminal
  ### Option 3: Train the model using Google Colab (cloud)
    - Download ~src/collab_rag.ipynb~ on your computer
    - Upload that file to Google Collab
    - On the top menu click ~Runtime -> Run All~
  ### Option 4: Train the model using a provider such as Microsoft Azure/AWS
    - This may be necessary if more data needs to be processed
    - Can get expesive really fast so I didn't test this option
  ### Upon Training
    - The vector database, and a visualization dataframe will be stored under ~rag_index~
    - If training takes too long or shows ~Terminated~, try the following steps
      - Training on a different computer 
      - Training on differnt Google Collab Runtime
      - Reducing the ~batch_size~ variable in ~src/app.py~ or ~scr/collab_rag.ipyn~
      - Reducing/Cleaning up the data files under ~/data~
    - You can push any changes to Github as required, the index database will be skipped as it can't be pushed to github
    - To uninstall any libries, run ~pip uninstall -r requirements.txt~ on your/Codespace terminal