# Financial Analysis Rag

### Author: Ayaan Merchant

# The Project
GoProjections, throught their website takes in 60-months worth of business projections from users. These projections are then meant to be fed in, through a CSV format to the "AI Business Analyzer" platform. 

Working with the data, the plaform then generates insights to identify financial anaomlies, outliers, and more. This insight is used to provide financial advise to the user, and ask any questions.

# The Platform
The GoProjections website provides this plaform with **pre-cleaned** data, and 60-month worth of projections (context) for each of the following:
- Revenue
- Expenses
- Profits
- Cashflow
- Capital Needed

The platform must generate actionable insights after taking in financial projections from user.

This default program uses FinBerg to analyze the sentiment of financial projections, and uses Flan-t5 LLM to generate and output. More information about these models and why we chose this route is under [implemntationinfo.md](/implementationInfo.md).

# How to work this model
- The vector database is too large to push through Github or any other remote server, perhaps AWS/Azure might work
- To run this model you will have to train it on your computer in order to get the vector database

# (Re)training the Model (time taken by all the steps depend heavily on GPU specs)
- Any data must be stored under the `/data` folder
- Currently the model can only be trained on pdfs, to change and process files of other format, edit `index_creation/pdf_load.py`
- Vector database was created using FinBerg, if you wish to use another model, go to `index_creation/index_create.py` or  `index_creation/collab_rag.ipynb` and change the `MODEL_NAME` variable to a model of your choice

## Option 1: Train the model locally
- Open up this Github repository in any IDE
- Run `pip install -r requirements.txt` on your terminal
- Run `python3 index_creation/index_create.py` on your terminal
## Option 2: Train the model using Github codespaces (cloud)
- On the github repository for this project open up Github Codespaces for main
- Run `pip install -r requirements.txt` on the Codespace terminal
- Run `python3 index_creation/index_create.py` on the Codespace terminal
## Option 3: Train the model using Google Colab (cloud)
- Download `index_creation/collab_rag.ipynb` on your computer
- Upload that file to Google Collab
- On the top menu click ~Runtime -> Run All~
## Option 4: Train the model using a provider such as Microsoft Azure/AWS
- This may be necessary if more data needs to be processed
- Can get expesive really fast so I didn't test this option
## Upon Training
- The vector database, and a visualization dataframe will be stored under `rag_index`
- If training takes too long or shows `Terminated`, try the following steps:
  - Training on a different computer 
  - Training on differnt Google Collab Runtime
  - Reducing the `batch_size` variable in `python3 index_creation/index_create.py` or `index_creation/collab_rag.ipynb`
  - Reducing/Cleaning up the data files under `/data` manually or through python
- You can push any changes to Github as required, the index database will be skipped as it can't be pushed to github
- To uninstall any libries, run `pip uninstall -r requirements.txt` on your/Codespace terminal
  - this isn't reccomended as you'll need these libraries to use the model

# Running The Model (time taken havily depends on GPU specs)
- Ensure inputed financial projections is titled `financial_projections.csv`
- Go to `main.py` and run 
  - way to run python script depends on IDE of choice

## Changing the prompt
- Go to `backEndLLM.py` and change the `prompt` variable
- The "###" must be kept there so the model adhers to our rules
- Ensure nothing from "### Inputs:" is deleated, can add things if needed
  - This rule encapsulates our financial projections, sentiment, and original data
## Changing the sentiment analysis model
- Go to `main.py` and change the `MODEL_NAME` variable to the model of your choice
- Go to `backEndLLM.py` and change the `MODEL_NAME_SENTIMENT` variable to the model of your choice
- Check [huggingface](https://huggingface.co) to see what libraries need to be imported
- NOTE: these 2 models must be the same as each other, and the same as what was used to train the vector database
## Changing the LLM Model
- Go to `backEndLLM.py` and change the `MODEL_NAME_LLM` varaiable to a model of your choice
- Check [huggingface](https://huggingface.co) to see what libraries need to be imported
- NOTE: It is reccomended that you use a finance based LLM, but this might require more time and a greater GPU

# Gibberish Output Problem
- Currently, when `main.py` is run using the `financial_projections.csv` in this repository, we get no usable outputs
- The LLM will return outputs that are basically restating our input
- This is becasue the data that is inputed is useless to our model as its not numeric or quantitative
- In order to change this data, replace files in `data` folder and retrain the model
### Image of current output (with problem)
![Image of current output (with problem)][img/problamatic.png])

### Image when we use custom data and manipulate prompt (shows the type of output we desire)
![Image of current output (with problem)][img/proper.png])


