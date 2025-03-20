import os
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def open_financial_projections():
    '''
    opens finacial projections csv, formats as needed, adds to database
    '''
    path_finances = 'data/finances/' #finances folder with document statements

    for filename in os.listdir(path_finances):
        file_path = os.path.join(path_finances, filename)

        csv = CSVLoader(file_path=file_path).load()

        #split docs
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        split_docs = text_splitter.split_documents(csv)

        print(split_docs[0])  #preview of doc

open_financial_projections()