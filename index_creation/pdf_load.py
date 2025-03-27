'''
author: Ayaan
This file is to load the pdfs, which will later be tokenized and turned into vectors
please note, to load the file, the filename must end with ".pdf"
'''

import os, re
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def loadPDF():
    '''
    opens pdfs, and format as needed
    '''
    data_path = 'data/' #finances folder with document statements
    all_docs = []

    for filename in os.listdir(data_path):
        #re init at 0 for each pdg
        pages_num = 0
        total_token_count = 0
        total_char_count = 0

        file_path = os.path.join(data_path, filename)

        if filename.endswith('.pdf'): #only if its a pdf - double check to avoid errors
            file_path = os.path.join(data_path, filename)

            loader = PyPDFLoader(file_path)
            doc = loader.load()

            pages_num = len(doc)

            for page in doc:
                page.page_content = format_doc(page.page_content)
                total_char_count += len(page.page_content)
                total_token_count += len(page.page_content) / 4 #since 1 token is 4 chars

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=35) #chunk pdf for better results, overlap to maintain context
            split_docs = text_splitter.split_documents(doc)
            all_docs.extend(split_docs)

            print(f"-------- PDF INFO for {filename} --------")
            print(f'''Number of Pages in PDF: {pages_num}
Number of Characters in PDF: {total_char_count}
Number of Tokens in PDF: {total_token_count}''')

    return all_docs

def format_doc(doc):
    '''
    format each page of pdf for easier reading
    '''
    doc = re.sub(r'\n+', '\n', doc) ##replace multiple newlines with single
    doc = re.sub(r'\s+', ' ', doc) ##replace multiple spaces with single space
    doc = re.sub(r'[^\x00-\x7F]+', ' ', doc)  # Remove non-ASCII chars - assume they can't be processed by text
    doc = doc.strip()
    return doc

