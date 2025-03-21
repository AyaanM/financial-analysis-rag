import os, re
from langchain_community.document_loaders import CSVLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def loadPDF():
    '''
    opens pdfs, and format as needed
    '''
    data_path = 'data/' #finances folder with document statements
    all_docs = []
    pages_num = 0
    total_token_count = 0
    total_char_count = 0

    """ for filename in os.listdir(data_path):
        if filename.endswith('.pdf'):
            file_path = os.path.join(data_path, filename)

            loader = PyPDFLoader(file_path)
            doc = loader.load()

            for page in doc:
                page.page_content = format_doc(page.page_content)

            # Split and add PDF documents
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=25)
            split_docs = text_splitter.split_documents(pdf_docs)
            all_docs.extend(split_docs)

            print(f"-------- PDF INFO for {filename}.pdf --------")
            print(f'''Number of Pages in PDF: {pages_num}
Number of Characters in PDF: {total_char_count}
Number of Tokens in PDF: {total_token_count}''') """

    ### TESTING SINCE MANY DOCS TAKE TOO LONG
    file_path = os.path.join(data_path, "Building Financial Models (John Tjia) (Z-Library).pdf")

    loader = PyPDFLoader(file_path)
    doc = loader.load()

    for page in doc:
        page.page_content = format_doc(page.page_content)
        pages_num += 1
        total_char_count += len(page.page_content)
        total_token_count += len(page.page_content) / 4 #since 1 token is 4 chars
        
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=25)
    split_docs = text_splitter.split_documents(doc)

    all_docs.extend(split_docs)

    print("-------- PDF INFO for Building Financial Models (John Tjia) (Z-Library).pdf --------")
    print(f'''Number of Pages in PDF: {pages_num}
Number of Characters in PDF: {total_char_count}
Number of Tokens in PDF: {total_token_count}''') #122824.75 tokens, cost = (122824.75 / 1,000) * 0.0001 = 0.012282475 

    return all_docs

def format_doc(doc):
    '''
    format each page of pdf for easier reading
    '''
    re.sub(r'\n+', '\n', doc) ##replace multiple newlines with single
    re.sub(r'\s+', ' ', doc) ##replace multiple spaces with single space
    re.sub(r'[^\x00-\x7F]+', ' ', doc)  # Remove non-ASCII chars - assume they can't be processed by text
    doc = doc.strip()
    return doc

#processData()
loadPDF()

