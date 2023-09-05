from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS
from datetime import datetime

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/"
DPATH = ""
DBPATH = ""
lst_books = [
    "aud",
    "sbks",
    "bg",
    "sb",
    "cc", 
    "sbks_bg",
    "sbks_bg_sb",
    "sbks_bg_sb_cc", 
    "sbks_bg_sb_cc_aud",
    "sbks_bg_aud",
    "sbks_aud"
]
lst_HFIE = [
    "hkunlp/instructor-xl",
    "hkunlp/instructor-large",
    "hkunlp/instructor-base"
]
lst_HFE = [
    "sentence-transformers/all-MiniLM-L6-v2",
    "microsoft/MiniLM-L12-H384-uncased",
    "nreimers/MiniLM-L6-H384-uncased",
    "thenlper/gte-base",
    "thenlper/gte-small",
    "thenlper/gte-large"
]
lst_HFBE = [
    "BAAI/bge-large-en",
    "BAAI/bge-large-en"
]

def extractEMB(string):
  last_slash_index = string.rfind("/")
  if last_slash_index == -1:
    return ""
  else:
    return string[last_slash_index + 1:]

def create_vector_db():
    DPATH = DATA_PATH + "sbks_bg_sb_cc_aud"
    print(DPATH + "__" + "hkunlp/instructor-xl" + " is being generated .... \n")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)
    loader = DirectoryLoader(DPATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    texts = text_splitter.split_documents(documents)

    print("loaded all the document ...")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")

    print("creating embeddings and storing in cpu memory of db ...")
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    db = FAISS.from_documents(texts, embeddings)

    DB_PATH = DB_FAISS_PATH + "sbks_bg_sb_cc" + "___" + "instructor-xl"
    db.save_local(DB_PATH)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)



    

if __name__ == '__main__':
    create_vector_db()