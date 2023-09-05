from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

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

    for bk in lst_books:
        for emb in lst_HFIE:
            DPATH = DATA_PATH + bk
            print(DPATH + "__" + extractEMB(emb) + " is being generated .... \n")
            loader = DirectoryLoader(DPATH, glob='*.pdf', loader_cls=PyPDFLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            texts = text_splitter.split_documents(documents)

            embeddings = HuggingFaceInstructEmbeddings(model_name=emb)

            db = FAISS.from_documents(texts, embeddings)

            DB_PATH = DB_FAISS_PATH + bk + "___" + extractEMB(emb)
            db.save_local(DB_PATH)

        for emb in lst_HFE:
            DPATH = DATA_PATH + bk
            print(DPATH + "__" + extractEMB(emb) + " is being generated .... \n")
            loader = DirectoryLoader(DPATH, glob='*.pdf', loader_cls=PyPDFLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            texts = text_splitter.split_documents(documents)

            embeddings = HuggingFaceEmbeddings(
               model_name=emb,
               model_kwargs = {'device': 'cpu'}
            )

            db = FAISS.from_documents(texts, embeddings)

            DB_PATH = DB_FAISS_PATH + bk + "___" + extractEMB(emb)
            db.save_local(DB_PATH)

        for emb in lst_HFBE:
            DPATH = DATA_PATH + bk
            print(DPATH + "__" + extractEMB(emb) + " is being generated .... \n")
            loader = DirectoryLoader(DPATH, glob='*.pdf', loader_cls=PyPDFLoader)
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
            texts = text_splitter.split_documents(documents)

            model_kwargs = {'device': 'cpu'}
            encode_kwargs = {'normalize_embeddings': True}
            embeddings = HuggingFaceBgeEmbeddings(
                model_name=emb,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )

            db = FAISS.from_documents(texts, embeddings)

            DB_PATH = DB_FAISS_PATH + bk + "___" + extractEMB(emb)
            db.save_local(DB_PATH)



    

if __name__ == '__main__':
    create_vector_db()