from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings, HuggingFaceEmbeddings, HuggingFaceBgeEmbeddings
from langchain.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/"
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

def create_vector_db():

    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=128)
    texts = text_splitter.split_documents(documents)

    #embeddings = HuggingFaceInstructEmbeddings(
    #    model_name='hkunlp/instructor-xl')

    #embeddings = HuggingFaceEmbeddings(
    #    model_name='sentence-transformers/all-MiniLM-L6-v2',
    #    model_kwargs = {'device': 'cpu'}
    #)

    embeddings = HuggingFaceEmbeddings(
        model_name='thenlper/gte-large', #nreimers/MiniLM-L6-H384-uncased, thenlper/gte-base,small,large
        model_kwargs = {'device': 'cpu'} #microsoft/MiniLM-L12-H384-uncased
    )

    #model_name = "BAAI/bge-large-en" #small
    #model_kwargs = {'device': 'cpu'}
    #encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    #embeddings = HuggingFaceBgeEmbeddings(
    #    model_name=model_name,
    #    model_kwargs=model_kwargs,
    #    encode_kwargs=encode_kwargs
    #)
    
    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()