from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_openai.embeddings import OpenAIEmbeddings
#from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_chroma import Chroma
from uuid import uuid4

#data and db configuration
DATA_PATH=r"data"
CHROMA_PATH=r"chroma_db"

#embeddings model
embeddings_model=SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

#db initialization
vector_store= Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

#load doc
loader= PyPDFDirectoryLoader(DATA_PATH)
docs = loader.load()

#split the doc
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(docs)

#creating and storing vectors
uuids = [str(uuid4()) for _ in range(len(chunks))]

vector_store.add_documents(documents=chunks, ids=uuids)
